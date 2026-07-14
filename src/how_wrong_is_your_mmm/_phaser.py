"""Budget phasing recommender.

The core idea: collinearity comes from all channels tracking the same demand
signal. The fix is to introduce *independent* variation in the weekly channel
mix — some weeks deliberately lean into TV, others into Meta or Search —
while keeping monthly budgets intact.

BudgetPhaser takes:
  - history_df: multi-year spend history (fixed, cannot be changed)
  - plan_df:    the upcoming year's budget (this is what gets phased)

It grid-searches over a phasing amplitude alpha ∈ [0, 1]:

  alpha = 0  →  no change from original plan
  alpha = 1  →  maximum allowed variation under the channel constraint

For each alpha it generates a phased plan schedule (monthly totals preserved per
channel), concatenates it with the history, fits a CollinearityDiagnostic on the
combined dataset, and measures the max CV across channels. The recommended alpha
minimises max CV.

Weighting schemes (uniform / binary / decay) were evaluated in a research study
(session 7) and dropped: uniform weighting always outperformed upweighting the
plan year, so the evaluation is plain OLS on history + phased plan throughout.

The output is a concrete plan-year weekly spend schedule the practitioner can
hand to their media agency, with monthly totals unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import _DEFAULT_ELASTICITIES
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic


def _get_month_labels(spend_df: pd.DataFrame) -> np.ndarray:
    """Return an array of year-month Period labels, one per row in spend_df.

    Parameters
    ----------
    spend_df:
        DataFrame with a DatetimeIndex.

    Returns
    -------
    np.ndarray of pandas Period objects (monthly frequency).
    """
    if not isinstance(spend_df.index, pd.DatetimeIndex):
        raise ValueError(
            "spend_df must have a DatetimeIndex. "
            "Use simulate_spend(start_date=...) or set a DatetimeIndex on your data."
        )
    return spend_df.index.to_period("M").to_numpy()


class Blackout:
    """Marker for blackout-mode phasing on a channel.

    Each week is drawn as either fully blacked out (0% of that week's
    planned spend) or left unchanged (100%), never anything in between.
    This is a different sampling mechanism from a symmetric +/-X range,
    not a special case of one — a channel is either range-based or
    blackout-mode, not both.

    Like a symmetric range, monthly totals are still preserved exactly.
    Unlike a symmetric range, Blackout's deviation shape is skewed towards
    zero whenever prob is high (most weeks dark, a few weeks carrying the
    load), so the rescale needed to hit the monthly total can be large:
    weeks that stay "on" absorb the budget freed up by the weeks that went
    dark and can end up well above their own original plan to compensate.
    This mirrors ordinary media flighting or pulsing (full spend some
    weeks, dark others), so the "on" weeks running hot is an expected
    consequence of that strategy rather than an arbitrary side effect. See
    _generate_phased_schedule for the mechanism. A dark week's spend is
    guaranteed to land at exactly zero, not just close to it — zero times
    any rescale factor is still zero.

    At least one week per month is always kept "on": with nothing left
    "on", there'd be nowhere for the month's budget to land, and the
    channel would silently end up completely untouched instead of blacked
    out.

    By default (max_dark_weeks_per_month=None) each week is an independent
    draw, so a month can land several dark weeks at once — the more weeks
    go dark, the fewer are left to absorb the month's budget, and the
    spike on those survivors gets correspondingly larger (several dark
    weeks in one month can force a single surviving week to several times
    its original plan). Setting max_dark_weeks_per_month caps how many
    weeks any one month can lose, which caps the spike too: with a cap of
    1, at most one week's budget ever needs to be redistributed across the
    rest of that month, so the "on" weeks see a modest, proportional bump
    rather than an extreme one. Recommended whenever the resulting spend
    increase needs to stay plausible for a media buyer to actually deploy.

    Parameters
    ----------
    prob:
        Maximum probability, at alpha=1, that a week (or, if
        max_dark_weeks_per_month is set, a month's blackout slot) is used.
        Default 1.0. Scales linearly with alpha, same as every other
        deviation shape: at alpha=0 the probability is 0 (no blackout,
        matches every other spec's "no change" fixed point).
    max_dark_weeks_per_month:
        Maximum number of weeks any single month may lose to blackout.
        Default None: every week in the month is an independent draw, with
        no cap (the original behaviour) — several weeks in the same month
        can go dark together, and the survivors absorb correspondingly
        more. If set (e.g. 1), each month independently "activates"
        blackout with probability prob (scaled by alpha), and if it does,
        exactly min(max_dark_weeks_per_month, n_weeks - 1) weeks in that
        month are chosen at random to go dark — always leaving at least
        one week "on".
    """

    def __init__(
        self,
        prob: float = 1.0,
        max_dark_weeks_per_month: int | None = None,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"Blackout prob must be between 0 and 1, got {prob}.")
        if max_dark_weeks_per_month is not None and max_dark_weeks_per_month < 1:
            raise ValueError(
                "Blackout max_dark_weeks_per_month must be >= 1, got "
                f"{max_dark_weeks_per_month}."
            )
        self.prob = float(prob)
        self.max_dark_weeks_per_month = max_dark_weeks_per_month

    def __repr__(self) -> str:
        return (
            f"Blackout(prob={self.prob}, "
            f"max_dark_weeks_per_month={self.max_dark_weeks_per_month})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Blackout)
            and self.prob == other.prob
            and self.max_dark_weeks_per_month == other.max_dark_weeks_per_month
        )


DeviationSpec = float | Blackout
ResolvedSpec = tuple[float, float] | Blackout


def _resolve_channel_specs(
    max_weekly_deviation_pct: DeviationSpec | dict[str, DeviationSpec],
    channels: list[str],
) -> dict[str, ResolvedSpec]:
    """Normalise max_weekly_deviation_pct into a per-channel spec dict.

    A channel's deviation shape isn't always the same across channels. An
    agency might allow a channel to move freely both up and down
    (symmetric range), lock a channel entirely (0), or want a hard on/off
    switch instead of a continuous range — see Blackout.

    Only symmetric ranges and Blackout are supported. An earlier version
    of this function also accepted an explicit one-sided (low, high) range
    (e.g. Search: (-100, 0), "never increase, sometimes blacked out"), but
    this was dropped: monthly totals are always preserved exactly (see
    _generate_phased_schedule), and preserving the total while biasing the
    raw draw towards a full blackout forces some weeks to spend well above
    their own original plan to compensate — visually, this reads as a
    broken promise ("I said never above plan, why is it above plan"),
    because a continuous partial-reduction range doesn't map to any
    familiar media-planning concept that would make the overshoot
    expected. Blackout has the identical mathematical trade-off (skewed
    draws still force a redistribution spike), but reads better because it
    maps onto ordinary media flighting/pulsing (full spend some weeks,
    dark others), so the "on" weeks running hot is an expected consequence
    of a recognisable strategy rather than an arbitrary side effect. Use
    Blackout (optionally with max_dark_weeks_per_month) for any "turn it
    down or off" ask instead.

    Accepts, at the top level or per channel in a dict:
    - a single float X: symmetric +/-X, i.e. bounds (-X, X). Backward
      compatible with the original single-number-for-everyone signature.
    - a Blackout instance: binary per-week on/off instead of a continuous
      range.

    A dict can mix both forms per channel, e.g.
    {"tv": 0, "meta": 60, "search": Blackout()} — TV locked, Meta free to
    move +/-60% either way, Search either at its original plan or fully
    dark in any given week.

    Parameters
    ----------
    max_weekly_deviation_pct:
        Single float, single Blackout, or dict[channel, float | Blackout].
    channels:
        The channels that must be covered (typically plan_df.columns).

    Returns
    -------
    dict[str, tuple[float, float] | Blackout] with one entry per channel.
    """

    def _as_spec(spec: DeviationSpec, ch: str) -> ResolvedSpec:
        if isinstance(spec, Blackout):
            return spec

        if isinstance(spec, tuple | list):
            raise TypeError(
                f"max_weekly_deviation_pct for channel {ch!r} got {spec!r}: "
                "explicit (low, high) ranges are no longer supported (they "
                "read as a 'never above plan' promise the monthly-total "
                "guarantee can't keep). Use a single float for a symmetric "
                "+/-X range, or Blackout() for a hard on/off switch."
            )

        magnitude = float(spec)
        if not 0.0 <= magnitude <= 100.0:
            raise ValueError(
                f"max_weekly_deviation_pct for channel {ch!r} must be "
                f"between 0 and 100 (inclusive), got {magnitude}."
            )
        return -magnitude, magnitude

    if isinstance(max_weekly_deviation_pct, dict):
        missing = set(channels) - set(max_weekly_deviation_pct)
        if missing:
            raise ValueError(
                f"max_weekly_deviation_pct dict is missing channels: {sorted(missing)}"
            )
        return {ch: _as_spec(max_weekly_deviation_pct[ch], ch) for ch in channels}

    return {ch: _as_spec(max_weekly_deviation_pct, ch) for ch in channels}


def _generate_phased_schedule(
    spend_df: pd.DataFrame,
    month_labels: np.ndarray,
    alpha: float,
    max_weekly_deviation_pct: DeviationSpec | dict[str, DeviationSpec],
    seed: int,
) -> pd.DataFrame:
    """Generate one phased weekly schedule for a given amplitude alpha.

    For each month and each channel independently:
    1. Draw a raw per-week deviation. For a symmetric range: uniform
       between -alpha x magnitude and +alpha x magnitude. For Blackout:
       either every week is an independent -100% (dark) draw with
       probability alpha x prob, or (if max_dark_weeks_per_month is set)
       the month activates blackout with probability alpha x prob and, if
       so, exactly that many weeks (chosen at random) go dark.
    2. Rescale so the monthly total is exactly preserved. NOTE: this rescale
       is applied across all weeks in the month together, so Blackout mode
       does not guarantee individual weeks stay within their raw draw after
       this step — see _resolve_channel_specs for why that's mathematically
       unavoidable, not a bug. A symmetric range's mean deviation is zero,
       so its rescale factor stays close to 1 and this effect is negligible
       there.
    3. Apply to original spend.

    Parameters
    ----------
    spend_df:
        NxK DataFrame with DatetimeIndex (the plan year).
    month_labels:
        Array of Period labels (one per week), from _get_month_labels.
    alpha:
        Phasing amplitude in [0, 1].
    max_weekly_deviation_pct:
        Maximum per-channel weekly deviation (%) at alpha=1. A single float
        (symmetric +/-, applied to every channel), a single Blackout, or a
        dict[channel, float | Blackout] for per-channel specs — e.g. a
        channel an agency won't let move at all gets 0, and one that
        should be a hard on/off switch gets Blackout(). See
        _resolve_channel_specs.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame with the same shape and index as spend_df.
    """
    rng = np.random.default_rng(seed)
    channels = list(spend_df.columns)
    channel_specs = _resolve_channel_specs(max_weekly_deviation_pct, channels)
    new_spend = spend_df.to_numpy().copy().astype(float)

    for month in np.unique(month_labels):
        mask = np.where(month_labels == month)[0]
        n_weeks = len(mask)
        for ci, ch in enumerate(channels):
            spec = channel_specs[ch]
            if isinstance(spec, Blackout):
                p = alpha * spec.prob
                cap = spec.max_dark_weeks_per_month
                dark = np.zeros(n_weeks, dtype=bool)
                if cap is None:
                    # legacy behaviour: every week is an independent draw,
                    # no limit on how many weeks in the month go dark
                    dark = rng.random(n_weeks) < p
                    if dark.all():
                        # keep at least one week on: with nothing left
                        # "on", there is nowhere for the month's budget to
                        # land, and the channel would silently end up
                        # untouched instead of blacked out — see the
                        # sum <= 0 fallback below.
                        dark[rng.integers(n_weeks)] = False
                else:
                    # capped behaviour: the month either activates its
                    # blackout slot (probability p) or doesn't; if it
                    # does, exactly n_dark weeks (never all of them) go
                    # dark, chosen at random — bounds how much budget any
                    # one month can divert, and so how large the spike on
                    # the surviving weeks can get.
                    n_dark = min(cap, n_weeks - 1)
                    if n_dark > 0 and rng.random() < p:
                        idx = rng.choice(n_weeks, size=n_dark, replace=False)
                        dark[idx] = True
                raw = np.where(dark, -1.0, 0.0)
            else:
                low_pct, high_pct = spec
                low_dev = alpha * low_pct / 100.0
                high_dev = alpha * high_pct / 100.0
                raw = rng.uniform(low_dev, high_dev, size=n_weeks)

            orig_weeks = spend_df.iloc[mask, ci].to_numpy()
            monthly_total = orig_weeks.sum()
            new_weeks = orig_weeks * (1.0 + raw)
            # rescale to preserve monthly total exactly
            if new_weeks.sum() > 0:
                new_spend[mask, ci] = new_weeks * (monthly_total / new_weeks.sum())
            else:
                new_spend[mask, ci] = orig_weeks

    return pd.DataFrame(new_spend, index=spend_df.index, columns=spend_df.columns)


def _max_monthly_deviation(
    original: pd.DataFrame,
    phased: pd.DataFrame,
    month_labels: np.ndarray,
) -> float:
    """Return the max fractional monthly deviation across all channels and months."""
    orig_arr = original.to_numpy()
    new_arr = phased.to_numpy()
    max_dev = 0.0
    for month in np.unique(month_labels):
        mask = np.where(month_labels == month)[0]
        for ci in range(orig_arr.shape[1]):
            orig_sum = orig_arr[mask, ci].sum()
            if orig_sum > 0:
                dev = abs(new_arr[mask, ci].sum() - orig_sum) / orig_sum
                max_dev = max(max_dev, dev)
    return max_dev


class BudgetPhaser:
    """Recommend the weekly spend phasing needed to reduce elasticity uncertainty.

    Takes a multi-year spend history and a plan-year budget. Grid-searches over
    phasing amplitude to find the plan-year schedule that minimises max CV across
    channels (under plain OLS on history + phased plan), while preserving
    monthly budgets.

    Parameters
    ----------
    history_df:
        Multi-year spend history (e.g. 4 years = 208 weeks) with a weekly
        DatetimeIndex. One column per channel. Fixed — not modified by phasing.
    plan_df:
        One-year spend plan (e.g. 52 weeks) with a weekly DatetimeIndex.
        Same columns as history_df. This is the data that gets phased.
    true_elasticities:
        Dict mapping channel name to true elasticity. Defaults to
        {"tv": 0.3, "meta": 0.5, "search": 0.4}.
    max_monthly_deviation_pct:
        Maximum allowed fractional deviation in monthly totals per channel (%).
        Default 1.0. Enforced by construction (rescaling).
    max_weekly_deviation_pct:
        Maximum per-channel weekly deviation from original plan spend at
        alpha=1 (%). Default 40.0 (symmetric +/-40%). A channel's allowed
        deviation isn't always the same shape in practice, so this accepts,
        at the top level or per channel in a dict:
          - a single float X: symmetric +/-X.
          - a Blackout instance: a hard per-week (or, with
            max_dark_weeks_per_month, per-month-capped) on/off switch (0%
            or 100% of plan) instead of a continuous range — see Blackout.
          - a dict[channel, float | Blackout] mixing both, e.g.
            {"tv": 0, "meta": 60, "search": Blackout()} for an agency that
            won't move TV at all, allows Meta +/-60% either way, and wants
            Search either at plan or fully dark, nothing in between.
        NOTE: neither form is a hard ceiling on the final schedule.
        Monthly totals are always preserved exactly. A symmetric range's
        mean deviation is zero, so its rescale stays close to 1 and this
        is negligible in practice; Blackout's deviation is skewed, so the
        budget freed up by dark weeks lands on that channel's "on" weeks
        instead, which can then spend above their own original plan — set
        max_dark_weeks_per_month on the Blackout to keep that spike
        bounded to a realistic size. See _resolve_channel_specs and
        Blackout. A dict must cover every channel in plan_df.
    seed:
        Base random seed.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        true_elasticities: dict[str, float] | None = None,
        max_monthly_deviation_pct: float = 1.0,
        max_weekly_deviation_pct: DeviationSpec | dict[str, DeviationSpec] = 40.0,
        seed: int = 0,
    ) -> None:
        _get_month_labels(history_df)  # validates DatetimeIndex
        _get_month_labels(plan_df)  # validates DatetimeIndex

        if list(history_df.columns) != list(plan_df.columns):
            raise ValueError(
                "history_df and plan_df must have the same columns. "
                f"Got {list(history_df.columns)} vs {list(plan_df.columns)}."
            )

        _resolve_channel_specs(
            max_weekly_deviation_pct, list(plan_df.columns)
        )  # validates shape and bounds, fails fast

        self.history_df = history_df
        self.plan_df = plan_df
        self.true_elasticities = (
            true_elasticities
            if true_elasticities is not None
            else _DEFAULT_ELASTICITIES
        )
        self.max_monthly_deviation_pct = max_monthly_deviation_pct
        self.max_weekly_deviation_pct = max_weekly_deviation_pct
        self.seed = seed

        self._plan_month_labels = _get_month_labels(plan_df)
        self.results_: pd.DataFrame | None = None
        self.recommended_schedule_: pd.DataFrame | None = None

    def fit(
        self,
        n_sims: int = 50,
        grid_steps: int = 20,
        n_phasing_seeds: int = 3,
        fast_mode: bool = False,
    ) -> BudgetPhaser:
        """Grid-search over phasing amplitude and store results.

        For each alpha:
          1. Generate n_phasing_seeds independent phased plan schedules.
          2. For each: concatenate history + phased plan, run
             CollinearityDiagnostic, record per-channel CVs.
          3. Average CVs across phasing seeds — this smooths the CV curve
             so the grid search isn't driven by a single lucky/unlucky draw.
          4. Record the alpha with the lowest averaged max CV as the recommendation.

        Parameters
        ----------
        n_sims:
            Number of noise seeds per grid point for CollinearityDiagnostic.
        grid_steps:
            Number of alpha levels to evaluate.
        n_phasing_seeds:
            Number of independent phased schedules to generate per alpha level.
            CVs are averaged across seeds before selecting the best alpha.
            Default 3. Set to 1 to match the single-seed behaviour of v2.
        fast_mode:
            If True, uses n_sims=10, grid_steps=10, n_phasing_seeds=1.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            grid_steps = 10
            n_phasing_seeds = 1

        alphas = np.linspace(0, 1, grid_steps)
        channels = list(self.plan_df.columns)
        rows = []

        for i, alpha in enumerate(alphas):
            seed_results = []

            for j in range(n_phasing_seeds):
                phased_plan = _generate_phased_schedule(
                    self.plan_df,
                    self._plan_month_labels,
                    alpha=float(alpha),
                    max_weekly_deviation_pct=self.max_weekly_deviation_pct,
                    seed=self.seed + i * n_phasing_seeds + j,
                )

                monthly_dev = _max_monthly_deviation(
                    self.plan_df, phased_plan, self._plan_month_labels
                )

                combined = pd.concat([self.history_df, phased_plan])

                diag = CollinearityDiagnostic(
                    spend_df=combined,
                    true_elasticities=self.true_elasticities,
                )
                diag.fit(n_sims=n_sims)
                summ = diag.summary().set_index("channel")

                seed_results.append(
                    {
                        "actual_correlation": diag.actual_correlation,
                        "monthly_dev": monthly_dev,
                        **{
                            ch: float(summ.loc[ch, "coef_of_variation"])
                            for ch in channels
                        },
                    }
                )

            # Average across phasing seeds to smooth the CV curve
            avg_corr = float(np.mean([r["actual_correlation"] for r in seed_results]))
            avg_monthly_dev = float(np.mean([r["monthly_dev"] for r in seed_results]))
            avg_cv = {
                ch: float(np.mean([r[ch] for r in seed_results])) for ch in channels
            }
            max_cv = max(avg_cv.values())

            row: dict = {
                "alpha": round(float(alpha), 4),
                "actual_correlation": round(avg_corr, 4),
                "max_cv": round(max_cv, 4),
                "max_monthly_deviation_pct": round(avg_monthly_dev * 100, 6),
            }
            for ch in channels:
                row[f"cv_{ch}"] = round(avg_cv[ch], 4)

            rows.append(row)

        self.results_ = pd.DataFrame(rows)

        # generate the recommended schedule at the best alpha
        best_alpha = float(self.results_.loc[self.results_["max_cv"].idxmin(), "alpha"])
        self.recommended_schedule_ = _generate_phased_schedule(
            self.plan_df,
            self._plan_month_labels,
            alpha=best_alpha,
            max_weekly_deviation_pct=self.max_weekly_deviation_pct,
            seed=self.seed + grid_steps * n_phasing_seeds,  # distinct from grid search
        )

        return self

    def recommend(self) -> pd.Series:
        """Return the grid point with the lowest max CV.

        Returns
        -------
        pd.Series with alpha, actual_correlation, max_cv, and per-channel CVs.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before recommend().")
        return self.results_.loc[self.results_["max_cv"].idxmin()]

    def summary(self) -> pd.DataFrame:
        """Return the full grid search results.

        Returns
        -------
        pd.DataFrame with one row per alpha level.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")
        return self.results_
