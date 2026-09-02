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

Weighting schemes (uniform / binary / decay) were evaluated in a research
study and dropped: uniform weighting always outperformed upweighting the
plan year, so the evaluation is plain OLS on history + phased plan
throughout.

The output is a concrete plan-year weekly spend schedule the practitioner can
hand to their media agency, with monthly totals unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import _DEFAULT_MARGINAL_RETURNS, simulate_demand
from how_wrong_is_your_mmm._diagnostic import (
    CollinearityDiagnostic,
    _validate_spend_data,
)


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


def _tile_plan(plan_df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Extend or truncate plan_df to n_weeks by repeating its weekly pattern.

    Used for horizon comparisons beyond (or short of) the supplied plan
    length — e.g. a 2-year view built from a 52-week plan, or a 3-month
    view built from the same plan truncated. There is no real data past
    the supplied plan, so a longer horizon is a projection that assumes
    next year repeats this year's weekly pattern, not a claim about an
    actual future plan — callers should treat it accordingly. Matches the
    tiling approach already used ad hoc in
    notebooks/02_phaser_walkthrough.ipynb for its own 2-year figures.

    Parameters
    ----------
    plan_df:
        Weekly spend plan with a DatetimeIndex.
    n_weeks:
        Target length in weeks. Can be shorter or longer than plan_df.

    Returns
    -------
    pd.DataFrame of length n_weeks, same columns as plan_df, with a
    DatetimeIndex continuing plan_df's own frequency from its start date.
    """
    n_orig = len(plan_df)
    reps = -(-n_weeks // n_orig)  # ceil division
    tiled_values = np.tile(plan_df.to_numpy(), (reps, 1))[:n_weeks]
    freq = plan_df.index.freqstr or pd.infer_freq(plan_df.index) or "W-MON"
    new_index = pd.date_range(start=plan_df.index[0], periods=n_weeks, freq=freq)
    return pd.DataFrame(tiled_values, index=new_index, columns=plan_df.columns)


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


NUDGE_SHAPES = ("uniform", "annulus", "edge")


def _validate_nudge_shape(nudge_shape: str) -> str:
    if nudge_shape not in NUDGE_SHAPES:
        raise ValueError(
            f"nudge_shape must be one of {NUDGE_SHAPES}, got {nudge_shape!r}."
        )
    return nudge_shape


def _shaped_nudge(
    rng: np.random.Generator,
    n_weeks: int,
    cap: float,
    nudge_shape: str,
    balance_signs: bool,
) -> np.ndarray:
    """Draw one month's per-week deviations for a symmetric +/-cap range.

    Splits the draw into a magnitude and a sign, because the two do
    different jobs and the useful variants change only one of them.

    magnitude (nudge_shape), as a fraction of the week's planned spend:
    - "uniform": |dev| ~ U(0, cap). The mean absolute value of a uniform
      draw is HALF its cap, so a nominal +/-20% setting moves a typical
      week by ~8% once the monthly rescale has also removed the month's
      common component. Most of the range the user negotiated is never
      spent. This is the shipped behaviour and stays the default.
    - "annulus": |dev| ~ U(cap/2, cap). Excludes the timid middle, so every
      week moves meaningfully, while still spreading spend over a range of
      levels rather than parking it at two.
    - "edge": |dev| = cap exactly. Maximises the variation injected per
      unit of cap, and gives up the spread of levels to get it.

    sign (balance_signs):
    - False: an independent fair coin per week.
    - True: within the month, equal numbers of up and down weeks (an odd
      week out is left at its plan). The rescale that follows preserves the
      monthly total by dividing through the month's mean deviation, so a
      sign-balanced month has almost nothing to divide out and the realised
      deviations stay close to the ones drawn. Unbalanced months are where
      the overshoot comes from: three weeks drawn down force the surviving
      week to absorb the whole month, and at cap=0.8 that can reach three
      times its planned spend.

    Returns an array of fractional deviations, one per week. Note that the
    cap binds on the DRAW, not on the realised deviation after the rescale
    -- see _generate_phased_schedule.
    """
    if cap <= 0.0:
        return np.zeros(n_weeks)

    if nudge_shape == "uniform":
        magnitude = rng.uniform(0.0, cap, size=n_weeks)
    elif nudge_shape == "annulus":
        magnitude = rng.uniform(cap / 2.0, cap, size=n_weeks)
    else:  # "edge"
        magnitude = np.full(n_weeks, cap, dtype=float)

    if balance_signs:
        signs = np.zeros(n_weeks)
        half = n_weeks // 2
        order = rng.permutation(n_weeks)
        signs[order[:half]] = 1.0
        signs[order[half : 2 * half]] = -1.0
    else:
        signs = rng.choice([-1.0, 1.0], size=n_weeks)

    return magnitude * signs


def _generate_phased_schedule(
    spend_df: pd.DataFrame,
    month_labels: np.ndarray,
    alpha: float,
    max_weekly_deviation_pct: DeviationSpec | dict[str, DeviationSpec],
    seed: int,
    nudge_shape: str = "uniform",
    balance_signs: bool = False,
) -> pd.DataFrame:
    """Generate one phased weekly schedule for a given amplitude alpha.

    For each month and each channel independently:
    1. Draw a raw per-week deviation. For a symmetric range: shaped by
       nudge_shape and balance_signs, defaulting to uniform between
       -alpha x magnitude and +alpha x magnitude. For Blackout:
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
    nudge_shape:
        How a symmetric range's per-week magnitude is drawn within the cap:
        "uniform" (default, |dev| ~ U(0, cap) -- the shipped behaviour),
        "annulus" (|dev| ~ U(cap/2, cap)) or "edge" (|dev| = cap). Ignored
        by Blackout channels, which are on/off by construction. See
        _shaped_nudge.
    balance_signs:
        If True, each month gets equal numbers of up and down weeks (an odd
        week out is left at its plan) instead of an independent coin flip
        per week, which keeps the monthly rescale factor near 1 and so keeps
        the realised deviations close to the drawn ones. Ignored by Blackout
        channels. Default False, the shipped behaviour.

    Returns
    -------
    pd.DataFrame with the same shape and index as spend_df.

    Notes
    -----
    max_weekly_deviation_pct binds on the DRAW, before step 2's rescale,
    so it is not a hard cap on the realised weekly deviation. Under the
    default uniform draw roughly 3-4% of weeks finish outside their nominal
    band; the more of the band a shape uses, the more the rescale has to
    move, unless balance_signs keeps the month's mean deviation near zero.
    Enforcing a cap on the realised deviation would need a repair step
    after the rescale, which this function does not do.

    Changing nudge_shape or balance_signs changes the generator's call
    sequence, so two schedules built with the same seed under different
    options are not the same draw seen two ways. Compare shapes in
    distribution, across seeds, not draw for draw. A channel locked at 0
    consumes no randomness on the shaped path (it does on the default
    one), so mixing locked and unlocked channels shifts the stream too.
    """
    _validate_nudge_shape(nudge_shape)
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
                if nudge_shape == "uniform" and not balance_signs:
                    # The default path deliberately keeps this exact RNG
                    # call, rather than routing through _shaped_nudge's
                    # equivalent magnitude-and-sign formulation. Splitting
                    # the draw changes the generator's call sequence, so
                    # every seeded schedule -- and every published number
                    # derived from one, in docs/overview.html and the
                    # notebooks -- would silently move.
                    raw = rng.uniform(low_dev, high_dev, size=n_weeks)
                else:
                    raw = _shaped_nudge(
                        rng, n_weeks, high_dev, nudge_shape, balance_signs
                    )

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
    """Recommend the weekly spend phasing needed to reduce marginal-return uncertainty.

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
    true_marginal_returns:
        Dict mapping channel name to true marginal return (£ revenue per
        £ spend, a.k.a. mROAS — not an economic elasticity, see _dgp.py).
        Defaults to {"tv": 0.5, "meta": 1.0, "search": 1.5}. As with
        CollinearityDiagnostic, there is no safe universal default here —
        prefer supplying your own per-channel values.
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
    base_sales:
        Base sales intercept forwarded to every internal
        CollinearityDiagnostic. Default 1_000.0, matching
        CollinearityDiagnostic's own default.
    revenue_noise_std:
        Standard deviation of sales noise (£), forwarded to every internal
        CollinearityDiagnostic used to score a candidate schedule. Default
        26_000.0. As with CollinearityDiagnostic, there is no universal
        default that's right for your data — set this from your own
        model's residual std (e.g. the residual std of a simple OLS fit
        on your actual sales/spend history), not the package default.
        Every CV this class reports scales ~linearly with this value, so
        an unexamined default here silently determines how alarming (or
        how reassuring) every downstream number looks.
    true_elasticities:
        Deprecated alias for `true_marginal_returns`, kept for backward
        compatibility. Raises ValueError if both are supplied. Emits a
        FutureWarning -- migrate to `true_marginal_returns`.
    nudge_shape:
        How much of the negotiated band each week's nudge actually uses.
        `max_weekly_deviation_pct` is a cap, and "uniform" spends only about
        half of it: the mean absolute value of a uniform draw is half its
        range, so a +/-20% setting moves a typical week by roughly 8%.
        "annulus" draws the magnitude from the outer half of the band
        (U(cap/2, cap)), and "edge" uses the cap exactly. Default "uniform",
        which is the historical behaviour -- every number published in
        `docs/overview.html` and the notebooks assumes it. Blackout channels
        are unaffected; they are on/off by construction. Research option:
        prefer "annulus" with `balance_signs=True` if you are exploring
        this, and read the Notes on cost below.
    balance_signs:
        If True, each month gets equal numbers of up and down weeks rather
        than an independent coin flip per week, which keeps the
        monthly-total rescale factor near 1 and so keeps realised weekly
        deviations close to the ones drawn. Without it an unbalanced month
        forces its surviving weeks to absorb the whole month, and at
        cap=80% a single week can reach several times its planned spend.
        Default False, the historical behaviour.
    demand:
        Optional latent demand series spanning history_df + plan_df (i.e.
        length len(history_df) + len(plan_df)), forwarded to every internal
        CollinearityDiagnostic this class creates. Supply your own, or
        leave it None to let this class draw one internally from
        demand_process/demand_seed whenever demand_coef is nonzero -- see
        CollinearityDiagnostic's own `demand` docstring, this mirrors it.
        Resolved once at construction (self.demand_), not redrawn per
        phasing draw or grid point, because it represents the one true
        state of the world every candidate schedule is measured against.
    demand_process:
        One of DEMAND_PROCESSES (see simulate_demand). Ignored when
        `demand` is supplied directly.
    demand_coef:
        Coefficient on demand in the sales equation -- what actually
        creates omitted-variable bias. 0.0 (default) reproduces this
        class's pre-existing behaviour exactly. Get a value in
        practitioner-legible units from calibrate_baseline.
    demand_seed:
        Random seed for the internal demand draw, when `demand` isn't
        supplied directly.
    saturation:
        Forwarded to every internal CollinearityDiagnostic -- see
        simulate_sales's own docstring. None (default) is linear,
        reproducing prior behaviour.
    adstock:
        Forwarded to every internal CollinearityDiagnostic -- see
        simulate_sales's own docstring. None (default) is no carryover,
        reproducing prior behaviour.
    reference_spend:
        Forwarded to every internal CollinearityDiagnostic. Only matters
        when saturation is active. Defaults to plan_df's own per-channel
        mean spend (the ORIGINAL, unphased plan) rather than
        simulate_sales's own per-call default (each candidate schedule's
        own mean) -- deliberately, so every phasing candidate is priced
        against the SAME response curve. Leaving this at simulate_sales's
        own default would make the curve move with each candidate
        schedule, which would make comparing them not like for like.
    controls:
        What every internal CollinearityDiagnostic's OLS fit controls for,
        forwarded to its fit(). None or False (default): omit -- reproduces
        prior behaviour exactly. True: control with this instance's own
        true demand_ (requires a demand series). A DataFrame or Series: an
        explicit proxy. One fixed setting for the whole instance (not a
        per-fit()-call choice, unlike CollinearityDiagnostic) because every
        method here (fit, channel_sensitivity, recommend_levers,
        impact_over_horizons) needs to score candidates under the same
        measurement the client will actually see.

    Notes
    -----
    `nudge_shape` and `balance_signs` are research options, not tuning
    knobs with a known best setting. Using more of the band buys tighter
    estimates, and under a saturating (concave) response it also gives up
    revenue, because the same monthly budget spread more unevenly across a
    curve that bends produces less output. The package's own DGP is linear
    in spend, so it cannot price that trade-off for you: nothing here
    charges you for an aggressive schedule. Treat the defaults as the
    supported path until that cost is quantified for your own response
    curves.

    An earlier version also accepted `max_monthly_deviation_pct`, a
    "maximum allowed deviation" input that was stored but never actually
    enforced or read anywhere (the identically-named `max_monthly_deviation_pct`
    column in `results_`/`summary()` is a *measured* quantity from
    `_max_monthly_deviation`, not this parameter — monthly totals are
    preserved exactly by construction regardless, see
    `_generate_phased_schedule`). It has been removed rather than wired up,
    since there was nothing for it to constrain that isn't already exact.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        true_marginal_returns: dict[str, float] | None = None,
        max_weekly_deviation_pct: DeviationSpec | dict[str, DeviationSpec] = 40.0,
        seed: int = 0,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 26_000.0,
        true_elasticities: dict[str, float] | None = None,
        nudge_shape: str = "uniform",
        balance_signs: bool = False,
        demand: np.ndarray | pd.Series | None = None,
        demand_process: str = "white_noise",
        demand_coef: float = 0.0,
        demand_seed: int = 0,
        saturation: dict[str, float] | float | None = None,
        adstock: dict[str, float] | float | None = None,
        reference_spend: dict[str, float] | None = None,
        controls: pd.DataFrame | pd.Series | bool | None = None,
    ) -> None:
        _get_month_labels(history_df)  # validates DatetimeIndex
        _get_month_labels(plan_df)  # validates DatetimeIndex

        if list(history_df.columns) != list(plan_df.columns):
            raise ValueError(
                "history_df and plan_df must have the same columns. "
                f"Got {list(history_df.columns)} vs {list(plan_df.columns)}."
            )

        # Same validation CollinearityDiagnostic.fit() runs on real spend,
        # applied eagerly here (on the concatenation fit() will actually
        # build internally) so a data-quality problem fails fast at
        # construction, not deep inside the first grid-search iteration.
        _validate_spend_data(pd.concat([history_df, plan_df]))

        _resolve_channel_specs(
            max_weekly_deviation_pct, list(plan_df.columns)
        )  # validates shape and bounds, fails fast

        _validate_nudge_shape(nudge_shape)

        if true_elasticities is not None:
            if true_marginal_returns is not None:
                raise ValueError(
                    "Pass only one of true_marginal_returns or the "
                    "deprecated true_elasticities, not both."
                )
            warnings.warn(
                "true_elasticities is deprecated and will be removed in a "
                "future release -- these are marginal returns (£ revenue "
                "per £ spend), not elasticities. Use true_marginal_returns "
                "instead.",
                FutureWarning,
                stacklevel=2,
            )
            true_marginal_returns = true_elasticities

        self.history_df = history_df
        self.plan_df = plan_df
        self.true_marginal_returns = (
            true_marginal_returns
            if true_marginal_returns is not None
            else _DEFAULT_MARGINAL_RETURNS
        )
        self.max_weekly_deviation_pct = max_weekly_deviation_pct
        self.nudge_shape = nudge_shape
        self.balance_signs = balance_signs
        self.seed = seed
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std
        self.demand = demand
        self.demand_process = demand_process
        self.demand_coef = demand_coef
        self.demand_seed = demand_seed
        self.saturation = saturation
        self.adstock = adstock
        # Defaults to the ORIGINAL, unphased plan's own mean spend rather
        # than simulate_sales's own per-call default (each candidate
        # schedule's own mean) -- so every phasing candidate this class
        # evaluates is priced against the SAME response curve. See the
        # reference_spend docstring above.
        if reference_spend is None and saturation is not None:
            reference_spend = {ch: float(plan_df[ch].mean()) for ch in plan_df.columns}
        self.reference_spend = reference_spend
        self.controls = controls

        n_total = len(history_df) + len(plan_df)
        if demand is not None:
            demand_arr = np.asarray(demand, dtype=float)
            if demand_arr.shape != (n_total,):
                raise ValueError(
                    "demand must be a 1-D series of length "
                    f"len(history_df) + len(plan_df) = {n_total}, got shape "
                    f"{demand_arr.shape}"
                )
        elif demand_coef:
            demand_arr = simulate_demand(
                n_total, process=demand_process, seed=demand_seed
            )
        else:
            demand_arr = None
        # Resolved once here, not redrawn per phasing draw or grid point --
        # it represents the one true state of the world every candidate
        # schedule is measured against. See the demand docstring above.
        self.demand_ = demand_arr

        self._plan_month_labels = _get_month_labels(plan_df)
        self.results_: pd.DataFrame | None = None
        self.confirmation_: pd.DataFrame | None = None
        self.recommended_schedule_: pd.DataFrame | None = None
        self.recommended_draws_: pd.DataFrame | None = None
        self.recommended_schedule_median_cv_: float | None = None

    @property
    def true_elasticities(self) -> dict[str, float]:
        """Deprecated alias for `true_marginal_returns`. See __init__."""
        warnings.warn(
            "BudgetPhaser.true_elasticities is deprecated, use "
            "true_marginal_returns instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.true_marginal_returns

    def _evaluate_spec_at_alpha(
        self,
        spec: DeviationSpec | dict[str, DeviationSpec],
        alpha: float,
        n_sims: int,
        n_phasing_seeds: int,
        seed_offset: int,
        noise_seed_offset: int = 0,
    ) -> dict:
        """Run n_phasing_seeds phased-schedule draws at one (spec, alpha)
        setting, average the resulting per-channel CVs, and return one row.

        Shared by fit()'s joint alpha grid search (spec =
        self.max_weekly_deviation_pct, every channel scaled together) and
        channel_sensitivity()'s per-channel marginal sweep (spec has every
        channel but one locked at 0) — both go through identical simulation
        machinery, only the spec being evaluated differs. Keeping this in
        one place means the two can't quietly drift out of sync.

        noise_seed_offset is forwarded to CollinearityDiagnostic.fit(). It
        defaults to 0, which reproduces the original always-seed-0..n_sims-1
        noise behaviour at every phasing seed. fit()'s confirmation pass
        (see its own docstring) passes a distinct, non-zero offset so its
        "honest" re-evaluation of the grid's candidates draws genuinely
        fresh noise, not just fresh phasing seeds over the same n_sims
        noise draws the grid search itself could have overfit to.
        """
        channels = list(self.plan_df.columns)
        seed_results = []

        for j in range(n_phasing_seeds):
            phased_plan = _generate_phased_schedule(
                self.plan_df,
                self._plan_month_labels,
                alpha=alpha,
                max_weekly_deviation_pct=spec,
                seed=self.seed + seed_offset + j,
                nudge_shape=self.nudge_shape,
                balance_signs=self.balance_signs,
            )

            monthly_dev = _max_monthly_deviation(
                self.plan_df, phased_plan, self._plan_month_labels
            )

            combined = pd.concat([self.history_df, phased_plan])

            diag = CollinearityDiagnostic(
                spend_df=combined,
                true_marginal_returns=self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                demand=self.demand_,
                demand_coef=self.demand_coef,
                saturation=self.saturation,
                adstock=self.adstock,
                reference_spend=self.reference_spend,
            )
            diag.fit(
                n_sims=n_sims,
                noise_seed_offset=noise_seed_offset,
                controls=self.controls,
            )
            summ = diag.summary().set_index("channel")

            seed_results.append(
                {
                    "actual_correlation": diag.actual_correlation,
                    "monthly_dev": monthly_dev,
                    **{ch: float(summ.loc[ch, "coef_of_variation"]) for ch in channels},
                }
            )

        # Average across phasing seeds to smooth the CV curve
        avg_corr = float(np.mean([r["actual_correlation"] for r in seed_results]))
        avg_monthly_dev = float(np.mean([r["monthly_dev"] for r in seed_results]))
        avg_cv = {ch: float(np.mean([r[ch] for r in seed_results])) for ch in channels}

        row: dict = {
            "alpha": round(float(alpha), 4),
            "actual_correlation": round(avg_corr, 4),
            "max_cv": round(max(avg_cv.values()), 4),
            "max_monthly_deviation_pct": round(avg_monthly_dev * 100, 6),
        }
        for ch in channels:
            row[f"cv_{ch}"] = round(avg_cv[ch], 4)
        return row

    def fit(
        self,
        n_sims: int = 50,
        grid_steps: int = 20,
        n_phasing_seeds: int = 3,
        fast_mode: bool = False,
        confirm_top_k: int = 3,
        confirm_n_phasing_seeds: int | None = None,
        n_recommended_draws: int = 10,
    ) -> BudgetPhaser:
        """Grid-search over phasing amplitude and store results.

        For each alpha:
          1. Generate n_phasing_seeds independent phased plan schedules.
          2. For each: concatenate history + phased plan, run
             CollinearityDiagnostic, record per-channel CVs.
          3. Average CVs across phasing seeds — this smooths the CV curve
             so the grid search isn't driven by a single lucky/unlucky draw.
          4. Record the alpha with the lowest averaged max CV as a candidate.

        Selection-bias correction ("optimizer's curse"): step 4's argmin is
        itself picked from `grid_steps` noisy estimates, so it's biased
        towards whichever alpha happened to get a lucky draw — confirmed
        empirically: independently re-evaluating a grid "winner" regresses
        its CV back up towards the honest value, sometimes substantially.
        To correct for this, the top
        `confirm_top_k` candidates by max_cv are re-evaluated with fresh
        phasing seeds *and* fresh noise seeds (both well clear of the grid
        search's own ranges -- see noise_seed_offset on
        CollinearityDiagnostic.fit; without this, the confirmation pass
        would re-use the same n_sims noise draws at every phasing seed and
        grid point, so an alpha that happened to fit those particular noise
        draws well would still look artificially good even after
        "confirmation") and a larger sample (`confirm_n_phasing_seeds`), and
        *that* honestly-measured winner's alpha becomes the actual
        recommendation. Both the raw grid (`results_`) and the confirmation
        pass (`confirmation_`) are kept, so the size of the correction stays
        inspectable rather than silently overwritten.

        recommended_schedule_ itself goes through a further, separate
        evaluation: draw-to-draw spread at a fixed alpha is real (a single
        phased-schedule draw's own max CV can swing 25-35% from the best to
        the worst of 20 otherwise-identical draws), so handing over the
        very first schedule generated at the recommended alpha -- with no
        evaluation at all -- would silently ship whichever draw got lucky
        or unlucky. Instead, `n_recommended_draws` independent schedules are
        generated at the recommended alpha, each is evaluated the same way
        the grid search evaluates a candidate, and the best-evaluated one
        (lowest max CV) becomes `recommended_schedule_`. Because "best of N
        evaluated draws" is itself a mild form of the same selection bias
        the grid-search confirmation exists to correct, the max CV actually
        printed in any report is the *median* max CV across those N draws
        (`recommended_schedule_median_cv_`), not the shipped draw's own
        (optimistic) evaluation -- see `recommended_draws_`.

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
            If True, uses n_sims=10, grid_steps=10, n_phasing_seeds=1,
            confirm_top_k=1, n_recommended_draws=1 (skips the extra
            confirmation cost — this mode is for iterating on report
            layout, not for numbers to hand a client).
        confirm_top_k:
            Number of the grid's lowest-max_cv candidates to re-evaluate
            independently before picking the final recommendation. Default
            3. Set to 1 to just re-confirm the raw grid winner without
            considering its near neighbours (cheaper, less protection
            against a genuinely close second place being the honest winner).
        confirm_n_phasing_seeds:
            Phasing seeds used for the confirmation pass. Defaults to
            3x n_phasing_seeds — independently *and* more precisely
            measured than the search itself, per the standard fix for
            optimizer's-curse-style selection bias (evaluate the winner on
            a fresh, larger sample rather than trust the value that won the
            search that picked it).
        n_recommended_draws:
            Number of independent phased-schedule draws generated and
            evaluated at the recommended alpha before picking the one
            shipped as `recommended_schedule_`. Default 10. See docstring
            above.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            grid_steps = 10
            n_phasing_seeds = 1
            confirm_top_k = 1
            n_recommended_draws = 1

        if confirm_n_phasing_seeds is None:
            confirm_n_phasing_seeds = n_phasing_seeds * 3

        # Distinct noise-seed ranges for grid search vs. confirmation, so
        # the confirmation pass can't just be re-confirming a fit to the
        # same n_sims noise draws the grid search already saw. See
        # noise_seed_offset on CollinearityDiagnostic.fit and the P1.7 note
        # in this method's docstring.
        grid_noise_offset = 0
        confirm_noise_offset = 500_000

        alphas = np.linspace(0, 1, grid_steps)
        rows = [
            self._evaluate_spec_at_alpha(
                self.max_weekly_deviation_pct,
                float(alpha),
                n_sims,
                n_phasing_seeds,
                seed_offset=i * n_phasing_seeds,
                noise_seed_offset=grid_noise_offset,
            )
            for i, alpha in enumerate(alphas)
        ]

        self.results_ = pd.DataFrame(rows)

        # Selection-bias correction: re-evaluate the top confirm_top_k
        # candidates (by the grid's own noisy max_cv) independently, with a
        # larger sample and seeds well clear of the grid search's own range,
        # then recommend whichever of THOSE is best — not the raw grid
        # argmin, which is systematically optimistic. See docstring above.
        k = min(confirm_top_k, len(self.results_))
        candidates = self.results_.nsmallest(k, "max_cv")
        confirm_rows = [
            self._evaluate_spec_at_alpha(
                self.max_weekly_deviation_pct,
                float(cand["alpha"]),
                n_sims,
                confirm_n_phasing_seeds,
                seed_offset=100_000 + i * confirm_n_phasing_seeds,
                noise_seed_offset=confirm_noise_offset,
            )
            for i, (_, cand) in enumerate(candidates.iterrows())
        ]
        self.confirmation_ = pd.DataFrame(confirm_rows)

        best_alpha = float(
            self.confirmation_.loc[self.confirmation_["max_cv"].idxmin(), "alpha"]
        )

        # Evaluate n_recommended_draws independent schedules at best_alpha
        # and ship the best-evaluated one, rather than one arbitrary,
        # never-evaluated draw. See docstring above.
        draw_records = []
        draw_schedules = []
        for m in range(n_recommended_draws):
            draw_seed = self.seed + 900_000 + m * 7919  # distinct, well-spaced
            schedule_m = _generate_phased_schedule(
                self.plan_df,
                self._plan_month_labels,
                alpha=best_alpha,
                max_weekly_deviation_pct=self.max_weekly_deviation_pct,
                seed=draw_seed,
                nudge_shape=self.nudge_shape,
                balance_signs=self.balance_signs,
            )
            combined_m = pd.concat([self.history_df, schedule_m])
            diag_m = CollinearityDiagnostic(
                spend_df=combined_m,
                true_marginal_returns=self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                demand=self.demand_,
                demand_coef=self.demand_coef,
                saturation=self.saturation,
                adstock=self.adstock,
                reference_spend=self.reference_spend,
            )
            diag_m.fit(
                n_sims=n_sims,
                noise_seed_offset=confirm_noise_offset + 1,
                controls=self.controls,
            )
            summ_m = diag_m.summary().set_index("channel")["coef_of_variation"]
            draw_schedules.append(schedule_m)
            record = {"draw": m, "seed": draw_seed, "max_cv": float(summ_m.max())}
            for ch in summ_m.index:
                record[f"cv_{ch}"] = float(summ_m[ch])
            draw_records.append(record)

        self.recommended_draws_ = pd.DataFrame(draw_records)
        best_draw_idx = int(self.recommended_draws_["max_cv"].idxmin())
        self.recommended_schedule_ = draw_schedules[best_draw_idx]
        # The shipped draw is the best of n_recommended_draws -- itself a
        # mild selection effect -- so the number reported alongside it is
        # the MEDIAN max CV across all evaluated draws, not the shipped
        # draw's own (optimistic) evaluation.
        self.recommended_schedule_median_cv_ = float(
            self.recommended_draws_["max_cv"].median()
        )

        return self

    def channel_sensitivity(
        self,
        channel: str,
        alphas: list[float] | None = None,
        magnitude_pct: float = 40.0,
        blackout: Blackout | None = None,
        n_sims: int = 50,
        n_phasing_seeds: int = 3,
        fast_mode: bool = False,
    ) -> pd.DataFrame:
        """Marginal CV-vs-amplitude curve for one channel, others left unphased.

        Answers "how much does phasing THIS channel alone help", isolated
        from the other channels — every other channel's spec is locked at 0
        (no phasing) while `channel` sweeps a symmetric +/-`magnitude_pct`
        continuous range from 0% (alpha=0) up to full amplitude (alpha=1),
        via the same `_generate_phased_schedule` mechanism `fit()` uses. If
        `blackout` is given, one further point is added using `Blackout`
        instead of a continuous range at that channel, so the two levers
        are directly comparable for this specific channel.

        This is a *marginal* curve, not the joint one `fit()` computes: it
        holds every other channel fixed, so it doesn't capture interaction
        effects from phasing several channels at once (that's what
        `results_`'s `cv_<channel>` columns already measure, across a
        single shared alpha applied to every channel together). Use this
        chart to decide which lever a channel needs; use `fit()`'s
        recommendation for the actual schedule to hand over.

        Parameters
        ----------
        channel:
            Channel to vary. Must be a column of plan_df.
        alphas:
            Alpha levels to evaluate for the continuous sweep. Defaults to
            [0, 0.2, 0.4, 0.6, 0.8, 1.0].
        magnitude_pct:
            Symmetric +/-X% weekly deviation at alpha=1 for the continuous
            sweep. Default 40.0, matching BudgetPhaser's own default.
        blackout:
            Optional Blackout spec to evaluate as one extra "Blackout" row
            at full amplitude, for comparison against the continuous
            sweep. Omit (None) to skip it.
        n_sims, n_phasing_seeds:
            Same meaning as in fit().
        fast_mode:
            If True, uses n_sims=10, n_phasing_seeds=1.

        Returns
        -------
        pd.DataFrame with one row per point: label, magnitude_pct
        (NaN for the Blackout row), is_blackout, cv (this channel's own
        coefficient of variation at that setting).
        """
        channels = list(self.plan_df.columns)
        if channel not in channels:
            raise ValueError(f"Unknown channel {channel!r}. Must be one of {channels}.")

        if fast_mode:
            n_sims = 10
            n_phasing_seeds = 1

        if alphas is None:
            alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        locked = dict.fromkeys(channels, 0.0)
        rows = []

        for i, alpha in enumerate(alphas):
            spec = {**locked, channel: magnitude_pct}
            result = self._evaluate_spec_at_alpha(
                spec,
                float(alpha),
                n_sims,
                n_phasing_seeds,
                seed_offset=10_000 + i * n_phasing_seeds,
            )
            rows.append(
                {
                    "label": f"{round(alpha * magnitude_pct)}%",
                    "magnitude_pct": round(alpha * magnitude_pct, 2),
                    "is_blackout": False,
                    "cv": result[f"cv_{channel}"],
                }
            )

        if blackout is not None:
            spec = {**locked, channel: blackout}
            result = self._evaluate_spec_at_alpha(
                spec, 1.0, n_sims, n_phasing_seeds, seed_offset=20_000
            )
            rows.append(
                {
                    "label": "Blackout",
                    "magnitude_pct": float("nan"),
                    "is_blackout": True,
                    "cv": result[f"cv_{channel}"],
                }
            )

        return pd.DataFrame(rows)

    def recommend_levers(
        self,
        magnitude_pct: float = 40.0,
        blackout: Blackout | None = Blackout(max_dark_weeks_per_month=1),
        improvement_threshold_pct: float = 10.0,
        n_sims: int = 50,
        n_phasing_seeds: int = 10,
        fast_mode: bool = False,
        channel_constraints: dict[str, DeviationSpec] | None = None,
    ) -> dict[str, DeviationSpec]:
        """Decide, per channel, whether Blackout is worth it over a continuous range.

        fit() applies a single shared lever to every channel by default,
        which can hide real per-channel differences: one channel might
        barely respond to a continuous +/-40% range while a hard on/off
        Blackout would meaningfully help it, and that gap is invisible
        unless someone actually checks each channel in isolation (this
        method exists because exactly that gap showed up in a real report
        — a headline channel that barely improved under the flat default
        lever, while another channel on the same report improved a lot
        more).

        For each channel, compares its own marginal CV (see
        channel_sensitivity — every other channel locked at 0) at full
        continuous amplitude (+/-magnitude_pct) against Blackout, and picks
        Blackout when it's at least improvement_threshold_pct relatively
        better for that channel.

        IMPORTANT — `improvement_threshold_pct` is a *policy* dial, not a
        noise filter, and it does not reliably distinguish "this channel
        needs Blackout" from "this channel can operationally tolerate
        Blackout." In practice Blackout tends to beat a +/-40% continuous
        range by 40%+ relative CV improvement for most channels (the two
        levers are different sampling mechanisms, not points on the same
        curve), so a default threshold of 10.0 rarely binds -- it will
        recommend Blackout for every channel that has *any* headroom left
        under a continuous range, including channels a media buyer
        considers operationally impossible to black out (a "must always be
        on" channel like a brand-safety or always-on search line). Do not
        treat this method's output as safe to apply unchanged: at minimum,
        review the returned dict before using it, and prefer passing
        `channel_constraints` below for any channel with a hard
        operational rule, rather than relying on the threshold to protect
        it.

        This is a one-shot, per-channel *marginal* decision, same caveat as
        channel_sensitivity(): it doesn't capture interaction effects from
        phasing several channels at once. Typical use: feed this method's
        output straight into a subsequent BudgetPhaser's own
        max_weekly_deviation_pct (ReportBuilder does this automatically by
        default — see its auto_lever parameter).

        Parameters
        ----------
        magnitude_pct:
            Symmetric +/-X% continuous option to compare against Blackout.
            Default 40.0, matching BudgetPhaser's own default lever.
        blackout:
            Blackout spec to compare against. If None, every channel keeps
            the continuous lever — nothing to compare against, so there's
            nothing to decide.
        improvement_threshold_pct:
            Minimum relative CV improvement (%) Blackout must show over the
            continuous option for a channel before it's picked instead. See
            the IMPORTANT note above — this rarely binds in practice, so
            don't rely on it alone to protect a channel that must never be
            blacked out; use `channel_constraints` for that. Default 10.0.
            At the default n_phasing_seeds=10, run-to-run sampling noise on
            this comparison is typically 2-5 percentage points, so a
            threshold decision (when it does bind) holds in the large
            majority of cases; noise widens somewhat at higher channel
            counts (seen up to ~5 points at 20 channels in testing), so
            this hasn't been proven safe at every scale, just at the
            counts checked so far. A much smaller n_phasing_seeds (e.g. 3)
            is not a safe setting to decide a lever with — noise there can
            reach 6-10 points and occasionally flip the decision on a real
            effect.
        n_sims, n_phasing_seeds:
            Same meaning as channel_sensitivity()/fit(). n_phasing_seeds
            defaults to 10 here (not fit()'s own default of 3) specifically
            so the lever decision above is stable enough to trust on its
            own, not just when called through ReportBuilder (which already
            passes 10, so this change doesn't alter its behaviour).
        fast_mode:
            If True, uses n_sims=10, n_phasing_seeds=1 — cheap, for
            iterating on the report itself, not a lever choice to actually
            hand a client.
        channel_constraints:
            Optional dict mapping channel name to a hard-pinned
            float | Blackout spec for that channel. Any channel listed
            here skips the CV comparison entirely and is returned with
            exactly this spec — e.g. {"tv": 0} to guarantee TV is never
            touched regardless of what the CV comparison would otherwise
            recommend, or {"brand_search": 20.0} to cap a channel to a
            gentler continuous range than `magnitude_pct` even if Blackout
            would score better. Channels not listed here are still decided
            by the usual CV comparison.

        Returns
        -------
        dict[str, float | Blackout], one entry per channel in plan_df,
        ready to pass as max_weekly_deviation_pct to a BudgetPhaser.
        """
        channels = list(self.plan_df.columns)
        if fast_mode:
            n_sims = 10
            n_phasing_seeds = 1
        channel_constraints = channel_constraints or {}

        spec: dict[str, DeviationSpec] = {}
        for ch in channels:
            if ch in channel_constraints:
                spec[ch] = channel_constraints[ch]
                continue
            if blackout is None:
                spec[ch] = magnitude_pct
                continue
            curve = self.channel_sensitivity(
                ch,
                alphas=[1.0],
                magnitude_pct=magnitude_pct,
                blackout=blackout,
                n_sims=n_sims,
                n_phasing_seeds=n_phasing_seeds,
            )
            cv_continuous = float(curve.loc[~curve["is_blackout"], "cv"].iloc[0])
            cv_blackout = float(curve.loc[curve["is_blackout"], "cv"].iloc[0])
            improvement = 100 * (cv_continuous - cv_blackout) / cv_continuous
            spec[ch] = (
                blackout if improvement >= improvement_threshold_pct else magnitude_pct
            )

        return spec

    def impact_over_horizons(
        self,
        horizons_weeks: list[int] | None = None,
        n_sims: int = 50,
        n_phasing_seeds: int = 3,
        fast_mode: bool = False,
        include_revenue: bool = False,
    ) -> pd.DataFrame:
        """Before-vs-after CV (and optionally £ revenue range) at several horizons.

        fit() must be called first — this reuses its recommended alpha
        rather than re-searching. For each horizon in `horizons_weeks`,
        plan_df is tiled to that length (see `_tile_plan`; shorter horizons
        truncate it, longer ones repeat its weekly pattern), then:
          - "today": history + tiled plan, unphased.
          - "after": history + tiled plan phased at the recommended alpha,
            averaged across n_phasing_seeds independent draws — the same
            averaging fit() already applies to its own CV curve, extended
            here across per-channel revenue ranges too (promoted from the
            single-draw-then-averaged pattern used in
            notebooks/02_phaser_walkthrough.ipynb).

        Parameters
        ----------
        horizons_weeks:
            Plan lengths to evaluate, in weeks. Defaults to [13, 52, 104]
            (roughly 3 months, 1 year, 2 years).
        n_sims, n_phasing_seeds:
            Same meaning as in fit().
        fast_mode:
            If True, uses n_sims=10, n_phasing_seeds=1.
        include_revenue:
            If True, also compute £ incremental-revenue ranges via
            CollinearityDiagnostic.summary(planned_spend=...), using
            plan_df's own total spend per channel as planned_spend at
            *every* horizon — not each horizon's own tiled-plan total.
            An earlier version of this priced each horizon against
            its own tiled spend, so a 104-week horizon (tiled to 2x the
            weeks) came out at roughly double the £ scale of the 52-week
            one — read by a client as "phase for longer, get more revenue,"
            when horizon here means "more experience/history has
            accumulated," not "more spend." This did not match
            notebooks/02_phaser_walkthrough.ipynb's own blackout_impact()
            helper (the source of overview.html's published £ figures),
            which always prices against the plan's own fixed annual total
            regardless of horizon — "same plan, £0 extra spend, just a
            tighter range" is the intended story throughout this package,
            and ReportBuilder's Impact section is meant to echo it. Without
            this parameter, only CV is returned (CV is dimensionless and
            was never affected by this).

        Returns
        -------
        pd.DataFrame with one row per (horizon, channel): horizon_weeks,
        channel, cv_today, cv_after, cv_reduction_pct, and — if
        include_revenue — revenue_today_p10/p90 and revenue_after_p10/p90.
        """
        if self.results_ is None or self.recommended_schedule_ is None:
            raise RuntimeError("Call fit() before impact_over_horizons().")

        if horizons_weeks is None:
            horizons_weeks = [13, 52, 104]
        if fast_mode:
            n_sims = 10
            n_phasing_seeds = 1

        best_alpha = float(self.recommend()["alpha"])
        channels = list(self.plan_df.columns)
        rows = []

        # Fixed across every horizon on purpose — see include_revenue's
        # docstring above. Revenue always prices against the plan's own
        # annual total; only the CV (and so the model-estimated range's
        # width) changes with horizon.
        fixed_planned_spend = self.plan_df.sum().to_dict() if include_revenue else None

        for h in horizons_weeks:
            tiled_plan = _tile_plan(self.plan_df, h)
            tiled_labels = _get_month_labels(tiled_plan)
            planned_spend = fixed_planned_spend

            today_combined = pd.concat([self.history_df, tiled_plan])
            today_diag = CollinearityDiagnostic(
                spend_df=today_combined,
                true_marginal_returns=self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
            )
            today_diag.fit(n_sims=n_sims)
            today_summ = today_diag.summary(planned_spend=planned_spend).set_index(
                "channel"
            )

            after_cv: dict[str, list[float]] = {ch: [] for ch in channels}
            after_p10: dict[str, list[float]] = {ch: [] for ch in channels}
            after_p90: dict[str, list[float]] = {ch: [] for ch in channels}

            for j in range(n_phasing_seeds):
                phased = _generate_phased_schedule(
                    tiled_plan,
                    tiled_labels,
                    alpha=best_alpha,
                    max_weekly_deviation_pct=self.max_weekly_deviation_pct,
                    seed=self.seed + 30_000 + h * 100 + j,
                    nudge_shape=self.nudge_shape,
                    balance_signs=self.balance_signs,
                )
                combined = pd.concat([self.history_df, phased])
                diag = CollinearityDiagnostic(
                    spend_df=combined,
                    true_marginal_returns=self.true_marginal_returns,
                    base_sales=self.base_sales,
                    revenue_noise_std=self.revenue_noise_std,
                )
                diag.fit(n_sims=n_sims)
                summ = diag.summary(planned_spend=planned_spend).set_index("channel")
                for ch in channels:
                    after_cv[ch].append(float(summ.loc[ch, "coef_of_variation"]))
                    if include_revenue:
                        after_p10[ch].append(
                            float(summ.loc[ch, "incremental_revenue_p10"])
                        )
                        after_p90[ch].append(
                            float(summ.loc[ch, "incremental_revenue_p90"])
                        )

            for ch in channels:
                cv_today = float(today_summ.loc[ch, "coef_of_variation"])
                cv_after = float(np.mean(after_cv[ch]))
                row = {
                    "horizon_weeks": h,
                    "channel": ch,
                    "cv_today": round(cv_today, 4),
                    "cv_after": round(cv_after, 4),
                    "cv_reduction_pct": round(
                        100 * (cv_today - cv_after) / cv_today, 2
                    ),
                }
                if include_revenue:
                    row["revenue_today_p10"] = float(
                        today_summ.loc[ch, "incremental_revenue_p10"]
                    )
                    row["revenue_today_p90"] = float(
                        today_summ.loc[ch, "incremental_revenue_p90"]
                    )
                    row["revenue_after_p10"] = float(np.mean(after_p10[ch]))
                    row["revenue_after_p90"] = float(np.mean(after_p90[ch]))
                rows.append(row)

        return pd.DataFrame(rows)

    def recommend(self) -> pd.Series:
        """Return the recommended alpha and its honestly re-evaluated CV.

        This is deliberately *not* the grid's own lowest-max_cv row — see
        fit()'s docstring on selection-bias correction. It's the best of
        the top confirm_top_k candidates after independent re-evaluation on
        a larger sample, and its alpha is what recommended_schedule_ is
        generated at. Note this row's own max_cv is a separate estimate
        from `recommended_schedule_median_cv_` (the median max CV across
        the n_recommended_draws schedules evaluated at this alpha before
        picking the one actually shipped) — the two should agree
        approximately but not exactly, since they come from independent
        simulation batches. Prefer `recommended_schedule_median_cv_` when
        describing the CV of the schedule actually being handed over. Use
        summary()/results_ to see the raw, noisier grid instead (useful for
        plotting the CV-vs-alpha curve, not for citing a specific channel's
        CV at the recommended alpha).

        Returns
        -------
        pd.Series with alpha, actual_correlation, max_cv, and per-channel CVs.
        """
        if self.confirmation_ is None:
            raise RuntimeError("Call fit() before recommend().")
        return self.confirmation_.loc[self.confirmation_["max_cv"].idxmin()]

    def summary(self) -> pd.DataFrame:
        """Return the full grid search results.

        Returns
        -------
        pd.DataFrame with one row per alpha level.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")
        return self.results_
