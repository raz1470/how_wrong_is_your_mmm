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
channel), concatenates it with the history, fits a weighted CollinearityDiagnostic
(so the plan year has more influence than the correlated history), and measures
the max CV across channels. The recommended alpha minimises max CV.

Three weighting schemes are supported:
  "uniform"  →  all observations weighted equally (baseline)
  "binary"   →  history gets weight 1, plan year gets plan_weight (default 5)
  "decay"    →  exponential decay from most recent week, parameterised by half_life

The output is a concrete plan-year weekly spend schedule the practitioner can
hand to their media agency, with monthly totals unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import _DEFAULT_ELASTICITIES
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic

_WEIGHTING_SCHEMES = ("uniform", "binary", "decay")


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


def _generate_phased_schedule(
    spend_df: pd.DataFrame,
    month_labels: np.ndarray,
    alpha: float,
    max_weekly_deviation_pct: float,
    seed: int,
) -> pd.DataFrame:
    """Generate one phased weekly schedule for a given amplitude alpha.

    For each month and each channel independently:
    1. Draw random weekly deviations bounded by alpha x max_weekly_deviation_pct.
    2. Rescale so the monthly total is exactly preserved.
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
        Maximum per-channel weekly deviation (%) at alpha=1.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame with the same shape and index as spend_df.
    """
    rng = np.random.default_rng(seed)
    n_channels = spend_df.shape[1]
    new_spend = spend_df.to_numpy().copy().astype(float)
    max_dev = alpha * max_weekly_deviation_pct / 100.0

    for month in np.unique(month_labels):
        mask = np.where(month_labels == month)[0]
        n_weeks = len(mask)
        for ci in range(n_channels):
            orig_weeks = spend_df.iloc[mask, ci].to_numpy()
            monthly_total = orig_weeks.sum()
            raw = rng.uniform(-max_dev, max_dev, size=n_weeks)
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


def _compute_weights(
    n_history: int,
    n_plan: int,
    weighting: str,
    plan_weight: float,
    half_life: int,
) -> np.ndarray:
    """Compute per-observation weights for the combined history + plan dataset.

    Parameters
    ----------
    n_history:
        Number of historical observations (rows in history_df).
    n_plan:
        Number of plan-year observations (rows in plan_df).
    weighting:
        One of "uniform", "binary", or "decay".
    plan_weight:
        Weight assigned to plan-year observations under "binary" scheme.
    half_life:
        Number of weeks over which weight halves under "decay" scheme.

    Returns
    -------
    np.ndarray of shape (n_history + n_plan,).
    """
    if weighting not in _WEIGHTING_SCHEMES:
        raise ValueError(
            f"weighting must be one of {_WEIGHTING_SCHEMES}. Got '{weighting}'."
        )
    n_total = n_history + n_plan
    if weighting == "uniform":
        return np.ones(n_total)
    if weighting == "binary":
        w = np.ones(n_total)
        w[n_history:] = plan_weight
        return w
    # decay: exponential, most recent observation has weight 1
    distances = np.arange(n_total - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2) / half_life * distances)


class BudgetPhaser:
    """Recommend the weekly spend phasing needed to reduce elasticity uncertainty.

    Takes a multi-year spend history and a plan-year budget. Grid-searches over
    phasing amplitude to find the plan-year schedule that minimises max CV across
    channels (under a weighted OLS that upweights the plan year), while preserving
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
    weighting:
        How to weight observations in the diagnostic OLS.
        "uniform"  — all observations equally weighted (baseline).
        "binary"   — history gets weight 1, plan year gets plan_weight.
        "decay"    — exponential decay from most recent week, half-life
                     controlled by half_life parameter.
        Default "binary".
    plan_weight:
        Weight assigned to plan-year observations under "binary" weighting.
        Default 5.0 (plan year counts 5x as much as a history week).
    half_life:
        Weeks over which weight halves under "decay" weighting. Default 52.
    max_monthly_deviation_pct:
        Maximum allowed fractional deviation in monthly totals per channel (%).
        Default 1.0. Enforced by construction (rescaling).
    max_weekly_deviation_pct:
        Maximum per-channel weekly deviation from original plan spend at
        alpha=1 (%). Default 40.0.
    seed:
        Base random seed.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        true_elasticities: dict[str, float] | None = None,
        weighting: str = "binary",
        plan_weight: float = 5.0,
        half_life: int = 52,
        max_monthly_deviation_pct: float = 1.0,
        max_weekly_deviation_pct: float = 40.0,
        seed: int = 0,
    ) -> None:
        _get_month_labels(history_df)  # validates DatetimeIndex
        _get_month_labels(plan_df)  # validates DatetimeIndex

        if list(history_df.columns) != list(plan_df.columns):
            raise ValueError(
                "history_df and plan_df must have the same columns. "
                f"Got {list(history_df.columns)} vs {list(plan_df.columns)}."
            )
        if weighting not in _WEIGHTING_SCHEMES:
            raise ValueError(
                f"weighting must be one of {_WEIGHTING_SCHEMES}. Got '{weighting}'."
            )

        self.history_df = history_df
        self.plan_df = plan_df
        self.true_elasticities = (
            true_elasticities
            if true_elasticities is not None
            else _DEFAULT_ELASTICITIES
        )
        self.weighting = weighting
        self.plan_weight = plan_weight
        self.half_life = half_life
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
        fast_mode: bool = False,
    ) -> BudgetPhaser:
        """Grid-search over phasing amplitude and store results.

        For each alpha:
          1. Generate a phased plan schedule (monthly totals preserved).
          2. Concatenate history + phased plan into a single dataset.
          3. Compute observation weights (upweighting the plan year).
          4. Run CollinearityDiagnostic with weighted OLS.
          5. Record max CV across channels.

        Parameters
        ----------
        n_sims:
            Number of noise seeds per grid point for CollinearityDiagnostic.
        grid_steps:
            Number of alpha levels to evaluate.
        fast_mode:
            If True, uses n_sims=10 and grid_steps=10.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            grid_steps = 10

        weights = _compute_weights(
            n_history=len(self.history_df),
            n_plan=len(self.plan_df),
            weighting=self.weighting,
            plan_weight=self.plan_weight,
            half_life=self.half_life,
        )

        alphas = np.linspace(0, 1, grid_steps)
        channels = list(self.plan_df.columns)
        rows = []

        for i, alpha in enumerate(alphas):
            phased_plan = _generate_phased_schedule(
                self.plan_df,
                self._plan_month_labels,
                alpha=float(alpha),
                max_weekly_deviation_pct=self.max_weekly_deviation_pct,
                seed=self.seed + i,
            )

            monthly_dev = _max_monthly_deviation(
                self.plan_df, phased_plan, self._plan_month_labels
            )

            combined = pd.concat([self.history_df, phased_plan])

            diag = CollinearityDiagnostic(
                spend_df=combined,
                true_elasticities=self.true_elasticities,
                weights=weights,
            )
            diag.fit(n_sims=n_sims)
            summ = diag.summary().set_index("channel")

            max_cv = float(summ["coef_of_variation"].max())

            row: dict = {
                "alpha": round(float(alpha), 4),
                "actual_correlation": round(diag.actual_correlation, 4),
                "max_cv": round(max_cv, 4),
                "max_monthly_deviation_pct": round(monthly_dev * 100, 6),
            }
            for ch in channels:
                row[f"cv_{ch}"] = round(float(summ.loc[ch, "coef_of_variation"]), 4)

            rows.append(row)

        self.results_ = pd.DataFrame(rows)

        # generate the recommended schedule at the best alpha
        best_alpha = float(self.results_.loc[self.results_["max_cv"].idxmin(), "alpha"])
        self.recommended_schedule_ = _generate_phased_schedule(
            self.plan_df,
            self._plan_month_labels,
            alpha=best_alpha,
            max_weekly_deviation_pct=self.max_weekly_deviation_pct,
            seed=self.seed + grid_steps,  # distinct seed from the grid search
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
