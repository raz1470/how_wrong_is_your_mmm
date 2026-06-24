"""Budget phasing recommender.

The core idea: collinearity comes from all channels tracking the same demand
signal. The fix is to introduce *independent* variation in the weekly channel
mix — some weeks deliberately lean into TV, others into Meta or Search —
while keeping monthly budgets intact.

BudgetPhaser works on a 52-week spend plan (DatetimeIndex required so it
knows which month each week belongs to). It grid-searches over a phasing
amplitude α ∈ [0, 1]:

  α = 0  →  no change from original plan
  α = 1  →  maximum allowed variation under the channel constraint

For each α it generates a phased schedule (monthly totals preserved per
channel), runs CollinearityDiagnostic, and measures the max CV across channels.
The recommended α is the one that minimises max CV.

The output is a concrete 52×N weekly spend schedule the practitioner can hand
to their media agency.
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


def _generate_phased_schedule(
    spend_df: pd.DataFrame,
    month_labels: np.ndarray,
    alpha: float,
    max_weekly_deviation_pct: float,
    seed: int,
) -> pd.DataFrame:
    """Generate one phased weekly schedule for a given amplitude alpha.

    For each month and each channel independently:
    1. Draw random weekly deviations bounded by alpha × max_weekly_deviation_pct.
    2. Rescale so the monthly total is exactly preserved.
    3. Apply to original spend.

    Parameters
    ----------
    spend_df:
        52×N DataFrame with DatetimeIndex.
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


class BudgetPhaser:
    """Recommend the weekly spend phasing needed to reduce elasticity uncertainty.

    Takes a 52-week spend plan (DatetimeIndex required) and grid-searches over
    phasing amplitude to find the schedule that minimises max CV across channels
    while preserving monthly budgets.

    Parameters
    ----------
    spend_df:
        52×N DataFrame with a weekly DatetimeIndex. One column per channel.
        Use simulate_spend(n_obs=52, start_date=...) for synthetic data.
    true_elasticities:
        Dict mapping channel name to true elasticity. Defaults to
        {"tv": 0.3, "meta": 0.5, "search": 0.4}.
    max_monthly_deviation_pct:
        Maximum allowed fractional deviation in monthly totals per channel (%).
        Default 1.0. Enforced by construction (rescaling); this param is used
        for validation and reporting only.
    max_weekly_deviation_pct:
        Maximum per-channel weekly deviation from original spend at alpha=1 (%).
        Default 40.0.
    seed:
        Base random seed.
    """

    def __init__(
        self,
        spend_df: pd.DataFrame,
        true_elasticities: dict[str, float] | None = None,
        max_monthly_deviation_pct: float = 1.0,
        max_weekly_deviation_pct: float = 40.0,
        seed: int = 0,
    ) -> None:
        self.spend_df = spend_df
        self.true_elasticities = (
            true_elasticities if true_elasticities is not None else _DEFAULT_ELASTICITIES
        )
        self.max_monthly_deviation_pct = max_monthly_deviation_pct
        self.max_weekly_deviation_pct = max_weekly_deviation_pct
        self.seed = seed

        self._month_labels = _get_month_labels(spend_df)
        self.results_: pd.DataFrame | None = None
        self.recommended_schedule_: pd.DataFrame | None = None

    def fit(
        self,
        n_sims: int = 50,
        grid_steps: int = 20,
        fast_mode: bool = False,
    ) -> BudgetPhaser:
        """Grid-search over phasing amplitude and store results.

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

        alphas = np.linspace(0, 1, grid_steps)
        channels = list(self.spend_df.columns)
        rows = []

        for i, alpha in enumerate(alphas):
            phased = _generate_phased_schedule(
                self.spend_df,
                self._month_labels,
                alpha=float(alpha),
                max_weekly_deviation_pct=self.max_weekly_deviation_pct,
                seed=self.seed + i,
            )

            monthly_dev = _max_monthly_deviation(
                self.spend_df, phased, self._month_labels
            )

            diag = CollinearityDiagnostic(
                spend_df=phased,
                true_elasticities=self.true_elasticities,
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
            self.spend_df,
            self._month_labels,
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
            raise RuntimeError("Call fit() first.")
        return self.results_.loc[self.results_["max_cv"].idxmin()]

    def summary(self) -> pd.DataFrame:
        """Return the full grid search results.

        Returns
        -------
        pd.DataFrame with one row per alpha level.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_
