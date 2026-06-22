"""Budget perturbation recommender.

The core idea: collinearity comes from both channels tracking the same demand
signal. The fix is to introduce *independent* variation in the TV/Meta split —
some weeks deliberately lean into TV, others into Meta, independent of what
demand is doing. This gives OLS the variation it needs to distinguish individual
channel effects.

BudgetPerturber quantifies how much independent variation you need to add to
bring elasticity uncertainty down to an acceptable level. It does this by:

1. Taking your current spend data (real or synthetic).
2. Adding increasing amounts of independent budget variation between channels
   (total spend per week is preserved — only the TV/Meta split varies).
3. Running CollinearityDiagnostic at each level and measuring the resulting
   elasticity CV.
4. Returning the perturbation level that minimises max CV across channels,
   along with the full curve so you can see the trade-off.

The output is a concrete recommendation: "vary your TV/Meta split by an
additional ±£X per week, independent of your campaign calendar."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic


def _perturb_spend(
    spend_df: pd.DataFrame,
    perturbation_std: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Add independent variation to the TV/Meta split.

    Shifts budget between channels by a random amount each week,
    independently of the underlying demand signal. Total spend per
    week is preserved — only the split changes.

    Parameters
    ----------
    spend_df:
        DataFrame with columns 'tv' and 'meta'.
    perturbation_std:
        Standard deviation of the weekly budget shift in £. At 0,
        spend is unchanged. Higher values introduce more independent
        variation in the channel split.
    seed:
        Random seed for the perturbation draws.

    Returns
    -------
    pd.DataFrame with columns 'tv' and 'meta' after perturbation.
    """
    rng = np.random.default_rng(seed)
    shift = rng.standard_normal(len(spend_df)) * perturbation_std
    return pd.DataFrame(
        {
            "tv": spend_df["tv"].to_numpy() + shift,
            "meta": spend_df["meta"].to_numpy() - shift,
        }
    )


class BudgetPerturber:
    """Recommend the budget variation needed to reduce elasticity uncertainty.

    Parameters
    ----------
    spend_df:
        DataFrame with columns 'tv' and 'meta'. Can be real spend data
        or synthetic spend from simulate_spend.
    true_elast_tv:
        True TV elasticity used to simulate sales in the diagnostic.
    true_elast_meta:
        True Meta elasticity used to simulate sales in the diagnostic.
    base_sales:
        Base sales intercept passed to simulate_sales.
    revenue_noise_std:
        Sales noise std passed to simulate_sales.
    max_perturbation_pct:
        Upper bound of the grid search, expressed as a fraction of mean
        weekly total spend. Default 0.5 means the grid goes up to a shift
        of ±50% of mean weekly budget.
    seed:
        Base random seed. Each grid point uses seed + grid_index so
        results are reproducible but vary across the grid.
    """

    def __init__(
        self,
        spend_df: pd.DataFrame,
        true_elast_tv: float = 0.3,
        true_elast_meta: float = 0.5,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 20_000.0,
        max_perturbation_pct: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.spend_df = spend_df
        self.true_elast_tv = true_elast_tv
        self.true_elast_meta = true_elast_meta
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std
        self.max_perturbation_pct = max_perturbation_pct
        self.seed = seed

        self.results_: pd.DataFrame | None = None
        self.mean_weekly_total_: float | None = None

    def fit(
        self,
        n_sims: int = 50,
        grid_steps: int = 20,
        fast_mode: bool = False,
    ) -> BudgetPerturber:
        """Run the grid search over perturbation levels.

        For each perturbation level, generates perturbed spend, runs
        CollinearityDiagnostic, and records the max CV across channels.

        Parameters
        ----------
        n_sims:
            Number of simulation seeds per grid point.
        grid_steps:
            Number of perturbation levels to evaluate.
        fast_mode:
            If True, uses n_sims=10 and grid_steps=10 for fast iteration.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            grid_steps = 10

        mean_total = float((self.spend_df["tv"] + self.spend_df["meta"]).mean())
        self.mean_weekly_total_ = mean_total

        max_std = self.max_perturbation_pct * mean_total
        perturbation_stds = np.linspace(0, max_std, grid_steps)

        rows = []
        for i, pert_std in enumerate(perturbation_stds):
            perturbed = _perturb_spend(self.spend_df, pert_std, seed=self.seed + i)
            diag = CollinearityDiagnostic(
                spend_df=perturbed,
                true_elast_tv=self.true_elast_tv,
                true_elast_meta=self.true_elast_meta,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
            )
            diag.fit(n_sims=n_sims)
            summ = diag.summary()

            tv_cv = float(summ.loc[summ["channel"] == "tv", "coef_of_variation"].iloc[0])
            meta_cv = float(summ.loc[summ["channel"] == "meta", "coef_of_variation"].iloc[0])

            rows.append(
                {
                    "perturbation_std": round(pert_std, 2),
                    "perturbation_pct": round(100 * pert_std / mean_total, 1),
                    "actual_correlation": round(diag.actual_correlation, 4),
                    "tv_cv": round(tv_cv, 4),
                    "meta_cv": round(meta_cv, 4),
                    "max_cv": round(max(tv_cv, meta_cv), 4),
                }
            )

        self.results_ = pd.DataFrame(rows)
        return self

    def recommend(self) -> pd.Series:
        """Return the grid point with the lowest max CV.

        Returns
        -------
        pd.Series with the recommended perturbation level and its diagnostics.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_.loc[self.results_["max_cv"].idxmin()]

    def summary(self) -> pd.DataFrame:
        """Return the full grid search results.

        Returns
        -------
        pd.DataFrame with one row per perturbation level.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_
