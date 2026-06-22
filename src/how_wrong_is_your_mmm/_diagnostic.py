"""Collinearity diagnostic: quantify elasticity unreliability across simulations.

The core insight: OLS is unbiased but unreliable under collinearity. The same
market with the same true elasticities produces very different OLS estimates
depending on which period of data you happen to observe. This class quantifies
that unreliability.

One pipeline, two entry points:
  - Synthetic spend: pass correlation, spend is generated internally.
  - Real spend: pass spend_df, only sales are simulated.

In both cases, n_sims sales columns are simulated with different noise seeds
and OLS is fit on each. The distribution of estimates is the diagnostic.
"""

from __future__ import annotations

import pandas as pd

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend
from how_wrong_is_your_mmm._mmm import fit_ols


class CollinearityDiagnostic:
    """Quantify how unreliable OLS elasticities are given a spend dataset.

    Parameters
    ----------
    correlation:
        Target correlation between TV and Meta spend. Used only when
        spend_df is None (synthetic spend path).
    spend_df:
        Real spend DataFrame with columns 'tv' and 'meta'. When supplied,
        synthetic spend generation is skipped and this data is used directly.
    n_obs:
        Number of observations for synthetic spend. Ignored when spend_df
        is supplied.
    spend_seed:
        Random seed for synthetic spend generation. Ignored when spend_df
        is supplied.
    true_elast_tv:
        True TV elasticity used to generate synthetic sales.
    true_elast_meta:
        True Meta elasticity used to generate synthetic sales.
    base_sales:
        Base sales intercept in the synthetic sales equation.
    revenue_noise_std:
        Standard deviation of sales noise.
    """

    def __init__(
        self,
        correlation: float = 0.7,
        spend_df: pd.DataFrame | None = None,
        n_obs: int = 104,
        spend_seed: int = 0,
        true_elast_tv: float = 0.3,
        true_elast_meta: float = 0.5,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 20_000.0,
    ) -> None:
        self.correlation = correlation
        self.spend_df = spend_df
        self.n_obs = n_obs
        self.spend_seed = spend_seed
        self.true_elast_tv = true_elast_tv
        self.true_elast_meta = true_elast_meta
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std

        self.spend_df_: pd.DataFrame | None = None
        self.results_: pd.DataFrame | None = None

    def fit(self, n_sims: int = 50, fast_mode: bool = False) -> CollinearityDiagnostic:
        """Run the diagnostic.

        Simulates n_sims sales columns (different noise seeds) from the spend
        data, fits OLS on each, and stores the distribution of estimates.

        Parameters
        ----------
        n_sims:
            Number of simulations (noise seeds).
        fast_mode:
            If True, overrides n_sims=10 for quick notebook iteration.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10

        # resolve spend — real or synthetic
        if self.spend_df is not None:
            self.spend_df_ = self.spend_df.copy()
        else:
            self.spend_df_ = simulate_spend(
                n_obs=self.n_obs,
                correlation=self.correlation,
                seed=self.spend_seed,
            )

        records = []
        for sim in range(n_sims):
            sales = simulate_sales(
                spend_df=self.spend_df_,
                true_elast_tv=self.true_elast_tv,
                true_elast_meta=self.true_elast_meta,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                seed=sim,
            )
            estimated = fit_ols(self.spend_df_, sales)
            for channel in ("tv", "meta"):
                true_e = self.true_elast_tv if channel == "tv" else self.true_elast_meta
                est_e = estimated[channel]
                records.append(
                    {
                        "sim": sim,
                        "channel": channel,
                        "true_elasticity": true_e,
                        "estimated_elasticity": est_e,
                        "error": est_e - true_e,
                        "error_pct": 100 * (est_e - true_e) / true_e,
                    }
                )

        self.results_ = pd.DataFrame(records)
        return self

    def summary(self) -> pd.DataFrame:
        """Return a summary of elasticity estimates across simulations.

        Returns
        -------
        pd.DataFrame with one row per channel showing true elasticity,
        mean and std of estimates, and coefficient of variation.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")

        summary = (
            self.results_.groupby("channel")
            .agg(
                true_elasticity=("true_elasticity", "first"),
                mean_estimated=("estimated_elasticity", "mean"),
                std_estimated=("estimated_elasticity", "std"),
                mean_error_pct=("error_pct", "mean"),
            )
            .reset_index()
        )
        summary["coef_of_variation"] = (
            summary["std_estimated"] / summary["mean_estimated"].abs()
        ).round(4)
        return summary.round(4)

    @property
    def actual_correlation(self) -> float:
        """Pearson correlation between TV and Meta in the spend data."""
        if self.spend_df_ is None:
            raise RuntimeError("Call fit() first.")
        return float(self.spend_df_["tv"].corr(self.spend_df_["meta"]))
