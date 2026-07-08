"""Collinearity diagnostic: quantify elasticity unreliability across simulations.

The core insight: OLS is unbiased but unreliable under collinearity. The same
market with the same true elasticities produces very different OLS estimates
depending on which period of data you happen to observe. This class quantifies
that unreliability.

One pipeline, two entry points:
  - Synthetic spend: pass correlation, spend is generated internally for N channels.
  - Real spend: pass spend_df, only sales are simulated.

In both cases, n_sims sales columns are simulated with different noise seeds
and OLS is fit on each. The distribution of estimates is the diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import (
    _DEFAULT_CHANNELS,
    _DEFAULT_ELASTICITIES,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._mmm import fit_ols


class CollinearityDiagnostic:
    """Quantify how unreliable OLS elasticities are given a spend dataset.

    Parameters
    ----------
    correlation:
        Target pairwise correlation between all channel pairs. Used only
        when spend_df is None (synthetic spend path).
    spend_df:
        Real spend DataFrame with one column per channel. When supplied,
        synthetic spend generation is skipped.
    channels:
        List of channel names for synthetic spend generation. Ignored when
        spend_df is supplied (channels are inferred from spend_df.columns).
        Defaults to ["tv", "meta", "search"].
    true_elasticities:
        Dict mapping channel name to true elasticity. Used to simulate sales.
        Must cover all channels in the spend data.
        Defaults to {"tv": 0.3, "meta": 0.5, "search": 0.4}.
    n_obs:
        Number of observations for synthetic spend. Ignored when spend_df
        is supplied.
    spend_seed:
        Random seed for synthetic spend generation. Ignored when spend_df
        is supplied.
    base_sales:
        Base sales intercept in the synthetic sales equation.
    revenue_noise_std:
        Standard deviation of sales noise.
    """

    def __init__(
        self,
        correlation: float = 0.7,
        spend_df: pd.DataFrame | None = None,
        channels: list[str] | None = None,
        true_elasticities: dict[str, float] | None = None,
        n_obs: int = 104,
        spend_seed: int = 0,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 20_000.0,
    ) -> None:
        self.correlation = correlation
        self.spend_df = spend_df
        self.channels = channels if channels is not None else _DEFAULT_CHANNELS
        self.true_elasticities = (
            true_elasticities
            if true_elasticities is not None
            else _DEFAULT_ELASTICITIES
        )
        self.n_obs = n_obs
        self.spend_seed = spend_seed
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std

        self.spend_df_: pd.DataFrame | None = None
        self.channels_: list[str] = []
        self.results_: pd.DataFrame | None = None

    def fit(self, n_sims: int = 50, fast_mode: bool = False) -> CollinearityDiagnostic:
        """Run the diagnostic.

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

        if self.spend_df is not None:
            self.spend_df_ = self.spend_df.copy()
            self.channels_ = list(self.spend_df.columns)
        else:
            self.spend_df_ = simulate_spend(
                n_obs=self.n_obs,
                correlation=self.correlation,
                channels=self.channels,
                seed=self.spend_seed,
            )
            self.channels_ = list(self.channels)

        records = []
        for sim in range(n_sims):
            sales = simulate_sales(
                spend_df=self.spend_df_,
                true_elasticities=self.true_elasticities,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                seed=sim,
            )
            estimated = fit_ols(self.spend_df_, sales)
            for channel in self.channels_:
                true_e = self.true_elasticities[channel]
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

    def summary(
        self,
        planned_spend: dict[str, float] | None = None,
        value_per_unit: float | None = None,
    ) -> pd.DataFrame:
        """Return a summary of elasticity estimates across simulations.

        Parameters
        ----------
        planned_spend:
            Optional dict mapping channel name to planned spend. When
            supplied, adds an incremental-revenue range (p10/p90) per
            channel, computed as the simulated elasticity distribution
            multiplied by planned spend (and by `value_per_unit`, if that
            is also given). Assumes sales is already a £ value (revenue)
            when `value_per_unit` is not supplied. Must cover every channel
            in the fitted data; extra keys are ignored.
        value_per_unit:
            Optional £ value per unit of "sales" — e.g. average LTV per new
            customer, for use when the sales column represents signups or
            conversions rather than £ revenue directly. When supplied,
            adds CAC (£ spend per unit of sales) and ROI (£ value per £
            spent) ranges (p10/p90) per channel, computed per simulation
            draw as ``cac = 1 / estimated_elasticity`` and
            ``roi = estimated_elasticity * value_per_unit``. In this
            linear DGP both are spend-independent channel properties —
            they don't depend on `planned_spend`. Draws with elasticity
            near zero can make CAC swing wildly or go negative; that
            instability is itself part of the diagnostic (an unreliable
            elasticity estimate makes for an unreliable CAC estimate too).
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

        if planned_spend is not None:
            missing = set(self.channels_) - set(planned_spend.keys())
            if missing:
                raise KeyError(
                    f"planned_spend is missing channel(s): {sorted(missing)}"
                )
            multiplier = 1.0 if value_per_unit is None else value_per_unit
            revenue = self.results_.copy()
            revenue["planned_spend"] = revenue["channel"].map(planned_spend)
            revenue["incremental_revenue"] = (
                revenue["estimated_elasticity"] * revenue["planned_spend"] * multiplier
            )
            revenue_range = (
                revenue.groupby("channel")["incremental_revenue"]
                .quantile([0.1, 0.9])
                .unstack()
                .rename(
                    columns={
                        0.1: "incremental_revenue_p10",
                        0.9: "incremental_revenue_p90",
                    }
                )
                .reset_index()
            )
            summary = summary.merge(revenue_range, on="channel")

        if value_per_unit is not None:
            derived = self.results_.copy()
            derived["cac"] = 1.0 / derived["estimated_elasticity"]
            derived["roi"] = derived["estimated_elasticity"] * value_per_unit
            derived_range = (
                derived.groupby("channel")[["cac", "roi"]]
                .quantile([0.1, 0.9])
                .unstack()
            )
            derived_range.columns = [
                f"{metric}_p{int(q * 100)}" for metric, q in derived_range.columns
            ]
            derived_range = derived_range.reset_index()
            summary = summary.merge(derived_range, on="channel")

        return summary.round(4)

    @property
    def actual_correlation(self) -> float:
        """Mean pairwise Pearson correlation across all channel pairs."""
        if self.spend_df_ is None:
            raise RuntimeError("Call fit() first.")
        corr = self.spend_df_.corr().to_numpy()
        n = len(self.channels_)
        pairs = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(pairs))

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Full Pearson correlation matrix across all channels."""
        if self.spend_df_ is None:
            raise RuntimeError("Call fit() first.")
        return self.spend_df_.corr()
