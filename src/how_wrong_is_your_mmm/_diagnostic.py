"""Collinearity diagnostic: quantify marginal-return unreliability across simulations.

The core insight: OLS is unbiased but unreliable under collinearity. The same
market with the same true marginal returns produces very different OLS
estimates depending on which period of data you happen to observe. This class
quantifies that unreliability.

Scope note: this diagnostic measures sampling variance under a model that is
correctly specified by construction (the DGP and the estimator share the same
linear functional form). It is silent on misspecification -- an omitted
driver (seasonality, a competitor event, adstock, saturation) can leave this
diagnostic looking healthy while the point estimate itself is badly biased.
Read the coefficient of variation (CV) below as "how identifiable is this
design", not "how correct is this model." See docs/research.html for the
full scope discussion.

One pipeline, two entry points:
  - Synthetic spend: pass correlation, spend is generated internally for N channels.
  - Real spend: pass spend_df, only sales are simulated.

In both cases, n_sims sales columns are simulated with different noise seeds
and OLS is fit on each. The distribution of estimates is the diagnostic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import (
    _DEFAULT_CHANNELS,
    _DEFAULT_MARGINAL_RETURNS,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._mmm import fit_ols

# Below this many observations per channel, OLS still runs but the estimate
# is wide enough to be more noise than signal -- worth a warning, not a hard
# stop (a user exploring a short pilot window is a legitimate use case).
_MIN_OBS_PER_CHANNEL_WARNING = 10


def _validate_spend_data(spend_df: pd.DataFrame) -> None:
    """Validate a real spend DataFrame before it's fit with OLS.

    Real spend data (unlike this package's own synthetic DGP output) can be
    missing values, constant for a channel, or too short relative to the
    number of channels -- all of which either crash `numpy.linalg.lstsq`
    with a cryptic LAPACK error, or let it return a finite but meaningless
    number silently. This turns those into one specific, actionable
    ValueError each, raised before any simulation runs.

    Called from every path that fits real spend: CollinearityDiagnostic's
    own real-spend entry point, and (via the combined history + phased
    plan DataFrame it builds internally) BudgetPhaser and ReportBuilder.
    """
    channels = list(spend_df.columns)

    non_numeric = [
        c for c in channels if not pd.api.types.is_numeric_dtype(spend_df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"spend_df column(s) {non_numeric} are not numeric. "
            "Every channel column must be a numeric spend series."
        )

    nan_cols = spend_df.columns[spend_df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"spend_df has missing (NaN) values in column(s) {nan_cols}. "
            "OLS cannot fit through missing data -- fill or drop these rows "
            "before running the diagnostic."
        )

    if not np.isfinite(spend_df.to_numpy(dtype=float)).all():
        raise ValueError(
            "spend_df contains non-finite values (inf or -inf). Check for "
            "a divide-by-zero or overflow upstream before running the "
            "diagnostic."
        )

    zero_variance = [c for c in channels if spend_df[c].std(ddof=0) == 0]
    if zero_variance:
        raise ValueError(
            f"spend_df channel(s) {zero_variance} have zero variance "
            "(constant spend, often all-zero for the period). An "
            "elasticity can't be estimated for a channel with no "
            "week-to-week variation -- exclude it or supply spend that "
            "actually varies."
        )

    min_required = len(channels) + 2  # +1 intercept, +1 residual degree of freedom
    if len(spend_df) < min_required:
        raise ValueError(
            f"spend_df has {len(spend_df)} observation(s) for "
            f"{len(channels)} channel(s), but OLS needs at least "
            f"{min_required} (one per channel, one for the intercept, and "
            "one left over so there's a residual to estimate against). "
            "Results below this are mathematically undefined, not just "
            "unreliable."
        )

    obs_per_channel = len(spend_df) / len(channels)
    if obs_per_channel < _MIN_OBS_PER_CHANNEL_WARNING:
        warnings.warn(
            f"spend_df has only {len(spend_df)} observations across "
            f"{len(channels)} channels ({obs_per_channel:.1f} per channel). "
            f"Below {_MIN_OBS_PER_CHANNEL_WARNING} observations per channel, "
            "elasticity estimates tend to be very wide and unstable even "
            "before collinearity is a factor -- treat results with extra "
            "caution, and prefer more history if it's available.",
            stacklevel=3,
        )


class CollinearityDiagnostic:
    """Quantify how unreliable OLS marginal-return estimates are for a spend dataset.

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
    true_marginal_returns:
        Dict mapping channel name to true marginal return (£ revenue per
        £ spend, a.k.a. mROAS -- NOT an economic elasticity, see _dgp.py).
        Used to simulate sales. Must cover all channels in the spend data.
        Defaults to {"tv": 2.0, "meta": 3.5, "search": 6.0}. These are a
        defensible illustrative starting point, not a claim about any real
        market -- for your own data, supply your own per-channel values
        (e.g. from a prior model, a plausible ROI range from finance, or a
        held-out incrementality test). There is no safe universal default:
        CV below is exactly inversely proportional to whatever value you
        supply (see coef_of_variation), so an unrealistic value shifts
        where the model-estimated range sits, even though it doesn't
        change how much phasing narrows it in relative terms.
    n_obs:
        Number of observations for synthetic spend. Ignored when spend_df
        is supplied.
    spend_seed:
        Random seed for synthetic spend generation. Ignored when spend_df
        is supplied.
    base_sales:
        Base sales intercept in the synthetic sales equation.
    revenue_noise_std:
        Standard deviation of sales noise (£). No universal default is
        correct here either -- CV scales ~linearly with this value (see
        coef_of_variation). Set it from your own model's residual std, or
        the residual std of a simple OLS fit on your actual sales/spend
        history, rather than relying on this package's default.
    true_elasticities:
        Deprecated alias for `true_marginal_returns`, kept for backward
        compatibility. Raises ValueError if both are supplied. Emits a
        FutureWarning -- migrate to `true_marginal_returns`.
    """

    def __init__(
        self,
        correlation: float = 0.7,
        spend_df: pd.DataFrame | None = None,
        channels: list[str] | None = None,
        true_marginal_returns: dict[str, float] | None = None,
        n_obs: int = 104,
        spend_seed: int = 0,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 20_000.0,
        true_elasticities: dict[str, float] | None = None,
    ) -> None:
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

        self.correlation = correlation
        self.spend_df = spend_df
        self.channels = channels if channels is not None else _DEFAULT_CHANNELS
        self.true_marginal_returns = (
            true_marginal_returns
            if true_marginal_returns is not None
            else _DEFAULT_MARGINAL_RETURNS
        )
        self.n_obs = n_obs
        self.spend_seed = spend_seed
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std

        self.spend_df_: pd.DataFrame | None = None
        self.channels_: list[str] = []
        self.results_: pd.DataFrame | None = None

    @property
    def true_elasticities(self) -> dict[str, float]:
        """Deprecated alias for `true_marginal_returns`. See __init__."""
        warnings.warn(
            "CollinearityDiagnostic.true_elasticities is deprecated, use "
            "true_marginal_returns instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.true_marginal_returns

    def fit(
        self,
        n_sims: int = 50,
        fast_mode: bool = False,
        noise_seed_offset: int = 0,
    ) -> CollinearityDiagnostic:
        """Run the diagnostic.

        Parameters
        ----------
        n_sims:
            Number of simulations (noise seeds).
        fast_mode:
            If True, overrides n_sims=10 for quick notebook iteration.
        noise_seed_offset:
            Shift applied to every noise seed used to simulate sales
            (seed = noise_seed_offset + sim, for sim in range(n_sims)).
            Default 0, which reproduces the original seed 0..n_sims-1
            behaviour. Callers that re-evaluate the same alpha/spec at
            multiple stages of a search (see BudgetPhaser.fit's
            selection-bias correction) should give the confirmation pass
            a different offset than the grid search used, so the "honest"
            re-check draws genuinely fresh noise rather than scoring
            itself against the exact same n_sims draws its own search
            already saw and could have overfit to.

        Raises
        ------
        ValueError
            If spend_df is supplied and contains non-numeric columns, NaN
            or non-finite values, a channel with zero spend variance, or
            fewer observations than OLS needs to fit at all. See
            _validate_spend_data.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10

        if self.spend_df is not None:
            _validate_spend_data(self.spend_df)
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
                true_marginal_returns=self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                seed=noise_seed_offset + sim,
            )
            estimated = fit_ols(self.spend_df_, sales)
            for channel in self.channels_:
                true_r = self.true_marginal_returns[channel]
                est_r = estimated[channel]
                records.append(
                    {
                        "sim": sim,
                        "channel": channel,
                        "true_marginal_return": true_r,
                        "estimated_marginal_return": est_r,
                        "error": est_r - true_r,
                        "error_pct": 100 * (est_r - true_r) / true_r,
                    }
                )

        self.results_ = pd.DataFrame(records)
        return self

    def analytic_cv(self) -> pd.Series:
        """Closed-form coefficient of variation per channel, no simulation.

        Because the DGP and the estimator share the same linear functional
        form, OLS variance has an exact closed form:
        ``Var(beta_hat_j) = sigma^2 * [(X'X)^-1]_jj``, so
        ``CV_j = sigma * sqrt([(X'X)^-1]_jj) / beta_j``. This requires no
        noise simulation at all -- it is exact, about 50x faster than
        `fit(n_sims=50)`, and free of the noise-draw selection bias that a
        finite `n_sims` Monte Carlo estimate carries (a given `n_sims`-seed
        block over- or under-estimates the true CV by chance; the analytic
        value is what the Monte Carlo estimate converges to as
        n_sims -> infinity). Uses `self.spend_df_` and
        `self.true_marginal_returns` -- call `fit()` first so spend_df_ is
        populated (synthetic or real), though the noise draws `fit()` ran
        are not used here.

        Returns
        -------
        pd.Series indexed by channel, the analytic CV for each.
        """
        if self.spend_df_ is None:
            raise RuntimeError("Call fit() first (to populate spend_df_).")

        x = np.column_stack(
            [np.ones(len(self.spend_df_))]
            + [self.spend_df_[c].to_numpy() for c in self.channels_]
        )
        xtx_inv = np.linalg.inv(x.T @ x)
        cv = {}
        for i, ch in enumerate(self.channels_):
            var_beta = (self.revenue_noise_std**2) * xtx_inv[i + 1, i + 1]
            beta = self.true_marginal_returns[ch]
            cv[ch] = float(np.sqrt(var_beta) / abs(beta))
        return pd.Series(cv, name="analytic_cv")

    def summary(
        self,
        planned_spend: dict[str, float] | None = None,
        value_per_unit: float | None = None,
    ) -> pd.DataFrame:
        """Return a summary of marginal-return estimates across simulations.

        Note on `coef_of_variation`: it measures sampling variance under a
        model that is correctly specified by construction. A low CV means
        this spend design can identify the channel's effect precisely --
        it is a statement about design informativeness, not about whether
        the underlying model is correctly specified. An omitted driver
        (seasonality, adstock, saturation, a competitor event) can bias the
        point estimate a lot while CV stays low, because that same omitted
        driver typically inflates the mean estimate (CV's denominator) as
        much as or more than it inflates the spread (CV's numerator). Don't
        read a green CV as "trustworthy," read it as "identifiable."

        Parameters
        ----------
        planned_spend:
            Optional dict mapping channel name to planned spend. When
            supplied, adds an incremental-revenue range (p10/p90) per
            channel, computed as the simulated marginal-return distribution
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
            draw as ``cac = 1 / estimated_marginal_return`` and
            ``roi = estimated_marginal_return * value_per_unit``. In this
            linear DGP both are spend-independent channel properties —
            they don't depend on `planned_spend`. Draws with a
            marginal-return estimate near zero can make CAC swing wildly
            or go negative; that instability is itself part of the
            diagnostic (an unreliable marginal-return estimate makes for
            an unreliable CAC estimate too).
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")

        summary = (
            self.results_.groupby("channel")
            .agg(
                true_marginal_return=("true_marginal_return", "first"),
                mean_estimated=("estimated_marginal_return", "mean"),
                std_estimated=("estimated_marginal_return", "std"),
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
                revenue["estimated_marginal_return"]
                * revenue["planned_spend"]
                * multiplier
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
            derived["cac"] = 1.0 / derived["estimated_marginal_return"]
            derived["roi"] = derived["estimated_marginal_return"] * value_per_unit
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
