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
design", not "how correct is this model." See docs/collinearity_research.html
for the full scope discussion.

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
    simulate_demand,
    simulate_demand_proxy,
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
        Defaults to {"tv": 0.5, "meta": 1.0, "search": 1.5}. These are a
        defensible illustrative starting point, not a claim about any real
        market -- for your own data, supply your own per-channel values
        (e.g. from a prior model, a plausible ROI range from finance, or a
        held-out incrementality test). There is no safe universal default:
        CV below is exactly inversely proportional to whatever value you
        supply (see coef_of_variation), so an unrealistic value shifts
        where the model-estimated range sits. A *uniform* rescaling of
        every channel's value (unsure of the overall scale, confident in
        the channels' proportions to each other) leaves BudgetPhaser's
        percentage CV reduction unchanged too -- but changing the
        *relative* proportions between channels can shift which channel
        looks least identified, which can change the phasing intensity
        BudgetPhaser's auto_lever recommends for it, and so the reduction
        percentage it actually delivers. Only the absolute £ width of the
        range here is unconditionally invariant to your assumption.
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
    demand:
        Optional latent demand series. On the real-spend path (spend_df
        supplied) this is the only way to give fit() a demand series at
        all -- there's no internal draw to fall back on, so it's required
        whenever demand_coef is nonzero. On the synthetic path it's
        optional: supply your own (e.g. to reuse one demand draw across
        several CollinearityDiagnostic instances, the way BudgetPhaser
        does internally) or leave it None to let fit() draw one from
        demand_process/demand_seed whenever demand_coef is nonzero.
    demand_process:
        One of DEMAND_PROCESSES (see simulate_demand), used for the
        internal demand draw on the synthetic path. Ignored when `demand`
        is supplied directly.
    demand_share:
        Forwarded to simulate_spend on the synthetic path: share of
        channels' common variance attributable to demand vs. an
        independent planning factor. See simulate_spend's own docstring --
        this only changes anything when demand_share < 1.
    demand_coef:
        Coefficient on demand in the sales equation -- what actually
        creates omitted-variable bias. 0.0 (default) means demand doesn't
        affect sales at all and reproduces this class's pre-existing
        behaviour exactly. Get a value in practitioner-legible units from
        calibrate_baseline rather than guessing one directly.
    demand_seed:
        Random seed for the internal demand draw (synthetic path only,
        when `demand` isn't supplied directly). Deliberately separate from
        spend_seed so changing one doesn't shift the other's draw.
    saturation:
        Forwarded to simulate_sales -- see its own docstring. A float
        applied to every channel, or a dict of channel -> exponent b in
        (0, 1]; None (default) is linear, reproducing prior behaviour.
    adstock:
        Forwarded to simulate_sales -- see its own docstring. A float
        applied to every channel, or a dict of channel -> decay in [0, 1);
        None (default) is no carryover, reproducing prior behaviour.
    reference_spend:
        Forwarded to simulate_sales -- see its own docstring. Only matters
        when saturation is active; defaults to each channel's own mean
        spend, same as simulate_sales's own default.
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
        revenue_noise_std: float = 26_000.0,
        true_elasticities: dict[str, float] | None = None,
        demand: np.ndarray | pd.Series | None = None,
        demand_process: str = "white_noise",
        demand_share: float = 1.0,
        demand_coef: float = 0.0,
        demand_seed: int = 0,
        saturation: dict[str, float] | float | None = None,
        adstock: dict[str, float] | float | None = None,
        reference_spend: dict[str, float] | None = None,
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
        self.demand = demand
        self.demand_process = demand_process
        self.demand_share = demand_share
        self.demand_coef = demand_coef
        self.demand_seed = demand_seed
        self.saturation = saturation
        self.adstock = adstock
        self.reference_spend = reference_spend

        self.spend_df_: pd.DataFrame | None = None
        self.channels_: list[str] = []
        self.results_: pd.DataFrame | None = None
        self.demand_: np.ndarray | None = None
        self.controls_: pd.DataFrame | pd.Series | None = None

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
        controls: pd.DataFrame | pd.Series | bool | float | None = None,
        proxy_seed: int = 0,
    ) -> CollinearityDiagnostic:
        """Run the diagnostic.

        Parameters
        ----------
        n_sims:
            Number of simulations (noise seeds).
        fast_mode:
            If True, overrides n_sims=10 for quick notebook iteration.
        controls:
            What the OLS fit controls for, forwarded to fit_ols. None or
            False (default): omit -- reproduces prior behaviour exactly,
            and is what every real MMM does, so it's what the resulting CV
            actually means (see the class/coef_of_variation docstrings on
            what a low CV does and doesn't tell you). True: control with
            this instance's own true demand series (self.demand_, from
            `demand`/demand_coef) -- the correctly-specified world, used to
            measure how much controlling removes rather than to represent
            a real analysis (a practitioner has a proxy, not the truth).
            Raises ValueError if True is passed with no demand series
            available. A float in (0, 1]: control with a measurement-error
            proxy of that quality, built from self.demand_ via
            simulate_demand_proxy (see `proxy_seed`) -- the client-facing
            case, standing in for a real proxy a practitioner would supply
            (a category search-trend index, a seasonality index, an
            existing model's baseline). A DataFrame or Series: an explicit
            proxy (or any other control) to use instead, e.g. a real one.
            Stored as `self.controls_` after fit() for reuse (e.g. by
            analytic_cv(), or to hand the same resolved series to another
            call).
        proxy_seed:
            Random seed forwarded to simulate_demand_proxy when `controls`
            is a float quality. Ignored otherwise.
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
            if self.demand_coef and self.demand is None:
                raise ValueError(
                    "demand_coef is set but no demand series was supplied. "
                    "On the real-spend path (spend_df given) there is no "
                    "internal demand draw to fall back on -- pass demand= "
                    "explicitly, aligned to spend_df."
                )
            self.demand_ = (
                np.asarray(self.demand, dtype=float)
                if self.demand is not None
                else None
            )
        else:
            if self.demand is not None:
                demand_arr = np.asarray(self.demand, dtype=float)
                if demand_arr.shape != (self.n_obs,):
                    raise ValueError(
                        f"demand must be a 1-D series of length n_obs="
                        f"{self.n_obs}, got shape {demand_arr.shape}"
                    )
            elif self.demand_coef:
                demand_arr = simulate_demand(
                    self.n_obs, process=self.demand_process, seed=self.demand_seed
                )
            else:
                demand_arr = None
            self.demand_ = demand_arr
            self.spend_df_ = simulate_spend(
                n_obs=self.n_obs,
                correlation=self.correlation,
                channels=self.channels,
                seed=self.spend_seed,
                demand=demand_arr,
                demand_share=self.demand_share,
            )
            self.channels_ = list(self.channels)

        if controls is True:
            if self.demand_ is None:
                raise ValueError(
                    "controls=True requires a demand series -- supply "
                    "demand= or set demand_coef (synthetic path) so fit() "
                    "has a true demand series to control with."
                )
            resolved_controls = pd.Series(
                self.demand_, index=self.spend_df_.index, name="demand"
            )
        elif controls is False or controls is None:
            resolved_controls = None
        elif isinstance(controls, (int, float)):
            if self.demand_ is None:
                raise ValueError(
                    "controls=<quality> requires a demand series to build "
                    "a proxy from -- supply demand= or set demand_coef "
                    "(synthetic path)."
                )
            proxy = simulate_demand_proxy(
                self.demand_, quality=float(controls), seed=proxy_seed
            )
            resolved_controls = pd.Series(
                proxy, index=self.spend_df_.index, name="demand_proxy"
            )
        else:
            resolved_controls = controls
        self.controls_ = resolved_controls

        records = []
        for sim in range(n_sims):
            sales = simulate_sales(
                spend_df=self.spend_df_,
                true_marginal_returns=self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                seed=noise_seed_offset + sim,
                demand=self.demand_,
                demand_coef=self.demand_coef,
                saturation=self.saturation,
                adstock=self.adstock,
                reference_spend=self.reference_spend,
            )
            estimated = fit_ols(self.spend_df_, sales, controls=resolved_controls)
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

        cols = [np.ones(len(self.spend_df_))] + [
            self.spend_df_[c].to_numpy() for c in self.channels_
        ]
        if self.controls_ is not None:
            # Match fit_ols's own column assembly, so the closed form sees
            # exactly the design matrix fit_ols actually fit -- otherwise
            # analytic_cv would silently go stale the moment controls are
            # used (the whole point of the closed form is that it's exact).
            controls_df = (
                self.controls_.to_frame(name=self.controls_.name or "control")
                if isinstance(self.controls_, pd.Series)
                else self.controls_
            )
            cols += [controls_df[c].to_numpy() for c in controls_df.columns]
        x = np.column_stack(cols)
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
