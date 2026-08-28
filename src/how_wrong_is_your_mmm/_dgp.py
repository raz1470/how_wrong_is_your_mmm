"""Data generating process for collinearity simulation.

Two functions with a clean separation of concerns:

- simulate_spend: generates synthetic correlated spend for N channels via a
  latent demand signal. All pairwise correlations are equal (single rho param).

- simulate_sales: creates a synthetic sales column from a spend DataFrame
  (real or synthetic) using known marginal returns (mROAS): £ revenue per
  £ spend, not economic elasticities. This step is identical regardless of
  whether spend is real or synthetic.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default spend scale per channel: (mean, std) in £/week.
_CHANNEL_SCALE: dict[str, tuple[float, float]] = {
    "tv": (100_000, 20_000),
    "meta": (80_000, 15_000),
    "search": (60_000, 12_000),
}
_DEFAULT_SCALE = (80_000, 15_000)
_DEFAULT_CHANNELS = ["tv", "meta", "search"]

# Marginal return (a.k.a. mROAS): £ of incremental revenue per £ of spend.
# These are NOT elasticities (a unitless %-response-to-%-spend measure) --
# the DGP is linear in raw spend, so the DGP/estimator coefficient is a
# marginal £-per-£ return. Values below are a defensible illustrative
# starting point, not a claim about any real market -- chosen to land in
# the range practitioners usually mean by "ROI" (revenue / spend) rather
# than an inflated mROAS, so the illustrative CVs stay honestly wide.
_DEFAULT_MARGINAL_RETURNS: dict[str, float] = {"tv": 0.5, "meta": 1.0, "search": 1.5}
# Deprecated alias -- kept only so old code importing the private name
# doesn't hard-crash. Prefer _DEFAULT_MARGINAL_RETURNS.
_DEFAULT_ELASTICITIES = _DEFAULT_MARGINAL_RETURNS

# Entropy for the independent "planning factor" stream used when
# demand_share < 1. Deliberately a SEPARATE generator from the main one so
# that lowering demand_share does not disturb the channel-noise draws --
# see the note in simulate_spend.
_PLANNING_STREAM = 20_260_828

# Demand processes available to simulate_demand.
DEMAND_PROCESSES = ("white_noise", "ar1", "seasonal", "seasonal_ar1")


def _noise_std_from_correlation(correlation: float) -> float:
    """Return per-channel noise std that produces the target pairwise correlation.

    If channel_i = demand + noise_i, channel_j = demand + noise_j, with demand
    and all noise terms N(0, 1), then Corr(i, j) = 1 / (1 + sigma^2).
    Solving: sigma = sqrt((1 - corr) / corr).

    This gives equal pairwise correlation for all channel pairs.
    """
    if not 0 < correlation < 1:
        raise ValueError("correlation must be strictly between 0 and 1")
    return float(np.sqrt((1 - correlation) / correlation))


def simulate_demand(
    n_obs: int = 104,
    process: str = "white_noise",
    seed: int = 0,
    ar_coef: float = 0.8,
    season_period: float = 52.0,
    season_weight: float = 0.7,
) -> np.ndarray:
    """Generate a latent demand series, standardised to mean 0 / sd 1.

    Standardisation is deliberate: it makes `demand_coef` mean the same thing
    across processes, so comparing a white-noise world against a seasonal one
    compares the *shape* of demand rather than accidentally comparing its
    amplitude.

    Parameters
    ----------
    n_obs:
        Number of observations (weeks).
    process:
        One of DEMAND_PROCESSES:

        - "white_noise": iid draws. Demand varies week to week, so weekly
          reshuffling of spend can decouple from it.
        - "ar1": persistent demand, x_t = ar_coef * x_{t-1} + eps.
        - "seasonal": a pure sinusoid at season_period weeks.
        - "seasonal_ar1": season_weight * seasonal + (1 - season_weight) * ar1.

        The distinction matters: BudgetPhaser preserves monthly totals and moves
        spend only *within* a month, so it can only attack spend-demand coupling
        that lives inside that window. A white-noise-only study would overstate
        how much phasing fixes bias.
    seed:
        Random seed.
    ar_coef:
        AR(1) coefficient, used by "ar1" and "seasonal_ar1".
    season_period:
        Period of the seasonal component, in observations (52 = annual weekly).
    season_weight:
        Weight on the seasonal component in "seasonal_ar1", in [0, 1].

    Returns
    -------
    np.ndarray of length n_obs, standardised to sample mean 0 and sd 1.
    """
    if process not in DEMAND_PROCESSES:
        raise ValueError(f"process must be one of {DEMAND_PROCESSES}, got {process!r}")
    if not -1.0 < ar_coef < 1.0:
        raise ValueError("ar_coef must be strictly between -1 and 1 for stationarity")
    if not 0.0 <= season_weight <= 1.0:
        raise ValueError("season_weight must be between 0 and 1 inclusive")
    if season_period <= 0:
        raise ValueError("season_period must be positive")

    rng = np.random.default_rng(seed)

    def _ar1() -> np.ndarray:
        burn = 200
        eps = rng.standard_normal(n_obs + burn)
        out = np.zeros(n_obs + burn)
        for t in range(1, n_obs + burn):
            out[t] = ar_coef * out[t - 1] + eps[t]
        return out[burn:]

    def _seasonal() -> np.ndarray:
        return np.sin(2 * np.pi * np.arange(n_obs) / season_period)

    if process == "white_noise":
        series = rng.standard_normal(n_obs)
    elif process == "ar1":
        series = _ar1()
    elif process == "seasonal":
        series = _seasonal()
    else:
        seasonal = _seasonal()
        ar = _ar1()
        seasonal = (seasonal - seasonal.mean()) / (seasonal.std() or 1.0)
        ar = (ar - ar.mean()) / (ar.std() or 1.0)
        series = season_weight * seasonal + (1 - season_weight) * ar

    sd = series.std()
    if sd == 0:
        raise ValueError(
            "demand series has zero variance -- check season_period against n_obs"
        )
    return (series - series.mean()) / sd


@dataclass(frozen=True)
class BaselineCalibration:
    """Result of calibrating a synthetic world against MMM-legible inputs.

    Attributes
    ----------
    baseline_level:
        Mean weekly baseline (non-media) sales.
    demand_coef:
        Coefficient on a standardised demand series, i.e. baseline_level *
        baseline_cv. This is what drives omitted-variable bias -- note it comes
        from the *volatility* of the baseline, not its level. A perfectly flat
        baseline, however large, is absorbed by the intercept and biases nothing.
    total_sales:
        Implied mean weekly total sales.
    channel_shares:
        Each channel's implied share of total sales, derived from its ROI and
        its mean spend. Show this back to the user: it is the reconciliation
        against what their own MMM decomposition reports.
    contributions:
        Each channel's implied mean weekly contribution, in currency.
    implied_totals:
        Only populated when `reported_shares` is supplied. Total sales implied
        by each channel separately. ROI and share are NOT independent inputs --
        each channel pins down its own total -- so these agreeing is a check on
        the user's MMM, not an accident.
    share_spread_pct:
        Spread of `implied_totals` as a percentage of their mean. Zero means the
        reported shares reconcile exactly with the ROIs and the spend.
    """

    baseline_level: float
    demand_coef: float
    total_sales: float
    channel_shares: dict[str, float]
    contributions: dict[str, float]
    implied_totals: dict[str, float] = field(default_factory=dict)
    share_spread_pct: float | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return the decomposition as a DataFrame, the way an MMM reports it."""
        rows = [
            {
                "channel": ch,
                "contribution": self.contributions[ch],
                "share_of_sales": self.channel_shares[ch],
                "implied_total_sales": self.implied_totals.get(ch, np.nan),
            }
            for ch in self.channel_shares
        ]
        rows.append(
            {
                "channel": "baseline",
                "contribution": self.baseline_level,
                "share_of_sales": self.baseline_level / self.total_sales,
                "implied_total_sales": np.nan,
            }
        )
        return pd.DataFrame(rows)


def calibrate_baseline(
    spend_df: pd.DataFrame,
    true_marginal_returns: dict[str, float] | None = None,
    baseline_share: float = 0.7,
    baseline_cv: float = 0.0,
    reported_shares: dict[str, float] | None = None,
) -> BaselineCalibration:
    """Derive a synthetic world from inputs a practitioner reads off their MMM.

    The user supplies ROI per channel and the baseline's share of sales -- both
    standard MMM decomposition outputs -- rather than an arbitrary intercept.
    Everything else follows:

        contribution_c = ROI_c * mean(spend_c)
        total_sales    = sum(contributions) / (1 - baseline_share)
        baseline_level = baseline_share * total_sales
        demand_coef    = baseline_level * baseline_cv

    Note what drives bias. `baseline_share` sets the *level*, which OLS absorbs
    into the intercept and which therefore biases nothing on its own. It is
    `baseline_cv` -- how much that baseline moves week to week -- that creates
    omitted-variable bias, and because the baseline is typically the largest
    term in the decomposition, small relative wobbles are large absolute swings.

    Parameters
    ----------
    spend_df:
        DataFrame with one column per channel.
    true_marginal_returns:
        ROI (£ revenue per £ spend) per channel. Defaults to the package's
        illustrative 0.5 / 1.0 / 1.5.
    baseline_share:
        Baseline (non-media) share of total sales, in [0, 1). Real MMMs
        typically report 0.6-0.8.
    baseline_cv:
        Coefficient of variation of the baseline week to week. 0.0 gives a flat
        baseline and therefore no omitted-variable bias.
    reported_shares:
        Optional per-channel shares of sales as reported by the user's own MMM.
        When supplied, the returned object carries the total sales each channel
        separately implies, and their spread -- a consistency check on the
        user's numbers rather than an input the simulation needs.

    Returns
    -------
    BaselineCalibration
    """
    if true_marginal_returns is None:
        true_marginal_returns = _DEFAULT_MARGINAL_RETURNS
    if not 0.0 <= baseline_share < 1.0:
        raise ValueError("baseline_share must be in [0, 1)")
    if baseline_cv < 0.0:
        raise ValueError("baseline_cv must be non-negative")

    contributions: dict[str, float] = {}
    for ch in spend_df.columns:
        if ch not in true_marginal_returns:
            raise ValueError(
                f"Channel '{ch}' in spend_df has no entry in true_marginal_returns."
            )
        contributions[ch] = float(true_marginal_returns[ch] * spend_df[ch].mean())

    media = sum(contributions.values())
    if media <= 0:
        raise ValueError(
            "Total media contribution must be positive to calibrate a baseline."
        )
    total = media / (1.0 - baseline_share)
    level = baseline_share * total

    implied_totals: dict[str, float] = {}
    spread: float | None = None
    if reported_shares is not None:
        for ch in spend_df.columns:
            share = reported_shares.get(ch)
            if share is None:
                raise ValueError(f"reported_shares has no entry for channel '{ch}'.")
            if not 0.0 < share < 1.0:
                raise ValueError(
                    f"reported_shares['{ch}'] must be strictly between 0 and 1."
                )
            implied_totals[ch] = contributions[ch] / share
        values = np.array(list(implied_totals.values()), dtype=float)
        spread = float(100 * (values.max() - values.min()) / values.mean())

    return BaselineCalibration(
        baseline_level=float(level),
        demand_coef=float(level * baseline_cv),
        total_sales=float(total),
        channel_shares={ch: contributions[ch] / total for ch in contributions},
        contributions=contributions,
        implied_totals=implied_totals,
        share_spread_pct=spread,
    )


def simulate_spend(
    n_obs: int = 104,
    correlation: float = 0.7,
    channels: list[str] | None = None,
    seed: int = 0,
    start_date: str | None = None,
    demand: np.ndarray | pd.Series | None = None,
    demand_share: float = 1.0,
    return_demand: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic correlated spend for N channels via a latent demand signal.

    All channels track the same underlying demand index with independent noise.
    The noise level is set so that all pairwise correlations equal `correlation`.

    Parameters
    ----------
    n_obs:
        Number of observations (weeks).
    correlation:
        Target Pearson correlation between any pair of channels.
    channels:
        List of channel names. Defaults to ["tv", "meta", "search"].
    seed:
        Random seed for reproducibility.
    start_date:
        If provided (e.g. "2023-01-02"), the DataFrame will have a weekly
        DatetimeIndex anchored on Mondays starting from this date. Required
        when using the output with BudgetPhaser.
    demand:
        Optional externally supplied latent demand series of length n_obs,
        e.g. from simulate_demand. Use this to pair one demand series with
        several spend frames -- notably a plan and its phased version, which
        BudgetPhaser produces and this function never sees. The internal draw
        still happens and is discarded, so supplying the series this function
        would have drawn anyway reproduces its output exactly.
    demand_share:
        Share of the channels' *common* variance attributable to demand, in
        [0, 1]. The remainder comes from an independent "planning factor" that
        correlates the channels but never enters sales.

        This separates two things the DGP previously conflated. At the default
        1.0 every bit of channel-channel correlation comes from demand, which
        forces corr(spend_c, demand) = sqrt(correlation) -- the maximal-bias
        case. Lower it to represent the common real situation where channels
        move together because one planning cycle sets both budgets, and neither
        closely tracks demand.

        Pairwise channel correlation stays exactly `correlation` at every
        setting; what changes is corr(spend_c, demand) = sqrt(demand_share *
        correlation). **The observed correlation matrix cannot reveal the
        split**, which is why omitted-variable bias is not identifiable from
        spend data alone and needs either a demand proxy or a sensitivity band.
    return_demand:
        If True, return (spend_df, demand_series) instead of just spend_df.

    Returns
    -------
    pd.DataFrame with one column per channel. If start_date is provided,
    the index is a weekly DatetimeIndex; otherwise it is the default integer index.
    If return_demand is True, returns (spend_df, demand_series) instead.
    """
    if channels is None:
        channels = _DEFAULT_CHANNELS
    if not 0.0 <= demand_share <= 1.0:
        raise ValueError("demand_share must be between 0 and 1 inclusive")

    rng = np.random.default_rng(seed)
    noise_std = _noise_std_from_correlation(correlation)

    # The internal draw ALWAYS happens, even when `demand` is supplied, so that
    # supplying a series does not shift the channel-noise draws below. Session
    # 41's nudge_shape lesson: changing a generator's call sequence silently
    # moves every seeded number already published from this function.
    drawn_demand = rng.standard_normal(n_obs)
    if demand is None:
        demand_arr = drawn_demand
    else:
        demand_arr = np.asarray(demand, dtype=float)
        if demand_arr.shape != (n_obs,):
            raise ValueError(
                f"demand must be a 1-D series of length n_obs={n_obs}, "
                f"got shape {demand_arr.shape}"
            )

    if demand_share < 1.0:
        # Drawn from a SEPARATE generator, deliberately. Taking it from `rng`
        # would shift every channel-noise draw below, so two demand_share
        # settings at the same seed would not be comparable draw for draw --
        # exactly the caveat nudge_shape has to carry. This way they are.
        planning = np.random.default_rng([seed, _PLANNING_STREAM]).standard_normal(
            n_obs
        )
        common = (
            np.sqrt(demand_share) * demand_arr + np.sqrt(1.0 - demand_share) * planning
        )
    else:
        common = demand_arr

    data = {}
    for ch in channels:
        mean, std = _CHANNEL_SCALE.get(ch, _DEFAULT_SCALE)
        signal = common + noise_std * rng.standard_normal(n_obs)
        data[ch] = mean + std * signal

    df = pd.DataFrame(data)
    if start_date is not None:
        df.index = pd.date_range(start=start_date, periods=n_obs, freq="W-MON")
    if return_demand:
        return df, pd.Series(demand_arr, index=df.index, name="demand")
    return df


def simulate_sales(
    spend_df: pd.DataFrame,
    true_marginal_returns: dict[str, float] | None = None,
    base_sales: float = 1_000.0,
    revenue_noise_std: float = 26_000.0,
    seed: int = 0,
    true_elasticities: dict[str, float] | None = None,
    demand: np.ndarray | pd.Series | None = None,
    demand_coef: float = 0.0,
) -> pd.Series:
    """Create a synthetic sales column from a spend DataFrame.

    Applies known marginal returns to the spend columns and adds noise.
    Works identically whether spend_df is synthetic or real.

    Model: sales = base + sum(beta[c] * spend[c] for c in channels) + noise

    Each beta[c] here is a marginal return (a.k.a. mROAS): £ of
    incremental sales per £ of spend on channel c. Because the DGP is
    linear in raw £ spend (no log-log transform), this is NOT an
    elasticity in the economic sense (%-response to %-spend) -- calling
    it that overstates how "typical" the numbers look to an MMM
    practitioner, and read as an ROI it makes even a healthy true effect
    look catastrophic.

    Parameters
    ----------
    spend_df:
        DataFrame with one column per channel. Can be synthetic or real.
    true_marginal_returns:
        Dict mapping channel name to true marginal return (£ revenue per
        £ spend). Defaults to {"tv": 0.5, "meta": 1.0, "search": 1.5}.
        All columns in spend_df must have an entry.
    base_sales:
        Base sales intercept.
    revenue_noise_std:
        Standard deviation of sales noise.
    seed:
        Random seed for the noise draw.
    true_elasticities:
        Deprecated alias for `true_marginal_returns`, kept for backward
        compatibility. Raises ValueError if both are supplied. Emits a
        FutureWarning -- migrate to `true_marginal_returns`.
    demand:
        Optional latent demand series contributing to sales. Because spend
        follows demand in this DGP, a model fitted WITHOUT this term is
        misspecified and its channel coefficients carry omitted-variable bias.
        Fit with it (see fit_ols's `controls`) and the model is correctly
        specified. Toggling between the two is how the bias is measured.
    demand_coef:
        Coefficient on `demand`. Ignored when demand is None. Get a value in
        practitioner-legible units from calibrate_baseline, which derives it
        from the baseline's share of sales and its volatility.

    Returns
    -------
    pd.Series of simulated sales values.

    Notes
    -----
    The demand term is added AFTER the noise draw, so supplying it does not
    disturb the random stream: at demand_coef=0 (or demand=None) this function
    reproduces its previous output exactly.
    """
    if true_elasticities is not None:
        if true_marginal_returns is not None:
            raise ValueError(
                "Pass only one of true_marginal_returns or the deprecated "
                "true_elasticities, not both."
            )
        warnings.warn(
            "true_elasticities is deprecated and will be removed in a "
            "future release -- these are marginal returns (£ revenue per "
            "£ spend), not elasticities. Use true_marginal_returns instead.",
            FutureWarning,
            stacklevel=2,
        )
        true_marginal_returns = true_elasticities

    if true_marginal_returns is None:
        true_marginal_returns = _DEFAULT_MARGINAL_RETURNS

    rng = np.random.default_rng(seed)
    sales = base_sales + revenue_noise_std * rng.standard_normal(len(spend_df))

    if demand is not None and demand_coef:
        demand_arr = np.asarray(demand, dtype=float)
        if demand_arr.shape != (len(spend_df),):
            raise ValueError(
                f"demand must be a 1-D series of length {len(spend_df)} to match "
                f"spend_df, got shape {demand_arr.shape}"
            )
        sales = sales + demand_coef * demand_arr

    for ch in spend_df.columns:
        if ch not in true_marginal_returns:
            raise ValueError(
                f"Channel '{ch}' in spend_df has no entry in "
                "true_marginal_returns. Provide true_marginal_returns for "
                f"all channels: {list(spend_df.columns)}"
            )
        sales = sales + true_marginal_returns[ch] * spend_df[ch].to_numpy()

    return pd.Series(sales, name="sales")
