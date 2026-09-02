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

# Entropy for simulate_demand_proxy's noise draw. Deliberately separate from
# every other stream in this module (same reasoning as _PLANNING_STREAM) so a
# given seed's proxy noise never collides with spend/planning noise drawn at
# that same seed value.
_PROXY_STREAM = 74_190_233

# Demand processes available to simulate_demand.
DEMAND_PROCESSES = ("white_noise", "ar1", "seasonal", "seasonal_ar1", "trend")


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
    trend_drift: float = 0.15,
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
        - "trend": a random walk with drift, x_t = x_{t-1} + trend_drift + eps.
          A genuine *non-stationary* trend -- unlike "ar1", which however
          persistent (ar_coef < 1 is enforced) always mean-reverts. This is
          the sharpest test for adstock, a low-pass filter: integration
          concentrates a random walk's energy near zero frequency even more
          than a raw AR(1) does, so carryover should leave it comparatively
          untouched while destroying the high-frequency variation phasing
          adds. Depends on `seed`, like "white_noise"/"ar1" -- each draw is a
          different path, not a fixed shape.

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
    trend_drift:
        Per-step drift for "trend", in units of the step innovation's own std
        (each step is trend_drift + N(0, 1)). 0 gives a pure driftless random
        walk; larger magnitudes give a more visibly directional path. Sign
        sets direction -- positive for growth, negative for decline.

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

    def _trend() -> np.ndarray:
        """Random walk with drift: x_t = x_{t-1} + trend_drift + eps_t, x_0 = 0.

        Deliberately NOT a deterministic ramp. A noiseless straight line is an
        unrealistic caricature of demand, and (found the hard way) it is a
        non-stationary shape squeezed through code elsewhere in this project
        that draws one series over a long window and then standardises and
        slices it -- a deterministic ramp's variance in a short sub-window is
        a tiny, geometry-dependent fraction of its full-window variance,
        which silently breaks any correlation targeting calibrated against
        the full window. A stochastic walk has the same non-stationarity (by
        design -- see the docstring above) but does not hand callers a
        pathologically clean signal.
        """
        steps = trend_drift + rng.standard_normal(n_obs)
        return np.cumsum(steps)

    if process == "white_noise":
        series = rng.standard_normal(n_obs)
    elif process == "ar1":
        series = _ar1()
    elif process == "seasonal":
        series = _seasonal()
    elif process == "trend":
        series = _trend()
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


def simulate_demand_proxy(
    demand: np.ndarray | pd.Series,
    quality: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Measurement-error proxy for a true demand series.

    A practitioner never has the true demand series that simulate_sales's
    `demand`/`demand_coef` (or CollinearityDiagnostic/BudgetPhaser's own
    `demand`/`demand_coef`) bias sales toward -- what they have, at best, is
    a proxy: a category (not brand) search-trend index, a seasonality index,
    or a sales baseline from an existing model. This models the classic
    measurement-error case: the truth plus noise, scaled so the resulting
    series correlates with the truth at exactly `quality` in expectation.

    *Assumption stated once, and it is a real one:* the proxy is treated as
    exogenous to the advertiser's own marketing. A brand-search series would
    not be -- it is partly caused by the very channel being measured, and
    controlling for it would absorb the effect under estimation. Category-
    level demand indices are the intended case.

    For a STRUCTURAL proxy instead -- e.g. a category index that tracks only
    the seasonal cycle and misses idiosyncratic shocks entirely -- there is
    no separate function: just call simulate_demand with a different process
    than the one actually driving sales (see notebooks/07 section 6's
    seasonal_proxy). That is a mismatched generative process, not
    measurement error, so it does not fit this function's quality dial.

    Parameters
    ----------
    demand:
        The true demand series to build a proxy for -- typically what was
        passed as `demand` to simulate_sales, or a CollinearityDiagnostic /
        BudgetPhaser instance's own `demand_` after fit()/construction.
    quality:
        Target correlation with the truth, in (0, 1]. 1.0 (default) returns
        the series standardised but otherwise unchanged. There is no
        realistic default -- 0.9 is already an unusually good real-world
        proxy (see notebooks/07 section 6's sweep), so pick a value you can
        defend for your own data source rather than relying on this one.
        A given noise draw's REALISED correlation with the truth is close
        to but not exactly `quality` -- it is the expectation, not a
        guarantee, of any one draw.
    seed:
        Random seed for the noise draw, drawn from a stream (_PROXY_STREAM)
        kept separate from every other seeded draw in this module -- see
        _PLANNING_STREAM's docstring note for why that separation matters.

    Returns
    -------
    np.ndarray, standardised to mean 0 / sd 1, same length as `demand`.
    """
    if not 0.0 < quality <= 1.0:
        raise ValueError("quality must be in (0, 1]")

    demand_arr = np.asarray(demand, dtype=float)

    if quality >= 1.0:
        noisy = demand_arr
    else:
        rng = np.random.default_rng([seed, _PROXY_STREAM])
        scale = np.sqrt((1 - quality**2) / quality**2)
        noisy = demand_arr + scale * rng.standard_normal(len(demand_arr))

    sd = noisy.std()
    if sd == 0:
        raise ValueError("proxy series has zero variance")
    return (noisy - noisy.mean()) / sd


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


def _saturation_for(saturation: dict[str, float] | float | None, channel: str) -> float:
    """Resolve the saturation exponent for one channel, validating as it goes.

    Returns 1.0 (linear) when no curvature applies to this channel, which is
    the value the caller uses to take the original linear code path.
    """
    if saturation is None:
        return 1.0
    if isinstance(saturation, dict):
        if channel not in saturation:
            return 1.0
        b = saturation[channel]
    else:
        b = saturation
    b = float(b)
    if not 0.0 < b <= 1.0:
        raise ValueError(
            f"saturation exponent for '{channel}' must be in (0, 1], got {b}. "
            "b = 1 is linear; smaller means stronger diminishing returns."
        )
    return b


def _adstock_for(adstock: dict[str, float] | float | None, channel: str) -> float:
    """Resolve the geometric adstock decay for one channel, validating.

    Returns 0.0 (no carryover) when none applies, which is the value the caller
    uses to take the original no-carryover code path.
    """
    if adstock is None:
        return 0.0
    if isinstance(adstock, dict):
        if channel not in adstock:
            return 0.0
        lam = adstock[channel]
    else:
        lam = adstock
    lam = float(lam)
    if not 0.0 <= lam < 1.0:
        raise ValueError(
            f"adstock decay for '{channel}' must be in [0, 1), got {lam}. "
            "0 is no carryover; approaching 1 spreads spend over more weeks."
        )
    return lam


def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Normalised geometric adstock: a[t] = (1 - decay) * x[t] + decay * a[t-1].

    The (1 - decay) factor is what makes this normalised, and it matters for
    this package specifically. Without it, carryover inflates total adstocked
    spend by 1 / (1 - decay), so the supplied marginal return would silently
    mean something different at every decay and no two settings would be
    comparable. Normalised, a constant spend level passes through unchanged --
    so a channel's steady-state level, and therefore the reference point the
    marginal return is calibrated at, does not move with decay at all.

    Seeded with a[0] = x[0] rather than 0, which is the same steady-state
    reasoning: a zero seed injects a warm-up ramp that is an artefact of where
    the array happens to start, and would read as a real spend pattern.
    """
    if decay == 0.0:
        return spend
    out = np.empty_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = (1.0 - decay) * spend[t] + decay * out[t - 1]
    return out


def simulate_sales(
    spend_df: pd.DataFrame,
    true_marginal_returns: dict[str, float] | None = None,
    base_sales: float = 1_000.0,
    revenue_noise_std: float = 26_000.0,
    seed: int = 0,
    true_elasticities: dict[str, float] | None = None,
    demand: np.ndarray | pd.Series | None = None,
    demand_coef: float = 0.0,
    saturation: dict[str, float] | float | None = None,
    adstock: dict[str, float] | float | None = None,
    reference_spend: dict[str, float] | None = None,
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
    saturation:
        Optional diminishing returns. A float applied to every channel, or a
        dict of channel -> exponent b in (0, 1]; channels absent from the dict
        stay linear. The contribution becomes k * spend ** b, with k chosen so
        that the MARGINAL return at `reference_spend` is exactly the supplied
        `true_marginal_returns` value. So the supplied marginal return keeps
        its meaning -- it is the return on the next pound at the reference
        level, rather than an average over the curve.

        b = 1.0 is linear and takes the same code path as saturation=None,
        so it reproduces the linear output exactly rather than approximately.

        Curvature is only identified by variation in spend LEVEL. A plan that
        sits in a narrow band around its own average barely traces the curve
        out, which is why b is normally treated as an assumption rather than
        an estimate. Schedules that drive spend to zero trace out far more of
        it -- whether that is enough to estimate b is a question this
        parameter exists to let a notebook answer.
    adstock:
        Optional geometric carryover. A float applied to every channel, or a
        dict of channel -> decay in [0, 1); channels absent from the dict get
        no carryover. Applied BEFORE saturation, matching the convention in
        Robyn, Meridian and PyMC-Marketing. Normalised, so a constant spend
        level passes through unchanged and the supplied marginal return means
        the same thing at every decay -- see apply_adstock.

        decay = 0.0 is no carryover and takes the same code path as
        adstock=None, so it reproduces the no-carryover output exactly.

        Carryover matters to this package more than it looks. The phasing
        lever works by adding high-frequency spend variation that demand
        cannot explain; adstock is a low-pass filter and attenuates exactly
        that component. So it is a sensitivity that can eat the benefit,
        not merely another knob.
    reference_spend:
        Spend level at which the supplied marginal returns are exact, per
        channel. Defaults to each channel's mean spend in `spend_df`. Supply
        it explicitly when comparing schedules, so that every schedule is
        calibrated against the SAME curve rather than against its own mean --
        otherwise the curve moves with the schedule and the comparison is
        not like for like. Ignored when the channel is linear.

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
        x = spend_df[ch].to_numpy()
        lam = _adstock_for(adstock, ch)
        b = _saturation_for(saturation, ch)
        if lam == 0.0 and b == 1.0:
            # Same expression as before either transform existed, so the
            # default reproduces earlier output digit for digit.
            sales = sales + true_marginal_returns[ch] * x
            continue
        # Adstock first, then saturation -- the order Robyn, Meridian and
        # PyMC-Marketing all use. apply_adstock is the identity at lam == 0.
        x = apply_adstock(x, lam)
        if b == 1.0:
            sales = sales + true_marginal_returns[ch] * x
        else:
            x_ref = (
                float(x.mean())
                if reference_spend is None
                else float(reference_spend[ch])
            )
            if x_ref <= 0:
                raise ValueError(
                    f"reference_spend for '{ch}' must be positive to calibrate "
                    f"a saturating response, got {x_ref}"
                )
            if (x < 0).any():
                raise ValueError(
                    f"channel '{ch}' has negative spend, which a saturating "
                    "response is not defined for"
                )
            # k chosen so d(contribution)/dx at x_ref equals the supplied
            # marginal return: k * b * x_ref**(b-1) == mr.
            k = true_marginal_returns[ch] / (b * x_ref ** (b - 1.0))
            sales = sales + k * x**b

    return pd.Series(sales, name="sales")
