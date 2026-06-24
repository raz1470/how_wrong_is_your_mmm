"""Data generating process for collinearity simulation.

Two functions with a clean separation of concerns:

- simulate_spend: generates synthetic correlated spend for N channels via a
  latent demand signal. All pairwise correlations are equal (single rho param).

- simulate_sales: creates a synthetic sales column from a spend DataFrame
  (real or synthetic) using known elasticities. This step is identical
  regardless of whether spend is real or synthetic.
"""

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
_DEFAULT_ELASTICITIES: dict[str, float] = {"tv": 0.3, "meta": 0.5, "search": 0.4}


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


def simulate_spend(
    n_obs: int = 104,
    correlation: float = 0.7,
    channels: list[str] | None = None,
    seed: int = 0,
    start_date: str | None = None,
) -> pd.DataFrame:
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
        when using the output with BudgetPerturber.

    Returns
    -------
    pd.DataFrame with one column per channel. If start_date is provided,
    the index is a weekly DatetimeIndex; otherwise it is the default integer index.
    """
    if channels is None:
        channels = _DEFAULT_CHANNELS

    rng = np.random.default_rng(seed)
    noise_std = _noise_std_from_correlation(correlation)
    demand = rng.standard_normal(n_obs)

    data = {}
    for ch in channels:
        mean, std = _CHANNEL_SCALE.get(ch, _DEFAULT_SCALE)
        signal = demand + noise_std * rng.standard_normal(n_obs)
        data[ch] = mean + std * signal

    df = pd.DataFrame(data)
    if start_date is not None:
        df.index = pd.date_range(start=start_date, periods=n_obs, freq="W-MON")
    return df


def simulate_sales(
    spend_df: pd.DataFrame,
    true_elasticities: dict[str, float] | None = None,
    base_sales: float = 1_000.0,
    revenue_noise_std: float = 20_000.0,
    seed: int = 0,
) -> pd.Series:
    """Create a synthetic sales column from a spend DataFrame.

    Applies known elasticities to the spend columns and adds noise.
    Works identically whether spend_df is synthetic or real.

    Model: sales = base + sum(elast[c] * spend[c] for c in channels) + noise

    Parameters
    ----------
    spend_df:
        DataFrame with one column per channel. Can be synthetic or real.
    true_elasticities:
        Dict mapping channel name to true elasticity. Defaults to
        {"tv": 0.3, "meta": 0.5, "search": 0.4}.
        All columns in spend_df must have an entry.
    base_sales:
        Base sales intercept.
    revenue_noise_std:
        Standard deviation of sales noise.
    seed:
        Random seed for the noise draw.

    Returns
    -------
    pd.Series of simulated sales values.
    """
    if true_elasticities is None:
        true_elasticities = _DEFAULT_ELASTICITIES

    rng = np.random.default_rng(seed)
    sales = base_sales + revenue_noise_std * rng.standard_normal(len(spend_df))
    for ch in spend_df.columns:
        if ch not in true_elasticities:
            raise ValueError(
                f"Channel '{ch}' in spend_df has no entry in true_elasticities. "
                f"Provide true_elasticities for all channels: {list(spend_df.columns)}"
            )
        sales = sales + true_elasticities[ch] * spend_df[ch].to_numpy()

    return pd.Series(sales, name="sales")
