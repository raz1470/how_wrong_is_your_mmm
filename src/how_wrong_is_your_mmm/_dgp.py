"""Data generating process for collinearity simulation.

Two functions with a clean separation of concerns:

- simulate_spend: generates synthetic correlated TV and Meta spend via a
  latent demand signal. Used when no real spend data is available.

- simulate_sales: creates a synthetic sales column from a spend DataFrame
  (real or synthetic) using known elasticities. This is the step that is
  always run, regardless of whether spend is real or synthetic.
"""

import numpy as np
import pandas as pd


def _noise_std_from_correlation(correlation: float) -> float:
    """Return per-channel noise std that produces the target correlation.

    If TV = demand + noise_tv and Meta = demand + noise_meta, with demand
    and noise all N(0, 1), then Corr(TV, Meta) = 1 / (1 + sigma^2).
    Solving: sigma = sqrt((1 - corr) / corr).
    """
    if not 0 < correlation < 1:
        raise ValueError("correlation must be strictly between 0 and 1")
    return float(np.sqrt((1 - correlation) / correlation))


def simulate_spend(
    n_obs: int = 104,
    correlation: float = 0.7,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic correlated TV and Meta spend via a latent demand signal.

    Both channels track the same underlying demand index with added noise.
    The noise level is set to achieve the target correlation between channels.

    Parameters
    ----------
    n_obs:
        Number of observations (weeks).
    correlation:
        Target Pearson correlation between TV and Meta spend.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: tv, meta.
    """
    rng = np.random.default_rng(seed)
    noise_std = _noise_std_from_correlation(correlation)

    demand = rng.standard_normal(n_obs)
    tv = demand + noise_std * rng.standard_normal(n_obs)
    meta = demand + noise_std * rng.standard_normal(n_obs)

    # scale to realistic weekly spend ranges
    tv = 100_000 + 20_000 * tv
    meta = 80_000 + 15_000 * meta

    return pd.DataFrame({"tv": tv, "meta": meta})


def simulate_sales(
    spend_df: pd.DataFrame,
    true_elast_tv: float = 0.3,
    true_elast_meta: float = 0.5,
    base_sales: float = 1_000.0,
    revenue_noise_std: float = 20_000.0,
    seed: int = 0,
) -> pd.Series:
    """Create a synthetic sales column from a spend DataFrame.

    Applies known elasticities to the spend columns and adds noise.
    Works identically whether spend_df is synthetic or real — this is
    the shared step in both entry points of the pipeline.

    Model: sales = base + true_elast_tv * tv + true_elast_meta * meta + noise

    Parameters
    ----------
    spend_df:
        DataFrame with columns 'tv' and 'meta'. Can be synthetic (from
        simulate_spend) or real spend data supplied by the user.
    true_elast_tv:
        True TV elasticity (coefficient) in the sales equation.
    true_elast_meta:
        True Meta elasticity (coefficient) in the sales equation.
    base_sales:
        Base sales intercept (sales with zero spend).
    revenue_noise_std:
        Standard deviation of sales noise.
    seed:
        Random seed for the noise draw.

    Returns
    -------
    pd.Series of simulated sales values.
    """
    rng = np.random.default_rng(seed)
    sales = (
        base_sales
        + true_elast_tv * spend_df["tv"].to_numpy()
        + true_elast_meta * spend_df["meta"].to_numpy()
        + revenue_noise_std * rng.standard_normal(len(spend_df))
    )
    return pd.Series(sales, name="sales")
