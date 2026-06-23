"""Lightweight OLS MMM for the simulation loop.

Fits sales ~ intercept + channel_1 + channel_2 + ... using OLS and returns
estimated channel elasticities. Works for any number of channels.
No adstock, no saturation — placeholder for PyMC-Marketing in a later phase.
"""

import numpy as np
import pandas as pd


def fit_ols(spend_df: pd.DataFrame, sales: pd.Series) -> dict[str, float]:
    """Fit a simple OLS MMM and return estimated channel elasticities.

    Model: sales = intercept + sum(beta[c] * spend[c] for c in channels)

    Parameters
    ----------
    spend_df:
        DataFrame with one column per channel.
    sales:
        Series of sales values to fit against.

    Returns
    -------
    dict mapping channel name to estimated elasticity.
    """
    channels = list(spend_df.columns)
    x = np.column_stack(
        [np.ones(len(spend_df))] + [spend_df[c].to_numpy() for c in channels]
    )
    y = sales.to_numpy()
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {c: float(coeffs[i + 1]) for i, c in enumerate(channels)}
