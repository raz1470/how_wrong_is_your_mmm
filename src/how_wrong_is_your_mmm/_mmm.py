"""Lightweight OLS MMM for the simulation loop.

Fits sales ~ intercept + tv + meta using OLS and returns estimated
channel elasticities. No adstock, no saturation — placeholder for
PyMC-Marketing in a later phase.
"""

import numpy as np
import pandas as pd


def fit_ols(spend_df: pd.DataFrame, sales: pd.Series) -> dict[str, float]:
    """Fit a simple OLS MMM and return estimated channel elasticities.

    Model: sales = intercept + beta_tv * tv + beta_meta * meta

    Parameters
    ----------
    spend_df:
        DataFrame with columns 'tv' and 'meta'.
    sales:
        Series of sales values to fit against.

    Returns
    -------
    dict with keys 'tv' and 'meta' — the estimated elasticities.
    """
    x = np.column_stack(
        [np.ones(len(spend_df)), spend_df["tv"].to_numpy(), spend_df["meta"].to_numpy()]
    )
    y = sales.to_numpy()
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"tv": float(coeffs[1]), "meta": float(coeffs[2])}
