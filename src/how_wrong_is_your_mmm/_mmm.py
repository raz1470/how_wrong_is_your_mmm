"""Lightweight OLS MMM for the simulation loop.

Fits sales ~ intercept + channel_1 + channel_2 + ... using OLS and returns
estimated channel marginal returns (£ revenue per £ spend, a.k.a. mROAS --
NOT economic elasticities, since the model is linear in raw £ spend rather
than log-log). Works for any number of channels. No adstock, no saturation
— placeholder for PyMC-Marketing in a later phase.
"""

import numpy as np
import pandas as pd


def fit_ols(
    spend_df: pd.DataFrame,
    sales: pd.Series,
    controls: pd.DataFrame | pd.Series | None = None,
) -> dict[str, float]:
    """Fit a simple OLS MMM and return estimated channel marginal returns.

    Model: sales = intercept + sum(beta[c] * spend[c]) + sum(gamma[k] * control[k])

    Parameters
    ----------
    spend_df:
        DataFrame with one column per channel.
    sales:
        Series of sales values to fit against.
    controls:
        Optional extra regressors -- typically a latent demand series or a
        client-supplied proxy for it. Omitting a control that drives both spend
        and sales leaves the channel coefficients biased; including it removes
        the bias but widens them, because the control is collinear with spend
        by construction. Toggling this argument is how that trade is measured.

        Note the width cost is real: in this package's DGP a demand control has
        corr(spend_c, demand) = sqrt(demand_share * correlation), so at
        demand_share=1 it is MORE collinear with each channel than the channels
        are with each other.

    Returns
    -------
    dict mapping name to estimated coefficient, covering the channels and any
    controls. Callers that iterate over their own channel list are unaffected
    by the extra keys.
    """
    channels = list(spend_df.columns)
    names = list(channels)
    cols = [spend_df[c].to_numpy() for c in channels]

    if controls is not None:
        if isinstance(controls, pd.Series):
            controls = controls.to_frame(name=controls.name or "control")
        if len(controls) != len(spend_df):
            raise ValueError(
                f"controls has {len(controls)} rows but spend_df has "
                f"{len(spend_df)} -- they must match."
            )
        for c in controls.columns:
            if c in names:
                raise ValueError(
                    f"control column '{c}' collides with a channel name in "
                    "spend_df; rename it."
                )
            names.append(c)
            cols.append(controls[c].to_numpy())

    x = np.column_stack([np.ones(len(spend_df)), *cols])
    y = sales.to_numpy()
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {n: float(coeffs[i + 1]) for i, n in enumerate(names)}
