"""Loop 21: month-level orthogonal step design. Each month, each channel's
whole month is scaled by (1 +/- step_pct/100), signs from a Hadamard(12)
non-constant column (balanced 6 up / 6 down per channel, orthogonal across
channels). Weekly shape within a month is unchanged (no weekly noise, no
pauses), so budget changes occur only at month boundaries (<=12 per channel
per year). Annual total per channel is restored by a tiny rescale.
Applied per 52-week block, fresh row/column shuffle per block."""

import numpy as np
import pandas as pd
from hadamard import hadamard

from how_wrong_is_your_mmm._phaser import _get_month_labels


def month_step_schedule(plan_df, month_labels, seed, step_pct=40.0):
    out = plan_df.to_numpy().astype(float).copy()
    n, k = out.shape
    H = hadamard(12)[:, 1:]  # 12 x 11, columns orthogonal & balanced
    for b, a in enumerate(range(0, n, 52)):
        idx = np.arange(a, min(a + 52, n))
        rng = np.random.default_rng(seed + 1000 * b)
        ml = _get_month_labels(plan_df.iloc[idx])
        months = list(dict.fromkeys(ml))  # in order
        lens = {m: int((ml == m).sum()) for m in months}
        full = [m for m in months if lens[m] >= 2]
        rows = rng.permutation(12)
        cols = rng.permutation(11)[:k]
        orig = out[idx].sum(axis=0)
        new = out[idx].copy()
        for mi, m in enumerate(full):
            sel = ml == m
            sign = H[rows[mi % 12], cols]  # (k,)
            new[sel] = new[sel] * (1.0 + step_pct / 100.0 * sign)
        new *= orig / new.sum(axis=0)
        out[idx] = new
    return pd.DataFrame(out, index=plan_df.index, columns=plan_df.columns)
