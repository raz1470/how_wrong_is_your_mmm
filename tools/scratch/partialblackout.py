"""Loop 22: winner with a FLOOR instead of a full blackout. Dark weeks are cut
to floor_pct% of plan (not 0); freed budget = (1-floor)*plan redistributed
proportionally into the round-robin recipient month, then the same edge layer.
Structure copied from design.cross_month_blackout_roundrobin_schedule."""

import numpy as np
import pandas as pd


def partial_blackout_schedule(
    plan_df, month_labels, seed, dark_weeks=4, floor_pct=25.0, min_recipient_weeks=4
):
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: int((month_labels == m).sum()) for m in months}
    new = plan_df.to_numpy().copy().astype(float)
    elig_b = [m for m in months if month_len[m] > dark_weeks]
    elig_r = [m for m in months if month_len[m] >= min_recipient_weeks]
    bo = rng.permutation(len(elig_b))
    re = rng.permutation(len(elig_r))
    f = floor_pct / 100.0
    for ci in range(len(channels)):
        bm = elig_b[bo[ci % len(elig_b)]]
        pool = [m for m in elig_r if m != bm]
        rm = pool[re[(ci + len(elig_r) // 2) % len(elig_r)] % len(pool)]
        bmask = np.where(month_labels == bm)[0]
        nb = len(bmask)
        nd = min(dark_weeks, nb - 1)
        start = rng.integers(0, nb - nd + 1)
        di = bmask[start : start + nd]
        freed = plan_df.iloc[di, ci].to_numpy().sum() * (1 - f)
        new[di, ci] = plan_df.iloc[di, ci].to_numpy() * f
        rmask = np.where(month_labels == rm)[0]
        rw = plan_df.iloc[rmask, ci].to_numpy()
        new[rmask, ci] = rw + freed * (rw / rw.sum())
    return pd.DataFrame(new, index=plan_df.index, columns=plan_df.columns)


def partial_blackout_plus_edge(
    plan_df, month_labels, seed, floor_pct=25.0, edge_cap_pct=15.0
):
    from how_wrong_is_your_mmm._phaser import _generate_phased_schedule

    base = partial_blackout_schedule(plan_df, month_labels, seed, floor_pct=floor_pct)
    return _generate_phased_schedule(
        base,
        month_labels,
        alpha=1.0,
        max_weekly_deviation_pct={c: edge_cap_pct for c in plan_df.columns},
        seed=seed + 10_000,
        nudge_shape="edge",
        balance_signs=True,
    )
