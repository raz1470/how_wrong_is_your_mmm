"""Add annulus+balanced +/-40%/+/-80% to the whole-plan cost check.
Deterministic (concave_truth), no simulation needed -- mirrors section5_cost.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import calibrate_baseline, simulate_demand, simulate_spend
from how_wrong_is_your_mmm._phaser import Blackout, _generate_phased_schedule

N_HIST, N_PLAN = 208, 52
CHANNELS = ["tv", "meta", "search"]
TRUE_MR = {"tv": 0.5, "meta": 1.0, "search": 1.5}
BASELINE_SHARE = 0.72
BASELINE_CV = 0.05
N_DRAW_SEEDS = 200


def _window_standardised(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def build_world(demand_seed, correlation=0.7, demand_share=1.0):
    n = N_HIST + N_PLAN
    demand = simulate_demand(n, process="white_noise", seed=demand_seed)
    hist_demand = _window_standardised(demand[:N_HIST])
    plan_demand = _window_standardised(demand[N_HIST:])
    history = simulate_spend(
        n_obs=N_HIST, correlation=correlation, seed=1000 + demand_seed,
        start_date="2019-01-07", demand=hist_demand, demand_share=demand_share,
    )
    plan = simulate_spend(
        n_obs=N_PLAN, correlation=correlation, seed=2000 + demand_seed,
        start_date="2023-01-09", demand=plan_demand, demand_share=demand_share,
    )
    return plan


plan_df = build_world(demand_seed=0)
X0 = {ch: float(plan_df[ch].mean()) for ch in CHANNELS}
B = 0.6


def concave_truth(x, ch, b):
    mr0 = TRUE_MR[ch]
    x0 = X0[ch]
    k = mr0 / (b * x0 ** (b - 1.0))
    return k * np.asarray(x, dtype=float) ** b


def all_channels(nominal):
    return {ch: nominal for ch in CHANNELS}


def is_unphased(spec):
    return all(isinstance(v, float) and v == 0.0 for v in spec.values())


def schedule_for(plan_df, spec, seed, nudge_shape="uniform", balance_signs=False, freq="M"):
    if is_unphased(spec):
        return plan_df
    return _generate_phased_schedule(
        plan_df, plan_df.index.to_period(freq).to_numpy(), alpha=1.0,
        max_weekly_deviation_pct=spec, seed=seed,
        nudge_shape=nudge_shape, balance_signs=balance_signs,
    )


LEVERS = [
    ("unphased", all_channels(0.0), "uniform", False),
    ("+/-40% (annulus, balanced)", all_channels(40.0), "annulus", True),
    ("+/-80% (annulus, balanced)", all_channels(80.0), "annulus", True),
    ("+/-80% (edge, balanced)", all_channels(80.0), "edge", True),
    ("Blackout", {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}, "uniform", False),
]

baseline = sum(concave_truth(plan_df[ch].to_numpy(), ch, B).sum() for ch in CHANNELS)

rows = []
for label, spec, nudge_shape, balance_signs in LEVERS:
    if is_unphased(spec):
        totals = [baseline]
    else:
        totals = []
        for seed in range(N_DRAW_SEEDS):
            sched = schedule_for(plan_df, spec, seed, nudge_shape, balance_signs)
            totals.append(sum(concave_truth(sched[ch].to_numpy(), ch, B).sum() for ch in CHANNELS))
    mean_total = float(np.mean(totals))
    cost_pct = 100 * (baseline - mean_total) / baseline
    rows.append({"lever": label, "cost: revenue given up %": cost_pct})

frame = pd.DataFrame(rows).set_index("lever")
pd.set_option("display.width", 200)
print(f"b={B}, whole plan (all 3 channels phased together)\n")
print(frame.round(3).to_string())
