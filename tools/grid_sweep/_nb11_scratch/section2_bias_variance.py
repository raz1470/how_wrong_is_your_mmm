"""Section 2 probe: bias/variance for edge+balanced vs uniform vs Blackout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import (
    calibrate_baseline,
    fit_ols,
    simulate_demand,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._phaser import Blackout, _generate_phased_schedule

N_HIST, N_PLAN = 208, 52
CHANNELS = ["tv", "meta", "search"]
TRUE_MR = {"tv": 0.5, "meta": 1.0, "search": 1.5}
BASELINE_SHARE = 0.72
BASELINE_CV = 0.05

N_SIMS = 40
N_PHASING_SEEDS = 6
N_DEMAND_SEEDS = 6


def _window_standardised(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def build_world(
    process="white_noise", demand_seed=0, correlation=0.7, demand_share=1.0
):
    n = N_HIST + N_PLAN
    demand = simulate_demand(n, process=process, seed=demand_seed)
    hist_demand = _window_standardised(demand[:N_HIST])
    plan_demand = _window_standardised(demand[N_HIST:])
    history = simulate_spend(
        n_obs=N_HIST,
        correlation=correlation,
        seed=1000 + demand_seed,
        start_date="2019-01-07",
        demand=hist_demand,
        demand_share=demand_share,
    )
    plan = simulate_spend(
        n_obs=N_PLAN,
        correlation=correlation,
        seed=2000 + demand_seed,
        start_date="2023-01-09",
        demand=plan_demand,
        demand_share=demand_share,
    )
    index = history.index.append(plan.index)
    demand_series = pd.Series(
        np.concatenate([hist_demand, plan_demand]), index=index, name="demand"
    )
    calibration = calibrate_baseline(
        pd.concat([history, plan]),
        TRUE_MR,
        baseline_share=BASELINE_SHARE,
        baseline_cv=BASELINE_CV,
    )
    return plan, demand_series, calibration


def all_channels(nominal):
    return {ch: nominal for ch in CHANNELS}


LEVERS = [
    ("unphased", all_channels(0.0), "uniform", False),
    ("+/-80% (uniform)", all_channels(80.0), "uniform", False),
    ("+/-40% (edge, balanced)", all_channels(40.0), "edge", True),
    ("+/-80% (edge, balanced)", all_channels(80.0), "edge", True),
    (
        "Blackout",
        {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS},
        "uniform",
        False,
    ),
]


def is_unphased(spec):
    return all(isinstance(v, float) and v == 0.0 for v in spec.values())


def schedule_for(
    plan_df, spec, seed, nudge_shape="uniform", balance_signs=False, freq="M"
):
    if is_unphased(spec):
        return plan_df
    return _generate_phased_schedule(
        plan_df,
        plan_df.index.to_period(freq).to_numpy(),
        alpha=1.0,
        max_weekly_deviation_pct=spec,
        seed=seed,
        nudge_shape=nudge_shape,
        balance_signs=balance_signs,
    )


def fit_draws(spend, demand_series, calibration, n_sims):
    aligned = demand_series.loc[spend.index]
    demand_values = aligned.to_numpy()
    draws = {ch: [] for ch in CHANNELS}
    for sim in range(n_sims):
        sales = simulate_sales(
            spend,
            TRUE_MR,
            base_sales=calibration.baseline_level,
            seed=sim,
            demand=demand_values,
            demand_coef=calibration.demand_coef,
        )
        fitted = fit_ols(spend, sales, controls=None)
        for ch in CHANNELS:
            draws[ch].append(fitted[ch])
    return {ch: np.array(v) for ch, v in draws.items()}


def main():
    rows = []
    for label, spec, nudge_shape, balance_signs in LEVERS:
        bias = {ch: [] for ch in CHANNELS}
        cv = {ch: [] for ch in CHANNELS}
        for demand_seed in range(N_DEMAND_SEEDS):
            plan_df, dem, cal = build_world(demand_seed=demand_seed)
            means = {ch: [] for ch in CHANNELS}
            spreads = {ch: [] for ch in CHANNELS}
            seeds = [0] if is_unphased(spec) else range(N_PHASING_SEEDS)
            for seed in seeds:
                schedule = schedule_for(plan_df, spec, seed, nudge_shape, balance_signs)
                draws = fit_draws(schedule, dem, cal, N_SIMS)
                for ch in CHANNELS:
                    means[ch].append(draws[ch].mean())
                    spreads[ch].append(100 * draws[ch].std() / abs(draws[ch].mean()))
            for ch in CHANNELS:
                bias[ch].append(100 * (np.mean(means[ch]) - TRUE_MR[ch]) / TRUE_MR[ch])
                cv[ch].append(np.median(spreads[ch]))
        mean_bias = float(np.mean([np.mean(bias[ch]) for ch in CHANNELS]))
        mean_cv = float(np.mean([np.mean(cv[ch]) for ch in CHANNELS]))
        rows.append({"lever": label, "mean_bias_%": mean_bias, "mean_cv_%": mean_cv})
        print(f"  done {label}", flush=True)

    frame = pd.DataFrame(rows).set_index("lever")
    ref_bias = frame.loc["unphased", "mean_bias_%"]
    ref_cv = frame.loc["unphased", "mean_cv_%"]
    frame["bias_removed_%"] = 100 * (ref_bias - frame["mean_bias_%"]) / ref_bias
    frame["cv_narrowed_%"] = 100 * (ref_cv - frame["mean_cv_%"]) / ref_cv
    pd.set_option("display.width", 200)
    print(frame.round(2).to_string())


if __name__ == "__main__":
    main()
