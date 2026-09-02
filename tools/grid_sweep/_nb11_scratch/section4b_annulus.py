"""Add annulus+balanced +/-80% to the adstock robustness check.

Includes 'unphased' in this same run (not cross-run hardcoded) so
removed-%/survival-% are computed against a reference taken with the
exact same seeds/settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import (
    apply_adstock,
    calibrate_baseline,
    fit_ols,
    simulate_demand,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._phaser import _generate_phased_schedule

N_HIST, N_PLAN = 208, 52
CHANNELS = ["tv", "meta", "search"]
TRUE_MR = {"tv": 0.5, "meta": 1.0, "search": 1.5}
BASELINE_SHARE = 0.72
BASELINE_CV = 0.05

N_SIMS = 40
N_PHASING_SEEDS = 6
N_DEMAND_SEEDS = 16

DECAYS = (0.0, 0.3, 0.5, 0.7)


def _window_standardised(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def build_world(demand_seed, process="white_noise", correlation=0.7, demand_share=1.0):
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
    ("+/-80% (annulus, balanced)", all_channels(80.0), "annulus", True),
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


def adstocked_frame(spend, decay):
    if decay == 0.0:
        return spend
    out = spend.copy()
    for ch in CHANNELS:
        out[ch] = apply_adstock(spend[ch].to_numpy(), decay)
    return out


def main():
    rows = []
    for decay in DECAYS:
        for label, spec, nudge_shape, balance_signs in LEVERS:
            bias = {ch: [] for ch in CHANNELS}
            for demand_seed in range(N_DEMAND_SEEDS):
                plan_df, dem, cal = build_world(demand_seed)
                seeds = [0] if is_unphased(spec) else range(N_PHASING_SEEDS)
                means = {ch: [] for ch in CHANNELS}
                for phasing_seed in seeds:
                    sched = schedule_for(
                        plan_df, spec, phasing_seed, nudge_shape, balance_signs
                    )
                    dem_arr = dem.loc[sched.index].to_numpy()
                    design_frame = adstocked_frame(sched, decay)
                    draws = {ch: [] for ch in CHANNELS}
                    for sim in range(N_SIMS):
                        sales = simulate_sales(
                            sched,
                            TRUE_MR,
                            base_sales=cal.baseline_level,
                            seed=sim,
                            demand=dem_arr,
                            demand_coef=cal.demand_coef,
                            adstock=decay,
                        )
                        fitted = fit_ols(design_frame, sales, controls=None)
                        for ch in CHANNELS:
                            draws[ch].append(fitted[ch])
                    for ch in CHANNELS:
                        means[ch].append(np.mean(draws[ch]))
                for ch in CHANNELS:
                    bias[ch].append(
                        100 * (np.mean(means[ch]) - TRUE_MR[ch]) / TRUE_MR[ch]
                    )
            rows.append(
                {
                    "decay": decay,
                    "lever": label,
                    "mean_bias_%": float(
                        np.mean([np.mean(bias[ch]) for ch in CHANNELS])
                    ),
                }
            )
            print(f"  decay={decay} {label}", flush=True)

    frame = pd.DataFrame(rows)
    order = [lab for lab, _, _, _ in LEVERS]
    piv = frame.pivot(index="lever", columns="decay", values="mean_bias_%").loc[order]
    removed = 100 * (piv.loc["unphased"] - piv) / piv.loc["unphased"]
    pd.set_option("display.width", 200)
    print("\nmean bias %, by lever and decay:")
    print(piv.round(2).to_string())
    print("\nbias removed vs unphased, %:")
    print(removed.round(2).to_string())
    print("\nshare of the no-carryover gain surviving at each decay, %:")
    print((100 * removed.div(removed[0.0], axis=0)).round(2).to_string())


if __name__ == "__main__":
    main()
