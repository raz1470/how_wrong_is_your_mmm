"""Notebook 07's lever sweep, re-run across demand_share and correlation.

Notebook 07 sections 4-8 all sit at demand_share=1.0, correlation=0.7 -- the
maximal-coupling corner. This script re-runs the same measurement across a grid
of both, and records enough per-cell detail that bias, CV and lambda (the
marginal-return inflation behind the budget multiple) all derive from one pass.

Structure differs from the notebook deliberately: the notebook loops
lever-outer and rebuilds the world for every lever, which is wasteful once
there is a grid on top. Here the world is built once per
(demand_share, correlation, demand_seed) and reused across levers. The
aggregation is copied exactly from the notebook's `sweep()` so the
demand_share=1.0 / correlation=0.7 slice is directly comparable to the
published tables.

Output: one tidy CSV, one row per
(demand_share, correlation, lever, demand_seed, phasing_seed, channel),
carrying the mean and sd of the n_sims coefficient draws. Everything else is
derived downstream, so re-aggregating never means re-running.
"""

from __future__ import annotations

import itertools
import os
import time

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

# --- constants, copied verbatim from notebook 07 cell 2 -------------------

N_HIST, N_PLAN = 208, 52
CHANNELS = ["tv", "meta", "search"]
TRUE_MR = {"tv": 0.5, "meta": 1.0, "search": 1.5}
BASELINE_SHARE = 0.72
BASELINE_CV = 0.05

# --- sample sizes, overridable from the environment -----------------------

N_SIMS = int(os.environ.get("N_SIMS", 40))
N_PHASING_SEEDS = int(os.environ.get("N_PHASING_SEEDS", 6))
N_DEMAND_SEEDS = int(os.environ.get("N_DEMAND_SEEDS", 6))

# --- the grid -------------------------------------------------------------

DEMAND_SHARES = tuple(
    float(x)
    for x in os.environ.get("DEMAND_SHARES", "1.0,0.75,0.5,0.25,0.0").split(",")
)
CORRELATIONS = tuple(
    float(x) for x in os.environ.get("CORRELATIONS", "0.4,0.7,0.9").split(",")
)


def all_channels(nominal):
    return {ch: nominal for ch in CHANNELS}


LEVERS = [
    ("unphased", all_channels(0.0)),
    ("+/-20%", all_channels(20.0)),
    ("+/-40%", all_channels(40.0)),
    ("+/-80%", all_channels(80.0)),
    ("Blackout", {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}),
    ("TV alone +/-80%", {"tv": 80.0, "meta": 0.0, "search": 0.0}),
]


# --- world construction, threading the two grid dials through -------------


def _window_standardised(x):
    """Re-standardise a slice of a longer demand series to mean 0 / sd 1 in
    its OWN window -- see adstock_threat.py's docstring of the same name.
    Session-44 propagation: fixed there first, now here too."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def build_world(
    process="white_noise",
    demand_seed=0,
    correlation=0.7,
    demand_share=1.0,
    baseline_cv=BASELINE_CV,
):
    """Notebook 07 cell 4, unchanged except that the dials are already
    parameters there -- the notebook simply never varied them."""
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
        baseline_cv=baseline_cv,
    )
    return history, plan, demand_series, calibration


def is_unphased(levers):
    return all(isinstance(v, float) and v == 0.0 for v in levers.values())


def schedule_for(plan_df, levers, seed, freq="M"):
    if is_unphased(levers):
        return plan_df
    return _generate_phased_schedule(
        plan_df,
        plan_df.index.to_period(freq).to_numpy(),
        alpha=1.0,
        max_weekly_deviation_pct=levers,
        seed=seed,
    )


def fit_draws(spend, demand_series, calibration, n_sims):
    """Sampling distribution of every channel's coefficient over sales noise,
    with demand omitted from the fit. Notebook 07 cell 6 with control=False."""
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
    out_path = os.environ.get("OUT", "grid_raw.csv")
    started = time.time()
    records = []
    combos = list(itertools.product(DEMAND_SHARES, CORRELATIONS))
    total_cells = len(combos) * N_DEMAND_SEEDS
    done_cells = 0

    for demand_share, correlation in combos:
        for demand_seed in range(N_DEMAND_SEEDS):
            # Built once and reused across every lever -- the world does not
            # depend on the schedule.
            _hist, plan_df, dem, cal = build_world(
                demand_seed=demand_seed,
                correlation=correlation,
                demand_share=demand_share,
            )
            # Diagnostics a client could actually observe, recorded so the
            # "you cannot tell these worlds apart" claim can be checked rather
            # than asserted.
            corr_matrix = plan_df.corr().to_numpy()
            pairwise = float(corr_matrix[np.triu_indices(len(CHANNELS), k=1)].mean())
            coupling = float(
                np.corrcoef(
                    plan_df["tv"].to_numpy(), dem.loc[plan_df.index].to_numpy()
                )[0, 1]
            )

            for label, levers in LEVERS:
                seeds = [0] if is_unphased(levers) else range(N_PHASING_SEEDS)
                for phasing_seed in seeds:
                    schedule = schedule_for(plan_df, levers, phasing_seed)
                    draws = fit_draws(schedule, dem, cal, N_SIMS)
                    for ch in CHANNELS:
                        est = draws[ch]
                        records.append(
                            {
                                "demand_share": demand_share,
                                "correlation": correlation,
                                "lever": label,
                                "demand_seed": demand_seed,
                                "phasing_seed": phasing_seed,
                                "channel": ch,
                                "mean_est": float(est.mean()),
                                "sd_est": float(est.std()),
                                "n_sims": N_SIMS,
                                "true_mr": TRUE_MR[ch],
                                "pairwise_corr": pairwise,
                                "corr_tv_demand": coupling,
                            }
                        )
            done_cells += 1
            elapsed = time.time() - started
            rate = elapsed / done_cells
            print(
                f"[{done_cells}/{total_cells}] d={demand_share} rho={correlation} "
                f"seed={demand_seed} | {elapsed:.0f}s elapsed, "
                f"~{rate * (total_cells - done_cells):.0f}s left",
                flush=True,
            )

    frame = pd.DataFrame(records)
    frame.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}: {len(frame):,} rows in {time.time() - started:.0f}s")
    print(
        f"config: N_SIMS={N_SIMS} N_PHASING_SEEDS={N_PHASING_SEEDS} "
        f"N_DEMAND_SEEDS={N_DEMAND_SEEDS}"
    )


if __name__ == "__main__":
    main()
