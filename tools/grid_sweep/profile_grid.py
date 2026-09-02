"""Can a phased schedule identify saturation and adstock?

Both parameters enter nonlinearly and everything else is linear given them, so
this profiles: grid over (b, lambda), transform the spend columns, run the
linear fit, keep the residual sum of squares, take the argmin.

The recovered value is not really the answer. The answer is the SPREAD of the
recovered value across draws: a flat RSS valley means many curvatures fit
equally well, which is what "b is unmeasurable" actually means in practice.

Fits include the true demand series as a control throughout. That is
deliberate -- the question here is whether the SPEND PATTERN identifies the
response shape, which is separate from omitted-variable bias. Mixing the two
would make a null result unattributable.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import (
    apply_adstock,
    calibrate_baseline,
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

N_SIMS = int(os.environ.get("N_SIMS", 25))
N_PHASING_SEEDS = int(os.environ.get("N_PHASING_SEEDS", 6))
N_DEMAND_SEEDS = int(os.environ.get("N_DEMAND_SEEDS", 16))

# Truths to recover. Two settings so a result is not specific to one corner.
TRUTHS = [(0.6, 0.5), (0.8, 0.3)]

# Candidate grids for the profile.
# Deliberately wider than the plausible range: when a lever cannot identify b
# the estimates pile on the grid boundaries, which CAPS the measured spread and
# flatters the result. A wide grid lets a genuinely uninformative case look as
# uninformative as it is.
B_CANDIDATES = np.round(np.linspace(0.20, 1.00, 33), 4)
LAM_CANDIDATES = np.round(np.linspace(0.00, 0.90, 31), 4)


def all_channels(nominal):
    return {ch: nominal for ch in CHANNELS}


LEVERS = [
    ("unphased", all_channels(0.0)),
    ("+/-20%", all_channels(20.0)),
    ("+/-40%", all_channels(40.0)),
    ("+/-80%", all_channels(80.0)),
    ("Blackout", {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}),
]


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


def _window_standardised(x):
    """Re-standardise a slice of a longer demand series to mean 0 / sd 1 in
    its OWN window -- see adstock_threat.py's docstring of the same name for
    the full derivation. simulate_demand standardises over the combined
    history+plan window; simulate_spend's correlation targeting assumes unit
    variance in whatever window it's actually handed. For white_noise here
    the gap is modest (measured: plan-window sd ~1.20 before this fix) but
    it is the same bug, so it is fixed the same way. Session-44 propagation:
    was only in adstock_threat.py's build_world, now here too.
    """
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def build_world(demand_seed, correlation=0.7, demand_share=1.0):
    n = N_HIST + N_PLAN
    demand = simulate_demand(n, process="white_noise", seed=demand_seed)
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


def design(spend_arr, demand_arr, b, lam):
    """[1, adstock(x, lam)**b per channel, demand]."""
    n, k = spend_arr.shape
    X = np.empty((n, k + 2))
    X[:, 0] = 1.0
    for j in range(k):
        X[:, 1 + j] = apply_adstock(spend_arr[:, j], lam) ** b
    X[:, -1] = demand_arr
    return X


def profile(spend_arr, demand_arr, sales_matrix):
    """RSS surface over the (b, lambda) grid, and the argmin per sim.

    sales_matrix is (n_obs, n_sims): one column per noise draw. Solving all
    draws against one design matrix at once is what makes the grid cheap.
    """
    n_sims = sales_matrix.shape[1]
    best_rss = np.full(n_sims, np.inf)
    best_b = np.zeros(n_sims)
    best_lam = np.zeros(n_sims)
    surface = np.zeros((len(B_CANDIDATES), len(LAM_CANDIDATES)))

    for i, b in enumerate(B_CANDIDATES):
        for j, lam in enumerate(LAM_CANDIDATES):
            X = design(spend_arr, demand_arr, b, lam)
            beta, _res, _rank, _sv = np.linalg.lstsq(X, sales_matrix, rcond=None)
            resid = sales_matrix - X @ beta
            rss = (resid**2).sum(axis=0)
            surface[i, j] = rss.mean()
            better = rss < best_rss
            best_rss = np.where(better, rss, best_rss)
            best_b = np.where(better, b, best_b)
            best_lam = np.where(better, lam, best_lam)
    return best_b, best_lam, surface


def valley_width(surface, tol=0.01):
    """Fraction of the (b, lambda) grid within `tol` of the minimum RSS.

    A direct reading of identification: 1.0 means every candidate fits about
    as well as the best one, so nothing is pinned down.
    """
    lo = surface.min()
    return float((surface <= lo * (1.0 + tol)).mean())


def main():
    started = time.time()
    rows = []
    for b_true, lam_true in TRUTHS:
        for label, levers in LEVERS:
            rec_b, rec_lam, widths = [], [], []
            for demand_seed in range(N_DEMAND_SEEDS):
                plan_df, dem, cal = build_world(demand_seed)
                # Fixed across every lever and every truth, so all schedules
                # are calibrated against the SAME response curve.
                ref = {ch: float(plan_df[ch].mean()) for ch in CHANNELS}
                seeds = [0] if is_unphased(levers) else range(N_PHASING_SEEDS)
                for phasing_seed in seeds:
                    sched = schedule_for(plan_df, levers, phasing_seed)
                    dem_arr = dem.loc[sched.index].to_numpy()
                    sales_cols = [
                        simulate_sales(
                            sched,
                            TRUE_MR,
                            base_sales=cal.baseline_level,
                            seed=sim,
                            demand=dem_arr,
                            demand_coef=cal.demand_coef,
                            saturation=b_true,
                            adstock=lam_true,
                            reference_spend=ref,
                        ).to_numpy()
                        for sim in range(N_SIMS)
                    ]
                    sales_matrix = np.column_stack(sales_cols)
                    spend_arr = sched[CHANNELS].to_numpy()
                    bb, ll, surface = profile(spend_arr, dem_arr, sales_matrix)
                    rec_b.append(bb)
                    rec_lam.append(ll)
                    widths.append(valley_width(surface))
            rec_b = np.concatenate(rec_b)
            rec_lam = np.concatenate(rec_lam)
            rows.append(
                {
                    "b_true": b_true,
                    "lam_true": lam_true,
                    "lever": label,
                    "b_mean": rec_b.mean(),
                    "b_sd": rec_b.std(),
                    "b_bias": rec_b.mean() - b_true,
                    "lam_mean": rec_lam.mean(),
                    "lam_sd": rec_lam.std(),
                    "lam_bias": rec_lam.mean() - lam_true,
                    "valley_%": 100 * float(np.mean(widths)),
                }
            )
            print(
                f"  done b={b_true} lam={lam_true} {label} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv("profile_grid.csv", index=False)
    pd.set_option("display.width", 220)
    for (b_true, lam_true), block in frame.groupby(["b_true", "lam_true"]):
        print(f"\n=== true b = {b_true}, true lambda = {lam_true} ===")
        print(
            block[
                [
                    "lever",
                    "b_mean",
                    "b_sd",
                    "b_bias",
                    "lam_mean",
                    "lam_sd",
                    "lam_bias",
                    "valley_%",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
