"""Section 3 probe: saturation identifiability for edge+balanced vs uniform vs Blackout."""
from __future__ import annotations

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

TRUTHS = [(0.6, 0.5), (0.8, 0.3)]

N_SIMS = 25
N_PHASING_SEEDS = 6
N_DEMAND_SEEDS = 16

B_CANDIDATES = np.round(np.linspace(0.20, 1.00, 33), 4)
LAM_CANDIDATES = np.round(np.linspace(0.00, 0.90, 31), 4)


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
    index = history.index.append(plan.index)
    demand_series = pd.Series(np.concatenate([hist_demand, plan_demand]), index=index, name="demand")
    calibration = calibrate_baseline(
        pd.concat([history, plan]), TRUE_MR, baseline_share=BASELINE_SHARE, baseline_cv=BASELINE_CV,
    )
    return plan, demand_series, calibration


def all_channels(nominal):
    return {ch: nominal for ch in CHANNELS}


LEVERS = [
    ("unphased", all_channels(0.0), "uniform", False),
    ("+/-80% (uniform)", all_channels(80.0), "uniform", False),
    ("+/-40% (edge, balanced)", all_channels(40.0), "edge", True),
    ("+/-80% (edge, balanced)", all_channels(80.0), "edge", True),
    ("Blackout", {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}, "uniform", False),
]


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


def design(spend_arr, demand_arr, b, lam):
    n, k = spend_arr.shape
    X = np.empty((n, k + 2))
    X[:, 0] = 1.0
    for j in range(k):
        X[:, 1 + j] = apply_adstock(spend_arr[:, j], lam) ** b
    X[:, -1] = demand_arr
    return X


def profile(spend_arr, demand_arr, sales_matrix):
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
    lo = surface.min()
    return float((surface <= lo * (1.0 + tol)).mean())


def main():
    rows = []
    for b_true, lam_true in TRUTHS:
        for label, spec, nudge_shape, balance_signs in LEVERS:
            rec_b, rec_lam, widths = [], [], []
            for demand_seed in range(N_DEMAND_SEEDS):
                plan_df, dem, cal = build_world(demand_seed)
                ref = {ch: float(plan_df[ch].mean()) for ch in CHANNELS}
                seeds = [0] if is_unphased(spec) else range(N_PHASING_SEEDS)
                for phasing_seed in seeds:
                    sched = schedule_for(plan_df, spec, phasing_seed, nudge_shape, balance_signs)
                    dem_arr = dem.loc[sched.index].to_numpy()
                    sales_cols = [
                        simulate_sales(
                            sched, TRUE_MR, base_sales=cal.baseline_level, seed=sim,
                            demand=dem_arr, demand_coef=cal.demand_coef,
                            saturation=b_true, adstock=lam_true, reference_spend=ref,
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
            rows.append({
                "b_true": b_true, "lam_true": lam_true, "lever": label,
                "b_sd": rec_b.std(), "lam_sd": rec_lam.std(),
                "valley_%": 100 * float(np.mean(widths)),
            })
            print(f"  done b={b_true} lam={lam_true} {label}", flush=True)

    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for (b_true, lam_true), block in frame.groupby(["b_true", "lam_true"]):
        print(f"\n=== true b = {b_true}, true lambda = {lam_true} ===")
        print(block[["lever", "b_sd", "lam_sd", "valley_%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
