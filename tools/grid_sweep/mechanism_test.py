"""Is Blackout's identification advantage the ZEROS, or just the wide range?

Blackout beats +/-80% on recovering both b and lambda, even though +/-80% has a
wider top end. Two candidate explanations:

  (a) the dark weeks are weeks with no media at all, so they measure the
      baseline directly -- a holdout in all but name;
  (b) nothing special about zero, Blackout simply moves spend around more.

The test: take the Blackout schedule and lift its zeros onto a small floor,
rescaling within each month so the monthly total is preserved. Same shape, same
movement, no true zeros. If (a), identification collapses back toward +/-80%.
If (b), it barely moves.

Session 44 continued: run across both of profile_grid's TRUTHS, not just one
corner -- "go low, not dark" is about to become a recommendation and needs to
hold at (0.8, 0.3) as well as (0.6, 0.5) before it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from profile_grid import (
    B_CANDIDATES,
    CHANNELS,
    LAM_CANDIDATES,
    N_DEMAND_SEEDS,
    N_PHASING_SEEDS,
    N_SIMS,
    TRUE_MR,
    TRUTHS,
    Blackout,
    all_channels,
    build_world,
    profile,
    schedule_for,
    simulate_sales,
    valley_width,
)

BLACKOUT = {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}


def floored(schedule, plan_df, floor_frac):
    """Lift zeros onto floor_frac * that channel's plan mean, then rescale
    within each month so the monthly total is unchanged."""
    out = schedule.copy()
    months = schedule.index.to_period("M")
    for ch in CHANNELS:
        x = out[ch].to_numpy().astype(float)
        floor = floor_frac * float(plan_df[ch].mean())
        x = np.maximum(x, floor)
        for m in months.unique():
            mask = np.asarray(months == m)
            target = schedule[ch].to_numpy()[mask].sum()
            current = x[mask].sum()
            if current > 0:
                # Rescale the headroom above the floor so the floor survives.
                head = x[mask] - floor
                head_sum = head.sum()
                if head_sum > 0 and target > floor * mask.sum():
                    x[mask] = floor + head * (target - floor * mask.sum()) / head_sum
                else:
                    x[mask] = x[mask] * target / current
        out[ch] = x
    return out


ARMS = [
    ("Blackout (true zeros)", 0.00),
    ("Blackout, floor 2% of plan", 0.02),
    ("Blackout, floor 5% of plan", 0.05),
    ("Blackout, floor 15% of plan", 0.15),
    ("+/-80% (for reference)", None),
]


def main():
    rows = []
    for b_true, lam_true in TRUTHS:
        for label, floor_frac in ARMS:
            rec_b, rec_lam, widths, mins = [], [], [], []
            for demand_seed in range(N_DEMAND_SEEDS):
                plan_df, dem, cal = build_world(demand_seed)
                ref = {ch: float(plan_df[ch].mean()) for ch in CHANNELS}
                for phasing_seed in range(N_PHASING_SEEDS):
                    if floor_frac is None:
                        sched = schedule_for(plan_df, all_channels(80.0), phasing_seed)
                    else:
                        sched = schedule_for(plan_df, BLACKOUT, phasing_seed)
                        if floor_frac > 0:
                            sched = floored(sched, plan_df, floor_frac)
                    dem_arr = dem.loc[sched.index].to_numpy()
                    sales = np.column_stack(
                        [
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
                    )
                    spend_arr = sched[CHANNELS].to_numpy()
                    bb, ll, surface = profile(spend_arr, dem_arr, sales)
                    rec_b.append(bb)
                    rec_lam.append(ll)
                    widths.append(valley_width(surface))
                    mins.append(
                        np.mean(
                            [
                                sched[ch].min() / plan_df[ch].mean()
                                for ch in CHANNELS
                            ]
                        )
                    )
            rec_b = np.concatenate(rec_b)
            rec_lam = np.concatenate(rec_lam)
            rows.append(
                {
                    "b_true": b_true,
                    "lam_true": lam_true,
                    "arm": label,
                    "min spend, x plan mean": float(np.mean(mins)),
                    "b_sd": rec_b.std(),
                    "lam_sd": rec_lam.std(),
                    "valley_%": 100 * float(np.mean(widths)),
                }
            )
            print(f"  done b={b_true} lam={lam_true} {label}", flush=True)

    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(frame.round(3).to_string(index=False))
    frame.to_csv("mechanism_test.csv", index=False)


if __name__ == "__main__":
    main()
