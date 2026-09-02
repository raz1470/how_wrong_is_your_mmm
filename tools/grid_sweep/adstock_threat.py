"""How much of the phasing gain survives realistic carryover?

The bias reduction works by dilution: phasing adds high-frequency spend
variation that demand cannot explain, so the endogenous part of spend becomes a
smaller fraction of what the model has to work with. Adstock is a low-pass
filter and attenuates exactly that component. On real spend it takes the sd of
week-to-week change from 26,674 at decay 0 to 2,130 at decay 0.9.

So this is a threat test on a published headline, not a feature study.

The model here is CORRECTLY specified with respect to carryover -- the fit
applies the same decay the DGP used, as it would if a client supplied theirs.
That isolates the question. If phasing's benefit fell apart only because the
fitted model ignored adstock, that would be a statement about the analyst, not
about the lever. Demand stays omitted, because omitting it is the bias under
study.

Saturation is deliberately off (b = 1). One mechanism at a time.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from profile_grid import (
    BASELINE_CV,
    BASELINE_SHARE,
    CHANNELS,
    N_HIST,
    N_PLAN,
    TRUE_MR,
    Blackout,
    all_channels,
    is_unphased,
    schedule_for,
)

from how_wrong_is_your_mmm import (
    apply_adstock,
    calibrate_baseline,
    fit_ols,
    simulate_demand,
    simulate_sales,
    simulate_spend,
)

# Demand PROCESS matters here in a way it did not for phasing. Notebook 07
# found bias reduction was invariant to the demand process (the within-month
# covariance hypothesis, falsified). Adstock is different in kind: it is a
# low-pass filter, so its effect depends on where demand's own energy sits in
# the frequency band. A result measured only on white noise would not transfer.
#
# "trend" is the specific, falsifiable prediction from session 43: it should
# be the WORST case of all, because a trend's energy sits at zero frequency
# and passes through the adstock low-pass filter essentially untouched, while
# the high-frequency variation phasing adds is exactly what carryover
# destroys. If bias removal survives on trend as well as it does on white
# noise, that prediction is wrong and the page's adstock caveat needs a
# different justification.
PROCESSES = ("white_noise", "ar1", "seasonal_ar1", "trend")


def _window_standardised(x):
    """Re-standardise a slice of a longer demand series to mean 0 / sd 1 in
    its OWN window.

    `simulate_demand` standardises over the *combined* history+plan window
    (drawn as one series so that shape/phase carries across the boundary --
    important for "seasonal"'s cycle and "trend"'s direction). But
    `simulate_spend`'s correlation targeting assumes whatever demand it is
    handed has unit variance in the exact window it's given. For a
    stationary process the two are close enough not to matter (measured:
    white_noise's plan-window sd is 1.20, ar1's 0.96). For "trend" they are
    not even close -- variance of a sub-window of a random walk is a small,
    non-stationary, geometry-dependent fraction of the full window's
    variance, so the plan slice came out at sd=0.20 before this fix, which
    silently pulled trend's realised spend-demand correlation far below the
    requested `correlation` and made trend look like a far weaker confounder
    than it was asked to be. Renormalising each slice right before it's
    handed to simulate_spend fixes this for every process, not just trend.
    """
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


N_SIMS = int(os.environ.get("N_SIMS", 40))
N_PHASING_SEEDS = int(os.environ.get("N_PHASING_SEEDS", 6))
N_DEMAND_SEEDS = int(os.environ.get("N_DEMAND_SEEDS", 16))

DECAYS = (0.0, 0.3, 0.5, 0.7)  # capped at 0.7: 0.9 is past what "carryover basically
# gone by week 4" (session 44) means, and the negative-bias crossover for
# seasonal_ar1/continuous levers past ~0.85 lives entirely outside that band.

LEVERS = [
    ("unphased", all_channels(0.0)),
    ("+/-20%", all_channels(20.0)),
    ("+/-40%", all_channels(40.0)),
    ("+/-80%", all_channels(80.0)),
    ("Blackout", {ch: Blackout(max_dark_weeks_per_month=1) for ch in CHANNELS}),
]


def adstocked_frame(spend, decay):
    """The spend the correctly-specified analyst would regress on."""
    if decay == 0.0:
        return spend
    out = spend.copy()
    for ch in CHANNELS:
        out[ch] = apply_adstock(spend[ch].to_numpy(), decay)
    return out


def exogenous_variation(spend, demand_arr, decay):
    """Share of adstocked spend variance demand cannot explain, averaged over
    channels. This is the quantity the dilution mechanism runs on, so it is
    worth measuring directly rather than inferring from the bias."""
    frame = adstocked_frame(spend, decay)
    shares = []
    for ch in CHANNELS:
        x = frame[ch].to_numpy()
        r = np.corrcoef(x, demand_arr)[0, 1]
        shares.append(1.0 - r**2)
    return float(np.mean(shares))


def main():
    started = time.time()
    records = []
    for process in PROCESSES:
        for decay in DECAYS:
            for label, levers in LEVERS:
                bias = {ch: [] for ch in CHANNELS}
                exo = []
                for demand_seed in range(N_DEMAND_SEEDS):
                    plan_df, dem, cal = build_world(demand_seed, process)
                    seeds = [0] if is_unphased(levers) else range(N_PHASING_SEEDS)
                    means = {ch: [] for ch in CHANNELS}
                    for phasing_seed in seeds:
                        sched = schedule_for(plan_df, levers, phasing_seed)
                        dem_arr = dem.loc[sched.index].to_numpy()
                        design_frame = adstocked_frame(sched, decay)
                        exo.append(exogenous_variation(sched, dem_arr, decay))
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
                records.append(
                    {
                        "process": process,
                        "decay": decay,
                        "lever": label,
                        "mean_bias_%": float(
                            np.mean([np.mean(bias[ch]) for ch in CHANNELS])
                        ),
                        "exogenous_share": float(np.mean(exo)),
                    }
                )
                print(
                    f"  {process} decay={decay} {label} ({time.time() - started:.0f}s)",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    frame.to_csv("adstock_threat.csv", index=False)

    order = [lab for lab, _ in LEVERS]
    pd.set_option("display.width", 200)
    for process, block in frame.groupby("process", sort=False):
        piv = block.pivot(index="lever", columns="decay", values="mean_bias_%").loc[
            order
        ]
        removed = 100 * (piv.loc["unphased"] - piv) / piv.loc["unphased"]
        print(f"\n{'=' * 78}\nPROCESS: {process}\n{'=' * 78}")
        print("\nmean bias %, by lever and carryover:")
        print(piv.round(1).to_string())
        print("\nbias removed vs that column's own unphased row, %:")
        print(removed.round(1).to_string())
        print("\nshare of the no-carryover gain that survives, %:")
        print((100 * removed.div(removed[0.0], axis=0)).round(1).to_string())
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
