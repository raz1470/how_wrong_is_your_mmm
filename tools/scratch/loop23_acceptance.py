"""Loop 23 -- acceptance re-run of the whole default sweep through the REAL
DiscoveryReport class, now that Redistribute is a first-class strategy.

Scenario is loop 20's: 10 channels, 52wk warm-up history + 156wk plan,
correlation 0.7, per-channel saturation/adstock cycling the overview's four
(b, lambda) pairs. Loop 20 phased the scratch winner by harness code, per
52wk chunk, and the built-in levers across the full plan; here EVERY row,
including the four Redistribute rows, is phased by DiscoveryReport._phase
(Redistribute per 12-month block, inside _generate_phased_schedule), so the
comparison is like-for-like.

Usage:  PYTHONPATH=src python loop23_acceptance.py SEEDS OUT_CSV [--time-baseline]
  SEEDS: comma-separated report seeds, e.g. 0,1,2,3
  --time-baseline: also fit once WITHOUT the Redistribute rows (seed = first
    seed) to measure the runtime the family adds.
"""

import sys
import time

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _default_levers,
    _safe_improvement,
)

NCH = 10
CHANNELS = [f"ch{i}" for i in range(NCH)]
TRUE_MARGINAL_RETURNS = {c: 0.5 + 0.9 * i / (NCH - 1) for i, c in enumerate(CHANNELS)}
_OVERVIEW_SATURATION = [0.60, 0.75, 0.90, 0.70]
_OVERVIEW_ADSTOCK = [0.50, 0.30, 0.10, 0.20]
SATURATION = {c: _OVERVIEW_SATURATION[i % 4] for i, c in enumerate(CHANNELS)}
ADSTOCK = {c: _OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(CHANNELS)}

FIT_KW = dict(n_sims=20, n_phasing_seeds=2, id_n_sims=8)  # same as loop 20


def scenario():
    hist = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    ps = hist.index[-1] + pd.Timedelta(weeks=1)
    plan = simulate_spend(
        n_obs=156, correlation=0.7, channels=CHANNELS, seed=1, start_date=ps
    )
    return hist, plan


def make_report(hist, plan, seed, levers=None):
    return DiscoveryReport(
        hist,
        plan,
        true_marginal_returns=TRUE_MARGINAL_RETURNS,
        saturation=SATURATION,
        adstock=ADSTOCK,
        levers=levers,
        seed=seed,
    )


def rows_for(report, plan, seed, fit_seconds):
    combined = pd.concat([report.history_df, report.plan_df])
    true_c = channel_contributions(
        combined,
        report.true_marginal_returns,
        report.saturation,
        report.adstock,
        report.reference_spend_,
    )
    n = len(report.plan_df)
    true_rev = {ch: float(true_c[ch].iloc[-n:].sum()) for ch in report.channels_}
    rows = []
    for label, spec, *_ in report.levers_:
        sched = report.schedules_[label]
        c = channel_contributions(
            pd.concat([report.history_df, sched]),
            report.true_marginal_returns,
            report.saturation,
            report.adstock,
            report.reference_spend_,
        )
        rev = {ch: float(c[ch].iloc[-n:].sum()) for ch in report.channels_}
        cost = 100 * float(
            np.mean(
                [_safe_improvement(true_rev[ch], rev[ch]) for ch in report.channels_]
            )
        )
        sc = report.results_[label]["scores"]
        ratio = sched.to_numpy() / plan.to_numpy()
        rows.append(
            {
                "report_seed": seed,
                "lever": label,
                "variance_cv": sc["variance"],
                "bias_pct": sc["bias"],
                "identifiability": sc["identifiability"],
                "cost_pct": cost,
                "max_spike": 1.0 if label == "unphased" else float(ratio.max()),
                "fit_seconds": fit_seconds,
                "winner": report.winner_,
            }
        )
    return rows


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    out = sys.argv[2]
    time_baseline = "--time-baseline" in sys.argv
    hist, plan = scenario()
    all_rows = []
    for k, seed in enumerate(seeds):
        report = make_report(hist, plan, seed)
        t0 = time.time()
        report.fit(**FIT_KW)
        dt = time.time() - t0
        print(f"seed {seed}: fit {dt:.0f}s, winner={report.winner_}", flush=True)
        all_rows.extend(rows_for(report, plan, seed, dt))
        if k == 0:
            html = report.to_html()
            assert "redistribute" in html
            print(f"seed {seed}: to_html ok ({len(html)} chars)", flush=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        if time_baseline and k == 0:
            base_levers = [
                lv for lv in _default_levers(CHANNELS) if "redistribute" not in lv[0]
            ]
            assert len(base_levers) == 16
            r16 = make_report(hist, plan, seed, levers=base_levers)
            t0 = time.time()
            r16.fit(**FIT_KW)
            print(
                f"baseline (16 levers) fit {time.time() - t0:.0f}s "
                f"vs 20 levers {dt:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
