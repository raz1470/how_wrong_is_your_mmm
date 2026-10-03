"""Loop 24 -- acceptance re-run of the whole default sweep, now including the
MonthStep family, through the REAL DiscoveryReport class.

Every row is phased by DiscoveryReport._phase -> _generate_phased_schedule
(Redistribute and MonthStep per 12-CALENDAR-MONTH block inside the class; the
loop 21/23 scratch used 52-week blocks and a per-chunk calendar, so multi-year
numbers are not draw-for-draw comparable, only distributionally). An extra
"+/-50% (month step)" row is appended to the default sweep: loop 23 flagged it
as untested and expected it near the winner's identifiability at ~2% cost.

Scenario (as loops 20/23): NCH channels, 52wk warm-up history + YEARS*52wk
plan, correlation 0.7, overview (b, lambda) pairs cycled, marginal returns
0.5..1.4, fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8).

Usage: PYTHONPATH=src python loop24_monthstep_acceptance.py SEEDS OUT_CSV
           [--channels N] [--years Y] [--time-baseline]
  --time-baseline: after the first seed, also fit the sweep WITHOUT the
    MonthStep rows (the 20 levers main had) to measure what the family adds.

Extra column edits_per_ch_yr: mean over channels of the number of weeks per
year in which the phased/plan ratio CHANGES (a deviation change a media buyer
must action), so a weekly nudge is ~52 and a month step is <=12.
"""

import sys
import time

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import MonthStep, simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _default_levers,
    _peak_week_multiple,
    _safe_improvement,
)

_OVERVIEW_SATURATION = [0.60, 0.75, 0.90, 0.70]
_OVERVIEW_ADSTOCK = [0.50, 0.30, 0.10, 0.20]
FIT_KW = dict(n_sims=20, n_phasing_seeds=2, id_n_sims=8)  # same as loops 20/23


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def setup(nch, years):
    channels = [f"ch{i}" for i in range(nch)]
    mr = {c: 0.5 + 0.9 * i / max(nch - 1, 1) for i, c in enumerate(channels)}
    sat = {c: _OVERVIEW_SATURATION[i % 4] for i, c in enumerate(channels)}
    ads = {c: _OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(channels)}
    hist = simulate_spend(
        n_obs=52, correlation=0.7, channels=channels, seed=0, start_date="2023-01-02"
    )
    ps = hist.index[-1] + pd.Timedelta(weeks=1)
    plan = simulate_spend(
        n_obs=52 * years, correlation=0.7, channels=channels, seed=1, start_date=ps
    )
    return channels, mr, sat, ads, hist, plan


def make_report(cfg, seed, levers):
    channels, mr, sat, ads, hist, plan = cfg
    return DiscoveryReport(
        hist,
        plan,
        true_marginal_returns=mr,
        saturation=sat,
        adstock=ads,
        levers=levers,
        seed=seed,
    )


def sweep(channels, with_month_step=True, extra50=True):
    levers = _default_levers(channels)
    if not with_month_step:
        levers = [lv for lv in levers if "month step" not in lv[0]]
    if extra50:
        levers.append(
            (
                "+/-50% (month step) [extra]",
                {ch: MonthStep(step_pct=50.0) for ch in channels},
                "uniform",
                False,
            )
        )
    return levers


def edits_per_ch_yr(plan, sched):
    ratio = np.where(plan.to_numpy() > 0, sched.to_numpy() / plan.to_numpy(), 1.0)
    changed = (np.abs(np.diff(ratio, axis=0)) > 1e-6).sum(axis=0)
    return float(changed.mean() / (len(plan) / 52))


def rows_for(report, seed, fit_seconds, nch, years):
    plan = report.plan_df
    combined = pd.concat([report.history_df, plan])
    true_c = channel_contributions(
        combined,
        report.true_marginal_returns,
        report.saturation,
        report.adstock,
        report.reference_spend_,
    )
    n = len(plan)
    true_rev = {ch: float(true_c[ch].iloc[-n:].sum()) for ch in report.channels_}
    rows = []
    for label, *_ in report.levers_:
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
        rows.append(
            {
                "n_channels": nch,
                "years": years,
                "report_seed": seed,
                "lever": label,
                "variance_cv": sc["variance"],
                "bias_pct": sc["bias"],
                "identifiability": sc["identifiability"],
                "cost_pct": cost,
                "max_spike": _peak_week_multiple(plan, sched),
                "edits_per_ch_yr": edits_per_ch_yr(plan, sched),
                "fit_seconds": fit_seconds,
                "winner": report.winner_,
            }
        )
    return rows


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    out = sys.argv[2]
    nch, years = arg("--channels", 10), arg("--years", 3)
    cfg = setup(nch, years)
    channels = cfg[0]
    all_rows = []
    for k, seed in enumerate(seeds):
        report = make_report(cfg, seed, sweep(channels))
        t0 = time.time()
        report.fit(**FIT_KW)
        dt = time.time() - t0
        print(
            f"n={nch} y={years} seed {seed}: {len(report.levers_)} levers, "
            f"fit {dt:.0f}s, winner={report.winner_}",
            flush=True,
        )
        all_rows.extend(rows_for(report, seed, dt, nch, years))
        pd.DataFrame(all_rows).to_csv(out, index=False)
        if k == 0:
            html = report.to_html()
            assert "month step" in html
            print("to_html ok", len(html), flush=True)
            if "--time-baseline" in sys.argv:
                base = make_report(cfg, seed, sweep(channels, False, False))
                t0 = time.time()
                base.fit(**FIT_KW)
                print(
                    f"baseline ({len(base.levers_)} levers, no MonthStep, no extra) "
                    f"fit {time.time() - t0:.0f}s vs {dt:.0f}s for "
                    f"{len(report.levers_)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
