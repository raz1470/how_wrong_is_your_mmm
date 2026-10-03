"""Loop 19 final robustness (mrw default=4; channel count + multi-year, mrw=4). Based on loop 16: impact after 1/2/3 years of phasing every year (chunked per 52wk). Adapted from loop 14/13: does the winning design (`roundrobin_plus_edge_layer`,
dark_weeks=4, edge_cap_pct=15.0, min_recipient_weeks=4) hold up on a DIFFERENT plan draw, not
just a different report/phasing seed?

Everything through loop 12 used the same single canonical plan_df
(simulate_spend seed=1) -- only the report seed (which drives which
weeks get blacked out/bumped, and the simulated fitting noise) varied.
A different plan draw changes month lengths (how many 4- vs 5-week
months, whether there's a short partial first month) and the plan's own
spend levels/correlation realisation, both of which the design's
mechanics (round-robin month assignment, proportional redistribution,
edge-layer rescale) depend on. This checks the design isn't secretly
overfit to this one plan's calendar.

History is held fixed (seed=0, same as always) since it's just the
pre-period used to fit models, not what's being phased.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

import sys as _s

import loop_harness as H
import numpy as np
import pandas as pd
from design import roundrobin_plus_edge_layer
from loop_harness import make_report_class

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    _default_levers,
    _safe_improvement,
)
from how_wrong_is_your_mmm._phaser import _get_month_labels

NCH = 10
SEEDS = [int(x) for x in _s.argv[1].split(",")]
CHANNELS = H.CHANNELS
TRUE_MARGINAL_RETURNS = H.TRUE_MARGINAL_RETURNS
SATURATION = H.SATURATION
ADSTOCK = H.ADSTOCK


def set_channels(n):
    global CHANNELS, TRUE_MARGINAL_RETURNS, SATURATION, ADSTOCK
    CHANNELS = [f"ch{i}" for i in range(n)]
    H.CHANNELS = CHANNELS
    TRUE_MARGINAL_RETURNS = {
        c: 0.5 + 0.9 * i / max(n - 1, 1) for i, c in enumerate(CHANNELS)
    }  # same 0.5-1.4 range
    SATURATION = {c: H._OVERVIEW_SATURATION[i % 4] for i, c in enumerate(CHANNELS)}
    ADSTOCK = {c: H._OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(CHANNELS)}


# Loop 14: vary CALENDAR structure. History start dates chosen so the plan
# start lands on different weekdays-of-month / month-boundary phases, and
# plan lengths vary (52 canonical, plus 48 and 56).
HISTORY_STARTS = [
    "2023-01-02",
    "2023-01-09",
    "2023-01-16",
    "2023-01-23",
    "2023-01-30",
    "2023-02-06",
    "2023-03-06",
    "2023-05-01",
]
PLAN_LENGTHS = [52, 48, 56]
PLAN_SEED = 1
REPORT_SEEDS = SEEDS
CANDIDATE_LABEL = "roundrobin_plus_edge_layer"


def run_one(history_df, plan_df, month_labels, plan_seed, report_seed):
    def phase_fn(plan_df, month_labels, seed):
        # phase each 52-week year independently (fresh round-robin per year)
        parts = []
        for k, a in enumerate(range(0, len(plan_df), 52)):
            chunk = plan_df.iloc[a : a + 52]
            ml = _get_month_labels(chunk)
            parts.append(
                roundrobin_plus_edge_layer(
                    chunk,
                    ml,
                    seed + 1000 * k,
                    dark_weeks=4,
                    edge_cap_pct=15.0,
                    min_recipient_weeks=4,
                )
            )
        return pd.concat(parts)

    ReportClass = make_report_class(phase_fn)
    levers = [
        *_default_levers(CHANNELS),
        (CANDIDATE_LABEL, {c: "candidate" for c in CHANNELS}, "uniform", False),
    ]

    report = ReportClass(
        history_df,
        plan_df,
        true_marginal_returns=TRUE_MARGINAL_RETURNS,
        saturation=SATURATION,
        adstock=ADSTOCK,
        levers=levers,
        seed=report_seed,
    )
    report.fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8)

    combined_baseline = pd.concat([report.history_df, report.plan_df])
    true_contributions = channel_contributions(
        combined_baseline,
        report.true_marginal_returns,
        report.saturation,
        report.adstock,
        report.reference_spend_,
    )
    true_revenue = {
        ch: float(true_contributions[ch].iloc[-len(report.plan_df) :].sum())
        for ch in report.channels_
    }

    rows = []
    for label, *_ in levers:
        scores = report.results_[label]["scores"]
        lever_contributions = channel_contributions(
            pd.concat([report.history_df, report.schedules_[label]]),
            report.true_marginal_returns,
            report.saturation,
            report.adstock,
            report.reference_spend_,
        )
        lever_revenue = {
            ch: float(lever_contributions[ch].iloc[-len(report.plan_df) :].sum())
            for ch in report.channels_
        }
        cost = 100 * float(
            np.mean(
                [
                    _safe_improvement(true_revenue[ch], lever_revenue[ch])
                    for ch in report.channels_
                ]
            )
        )
        max_spike = (
            float((report.schedules_[label].to_numpy() / plan_df.to_numpy()).max())
            if not label.startswith("unphased")
            else 1.0
        )
        rows.append(
            {
                "plan_seed": plan_seed,
                "report_seed": report_seed,
                "lever": label,
                "n_months": len(np.unique(month_labels)),
                "variance_cv": scores["variance"],
                "bias_pct": scores["bias"],
                "identifiability": scores["identifiability"],
                "cost_pct": cost,
                "max_spike": max_spike,
            }
        )
    return rows


def main():
    set_channels(NCH)
    rows_all = []
    hist = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    ps = hist.index[-1] + pd.Timedelta(weeks=1)
    full = simulate_spend(
        n_obs=156, correlation=0.7, channels=CHANNELS, seed=1, start_date=ps
    )
    for years in [3]:
        plan_df = full.iloc[: 52 * years]
        ml = _get_month_labels(plan_df)
        for rs in REPORT_SEEDS:
            rows = run_one(hist, plan_df, ml, "y%d" % years, rs)
            for r in rows:
                r["years"] = years
                r["n_channels"] = NCH
            rows_all.extend(rows)
            c = [r for r in rows if r["lever"] == CANDIDATE_LABEL][0]
            print(
                f"all n={NCH} years={years} seed={rs}: variance={c['variance_cv']:.4f} bias={c['bias_pct']:.2f} id={c['identifiability']:.2f} cost={c['cost_pct']:.3f} spike={c['max_spike']:.2f}",
                flush=True,
            )
    df = pd.DataFrame(rows_all)
    df.to_csv(
        f"loop_data/loop20_allstrat_seeds{_s.argv[1].replace(',', '_')}_raw.csv",
        index=False,
    )
    print(
        df.groupby(["years", "lever"])[
            ["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]
        ]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
