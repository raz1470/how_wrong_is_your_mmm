"""Loop 14 (adapted from loop 13): does the winning design (`roundrobin_plus_edge_layer`,
dark_weeks=4, edge_cap_pct=15.0) hold up on a DIFFERENT plan draw, not
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

import numpy as np
import pandas as pd
from design import roundrobin_plus_edge_layer
from loop_harness import (
    ADSTOCK,
    CHANNELS,
    SATURATION,
    TRUE_MARGINAL_RETURNS,
    make_report_class,
)

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import _safe_improvement
from how_wrong_is_your_mmm._phaser import _get_month_labels

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
REPORT_SEEDS = [0, 1, 2]
CANDIDATE_LABEL = "roundrobin_plus_edge_layer"


def run_one(history_df, plan_df, month_labels, plan_seed, report_seed):
    def phase_fn(plan_df, month_labels, seed):
        return roundrobin_plus_edge_layer(
            plan_df, month_labels, seed, dark_weeks=4, edge_cap_pct=15.0
        )

    ReportClass = make_report_class(phase_fn)
    levers = [
        ("Unphased", {}, "uniform", False),
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
            if label != "Unphased"
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
    all_rows = []
    for hs in HISTORY_STARTS:
        history_df = simulate_spend(
            n_obs=104, correlation=0.7, channels=CHANNELS, seed=0, start_date=hs
        )
        plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)
        for L in PLAN_LENGTHS:
            plan_df = simulate_spend(
                n_obs=L,
                correlation=0.7,
                channels=CHANNELS,
                seed=PLAN_SEED,
                start_date=plan_start,
            )
            month_labels = _get_month_labels(plan_df)
            _, counts = np.unique(month_labels, return_counts=True)
            print(
                f"hist_start={hs} plan_start={plan_start.date()} L={L}: months={counts.tolist()}",
                flush=True,
            )
            for report_seed in REPORT_SEEDS:
                try:
                    rows = run_one(history_df, plan_df, month_labels, hs, report_seed)
                except Exception as e:
                    print(
                        f"  report_seed={report_seed}: FAILED {type(e).__name__}: {e}",
                        flush=True,
                    )
                    all_rows.append(
                        {
                            "plan_seed": hs,
                            "report_seed": report_seed,
                            "lever": CANDIDATE_LABEL,
                            "n_months": len(counts),
                            "error": str(e)[:200],
                            "plan_len": L,
                            "month_lengths": str(counts.tolist()),
                        }
                    )
                    continue
                for r in rows:
                    r["plan_len"] = L
                    r["month_lengths"] = str(counts.tolist())
                all_rows.extend(rows)
                cand = [r for r in rows if r["lever"] == CANDIDATE_LABEL][0]
                print(
                    f"  report_seed={report_seed}: variance={cand['variance_cv']:.4f} bias={cand['bias_pct']:.2f} id={cand['identifiability']:.2f} cost={cand['cost_pct']:.3f} spike={cand['max_spike']:.2f}",
                    flush=True,
                )

    df = pd.DataFrame(all_rows)
    df.to_csv("loop_data/loop14_calendar_robustness_raw.csv", index=False)
    if "error" in df:
        print("\nFAILURES:", int(df["error"].notna().sum()))
    ok = df[df["variance_cv"].notna()] if "variance_cv" in df else df
    cand = ok[ok["lever"] == CANDIDATE_LABEL]
    print(
        cand.groupby(["plan_seed", "plan_len"])[
            ["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]
        ]
        .mean()
        .to_string()
    )
    print(
        cand[["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]]
        .agg(["mean", "std", "min", "max"])
        .to_string()
    )
    print(
        ok[ok["lever"] == "Unphased"][["variance_cv", "bias_pct", "identifiability"]]
        .agg(["mean", "min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
