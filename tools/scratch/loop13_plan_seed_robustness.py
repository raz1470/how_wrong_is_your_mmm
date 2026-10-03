"""Loop 13: does the winning design (`roundrobin_plus_edge_layer`,
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

PLAN_SEEDS = [1, 2, 3, 4, 5]  # 1 = the canonical plan used in every prior loop
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
    history_df = simulate_spend(
        n_obs=104, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)

    all_rows = []
    for plan_seed in PLAN_SEEDS:
        plan_df = simulate_spend(
            n_obs=52,
            correlation=0.7,
            channels=CHANNELS,
            seed=plan_seed,
            start_date=plan_start,
        )
        month_labels = _get_month_labels(plan_df)
        _, counts = np.unique(month_labels, return_counts=True)
        print(f"plan_seed={plan_seed}: month lengths = {counts.tolist()}", flush=True)
        for report_seed in REPORT_SEEDS:
            rows = run_one(history_df, plan_df, month_labels, plan_seed, report_seed)
            all_rows.extend(rows)
            cand = [r for r in rows if r["lever"] == CANDIDATE_LABEL][0]
            print(
                f"  report_seed={report_seed}: variance={cand['variance_cv']:.4f} "
                f"bias={cand['bias_pct']:.2f} id={cand['identifiability']:.2f} "
                f"cost={cand['cost_pct']:.3f} spike={cand['max_spike']:.2f}",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    df.to_csv("proto/loop13_plan_seed_robustness_raw.csv", index=False)

    print("\n=== candidate mean by plan_seed ===")
    cand_df = df[df["lever"] == CANDIDATE_LABEL]
    print(
        cand_df.groupby("plan_seed")[
            [
                "n_months",
                "variance_cv",
                "bias_pct",
                "identifiability",
                "cost_pct",
                "max_spike",
            ]
        ]
        .mean()
        .to_string()
    )

    print("\n=== candidate overall mean/std across all plan draws ===")
    print(
        cand_df[["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]]
        .agg(["mean", "std", "min", "max"])
        .to_string()
    )

    print("\n=== Unphased mean by plan_seed (for context) ===")
    unph_df = df[df["lever"] == "Unphased"]
    print(
        unph_df.groupby("plan_seed")[["variance_cv", "bias_pct", "identifiability"]]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
