"""Seed-robustness check on the t=2/3/4 scattered-bump sweep (a flagged
open item): is t=3's edge over t=2/t=4 a real
effect or one lucky random draw of blackout months / bump weeks?

Same plan/history scenario as validate_via_discovery_report.py, same
real DiscoveryReport pipeline, but now run across many report seeds
(which drive which weeks get blacked out and bumped) and only the
Unphased baseline + the three t-variants, to keep runtime down.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

import numpy as np
import pandas as pd
from design import cross_month_blackout_scatter_schedule

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import DiscoveryReport, _safe_improvement
from how_wrong_is_your_mmm._phaser import _get_month_labels

CHANNELS = [f"ch{i}" for i in range(10)]
TRUE_MARGINAL_RETURNS = {c: 0.5 + 0.1 * i for i, c in enumerate(CHANNELS)}
SATURATION = {c: 0.7 for c in CHANNELS}


class ScatterOnlyReport(DiscoveryReport):
    def _phase(self, spec, nudge_shape, balance_signs, seed):
        marker = spec.get(CHANNELS[0]) if isinstance(spec, dict) else None
        if isinstance(marker, str) and marker.startswith("scatter_t"):
            t = int(marker.removeprefix("scatter_t"))
            return cross_month_blackout_scatter_schedule(
                self.plan_df, self._plan_month_labels, seed, dark_weeks=t
            )
        return super()._phase(spec, nudge_shape, balance_signs, seed)


def one_run(history_df, plan_df, month_lengths, report_seed):
    levers = [
        ("Unphased", {}, "uniform", False),
        *[
            (f"t={t}", {c: f"scatter_t{t}" for c in CHANNELS}, "uniform", False)
            for t in (2, 3, 4)
        ],
    ]
    report = ScatterOnlyReport(
        history_df,
        plan_df,
        true_marginal_returns=TRUE_MARGINAL_RETURNS,
        saturation=SATURATION,
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
        rows.append(
            {
                "seed": report_seed,
                "lever": label,
                "variance_cv": scores["variance"],
                "bias_pct": scores["bias"],
                "identifiability": scores["identifiability"],
                "cost_pct": cost,
            }
        )
    return rows


def main():
    history_df = simulate_spend(
        n_obs=104, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=1, start_date=plan_start
    )
    month_labels = _get_month_labels(plan_df)
    _, counts = np.unique(month_labels, return_counts=True)
    month_lengths = counts.tolist()

    all_rows = []
    n_seeds = 12
    for s in range(n_seeds):
        print(f"seed {s}/{n_seeds}...", flush=True)
        all_rows.extend(one_run(history_df, plan_df, month_lengths, report_seed=s))

    df = pd.DataFrame(all_rows)
    df.to_csv("proto/seed_robustness_raw.csv", index=False)

    summary = df.groupby("lever")[
        ["variance_cv", "bias_pct", "identifiability", "cost_pct"]
    ].agg(["mean", "std", "min", "max"])
    print(summary.to_string())

    # win-count: for each seed, which t has the best (lowest) value per metric
    t_rows = df[df["lever"] != "Unphased"].copy()
    for metric in ["variance_cv", "bias_pct", "identifiability"]:
        wins = t_rows.loc[
            t_rows.groupby("seed")[metric].idxmin(), "lever"
        ].value_counts()
        print(f"\n{metric} win count out of {n_seeds} seeds:\n{wins.to_string()}")


if __name__ == "__main__":
    main()
