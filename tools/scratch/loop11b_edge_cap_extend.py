"""Loop 11b: extend the edge_cap_pct sweep beyond 25% (loop 11 found no
crossover/plateau in 0-25% -- every point of extra edge_cap kept helping
all three rigor metrics at a small, steady cost. Push further to find
where it actually plateaus or the deployability cost stops being worth
it."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

from functools import partial

import pandas as pd
from design import roundrobin_plus_edge_layer
from loop_harness import canonical_scenario, evaluate_candidate, summarize

CAPS = [30.0, 40.0, 50.0, 70.0, 100.0]


def max_spike_ratio(plan_df, month_labels, seed, edge_cap_pct):
    sched = roundrobin_plus_edge_layer(
        plan_df, month_labels, seed, dark_weeks=4, edge_cap_pct=edge_cap_pct
    )
    return (sched.to_numpy() / plan_df.to_numpy()).max()


def main():
    history_df, plan_df, month_labels = canonical_scenario()

    all_rows = []
    for cap in CAPS:
        label = f"edge_cap_{cap:g}"
        fn = partial(roundrobin_plus_edge_layer, dark_weeks=4, edge_cap_pct=cap)
        df = evaluate_candidate(fn, label, n_seeds=6)
        df["edge_cap_pct"] = cap
        df["max_spike"] = [
            max_spike_ratio(plan_df, month_labels, s, cap) for s in df["seed"]
        ]
        all_rows.append(df)
        summarize(df, label)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv("proto/loop11b_edge_cap_extend_raw.csv", index=False)

    summary = full.groupby("edge_cap_pct")[
        ["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]
    ].mean()
    print("\n=== mean by edge_cap_pct (extended) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
