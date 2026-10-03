"""Loop 11: tune edge_cap_pct in the loop-10 winner
(roundrobin_plus_edge_layer, dark_weeks=4). Only 10% has been tested so
far -- sweep to see whether more/less of the edge layer helps further or
starts hurting cost/deployability.

Design principle motivating this: the edge layer's job (per the tracking
doc) is to add cross-channel orthogonal fine variation on top of the
structural round-robin redistribution, mainly to help variance/bias
further without disturbing the structural bias win. If a little (10%)
helps, more should help monotonically up to some point where cost and
deployability (max weekly spike) start to dominate -- that crossover is
what this sweep is looking for.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

from functools import partial

import pandas as pd
from design import roundrobin_plus_edge_layer
from loop_harness import evaluate_candidate, summarize

CAPS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def max_spike_ratio(plan_df, month_labels, seed, edge_cap_pct):
    sched = roundrobin_plus_edge_layer(
        plan_df, month_labels, seed, dark_weeks=4, edge_cap_pct=edge_cap_pct
    )
    ratio = (sched.to_numpy() / plan_df.to_numpy()).max()
    return ratio


def main():
    from loop_harness import canonical_scenario

    history_df, plan_df, month_labels = canonical_scenario()

    all_rows = []
    for cap in CAPS:
        label = f"edge_cap_{cap:g}"
        fn = partial(roundrobin_plus_edge_layer, dark_weeks=4, edge_cap_pct=cap)
        df = evaluate_candidate(fn, label, n_seeds=6)
        df["edge_cap_pct"] = cap
        # deployability proxy: max single-week spike ratio vs plan, per seed
        df["max_spike"] = [
            max_spike_ratio(plan_df, month_labels, s, cap) for s in df["seed"]
        ]
        all_rows.append(df)
        summarize(df, label)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv("proto/loop11_edge_cap_sweep_raw.csv", index=False)

    summary = full.groupby("edge_cap_pct")[
        ["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]
    ].mean()
    print("\n=== mean by edge_cap_pct ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
