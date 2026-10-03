"""Loop 12: is loop 11's edge_cap_pct effect really about MAGNITUDE, or
about FREQUENCY (how many weeks per month get touched), or both?

Holds a deviation "budget" H = frequency * cap_pct**2 roughly constant
(a variance-like quantity) while trading frequency against magnitude.
If bias/variance/identifiability move together with H regardless of the
split, frequency isn't doing anything independent of total magnitude
injected. If scores differ at matched H, frequency (breadth of
distinct touched weeks) matters on its own -- e.g. because more weeks
touched means more distinct spend *levels* for saturation
identifiability, or a wider spread of exogenous variance for bias.

Also runs frequency=1.0 at each condition's cap as a sanity check
against `roundrobin_plus_edge_layer` from loop 10/11 (should match
closely, since frequency=1.0 is the same mechanism as "edge" shape).
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

from functools import partial

import pandas as pd
from design import roundrobin_plus_freq_ablation
from loop_harness import canonical_scenario, evaluate_candidate, summarize

# matched-budget conditions: H = frequency * cap_pct**2 ~= 225 throughout
CONDITIONS = [
    ("freq1.00_cap15", 1.00, 15.0),
    ("freq0.75_cap17", 0.75, 17.3),
    ("freq0.50_cap21", 0.50, 21.2),
    ("freq0.25_cap30", 0.25, 30.0),
    ("freq0.125_cap42", 0.125, 42.4),
]


def max_spike_ratio(plan_df, month_labels, seed, cap_pct, frequency):
    sched = roundrobin_plus_freq_ablation(
        plan_df, month_labels, seed, dark_weeks=4, cap_pct=cap_pct, frequency=frequency
    )
    return (sched.to_numpy() / plan_df.to_numpy()).max()


def main():
    history_df, plan_df, month_labels = canonical_scenario()

    all_rows = []
    for label, freq, cap in CONDITIONS:
        fn = partial(
            roundrobin_plus_freq_ablation, dark_weeks=4, cap_pct=cap, frequency=freq
        )
        df = evaluate_candidate(fn, label, n_seeds=8)
        df["frequency"] = freq
        df["cap_pct"] = cap
        df["H"] = freq * cap**2
        df["max_spike"] = [
            max_spike_ratio(plan_df, month_labels, s, cap, freq) for s in df["seed"]
        ]
        all_rows.append(df)
        summarize(df, label)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv("proto/loop12_frequency_ablation_raw.csv", index=False)

    summary = full.groupby(["frequency", "cap_pct", "H"])[
        ["variance_cv", "bias_pct", "identifiability", "cost_pct", "max_spike"]
    ].mean()
    print("\n=== mean by (frequency, cap_pct) at matched H ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
