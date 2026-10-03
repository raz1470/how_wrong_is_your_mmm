"""Aggregate loop24 raw CSVs (dedupe by config+seed) into per-config tables.

Usage: python loop24_summarise.py [--md]
"""

import glob
import sys

import pandas as pd

files = sorted(glob.glob("loop_data/loop24_*_raw.csv"))
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(
    ["n_channels", "years", "report_seed", "lever"]
)
for (nch, yrs), d in df.groupby(["n_channels", "years"]):
    print(
        f"\n=== {nch} channels, {yrs} year(s): seeds {sorted(d.report_seed.unique())}"
    )
    g = d.groupby("lever")
    t = pd.DataFrame(
        {
            "variance": g.variance_cv.mean(),
            "bias %": g.bias_pct.mean(),
            "bias sd": g.bias_pct.std(),
            "identifiability": g.identifiability.mean(),
            "cost %": g.cost_pct.mean(),
            "spike mean": g.max_spike.mean(),
            "spike max": g.max_spike.max(),
            "edits/ch/yr": g.edits_per_ch_yr.mean(),
        }
    ).sort_values("bias %")
    print(t.round(3).to_markdown() if "--md" in sys.argv else t.round(3).to_string())
    print(
        "\nsweep winner by seed:\n",
        d.drop_duplicates("report_seed").winner.value_counts().to_string(),
    )
    print(
        "fit seconds (mean):",
        round(d.drop_duplicates("report_seed").fit_seconds.mean()),
    )
