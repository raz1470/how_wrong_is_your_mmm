"""Aggregate loop23 raw CSVs (dedupe by seed) into the loop-20-style table."""

import glob
import sys

import pandas as pd

files = sorted(glob.glob("loop_data/loop23_*_raw.csv"))
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(
    ["report_seed", "lever"]
)
n = df.report_seed.nunique()
print(f"{n} seeds: {sorted(df.report_seed.unique())}")
g = df.groupby("lever")
t = pd.DataFrame(
    {
        "variance": g.variance_cv.mean(),
        "bias %": g.bias_pct.mean(),
        "bias sd": g.bias_pct.std(),
        "identifiability": g.identifiability.mean(),
        "cost %": g.cost_pct.mean(),
        "spike mean": g.max_spike.mean(),
        "spike max": g.max_spike.max(),
    }
).sort_values("bias %")
print(t.round(3).to_markdown() if "--md" in sys.argv else t.round(3).to_string())
w = df.drop_duplicates("report_seed").winner.value_counts()
print("\nsweep winner by seed:\n", w.to_string())
