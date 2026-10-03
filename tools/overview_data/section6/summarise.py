"""Section 6 summary over four data seeds. Variance, saturation and adstock
come from fullfit.py (15 phasing draws). Bias comes from harness.py at
100 draws. Reads and writes results/. Run after both:
    uv run python tools/overview_data/section6/summarise.py
Improvement = 100 * (mean unphased - mean strategy) / mean unphased, with
the means taken over the four data seeds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).parent / "results"
NS = [5, 10, 15]
TAGS = ["base", "s1", "s2", "s3"]
KEY = {
    "Weekly nudge": "nudge",
    "Dark month": "darkmonth",
    "Peak month": "peakmonth",
    "Combined": "combined",
    "Month step": "monthstep",
    "Dark week": "darkweek",
}
M = ["variance", "bias", "saturation_range", "adstock_range"]


def full(n, t):
    return json.load(open(R / f"full_{n}_{t}.json"))["levers"]


def bias(n, t):
    df = pd.read_csv(R / f"bias_{n}_{t}_100.csv")
    return df.groupby(["lever", "channel"]).err.mean().abs().groupby("lever").mean()


raw = {}  # raw[n][tag][lever][metric]
for n in NS:
    raw[n] = {}
    for t in TAGS:
        L = full(n, t)
        b = bias(n, t)
        raw[n][t] = {
            lab: {
                "variance": L[lab]["variance"],
                "bias": float(b[lab]),
                "saturation_range": L[lab]["saturation_range"],
                "adstock_range": L[lab]["adstock_range"],
                "cost_pct": L[lab]["cost_pct"],
                "bias15": L[lab]["bias"],
            }
            for lab in L
        }
out = {k: {m: [] for m in M} for k in KEY.values()}
per_seed = {}
for lab, k in KEY.items():
    for n in NS:
        for m in M:
            u = np.mean([raw[n][t]["unphased"][m] for t in TAGS])
            r = np.mean([raw[n][t][lab][m] for t in TAGS])
            out[k][m].append(round(100 * (u - r) / u, 1))
            per_seed[(lab, m, n)] = [
                round(
                    100
                    * (raw[n][t]["unphased"][m] - raw[n][t][lab][m])
                    / raw[n][t]["unphased"][m],
                    1,
                )
                for t in TAGS
            ]
json.dump(out, open(R / "scale_summary_4seed.json", "w"))
json.dump({str(n): raw[n] for n in NS}, open(R / "scale_raw_4seed.json", "w"), indent=1)
for m in M:
    print(
        f"\n{m}: improvement % at 5/10/15 (4-seed mean) | per seed [base,s1,s2,s3] at 15"
    )
    for lab, k in KEY.items():
        print(f"  {lab:13s} {out[k][m]}   {per_seed[(lab, m, 15)]}")
print(
    "\nunphased levels (mean over seeds):",
    {
        m: [
            round(float(np.mean([raw[n][t]["unphased"][m] for t in TAGS])), 3)
            for n in NS
        ]
        for m in M
    },
)
print(
    "cost % (mean):",
    {
        lab: [
            round(float(np.mean([raw[n][t][lab]["cost_pct"] for t in TAGS])), 2)
            for n in NS
        ]
        for lab in KEY
    },
)
