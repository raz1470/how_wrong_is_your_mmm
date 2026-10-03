"""Peak month at 15 channels on the published dataset: two checks.

1. Funding cut against demand: does the peak month's position relative to
   demand explain the extra bias?
2. No clash: does bias recover when no two channels peak in the same month?

Per (draw, channel) rows for unphased, Peak month (15 channels peak) and
Peak month on 12 channels only (one per month, no clashes). Run from the
repo root, then peak_summary.py:
    uv run python tools/peak_month_bias/peak_checks.py 200 15
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "overview_data" / "section6"))
from harness import build

from how_wrong_is_your_mmm import Redistribute

warnings.filterwarnings("ignore")
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_CH = int(sys.argv[2]) if len(sys.argv) > 2 else 15
OUT = Path(__file__).parent

r = build(N_CH)
chans = list(r.plan_df.columns)
peak = Redistribute(dark_weeks=0, edge_cap_pct=0.0, high_month_pct=150.0)
levers = {
    "unphased": {c: 0.0 for c in chans},
    "peak_all": {c: peak for c in chans},
    "peak_12": {c: (peak if i < 12 else 0.0) for i, c in enumerate(chans)},
}
months = np.asarray(r._plan_month_labels)
uniq = list(dict.fromkeys(months))
nh = len(r.history_df)

rows = []
for j in range(N_DRAWS):
    demand = r._bias_demand(j)
    d_plan = pd.Series(demand[nh:]).groupby(months, sort=False).mean()
    d_rel = d_plan / d_plan.mean()
    for name, spec in levers.items():
        plan = r._phase(spec, "edge", True, r.seed + j)
        comb = pd.concat([r.history_df, plan])
        err, _ = r._bias_fit(comb, j, 50, j)
        ratio = (plan / r.plan_df).groupby(months, sort=False).mean()
        pk = {
            c: (uniq.index(ratio[c].idxmax()) if ratio[c].max() > 1.5 else -1)
            for c in chans
        }
        counts = pd.Series([m for m in pk.values() if m >= 0]).value_counts()
        for c in chans:
            m = pk[c]
            rows.append(
                dict(
                    lever=name,
                    draw=j,
                    channel=c,
                    err=float(err[c]),
                    peak_month=m,
                    n_sharing=int(counts.get(m, 0)) if m >= 0 else 0,
                    demand_rel=float(d_rel.iloc[m]) if m >= 0 else np.nan,
                    corr_demand=float(np.corrcoef(comb[c].to_numpy(), demand)[0, 1]),
                )
            )
    if j % 10 == 9:
        print(j + 1, flush=True)
pd.DataFrame(rows).to_csv(OUT / f"peak_checks_{N_CH}_{N_DRAWS}.csv", index=False)
