"""Full section 6 fit (variance, identifiability, cost) for one channel
count and one dataset, at the report's default settings:
    uv run python tools/overview_data/section6/fullfit.py <n_channels> <base|s1|s2|s3>
Writes results/full_<n>_<tag>.json. Skips a run whose file exists. About
23 minutes per 15-channel fit. Bias here is at 15 draws and is not used:
summarise.py takes bias from harness.py's 100-draw runs."""

import json
import sys
import time
import warnings
from pathlib import Path

import harness as H

warnings.filterwarnings("ignore")
SEEDS = {
    "base": {},
    "s1": dict(demand_seed=106, hist_seed=112, plan_seed=113),
    "s2": dict(demand_seed=206, hist_seed=212, plan_seed=213),
    "s3": dict(demand_seed=306, hist_seed=312, plan_seed=313),
}
n = int(sys.argv[1])
tag = sys.argv[2]
out = Path(__file__).parent / "results" / f"full_{n}_{tag}.json"
if out.exists():
    sys.exit(0)
t0 = time.time()
r = H.build(n, **SEEDS[tag]).fit(horizon_years=1, n_bias_draws=15)
un = sum(r.results_["unphased"]["revenue_mean"].values())
rows = {}
for lab, res in r.results_.items():
    s = dict(res["scores"])
    s["cost_pct"] = 100 * (un - sum(res["revenue_mean"].values())) / un
    rows[lab] = s
json.dump(
    {
        "n": n,
        "tag": tag,
        "winner": r.winner_,
        "seconds": time.time() - t0,
        "levers": rows,
    },
    open(out, "w"),
    indent=1,
    default=float,
)
print(n, tag, r.winner_, f"{time.time() - t0:.0f}s", flush=True)
