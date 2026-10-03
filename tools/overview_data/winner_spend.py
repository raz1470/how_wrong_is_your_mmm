"""Winner's phased schedule, all channels, example scenario (seed = report
seed, i.e. the report's representative schedule). Feeds the chart that
opens section 5. No fit needed. Writes results/winner_spend.json."""

import json
from pathlib import Path

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand, simulate_spend

CH = ["tv", "meta", "search_generic", "tiktok"]
D = simulate_demand(156, process="trend", seed=6)
h = simulate_spend(
    n_obs=104,
    correlation=0.7,
    channels=CH,
    seed=12,
    start_date="2025-01-06",
    demand=D[:104],
)
p = simulate_spend(
    n_obs=52,
    correlation=0.7,
    channels=CH,
    seed=13,
    start_date="2027-01-04",
    demand=D[104:],
)
r = DiscoveryReport(
    history_df=h,
    plan_df=p,
    true_marginal_returns={
        "tv": 0.5,
        "meta": 1.0,
        "search_generic": 1.5,
        "tiktok": 1.2,
    },
    saturation={"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70},
    adstock={"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20},
    client_name="Demo client",
    plan_year="2027",
)
lab, spec, shape, bal = [l for l in r.levers_ if l[0] == "Combined"][0]
s = r._phase(spec, shape, bal, r.seed)
mon = p.index.to_period("M")
ratio = s.groupby(mon).sum() / p.groupby(mon).sum()
for c in CH:
    print(
        c,
        "dark",
        str(ratio[c].idxmin()),
        "peak months",
        [str(m) for m in ratio.index[ratio[c] > 1.3]],
        round(s[c].max() / 1e3),
        round(float((s[c] / p[c]).max()), 2),
    )
json.dump(
    {
        "dates": [str(d.date()) for d in p.index],
        "plan": {c: [round(float(v)) for v in p[c]] for c in CH},
        "phased": {c: [round(float(v)) for v in s[c]] for c in CH},
    },
    open(Path(__file__).parent / "results" / "winner_spend.json", "w"),
)
