"""One phased tv schedule per strategy on the example scenario, plus its
peak week and whether month and year totals are kept. Feeds the phase
panels in section 4. No fit needed. Writes results/phase_panels.json."""

import json
import warnings
from pathlib import Path

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand, simulate_spend
from how_wrong_is_your_mmm._discovery_report import _peak_week_multiple

warnings.filterwarnings("ignore")
CH = ["tv", "meta", "search_generic", "tiktok"]
MR = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
SAT = {"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70}
ADS = {"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20}
NH, NP = 104, 52
D = simulate_demand(NH + NP, process="trend", seed=6)
h = simulate_spend(
    n_obs=NH,
    correlation=0.7,
    channels=CH,
    seed=12,
    start_date="2025-01-06",
    demand=D[:NH],
)
p = simulate_spend(
    n_obs=NP,
    correlation=0.7,
    channels=CH,
    seed=13,
    start_date="2027-01-04",
    demand=D[NH:],
)
r = DiscoveryReport(
    history_df=h,
    plan_df=p,
    true_marginal_returns=MR,
    saturation=SAT,
    adstock=ADS,
    client_name="Demo client",
    plan_year="2027",
)
out = {
    "plan": {c: [round(float(v)) for v in p[c]] for c in CH},
    "dates": [str(d.date()) for d in p.index],
    "strategies": {},
}
mon = p.index.to_period("M")
for label, spec, shape, bal in r.levers_:
    if label == "unphased":
        continue
    s = r._phase(spec, shape, bal, r.seed)
    mdiff = (s.groupby(mon).sum() - p.groupby(mon).sum()).abs().max().max()
    out["strategies"][label] = {
        "tv": [round(float(v)) for v in s["tv"]],
        "peak": _peak_week_multiple(p, s),
        "month_kept": bool(mdiff < 1),
        "year_diff": float((s.sum() - p.sum()).abs().max()),
        "dark_weeks_tv": int((s["tv"] < 1).sum()),
    }
    print(
        label,
        round(out["strategies"][label]["peak"], 2),
        out["strategies"][label]["month_kept"],
        out["strategies"][label]["year_diff"],
        out["strategies"][label]["dark_weeks_tv"],
    )
json.dump(out, open(Path(__file__).parent / "results" / "phase_panels.json", "w"))
