"""What each strategy costs on the example scenario, and how far that cost
moves with the saturation curve.

Cost is the plan-year revenue the channels drive under the unphased plan
minus the same under the phased plan, at the supplied response curves. It
is deterministic: no fit and no noise.

- By strategy: the mean of each channel's own % (the table's figure), the
  % of all media-driven revenue, GBP, and the % of total plan-year sales.
- Sensitivity: the same schedules, with every channel's saturation
  exponent set to each value across the range section 2 says cannot be
  told apart.

Run from the repo root:  uv run python tools/overview_data/cost.py
Writes results/cost.json.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand, simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions

warnings.filterwarnings("ignore")
CH = ["tv", "meta", "search_generic", "tiktok"]
MR = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
SAT = {"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70}
ADS = {"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20}
EXPONENTS = [0.2, 0.4, 0.6, 0.8, 1.0]

D = simulate_demand(156, process="trend", seed=6)
h = simulate_spend(
    n_obs=104,
    correlation=0.7,
    channels=CH,
    seed=37,
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
    true_marginal_returns=MR,
    saturation=SAT,
    adstock=ADS,
    client_name="Demo client",
    plan_year="2027",
)
schedules = {
    label: r._phase(spec, shape, bal, r.seed) for label, spec, shape, bal in r.levers_
}


def plan_year_revenue(plan, saturation):
    contributions = channel_contributions(
        pd.concat([h, plan]), MR, saturation, ADS, r.reference_spend_
    )
    return contributions.iloc[-len(p) :].sum()


total_sales = r.calibration_.total_sales * len(p)
base = plan_year_revenue(p, SAT)
out = {
    "plan_year_sales_m": total_sales / 1e6,
    "media_revenue_m": float(base.sum()) / 1e6,
    "strategies": {},
    "sensitivity": {},
}
for label, schedule in schedules.items():
    if label == "unphased":
        continue
    revenue = plan_year_revenue(schedule, SAT)
    lost = float(base.sum() - revenue.sum())
    out["strategies"][label] = {
        "mean_of_channel_pct": float((100 * (base - revenue) / base).mean()),
        "pct_of_media_revenue": 100 * lost / float(base.sum()),
        "gbp_m": lost / 1e6,
        "pct_of_total_sales": 100 * lost / total_sales,
    }
    out["sensitivity"][label] = {}
    for b in EXPONENTS:
        flat = {c: b for c in CH}
        base_b = plan_year_revenue(p, flat)
        cost_b = (100 * (base_b - plan_year_revenue(schedule, flat)) / base_b).mean()
        out["sensitivity"][label][str(b)] = float(cost_b)
OUT = Path(__file__).parent / "results" / "cost.json"
OUT.write_text(json.dumps(out, indent=1))
print(
    json.dumps(
        {
            k: {a: round(b, 2) for a, b in v.items()}
            for k, v in out["strategies"].items()
        },
        indent=1,
    )
)
print(
    {
        k: {a: round(b, 2) for a, b in v.items()}
        for k, v in out["sensitivity"].items()
        if k in ("Combined", "Dark month")
    }
)
