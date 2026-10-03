"""Every example-scenario number docs/overview.html uses, as JSON.

Scenario = tools/generate_example_report.py (104w trend history + 52w plan,
report defaults: demand_spend_corr 0.65, trend). Main fit uses default
fit() settings. Run from repo root:
    uv run python tools/overview_data/example_scenario.py
Writes results/example_scenario.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import (
    DiscoveryReport,
    simulate_demand,
    simulate_spend,
)
from how_wrong_is_your_mmm._dgp import apply_adstock, channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    _peak_week_multiple,
    _safe_improvement,
)

CH = ["tv", "meta", "search_generic", "tiktok"]
MR = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
SAT = {"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70}
ADS = {"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20}
N_HISTORY, N_PLAN = 104, 52
DEMAND = simulate_demand(N_HISTORY + N_PLAN, process="trend", seed=6)


def scenario(correlation=0.7):
    h = simulate_spend(
        n_obs=N_HISTORY,
        correlation=correlation,
        channels=CH,
        seed=37,
        start_date="2025-01-06",
        demand=DEMAND[:N_HISTORY],
    )
    p = simulate_spend(
        n_obs=N_PLAN,
        correlation=correlation,
        channels=CH,
        seed=13,
        start_date="2027-01-04",
        demand=DEMAND[N_HISTORY:],
    )
    return h, p


def make_report(h, p, **kw):
    return DiscoveryReport(
        history_df=h,
        plan_df=p,
        true_marginal_returns=MR,
        saturation=SAT,
        adstock=ADS,
        client_name="Demo client",
        plan_year="2027",
        **kw,
    )


def mean_pairwise(m):
    df = pd.DataFrame(m).loc[CH, CH].to_numpy()
    return float(df[np.triu_indices(len(CH), 1)].mean())


def true_revenue(r):
    tc = channel_contributions(
        pd.concat([r.history_df, r.plan_df]), MR, SAT, ADS, r.reference_spend_
    )
    return {c: float(tc[c].iloc[-len(r.plan_df) :].sum()) for c in CH}


def cost_pct(r, label, truth):
    lc = channel_contributions(
        pd.concat([r.history_df, r.schedules_[label]]), MR, SAT, ADS, r.reference_spend_
    )
    lev = {c: float(lc[c].iloc[-len(r.plan_df) :].sum()) for c in CH}
    return 100 * float(np.mean([_safe_improvement(truth[c], lev[c]) for c in CH]))


def forest_rows(r, res, truth):
    """Same fields _render_html builds, in £m for revenue."""
    out = {}
    for c in CH:
        out[c] = {
            "var": [
                res["revenue_p10"][c] / 1e6,
                res["revenue_p90"][c] / 1e6,
                res["revenue_mean"][c] / 1e6,
            ],
            "bias": [
                truth[c] * (1 + res["bias_pct_p10"][c] / 100) / 1e6,
                truth[c] * (1 + res["bias_pct_p90"][c] / 100) / 1e6,
                truth[c] * (1 + res["bias_pct"][c] / 100) / 1e6,
            ],
            "bias_pct": res["bias_pct"][c],
            "b": [
                res["b_p10"][c],
                res["b_p90"][c],
                res["identifiability"][c]["b_mean"],
            ],
            "lam": [
                res["lam_p10"][c],
                res["lam_p90"][c],
                res["identifiability"][c]["lam_mean"],
            ],
            "cv": res["variance_cv"][c],
        }
    return out


hist, plan = scenario()
out = {}

# ── Main report, default settings (= docs/example-report.html) ──
r = make_report(hist, plan)
r.fit()
truth = true_revenue(r)
u, w = r.results_["unphased"], r.results_[r.winner_]
labels = [lb for lb, *_ in r.levers_]
out["winner"] = r.winner_
out["revenue_noise_std"] = r.revenue_noise_std
out["total_sales"] = r.calibration_.total_sales
out["truth_m"] = {c: truth[c] / 1e6 for c in CH}
out["plan_spend_m"] = {c: r.planned_spend_[c] / 1e6 for c in CH}
out["plan_total_m"] = sum(r.planned_spend_.values()) / 1e6
out["unphased"] = forest_rows(r, u, truth)
out["winner_rows"] = forest_rows(r, w, truth)
out["corr_before"] = pd.DataFrame(u["correlation"]).loc[CH, CH].round(2).to_dict()
out["corr_after"] = pd.DataFrame(w["correlation"]).loc[CH, CH].round(2).to_dict()
out["corr_before_mean"] = mean_pairwise(u["correlation"])
out["corr_after_mean"] = mean_pairwise(w["correlation"])
out["realised_demand_corr"] = r.realised_demand_corr_
out["table"] = {
    lb: {
        "cv": r.results_[lb]["scores"]["variance"],
        "bias": r.results_[lb]["scores"]["bias"],
        "valley": r.results_[lb]["scores"]["identifiability"],
        "cost": cost_pct(r, lb, truth),
        "sat_w": float(
            np.mean(
                [r.results_[lb]["b_p90"][c] - r.results_[lb]["b_p10"][c] for c in CH]
            )
        ),
        "lam_w": float(
            np.mean(
                [
                    r.results_[lb]["lam_p90"][c] - r.results_[lb]["lam_p10"][c]
                    for c in CH
                ]
            )
        ),
        "peak": _peak_week_multiple(r.plan_df, r.schedules_[lb]),
    }
    for lb in labels
}
base = out["table"]["unphased"]
out["improve"] = {
    lb: {k: 100 * _safe_improvement(base[k], v[k]) for k in ("cv", "bias", "valley")}
    for lb, v in out["table"].items()
}
out["ttb"] = r.time_to_benefit_
out["mean_pairwise_spend"] = mean_pairwise(pd.concat([hist, plan]).corr())
out["plan_mean_pairwise_spend"] = mean_pairwise(plan.corr())
print("main done", r.winner_, flush=True)

# ── Section 1 scenario charts (was build_section1_data.py) ──
comb = pd.concat([hist, plan])
contrib = channel_contributions(comb, MR, SAT, ADS, r.reference_spend_)
baseline = r.calibration_.baseline_level + r.calibration_.demand_coef * r.demand_
max_lag = min(52, max(int(np.ceil(np.log(0.02) / np.log(lam))) for lam in ADS.values()))
imp = np.zeros(max_lag + 1)
imp[0] = 1.0
xs = np.linspace(0, max(r.reference_spend_.values()) * 1.5, 60)
sat_curves = {}
for c in CH:
    b, x0 = SAT[c], r.reference_spend_[c]
    k = MR[c] / (b * x0 ** (b - 1.0))
    sat_curves[c] = k * xs**b
out["scen"] = {
    "dates": [d.strftime("%Y-%m-%d") for d in comb.index],
    "nHist": len(hist),
    "spend": {c: [round(float(v)) for v in comb[c]] for c in CH},
    "adstock": {
        c: [round(float(v), 4) for v in apply_adstock(imp, ADS[c])] for c in CH
    },
    "satX": [round(float(v)) for v in xs],
    "sat": {c: [round(float(v)) for v in sat_curves[c]] for c in CH},
    "refSpend": {c: round(float(r.reference_spend_[c])) for c in CH},
    "baseline": [round(float(v)) for v in baseline],
    "contrib": {c: [round(float(v)) for v in contrib[c]] for c in CH},
    "baselineShare": r.baseline_share,
}
out["plan_weekly"] = {c: [round(float(v)) for v in plan[c]] for c in CH}

OUT = Path(__file__).parent / "results" / "example_scenario.json"
OUT.write_text(json.dumps(out, indent=1, default=float))
print(f"wrote {OUT}")
