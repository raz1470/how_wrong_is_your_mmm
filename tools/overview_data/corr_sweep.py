"""Section 3 sweeps. Writes results/corr_sweep.json.
bias:  channel corr fixed at 0.7, demand_spend_corr swept 0.1-0.9.
alpha: channel corr swept 0.0-0.9, demand's weight on the spend factor
       (alpha) held at its 0.7 / 0.65 value, so demand is equally tied to
       the plan at every point.
fixed: channel corr swept, demand built once at 0.7 and reused.
All unphased DiscoveryReport at report defaults, fixed-spread spend."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand
from how_wrong_is_your_mmm._dgp import _CHANNEL_SCALE, _DEFAULT_SCALE

warnings.filterwarnings("ignore")
CH = ["tv", "meta", "search_generic", "tiktok"]
MR = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
SAT = {"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70}
ADS = {"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20}
NH, NP = 104, 52
D = simulate_demand(NH + NP, process="trend", seed=6)
UNPH = [("Unphased", {ch: 0.0 for ch in CH}, "uniform", False)]


def spend(c):
    out = []
    for n, seed, start, dem in [
        (NH, 37, "2025-01-06", D[:NH]),
        (NP, 13, "2027-01-04", D[NH:]),
    ]:
        rng = np.random.default_rng(seed)
        rng.standard_normal(n)
        data = {}
        for ch in CH:
            m, s = _CHANNEL_SCALE.get(ch, _DEFAULT_SCALE)
            z = rng.standard_normal(n)
            data[ch] = m + s / np.sqrt(0.7) * (np.sqrt(c) * dem + np.sqrt(1 - c) * z)
        df = pd.DataFrame(data)
        df.index = pd.date_range(start, periods=n, freq="W-MON")
        out.append(df)
    return out


def mean_fcorr(h, p):
    s = pd.concat([h, p])
    z = (s - s.mean()) / s.std()
    f = z.mean(axis=1)
    return float(np.mean([np.corrcoef(f, s[c])[0, 1] for c in CH]))


def run(h, p, **kw):
    r = DiscoveryReport(
        history_df=h,
        plan_df=p,
        true_marginal_returns=MR,
        saturation=SAT,
        adstock=ADS,
        client_name="Demo client",
        plan_year="2027",
        levers=UNPH,
        **kw,
    ).fit()
    res = r.results_["Unphased"]
    return r, {
        "cv": {k: 100 * v for k, v in res["variance_cv"].items()},
        "bias": res["bias_pct"],
        "valley": pd.DataFrame(res["identifiability"]).T["valley_pct"].to_dict(),
        "dem_corr": float(np.mean(list(r.realised_demand_corr_.values()))),
    }


out = {"bias": {}, "alpha": {}, "fixed": {}}
h7, p7 = spend(0.7)
r7, _ = run(h7, p7)
alpha07 = r7.demand_link_.factor_weight
D07 = r7.demand_
for x in np.round(np.arange(0.1, 0.95, 0.1), 1):
    out["bias"][float(x)] = run(h7, p7, demand_spend_corr=float(x))[1]
for c in np.round(np.arange(0.0, 0.95, 0.1), 1):
    h, p = spend(float(c))
    out["alpha"][float(c)] = run(
        h, p, demand_spend_corr=min(1.0, alpha07 * mean_fcorr(h, p))
    )[1]
    out["fixed"][float(c)] = run(h, p, demand=D07)[1]
out["alpha07"] = alpha07
json.dump(
    out,
    open(Path(__file__).parent / "results" / "corr_sweep.json", "w"),
    default=float,
    indent=1,
)
