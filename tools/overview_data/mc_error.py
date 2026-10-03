"""Monte Carlo standard errors for the section 4 strategy table.

The table's figures are averages over a finite number of draws. This script
reruns the draws, keeps each one, and reports how far each figure would
move on a fresh set. Example scenario, report default settings.

- Variance, saturation and adstock: one score per phasing draw (15). The
  standard error of the table figure is their sd / sqrt(15). The unphased
  plan has no phasing draws, so its figure comes from one set of noise
  draws: the sd over 15 fresh sets is the standard error of that figure.
- Bias: one mean error per channel per demand draw (100). Standard errors
  are from 1,000 bootstrap resamples of the draws. The improvement on the
  unphased plan resamples the same draws for both, since they share demand.

Run from the repo root:  uv run python tools/overview_data/mc_error.py
Writes results/mc_error.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import (
    CollinearityDiagnostic,
    DiscoveryReport,
    IdentifiabilityDiagnostic,
    simulate_demand,
    simulate_spend,
)

warnings.filterwarnings("ignore")
CH = ["tv", "meta", "search_generic", "tiktok"]
MR = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
SAT = {"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70}
ADS = {"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20}
N_SEEDS, N_BIAS, N_SIMS, N_BOOT = 15, 100, 50, 1000

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
common = dict(
    true_marginal_returns=MR,
    base_sales=r.calibration_.baseline_level,
    revenue_noise_std=r.revenue_noise_std,
    demand_coef=r.calibration_.demand_coef,
    reference_spend=r.reference_spend_,
)


def variance_score(combined, noise_offset):
    d = CollinearityDiagnostic(
        spend_df=combined, demand=r.demand_, saturation=SAT, adstock=ADS, **common
    )
    d.fit(n_sims=N_SIMS, controls=True, noise_seed_offset=noise_offset)
    return float(d.summary()["coef_of_variation"].mean())


def range_scores(combined, noise_offset):
    d = IdentifiabilityDiagnostic(
        spend_df=combined,
        demand=r.demand_,
        true_saturation=SAT,
        true_adstock=ADS,
        **common,
    )
    d.fit(n_sims=N_SIMS, noise_seed_offset=noise_offset)
    width = lambda col: float(
        np.mean(
            [
                d.results_[c][col].quantile(0.9) - d.results_[c][col].quantile(0.1)
                for c in CH
            ]
        )
    )
    return width("recovered_b"), width("recovered_lam")


def se(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


out, bias_draws = {}, {}
for label, spec, shape, bal in r.levers_:
    unphased = label == "unphased"
    var, sat, ads = [], [], []
    for j in range(N_SEEDS):
        combined = pd.concat([r.history_df, r._phase(spec, shape, bal, r.seed + j)])
        # The report fits every phasing draw on the same noise draws. The
        # unphased plan has one schedule, so vary the noise draws instead.
        var.append(variance_score(combined, 1000 * j if unphased else 0))
        b, lam = range_scores(combined, r.seed + (1000 * j if unphased else j))
        sat.append(b)
        ads.append(lam)
    errs = np.array(
        [
            r._bias_fit(
                pd.concat([r.history_df, r._phase(spec, shape, bal, r.seed + k)]),
                k,
                N_SIMS,
                k,
            )[0][CH].to_numpy()
            for k in range(N_BIAS)
        ]
    )
    bias_draws[label] = errs
    scale = 1.0 if unphased else 1 / np.sqrt(N_SEEDS)
    out[label] = {
        "variance": float(np.mean(var)),
        "variance_se": float(np.std(var, ddof=1) * scale),
        "saturation": float(np.mean(sat)),
        "saturation_se": float(np.std(sat, ddof=1) * scale),
        "adstock": float(np.mean(ads)),
        "adstock_se": float(np.std(ads, ddof=1) * scale),
        "bias": float(np.abs(errs.mean(axis=0)).mean()),
        "bias_by_channel": dict(zip(CH, errs.mean(axis=0).round(3).tolist())),
        "bias_by_channel_se": dict(
            zip(CH, (errs.std(axis=0, ddof=1) / np.sqrt(N_BIAS)).round(3).tolist())
        ),
    }
    print(
        label,
        {k: round(v, 3) for k, v in out[label].items() if isinstance(v, float)},
        flush=True,
    )

rng = np.random.default_rng(0)
idx = rng.integers(0, N_BIAS, size=(N_BOOT, N_BIAS))
score = lambda errs: np.abs(errs.mean(axis=0)).mean()
base = bias_draws["unphased"]
for label, errs in bias_draws.items():
    boots = np.array([score(errs[i]) for i in idx])
    out[label]["bias_se"] = float(boots.std(ddof=1))
    if label != "unphased":
        imp = np.array([100 * (1 - score(errs[i]) / score(base[i])) for i in idx])
        out[label]["bias_improvement_pct"] = float(
            100 * (1 - score(errs) / score(base))
        )
        out[label]["bias_improvement_pct_se"] = float(imp.std(ddof=1))
        # Per channel: change in absolute bias, phased minus unphased.
        diff = np.array(
            [np.abs(errs[i].mean(axis=0)) - np.abs(base[i].mean(axis=0)) for i in idx]
        )
        out[label]["abs_bias_change_by_channel"] = dict(
            zip(
                CH,
                (np.abs(errs.mean(axis=0)) - np.abs(base.mean(axis=0)))
                .round(3)
                .tolist(),
            )
        )
        out[label]["abs_bias_change_by_channel_se"] = dict(
            zip(CH, diff.std(axis=0, ddof=1).round(3).tolist())
        )
OUT = Path(__file__).parent / "results" / "mc_error.json"
OUT.write_text(json.dumps(out, indent=1))
print(f"wrote {OUT}")
