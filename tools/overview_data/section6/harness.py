"""Section 6 setup, and a bias-only run of it.

Channels 'Channel 1..n' cycle the example's four (return, saturation,
adstock) archetypes. Noise is held at 2% of the four-channel example's
sales. Run as a script it runs only the bias fits (no variance, no
identifiability) and keeps every draw:
    uv run python tools/overview_data/section6/harness.py <n_channels> <n_draws> <base|s1|s2|s3>
Writes results/bias_<n>_<tag>_<draws>.csv.
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand, simulate_spend

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
MR = [0.5, 1.0, 1.5, 1.2]
SAT = [0.60, 0.75, 0.90, 0.70]
ADS = [0.50, 0.30, 0.10, 0.20]
NH, NP = 104, 52
NOISE = 23107.358063223437


def build(n, demand_seed=6, hist_seed=12, plan_seed=13, report_seed=None):
    D = simulate_demand(NH + NP, process="trend", seed=demand_seed)
    ch = [f"Channel {i + 1}" for i in range(n)]
    h = simulate_spend(
        n_obs=NH,
        correlation=0.7,
        channels=ch,
        seed=hist_seed,
        start_date="2025-01-06",
        demand=D[:NH],
    )
    p = simulate_spend(
        n_obs=NP,
        correlation=0.7,
        channels=ch,
        seed=plan_seed,
        start_date="2027-01-04",
        demand=D[NH:],
    )
    kw = {} if report_seed is None else {"seed": report_seed}
    return DiscoveryReport(
        history_df=h,
        plan_df=p,
        true_marginal_returns={c: MR[i % 4] for i, c in enumerate(ch)},
        saturation={c: SAT[i % 4] for i, c in enumerate(ch)},
        adstock={c: ADS[i % 4] for i, c in enumerate(ch)},
        client_name="Demo client",
        plan_year="2027",
        revenue_noise_std=NOISE,
        **kw,
    )


def bias_draws(r, n_draws, n_sims=50, levers=None):
    """rows: lever, draw, channel, mean_error_pct"""
    rows = []
    for label, spec, shape, bal in r.levers_:
        if levers and label not in levers:
            continue
        for j in range(n_draws):
            plan = r._phase(spec, shape, bal, r.seed + j)
            comb = pd.concat([r.history_df, plan])
            err, _ = r._bias_fit(comb, j, n_sims, j)
            for c, v in err.items():
                rows.append((label, j, c, float(v)))
    return pd.DataFrame(rows, columns=["lever", "draw", "channel", "err"])


if __name__ == "__main__":
    n = int(sys.argv[1])
    nd = int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "base"
    seeds = {
        "base": {},
        "s1": dict(demand_seed=106, hist_seed=112, plan_seed=113),
        "s2": dict(demand_seed=206, hist_seed=212, plan_seed=213),
        "s3": dict(demand_seed=306, hist_seed=312, plan_seed=313),
    }[tag]
    t0 = time.time()
    r = build(n, **seeds)
    if not hasattr(r, "levers_") or not hasattr(r, "calibration_"):
        print(
            "need fit-time attrs",
            [a for a in dir(r) if a.endswith("_") and not a.startswith("_")],
        )
    df = bias_draws(r, nd)
    df.to_csv(OUT / f"bias_{n}_{tag}_{nd}.csv", index=False)
    print(n, tag, nd, f"{time.time() - t0:.0f}s", flush=True)
