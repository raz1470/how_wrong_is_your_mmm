"""Channel-count x correlation grid, averaged over multiple draws, across
all four reliability metrics (variance, bias, adstock/saturation
identifiability) -- the DiscoveryReport-era successor to the old
notebooks/archive/04_channel_scaling_walkthrough.ipynb's "does the fix hold
across the grid" section, which produced overview.html's current chartGrid
(variance only, at n_sims=50/grid_steps=12/n_phasing_seeds=6/3 draws).

This is a genuine multi-minute run (~20 min at these settings), not a quick
notebook cell -- same reasoning as that old section's own comment ("cheaper
settings than Sections 2-3... because each of the 25 cells runs its own grid
search"), now with three more, noisier metrics on top of variance.

Resumable: each call processes cells until TIME_BUDGET_S is spent, writing
after every cell, then exits -- re-run the same command to continue where it
left off. (No persistent background-process support in this environment;
this stands in for "run once in the background".) notebooks/02_channel_scaling.ipynb
loads the finished CSV rather than recomputing it.

Usage: uv run python tools/channel_scaling_sweep.py   (repeat until it prints DONE)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import Blackout, DiscoveryReport, simulate_spend

OUT_PATH = Path(__file__).parent / "channel_scaling_grid.csv"

# Same rows as the old notebook's grid. Columns narrowed from that
# notebook's [0.1, 0.3, 0.5, 0.7, 0.9]: that version was variance-only
# (linear, no saturation), so low-correlation's occasional negative
# synthetic spend was harmless. This version profiles adstock/saturation
# identifiability too, which needs real curvature, and a saturating
# response is undefined for negative spend -- so the floor moves to 0.4.
CHANNEL_COUNTS = [3, 5, 10, 15, 20]
CORRELATIONS = [0.5, 0.6, 0.75, 0.9]
N_DRAWS = 3  # independent (history, plan) seeds averaged per cell, same as
# the old grid section's own SWEEP5_DRAWS
N_SIMS = 50  # same as the old notebook's "publication quality" N_SIMS
N_PHASING_SEEDS = 3  # phased-schedule draws averaged within each outer draw
ID_N_SIMS = 15
ID_GRID = 15
TIME_BUDGET_S = 25  # this call's wall-clock budget

# Cycled archetypes rather than one value for every synthetic channel:
# marginal returns from this page's own 0.5/1.0/1.5 (K=3 reproduces the
# tv/meta/search scenario exactly, same convention the old notebook used),
# saturation/adstock from the live docs/example-report.html's four channels.
MR_CYCLE = [0.5, 1.0, 1.5]
SAT_CYCLE = [0.60, 0.75, 0.90, 0.70]
ADS_CYCLE = [0.50, 0.30, 0.10, 0.20]


def scenario(n_channels):
    channels = [f"ch{i}" for i in range(n_channels)]
    mr = {c: MR_CYCLE[i % len(MR_CYCLE)] for i, c in enumerate(channels)}
    sat = {c: SAT_CYCLE[i % len(SAT_CYCLE)] for i, c in enumerate(channels)}
    ads = {c: ADS_CYCLE[i % len(ADS_CYCLE)] for i, c in enumerate(channels)}
    return channels, mr, sat, ads


def improvement(before, after):
    return 0.0 if before == 0 else 100 * (before - after) / before


def run_draw(n_channels, correlation, draw):
    channels, mr, sat, ads = scenario(n_channels)
    h = simulate_spend(
        n_obs=208,
        correlation=correlation,
        channels=channels,
        seed=draw,
        start_date="2019-01-07",
    )
    p = simulate_spend(
        n_obs=52,
        correlation=correlation,
        channels=channels,
        seed=100 + draw,
        start_date="2023-01-09",
    )
    levers = [
        ("unphased", {c: 0.0 for c in channels}, "uniform", False),
        (
            "Blackout",
            {c: Blackout(max_dark_weeks_per_month=1) for c in channels},
            "uniform",
            False,
        ),
    ]
    report = DiscoveryReport(
        history_df=h,
        plan_df=p,
        true_marginal_returns=mr,
        saturation=sat,
        adstock=ads,
        levers=levers,
        seed=draw,
    )
    report.fit(
        n_sims=N_SIMS,
        n_phasing_seeds=N_PHASING_SEEDS,
        id_n_sims=ID_N_SIMS,
        id_b_candidates=np.linspace(0.3, 1.0, ID_GRID),
        id_lam_candidates=np.linspace(0.0, 0.7, ID_GRID),
    )
    u, w = report.results_["unphased"], report.results_["Blackout"]
    return {
        "variance": np.mean(
            [improvement(u["variance_cv"][c], w["variance_cv"][c]) for c in channels]
        ),
        "bias": np.mean(
            [abs(u["bias_pct"][c]) - abs(w["bias_pct"][c]) for c in channels]
        ),  # pp reduction, not relative -- bias crosses zero
        "adstock": np.mean(
            [
                improvement(
                    u["identifiability"][c]["lam_sd"], w["identifiability"][c]["lam_sd"]
                )
                for c in channels
            ]
        ),
        "saturation": np.mean(
            [
                improvement(
                    u["identifiability"][c]["b_sd"], w["identifiability"][c]["b_sd"]
                )
                for c in channels
            ]
        ),
    }


def main():
    t_start = time.time()
    all_cells = [(k, r) for k in CHANNEL_COUNTS for r in CORRELATIONS]
    total_cells = len(all_cells)

    done = pd.DataFrame(columns=["channels", "correlation"])
    if OUT_PATH.exists():
        done = pd.read_csv(OUT_PATH)
    done_pairs = set(zip(done["channels"], done["correlation"], strict=False))
    records = done.to_dict("records")

    remaining = [(k, r) for k, r in all_cells if (k, r) not in done_pairs]
    if not remaining:
        print(
            f"DONE already -> {OUT_PATH} ({total_cells}/{total_cells} cells)",
            flush=True,
        )
        return

    print(f"resuming: {len(done_pairs)}/{total_cells} cells already done", flush=True)
    for k, r in remaining:
        if time.time() - t_start > TIME_BUDGET_S:
            print(
                f"time budget reached, {len(records)}/{total_cells} done -- "
                "re-run this script to continue",
                flush=True,
            )
            return
        cell_i = len(records) + 1
        t0 = time.time()
        draws = [run_draw(k, r, d) for d in range(N_DRAWS)]
        row = {"channels": k, "correlation": r}
        for metric in ["variance", "bias", "adstock", "saturation"]:
            vals = [d[metric] for d in draws]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        records.append(row)
        pd.DataFrame(records).to_csv(OUT_PATH, index=False)
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        print(
            f"[{cell_i}/{total_cells}] K={k:>2} r={r}  "
            f"var={row['variance_mean']:.1f} bias={row['bias_mean']:.1f} "
            f"adstock={row['adstock_mean']:.1f} sat={row['saturation_mean']:.1f}  "
            f"({elapsed:.1f}s, call total {total_elapsed / 60:.1f}min)",
            flush=True,
        )
    print(f"DONE -> {OUT_PATH} ({len(records)}/{total_cells} cells)", flush=True)


if __name__ == "__main__":
    main()
