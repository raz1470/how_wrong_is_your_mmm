"""Aggregate grid_raw.csv the way notebook 07 aggregates, and derive the
budget multiple from the same draws.

Three aggregations:

  bias / cv  -- notebook 07 `sweep()`: mean over phasing seeds first, then the
                bias is formed per demand seed and averaged; cv is the median
                across phasing seeds, then averaged across demand seeds.

  lambda     -- notebook 07 `inflation_draws()`: every draw pooled across
                channels, demand seeds and phasing seeds, then one mean. This
                is what feeds budget_multiple = lambda ** (1/(1-b)).

  per-client -- NOT in the notebook, and the point of this run. A demand seed
                is one client's whole world, and a client has one history. So
                the per-seed spread is the client-facing quantity: the
                systematic average can sit at zero while every individual
                client is badly wrong in an undeterminable direction. Section 3
                of the notebook makes exactly this point for bias; it has never
                been made for the budget multiple.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

B_GRID = (0.4, 0.6, 0.8)
CHANNELS = ["tv", "meta", "search"]
LEVER_ORDER = [
    "unphased",
    "+/-20%",
    "+/-40%",
    "+/-80%",
    "Blackout",
    "TV alone +/-80%",
]


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["lambda"] = raw["mean_est"] / raw["true_mr"]
    raw["cv_pct"] = 100 * raw["sd_est"] / raw["mean_est"].abs()

    rows = []
    keys = ["demand_share", "correlation", "lever"]
    for (demand_share, correlation, lever), block in raw.groupby(keys, sort=False):
        row = {
            "demand_share": demand_share,
            "correlation": correlation,
            "lever": lever,
            "pairwise_corr": block["pairwise_corr"].mean(),
            "corr_tv_demand": block["corr_tv_demand"].mean(),
        }

        # --- bias and cv, notebook `sweep()` order of operations ---
        per_channel_bias, per_channel_cv = {}, {}
        # per (channel, demand_seed) bias, kept for the spread measures below
        seed_bias = {}
        for ch in CHANNELS:
            ch_block = block[block["channel"] == ch]
            bias_by_seed, cv_by_seed = [], []
            for seed, seed_block in ch_block.groupby("demand_seed", sort=False):
                mean_est = seed_block["mean_est"].mean()
                true_mr = seed_block["true_mr"].iloc[0]
                b = 100 * (mean_est - true_mr) / true_mr
                bias_by_seed.append(b)
                seed_bias[(ch, seed)] = b
                cv_by_seed.append(np.median(seed_block["cv_pct"].to_numpy()))
            per_channel_bias[ch] = float(np.mean(bias_by_seed))
            per_channel_cv[ch] = float(np.mean(cv_by_seed))
            row[f"{ch}_bias_%"] = per_channel_bias[ch]
        row["mean_bias_%"] = float(np.mean(list(per_channel_bias.values())))
        row["mean_cv_%"] = float(np.mean(list(per_channel_cv.values())))

        # --- per-client spread on bias ---
        # One demand seed = one client's world. Average the channels within a
        # client first, so the number is "how wrong is this client's model".
        seeds = sorted({s for _ch, s in seed_bias})
        client_bias = np.array(
            [np.mean([seed_bias[(ch, s)] for ch in CHANNELS]) for s in seeds]
        )
        row["n_clients"] = len(seeds)
        row["bias_sd_across_clients_%"] = float(client_bias.std(ddof=1))
        row["bias_se_%"] = float(client_bias.std(ddof=1) / np.sqrt(len(seeds)))
        row["typical_abs_bias_%"] = float(np.abs(client_bias).mean())

        # --- lambda, notebook `inflation_draws()`: pool everything ---
        lam = float(block["lambda"].mean())
        row["mean_lambda"] = lam
        by_channel = block.groupby("channel")["lambda"].mean()
        row["lambda_spread"] = float(by_channel.max() - by_channel.min())
        for b in B_GRID:
            row[f"budget_x_b{b}"] = lam ** (1.0 / (1.0 - b))

        # --- per-client budget multiple: the band a real report would face ---
        client_lambda = np.array([1.0 + cb / 100.0 for cb in client_bias])
        row["lambda_p10"] = float(np.percentile(client_lambda, 10))
        row["lambda_p90"] = float(np.percentile(client_lambda, 90))
        for b in B_GRID:
            power = 1.0 / (1.0 - b)
            safe = np.maximum(client_lambda, 0.05)
            mult = safe**power
            row[f"budget_x_b{b}_p10"] = float(np.percentile(mult, 10))
            row[f"budget_x_b{b}_p90"] = float(np.percentile(mult, 90))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["lever"] = pd.Categorical(out["lever"], LEVER_ORDER, ordered=True)
    return out.sort_values(["correlation", "demand_share", "lever"]).reset_index(
        drop=True
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "grid_raw.csv"
    raw = pd.read_csv(path)
    agg = aggregate(raw)
    agg.to_csv(path.replace("_raw", "_agg"), index=False)
    pd.set_option("display.width", 250)

    print("=" * 96)
    print("A. THE LEVER GRID ACROSS demand_share, at correlation = 0.7")
    print("=" * 96)
    slice_ = agg[agg["correlation"] == 0.7]
    print(
        slice_.pivot(index="lever", columns="demand_share", values="mean_bias_%")
        .round(1)
        .to_string()
    )
    print("\nbias removed vs that column's own unphased row, %:")
    piv = slice_.pivot(index="lever", columns="demand_share", values="mean_bias_%")
    print(
        (100 * (piv.loc["unphased"] - piv) / piv.loc["unphased"]).round(1).to_string()
    )

    print("\n" + "=" * 96)
    print("B. WHAT A CLIENT CAN OBSERVE, AND WHAT THEY CANNOT (unphased)")
    print("=" * 96)
    unph = agg[(agg["lever"] == "unphased")]
    print(
        unph[
            [
                "correlation",
                "demand_share",
                "pairwise_corr",
                "corr_tv_demand",
                "mean_bias_%",
                "typical_abs_bias_%",
                "bias_sd_across_clients_%",
                "bias_se_%",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 96)
    print("C. THE BUDGET MULTIPLE, b = 0.6, across demand_share (correlation = 0.7)")
    print("=" * 96)
    print(
        slice_[
            [
                "lever",
                "demand_share",
                "mean_lambda",
                "budget_x_b0.6",
                "budget_x_b0.6_p10",
                "budget_x_b0.6_p90",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 96)
    print("D. THE FULL BAND ON THE HEADLINE NUMBER (unphased, correlation = 0.7)")
    print("=" * 96)
    row = agg[(agg["lever"] == "unphased") & (agg["correlation"] == 0.7)]
    for b in B_GRID:
        lo = row[f"budget_x_b{b}"].min()
        hi = row[f"budget_x_b{b}"].max()
        print(
            f"  b={b}: budget multiple ranges {lo:.2f}x -- {hi:.2f}x "
            f"across demand_share alone"
        )
    lo_all = min(row[f"budget_x_b{b}"].min() for b in B_GRID)
    hi_all = max(row[f"budget_x_b{b}"].max() for b in B_GRID)
    print(f"\n  across b AND demand_share together: {lo_all:.2f}x -- {hi_all:.2f}x")


if __name__ == "__main__":
    main()
