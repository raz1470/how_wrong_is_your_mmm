"""Prototype of the benefit table: one row per lever, one column per problem.

The point of the table is that no column agrees with another, so no single
lever is 'the recommendation'. Cost figures are notebook 07 section 7.2's
phasing_cost_pct at b=0.6; saturation visibility is not measured yet.
"""

import numpy as np
import pandas as pd

agg = pd.read_csv("grid_agg.csv")
COST = {  # notebook 07 sec 7.2, phasing cost as % of media contribution, b=0.6
    "unphased": 0.00,
    "+/-20%": 0.11,
    "+/-40%": 0.47,
    "+/-80%": 2.25,
    "Blackout": 9.90,
    "TV alone +/-80%": 0.52,
}
ORDER = ["unphased", "+/-20%", "+/-40%", "+/-80%", "Blackout", "TV alone +/-80%"]

ref = agg[(agg.correlation == 0.7) & (agg.demand_share == 1.0)].set_index("lever")
base_bias = ref.loc["unphased", "mean_bias_%"]
base_cv = ref.loc["unphased", "mean_cv_%"]

rows = []
for lev in ORDER:
    r = ref.loc[lev]
    # stability of the bias ranking across the grid: min/max bias-removed
    # across every demand_share >= 0.25 and every correlation
    grid = agg[(agg.demand_share >= 0.25)]
    removed = []
    for (d, c), blk in grid.groupby(["demand_share", "correlation"]):
        blk = blk.set_index("lever")
        u = blk.loc["unphased", "mean_bias_%"]
        removed.append(100 * (u - blk.loc[lev, "mean_bias_%"]) / u)
    rows.append(
        {
            "lever": lev,
            "variance: CV narrowed %": 100 * (base_cv - r["mean_cv_%"]) / base_cv,
            "bias: removed %": 100 * (base_bias - r["mean_bias_%"]) / base_bias,
            "bias removed, grid min": min(removed),
            "bias removed, grid max": max(removed),
            "saturation visibility": np.nan,
            "cost: revenue given up %": COST[lev],
        }
    )

t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.round(1).to_string(index=False))
print("\nbest in column (excluding unphased):")
body = t[t.lever != "unphased"]
print(f"  variance   -> {body.loc[body['variance: CV narrowed %'].idxmax(), 'lever']}")
print(f"  bias       -> {body.loc[body['bias: removed %'].idxmax(), 'lever']}")
print(f"  cost       -> {body.loc[body['cost: revenue given up %'].idxmin(), 'lever']}")
print("  saturation -> not measured yet")
