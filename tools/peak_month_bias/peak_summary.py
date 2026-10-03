"""Summary tables for FINDINGS.md from peak_checks.py's output."""

from pathlib import Path

import pandas as pd

d = pd.read_csv(Path(__file__).parent / "peak_checks_15_200.csv")
w = d.pivot_table(index=["draw", "channel"], columns="lever", values="err")


def score(col, chans=None):
    x = w[col].unstack("channel")
    if chans is not None:
        x = x[chans]
    return x.mean().abs().mean()


chs = [f"Channel {i}" for i in range(1, 16)]
first12 = chs[:12]
print(
    "mean |bias| all 15: unphased %.1f peak_all %.1f peak_12 %.1f"
    % (score("unphased"), score("peak_all"), score("peak_12"))
)
print(
    "mean |bias| first 12: unphased %.1f peak_all %.1f peak_12 %.1f"
    % (
        score("unphased", first12),
        score("peak_all", first12),
        score("peak_12", first12),
    )
)
print(
    "last 3: unphased %.1f peak_all %.1f peak_12 %.1f"
    % (
        score("unphased", chs[12:]),
        score("peak_all", chs[12:]),
        score("peak_12", chs[12:]),
    )
)
m = w.groupby("channel").mean().reindex(chs).round(1)
print(m.T.to_string())
pa = d[d.lever == "peak_all"].merge(
    d[d.lever == "unphased"][["draw", "channel", "err", "corr_demand"]],
    on=["draw", "channel"],
    suffixes=("", "_un"),
)
pa["delta_abs"] = pa.err.abs() - pa.err_un.abs()
pa["delta"] = pa.err - pa.err_un
print(pa.groupby("n_sharing")[["delta", "delta_abs"]].agg(["mean", "count"]).round(1))
for c in ["Channel 1", "Channel 5", "Channel 9", "Channel 13"]:
    s = pa[pa.channel == c]
    print(
        c,
        "unphased %.0f" % s.err_un.mean(),
        {int(k): round(v, 0) for k, v in s.groupby("n_sharing").err.mean().items()},
        "corr(delta, demand_rel)=%.2f" % s[["delta", "demand_rel"]].corr().iloc[0, 1],
        "by month:",
        s.groupby("peak_month").delta.mean().round(0).tolist(),
    )
print(
    "all channels corr(delta, demand_rel): %.2f ; corr(delta, peak_month): %.2f"
    % (
        pa[["delta", "demand_rel"]].corr().iloc[0, 1],
        pa[["delta", "peak_month"]].corr().iloc[0, 1],
    )
)
print("delta by peak month:", pa.groupby("peak_month").delta.mean().round(1).tolist())
pa["dcorr"] = pa.corr_demand - pa.corr_demand_un
print(
    "mean change in corr(spend, demand):",
    pa.dcorr.mean().round(3),
    " corr(delta, dcorr)=%.2f" % pa[["delta", "dcorr"]].corr().iloc[0, 1],
)
print(
    pa.groupby("channel")[["dcorr", "corr_demand_un"]]
    .mean()
    .reindex(chs)
    .round(3)
    .T.to_string()
)
