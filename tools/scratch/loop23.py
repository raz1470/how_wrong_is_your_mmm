"""Loop 23: hybrid (blackout/floor + month steps) and month-step robustness.
usage: loop23.py MODE NCH YEARS HIST_START SEEDS   (MODE: hybrid | robust)"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools/scratch")
import loop_harness as H
import numpy as np
import pandas as pd
from design import roundrobin_plus_edge_layer
from monthstep import month_step_schedule
from partialblackout import partial_blackout_schedule

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _safe_improvement,
)
from how_wrong_is_your_mmm._phaser import _get_month_labels

MODE, NCH, YEARS, HS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SEEDS = [int(x) for x in sys.argv[5].split(",")]
CH = [f"ch{i}" for i in range(NCH)]
H.CHANNELS = CH
TMR = {c: 0.5 + 0.9 * i / max(NCH - 1, 1) for i, c in enumerate(CH)}
SAT = {c: H._OVERVIEW_SATURATION[i % 4] for i, c in enumerate(CH)}
ADS = {c: H._OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(CH)}
if NCH == 10:
    TMR = H.TRUE_MARGINAL_RETURNS


def blocks(plan_df, seed, fn):
    parts = []
    for k, a in enumerate(range(0, len(plan_df), 52)):
        ch = plan_df.iloc[a : a + 52]
        parts.append(fn(ch, _get_month_labels(ch), seed + 1000 * k))
    return pd.concat(parts)


def make(spec_name):
    kind, *rest = spec_name.split("_")
    if kind == "winner":
        return lambda p, s: blocks(
            p,
            s,
            lambda c, m, sd: roundrobin_plus_edge_layer(
                c, m, sd, dark_weeks=4, edge_cap_pct=15.0
            ),
        )
    if kind == "ms":
        st = float(rest[0])
        return lambda p, s: month_step_schedule(p, _get_month_labels(p), s, st)
    if kind == "hyb":  # hyb_<floor>_<step>
        fl, st = float(rest[0]), float(rest[1])

        def f(p, s):
            base = blocks(
                p, s, lambda c, m, sd: partial_blackout_schedule(c, m, sd, floor_pct=fl)
            )
            return month_step_schedule(base, _get_month_labels(p), s + 555, st)

        return f
    raise ValueError(spec_name)


class R(DiscoveryReport):
    def _phase(self, spec, nudge_shape, balance_signs, seed):
        v = spec.get(CH[0]) if isinstance(spec, dict) else None
        if isinstance(v, str) and v.startswith("X:"):
            return make(v[2:])(self.plan_df, seed)
        return super()._phase(spec, nudge_shape, balance_signs, seed)


def main():
    hist = simulate_spend(n_obs=52, correlation=0.7, channels=CH, seed=0, start_date=HS)
    ps = hist.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=52 * YEARS, correlation=0.7, channels=CH, seed=1, start_date=ps
    )
    names = (
        ["ms_40", "ms_60", "hyb_0_20", "hyb_0_40", "hyb_25_20", "hyb_25_40"]
        if MODE == "hybrid"
        else ["winner", "ms_40", "ms_60"]
    )
    levers = [("unphased", {c: 0.0 for c in CH}, "uniform", False)] + [
        (n, {c: "X:" + n for c in CH}, "uniform", False) for n in names
    ]
    rows = []
    for s in SEEDS:
        rep = R(
            hist,
            plan_df,
            true_marginal_returns=TMR,
            saturation=SAT,
            adstock=ADS,
            levers=levers,
            seed=s,
        )
        rep.fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8)
        tc = channel_contributions(
            pd.concat([rep.history_df, rep.plan_df]),
            rep.true_marginal_returns,
            rep.saturation,
            rep.adstock,
            rep.reference_spend_,
        )
        tr = {c: float(tc[c].iloc[-len(plan_df) :].sum()) for c in rep.channels_}
        for label, *_ in levers:
            sc = rep.results_[label]["scores"]
            lc = channel_contributions(
                pd.concat([rep.history_df, rep.schedules_[label]]),
                rep.true_marginal_returns,
                rep.saturation,
                rep.adstock,
                rep.reference_spend_,
            )
            lr = {c: float(lc[c].iloc[-len(plan_df) :].sum()) for c in rep.channels_}
            cost = 100 * float(
                np.mean([_safe_improvement(tr[c], lr[c]) for c in rep.channels_])
            )
            sp = (
                float((rep.schedules_[label].to_numpy() / plan_df.to_numpy()).max())
                if label != "unphased"
                else 1.0
            )
            rows.append(
                dict(
                    mode=MODE,
                    n_channels=NCH,
                    years=YEARS,
                    hist_start=HS,
                    seed=s,
                    lever=label,
                    variance_cv=sc["variance"],
                    bias_pct=sc["bias"],
                    identifiability=sc["identifiability"],
                    cost_pct=cost,
                    max_spike=sp,
                )
            )
        print("seed", s, "done", flush=True)
    pd.DataFrame(rows).to_csv(
        f"loop_data/loop23_{MODE}_n{NCH}_y{YEARS}_{HS}_s{sys.argv[5].replace(',', '_')}_raw.csv",
        index=False,
    )


main()
