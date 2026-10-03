"""Loop 21: month-level step design vs winner vs +/-60/80% edge, 10ch, 3yr, same harness as loop20."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools/scratch")
import loop_harness as H
import numpy as np
import pandas as pd
from design import roundrobin_plus_edge_layer
from monthstep import month_step_schedule
from partialblackout import partial_blackout_plus_edge

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _default_levers,
    _safe_improvement,
)
from how_wrong_is_your_mmm._phaser import _get_month_labels

CH = H.CHANNELS
SEEDS = [int(x) for x in sys.argv[1].split(",")]
STEPS = [20, 40, 60, 80]


def winner(plan_df, seed, floor=0.0):
    parts = []
    for k, a in enumerate(range(0, len(plan_df), 52)):
        ch = plan_df.iloc[a : a + 52]
        ml = _get_month_labels(ch)
        parts.append(
            roundrobin_plus_edge_layer(
                ch, ml, seed + 1000 * k, dark_weeks=4, edge_cap_pct=15.0
            )
            if floor == 0
            else partial_blackout_plus_edge(ch, ml, seed + 1000 * k, floor_pct=floor)
        )
    return pd.concat(parts)


class R(DiscoveryReport):
    def _phase(self, spec, nudge_shape, balance_signs, seed):
        v = spec.get(CH[0]) if isinstance(spec, dict) else None
        if v == "winner":
            return winner(self.plan_df, seed)
        if isinstance(v, str) and v.startswith("floor"):
            return winner(self.plan_df, seed, float(v[5:]))
        if isinstance(v, str) and v.startswith("ms"):
            return month_step_schedule(
                self.plan_df, self._plan_month_labels, seed, float(v[2:])
            )
        return super()._phase(spec, nudge_shape, balance_signs, seed)


def main():
    hist = simulate_spend(
        n_obs=52, correlation=0.7, channels=CH, seed=0, start_date="2023-01-02"
    )
    ps = hist.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=156, correlation=0.7, channels=CH, seed=1, start_date=ps
    )
    keep = {"unphased"}
    base = [l for l in _default_levers(CH) if l[0] in keep]
    levers = (
        base
        + [("winner", {c: "winner" for c in CH}, "uniform", False)]
        + [
            (f"floor{p}", {c: f"floor{p}" for c in CH}, "uniform", False)
            for p in (10, 25, 50)
        ]
    )
    rows = []
    for s in SEEDS:
        rep = R(
            hist,
            plan_df,
            true_marginal_returns=H.TRUE_MARGINAL_RETURNS,
            saturation=H.SATURATION,
            adstock=H.ADSTOCK,
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
            sp = float((rep.schedules_[label].to_numpy() / plan_df.to_numpy()).max())
            rows.append(
                dict(
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
    df = pd.DataFrame(rows)
    df.to_csv(
        f"loop_data/loop22_floor_seeds{sys.argv[1].replace(',', '_')}_raw.csv",
        index=False,
    )


main()
