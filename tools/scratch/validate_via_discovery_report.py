"""Run the two-layer orthogonal design (and the discrete-levels idea)
through the REAL scoring pipeline, same pattern earlier work used:
subclass DiscoveryReport, override only _phase(), leave fit()/scores()
untouched.

Comparison set (2026-09-18):
  - Unphased (reference)
  - +/-20% edge+balanced (existing default -- kept for comparison even
    though it isn't considered client-suitable either)
  - +/-80% edge+balanced (existing default's ceiling -- not deployable,
    included anyway as an upper reference point)
  - Orthogonal (two-layer) -- item 8's design generator prototype
  - Random levels: each week, independently pick from
    {0%, -10%, -20%, +10%, +20%}

Not wired into _phaser.py. Scratch/throwaway.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

import numpy as np
import pandas as pd
from design import (
    apply_design,
    cross_month_blackout_scatter_schedule,
    cross_month_blackout_schedule,
    random_levels_schedule,
    two_layer_design,
)

from how_wrong_is_your_mmm import Blackout, simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import DiscoveryReport, _safe_improvement
from how_wrong_is_your_mmm._phaser import _get_month_labels

CHANNELS = [f"ch{i}" for i in range(10)]
TRUE_MARGINAL_RETURNS = {c: 0.5 + 0.1 * i for i, c in enumerate(CHANNELS)}
CAP_PCT = {c: 20.0 for c in CHANNELS}
# Per-channel saturation/adstock, cycling the four (b, lambda) pairs from
# the package's own canonical 4-channel scenario (docs/overview.html) --
# Note (2026-09-18): "take inspiration from the params used in the
# overview for adstock and saturation", replacing the flat
# saturation=0.7/adstock=0 placeholder used until then.
_OVERVIEW_SATURATION = [0.60, 0.75, 0.90, 0.70]  # tv, meta, search_generic, tiktok
_OVERVIEW_ADSTOCK = [0.50, 0.30, 0.10, 0.20]
SATURATION = {c: _OVERVIEW_SATURATION[i % 4] for i, c in enumerate(CHANNELS)}
ADSTOCK = {c: _OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(CHANNELS)}


class ExtendedDiscoveryReport(DiscoveryReport):
    def _phase(self, spec, nudge_shape, balance_signs, seed):
        if isinstance(spec, dict) and spec.get(CHANNELS[0]) == "orthogonal":
            patterns, _stats = two_layer_design(
                CHANNELS, self._month_lengths, seed=seed, n_iter=20
            )
            return apply_design(
                self.plan_df, self._plan_month_labels, patterns, CAP_PCT
            )
        if isinstance(spec, dict) and spec.get(CHANNELS[0]) == "random_levels":
            return random_levels_schedule(self.plan_df, self._plan_month_labels, seed)
        if isinstance(spec, dict) and spec.get(CHANNELS[0]) == "cross_month_blackout":
            return cross_month_blackout_schedule(
                self.plan_df, self._plan_month_labels, seed, dark_weeks=4
            )
        marker = spec.get(CHANNELS[0]) if isinstance(spec, dict) else None
        if isinstance(marker, str) and marker.startswith("scatter_t"):
            t = int(marker.removeprefix("scatter_t"))
            return cross_month_blackout_scatter_schedule(
                self.plan_df, self._plan_month_labels, seed, dark_weeks=t
            )
        return super()._phase(spec, nudge_shape, balance_signs, seed)


def main():
    history_df = simulate_spend(
        n_obs=104, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=1, start_date=plan_start
    )
    month_labels = _get_month_labels(plan_df)
    _, counts = np.unique(month_labels, return_counts=True)
    month_lengths = counts.tolist()

    levers = [
        ("Unphased", {}, "uniform", False),
        ("+/-20% edge+balanced", {c: 20.0 for c in CHANNELS}, "edge", True),
        ("+/-80% edge+balanced", {c: 80.0 for c in CHANNELS}, "edge", True),
        (
            "Orthogonal (two-layer)",
            {c: "orthogonal" for c in CHANNELS},
            "uniform",
            False,
        ),
        (
            "Random levels [0,-10,-20,+10,+20]%",
            {c: "random_levels" for c in CHANNELS},
            "uniform",
            False,
        ),
        (
            "Blackout (dark=4, prob=1.0)",
            {c: Blackout(max_dark_weeks_per_month=4, prob=1.0) for c in CHANNELS},
            "uniform",
            False,
        ),
        (
            "Cross-month blackout + redistribution",
            {c: "cross_month_blackout" for c in CHANNELS},
            "uniform",
            False,
        ),
        *[
            (
                f"Blackout t={t} + scattered +10/20% bumps",
                {c: f"scatter_t{t}" for c in CHANNELS},
                "uniform",
                False,
            )
            for t in (2, 3, 4)
        ],
    ]

    report = ExtendedDiscoveryReport(
        history_df,
        plan_df,
        true_marginal_returns=TRUE_MARGINAL_RETURNS,
        saturation=SATURATION,
        adstock=ADSTOCK,
        levers=levers,
        seed=0,
    )
    report._month_lengths = month_lengths
    report.fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8)

    # Cost column: % of true plan-period revenue given up by phasing,
    # same formula DiscoveryReport's own Section 3 table uses (see
    # _discovery_report.py ~line 1878) -- built here rather than
    # exposed by the class, since report.schedules_[lbl] (populated by
    # fit()) already has everything the formula needs.
    channels = report.channels_
    combined_baseline = pd.concat([report.history_df, report.plan_df])
    true_contributions = channel_contributions(
        combined_baseline,
        report.true_marginal_returns,
        report.saturation,
        report.adstock,
        report.reference_spend_,
    )
    true_revenue = {
        ch: float(true_contributions[ch].iloc[-len(report.plan_df) :].sum())
        for ch in channels
    }

    cost_pct = {}
    for lbl in report.levers_:
        label = lbl[0]
        lever_contributions = channel_contributions(
            pd.concat([report.history_df, report.schedules_[label]]),
            report.true_marginal_returns,
            report.saturation,
            report.adstock,
            report.reference_spend_,
        )
        lever_revenue = {
            ch: float(lever_contributions[ch].iloc[-len(report.plan_df) :].sum())
            for ch in channels
        }
        cost_pct[label] = 100 * float(
            np.mean(
                [
                    _safe_improvement(true_revenue[ch], lever_revenue[ch])
                    for ch in channels
                ]
            )
        )

    summary = report.summary()
    summary["cost_pct"] = summary["lever"].map(cost_pct)
    print(summary.to_string())


if __name__ == "__main__":
    main()
