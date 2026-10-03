"""Reusable harness for the design-tweak-test-decide loop: given a
candidate _phase() function, run it against Unphased across several
report seeds on the fixed canonical scenario, and return mean/std/min/max
for the three rigor metrics + cost -- directly comparable to the t=3
cross-month-blackout-scatter baseline established earlier
(variance 0.133, bias 19.2, identifiability 7.06, cost 1.90 at n_seeds=12,
n_sims=20/n_phasing_seeds=2/id_n_sims=8).
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "proto")

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm import simulate_spend
from how_wrong_is_your_mmm._dgp import channel_contributions
from how_wrong_is_your_mmm._discovery_report import DiscoveryReport, _safe_improvement
from how_wrong_is_your_mmm._phaser import _get_month_labels

CHANNELS = [f"ch{i}" for i in range(10)]
TRUE_MARGINAL_RETURNS = {c: 0.5 + 0.1 * i for i, c in enumerate(CHANNELS)}

# Per-channel saturation/adstock, cycling the four (b, lambda) pairs from
# the package's own canonical 4-channel scenario (docs/overview.html --
# tv/meta/search_generic/tiktok) across our 10 test channels, rather than
# the flat saturation=0.7/adstock=0 placeholder used until then.
# Note (2026-09-18): "take inspiration from the params used in the
# overview for adstock and saturation" -- overview's true values:
#   tv: b=0.60, lambda=0.50   meta: b=0.75, lambda=0.30
#   search_generic: b=0.90, lambda=0.10   tiktok: b=0.70, lambda=0.20
_OVERVIEW_SATURATION = [0.60, 0.75, 0.90, 0.70]
_OVERVIEW_ADSTOCK = [0.50, 0.30, 0.10, 0.20]
SATURATION = {c: _OVERVIEW_SATURATION[i % 4] for i, c in enumerate(CHANNELS)}
ADSTOCK = {c: _OVERVIEW_ADSTOCK[i % 4] for i, c in enumerate(CHANNELS)}


def canonical_scenario():
    history_df = simulate_spend(
        n_obs=104, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-02"
    )
    plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=1, start_date=plan_start
    )
    month_labels = _get_month_labels(plan_df)
    return history_df, plan_df, month_labels


def make_report_class(phase_fn):
    class CandidateReport(DiscoveryReport):
        def _phase(self, spec, nudge_shape, balance_signs, seed):
            if isinstance(spec, dict) and spec.get(CHANNELS[0]) == "candidate":
                return phase_fn(self.plan_df, self._plan_month_labels, seed)
            return super()._phase(spec, nudge_shape, balance_signs, seed)

    return CandidateReport


def evaluate_candidate(
    phase_fn,
    label: str,
    n_seeds: int = 8,
    n_sims: int = 20,
    n_phasing_seeds: int = 2,
    id_n_sims: int = 8,
) -> pd.DataFrame:
    """Run `phase_fn` (plan_df, month_labels, seed) -> phased_df against
    Unphased, across n_seeds report seeds. Returns one row per seed with
    variance_cv, bias_pct, identifiability, cost_pct."""
    history_df, plan_df, month_labels = canonical_scenario()
    ReportClass = make_report_class(phase_fn)
    levers = [
        ("Unphased", {}, "uniform", False),
        (label, {c: "candidate" for c in CHANNELS}, "uniform", False),
    ]

    rows = []
    for s in range(n_seeds):
        report = ReportClass(
            history_df,
            plan_df,
            true_marginal_returns=TRUE_MARGINAL_RETURNS,
            saturation=SATURATION,
            adstock=ADSTOCK,
            levers=levers,
            seed=s,
        )
        report.fit(n_sims=n_sims, n_phasing_seeds=n_phasing_seeds, id_n_sims=id_n_sims)

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
            for ch in report.channels_
        }

        scores = report.results_[label]["scores"]
        lever_contributions = channel_contributions(
            pd.concat([report.history_df, report.schedules_[label]]),
            report.true_marginal_returns,
            report.saturation,
            report.adstock,
            report.reference_spend_,
        )
        lever_revenue = {
            ch: float(lever_contributions[ch].iloc[-len(report.plan_df) :].sum())
            for ch in report.channels_
        }
        cost = 100 * float(
            np.mean(
                [
                    _safe_improvement(true_revenue[ch], lever_revenue[ch])
                    for ch in report.channels_
                ]
            )
        )
        rows.append(
            {
                "seed": s,
                "variance_cv": scores["variance"],
                "bias_pct": scores["bias"],
                "identifiability": scores["identifiability"],
                "cost_pct": cost,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> None:
    print(f"--- {label} (n={len(df)} seeds) ---")
    for col in ["variance_cv", "bias_pct", "identifiability", "cost_pct"]:
        print(f"  {col:16s} mean={df[col].mean():.4f}  std={df[col].std():.4f}")
