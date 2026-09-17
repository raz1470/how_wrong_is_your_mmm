"""Generate assets/readme-honest-ranges.png, the README's hero chart.

Same four-channel scenario as notebooks 01/02/03 and the live
docs/example-report.html (tv/meta/search_generic/tiktok). No
saturation/adstock, matching notebook 03's choice: this chart isolates
the timing question (how much does accumulated phased history tighten
the estimate for a fixed upcoming plan), the same way notebook 03 does
for CV.

Design, since the original generator script was never saved (see
NOTES.md): keep one £-plan fixed (the same `plan_df` throughout, "the
same plan" the README caption promises) and vary only the training
data fitted before evaluating it against that plan:
  - Today: fit on 208 weeks of natural (unphased, correlated) history.
  - After 1 year: fit on that history plus one more year of phased
    spend (the report's own winning lever as of session 56/item 7's
    widened sweep: Blackout, dark=4 weeks/month, prob=1.0 -- dominates
    +/-80% edge+balanced on all three rigor axes on this scenario, see
    notebooks/05_strategy_comparison.ipynb, at a materially higher cost
    which this chart does not show since it only tracks estimate width).
Each fit's estimated marginal returns are applied to the *same*
plan_df via CollinearityDiagnostic.summary(planned_spend=...), which
is what makes the two bars a range for one £X plan getting more
reliable, not two different plans. A two-year variant was tried and
dropped: the year-1 narrowing already carries the story, a third bar
per channel added clutter without a new finding.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from how_wrong_is_your_mmm import Blackout, simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._phaser import _generate_phased_schedule, _get_month_labels

OUT_PATH = Path(__file__).parent.parent / "assets" / "readme-honest-ranges.png"

CHANNELS = ["tv", "meta", "search_generic", "tiktok"]
TRUE_MARGINAL_RETURNS = {"tv": 0.5, "meta": 1.0, "search_generic": 1.5, "tiktok": 1.2}
CORRELATION = 0.7
N_SIMS = 50

CHANNEL_LABELS = {
    "tv": "TV",
    "meta": "Meta",
    "search_generic": "Search",
    "tiktok": "TikTok",
}
PERIOD_COLORS = {"Today": "#d1d5db", "After 1 year": "#2563eb"}


def phased_year(history: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One more year of Blackout(dark=4, prob=1.0) phased spend, appended
    right after `history` ends -- the report's own winning lever as of
    the item-7 sweep widening (see module docstring). seed picks the
    underlying natural 52-week draw that gets phased (not the phasing
    draw itself, which _generate_phased_schedule takes care of
    internally)."""
    start = history.index[-1] + pd.Timedelta(weeks=1)
    natural = simulate_spend(
        n_obs=52,
        correlation=CORRELATION,
        channels=CHANNELS,
        seed=seed,
        start_date=start,
    )
    month_labels = _get_month_labels(natural)
    return _generate_phased_schedule(
        natural,
        month_labels,
        alpha=1.0,
        max_weekly_deviation_pct={
            c: Blackout(prob=1.0, max_dark_weeks_per_month=4) for c in CHANNELS
        },
        seed=1000 + seed,
        nudge_shape="uniform",
        balance_signs=False,
    )


def revenue_range(train_df: pd.DataFrame, plan_df: pd.DataFrame) -> pd.DataFrame:
    diag = CollinearityDiagnostic(
        spend_df=train_df, true_marginal_returns=TRUE_MARGINAL_RETURNS
    )
    diag.fit(n_sims=N_SIMS)
    s = diag.summary(planned_spend=plan_df).set_index("channel")
    return s[
        [
            "incremental_revenue_p10",
            "incremental_revenue_mean",
            "incremental_revenue_p90",
        ]
    ]


def main() -> None:
    history_df = simulate_spend(
        n_obs=208,
        correlation=CORRELATION,
        channels=CHANNELS,
        seed=0,
        start_date="2019-01-07",
    )
    plan_start = history_df.index[-1] + pd.Timedelta(weeks=1)
    plan_df = simulate_spend(
        n_obs=52,
        correlation=CORRELATION,
        channels=CHANNELS,
        seed=1,
        start_date=plan_start,
    )
    plan_total = float(plan_df.values.sum())
    print(f"Annual plan total: £{plan_total:,.0f}")

    year1 = phased_year(history_df, seed=2)

    periods = {
        "Today": history_df,
        "After 1 year": pd.concat([history_df, year1]),
    }

    results = {
        label: revenue_range(train_df, plan_df) for label, train_df in periods.items()
    }

    true_revenue = {
        ch: TRUE_MARGINAL_RETURNS[ch] * plan_df[ch].sum() for ch in CHANNELS
    }

    # Width narrowing, today vs after 1 year -- for the caption.
    for ch in CHANNELS:
        w0 = (
            results["Today"].loc[ch, "incremental_revenue_p90"]
            - results["Today"].loc[ch, "incremental_revenue_p10"]
        )
        w1 = (
            results["After 1 year"].loc[ch, "incremental_revenue_p90"]
            - results["After 1 year"].loc[ch, "incremental_revenue_p10"]
        )
        narrower = 100 * (w0 - w1) / w0
        print(
            f"{ch:15s} width today=£{w0:,.0f}  after 1yr=£{w1:,.0f}  "
            f"narrower={narrower:.1f}%"
        )

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    _fig, ax = plt.subplots(figsize=(9, 5))

    n_periods = len(periods)
    for ch_idx, ch in enumerate(CHANNELS):
        base_y = (len(CHANNELS) - 1 - ch_idx) * (n_periods + 1)
        for p_idx, (label, table) in enumerate(results.items()):
            y = base_y + (n_periods - 1 - p_idx)
            lo = table.loc[ch, "incremental_revenue_p10"]
            mid = table.loc[ch, "incremental_revenue_mean"]
            hi = table.loc[ch, "incremental_revenue_p90"]
            ax.plot(
                [lo, hi],
                [y, y],
                color=PERIOD_COLORS[label],
                lw=6,
                solid_capstyle="round",
            )
            ax.plot(mid, y, "o", color="#111827", zorder=3, ms=4)
        ax.axvline(
            true_revenue[ch], color="#111827", ls="--", lw=1, ymin=0, ymax=1, alpha=0.0
        )

    # true-value dashed markers, drawn per channel row-block rather than
    # full-height axvlines (each channel has a different true value)
    for ch_idx, ch in enumerate(CHANNELS):
        base_y = (len(CHANNELS) - 1 - ch_idx) * (n_periods + 1)
        ax.plot(
            [true_revenue[ch], true_revenue[ch]],
            [base_y - 0.6, base_y + n_periods - 1 + 0.6],
            color="#111827",
            ls="--",
            lw=1,
        )

    yticks = [
        (len(CHANNELS) - 1 - i) * (n_periods + 1) + (n_periods - 1) / 2
        for i in range(len(CHANNELS))
    ]
    ax.set_yticks(yticks, [CHANNEL_LABELS[ch] for ch in CHANNELS], fontsize=12)
    ax.set_xlabel(
        "Model-estimated incremental revenue for next year's plan (£)", fontsize=10
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"£{x / 1e6:.1f}m" if abs(x) >= 1e6 else f"£{x / 1e3:.0f}k"
        )
    )
    ax.set_title(
        "A year of phasing tightens the estimated range for the same plan",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in PERIOD_COLORS.values()]
    handles.append(plt.Line2D([0], [0], color="#111827", ls="--", lw=1))
    ax.legend(
        handles,
        [*PERIOD_COLORS.keys(), "True marginal return, implied"],
        loc="lower right",
        fontsize=9,
        frameon=False,
    )
    ax.margins(y=0.03)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
