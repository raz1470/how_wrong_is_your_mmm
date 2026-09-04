"""DiscoveryReport: one shared phasing strategy applied to every channel,
swept across a small grid of candidate strategies, with impact shown
against each of the three reliability problems this package diagnoses.

This is the "which strategy should I even consider?" report -- distinct
from ReportBuilder, which is the "here's the phased CSV for the % change
you told me to use" report once a strategy has been picked (session 45's
two-report split; see NOTES.md).

Every candidate strategy is applied identically to every channel (unlike
ReportBuilder's per-channel max_weekly_deviation_pct dict) -- there is no
per-channel number to report here, only a single report-wide choice, so
the report picks ONE "highest impact" strategy (dominance check, else
worst-axis -- see _pick_winner) and uses it for every "impact from best
lever" callout.

Report structure: cover -> spend correlation, before vs after ->
variance problem + impact -> bias problem + impact -> identifiability
problem + impact (saturation and adstock each get their own chart) ->
cross-strategy table (every candidate, with a dropdown) -> appendix
(every input the client supplied: marginal return, adstock and saturation
per channel, the spend series, and the shared demand series).

Follows the package's shared-DGP design (session 44): one demand series
drives every simulated sales column in this report, and saturation/adstock
-- per channel since identifiability was made per-channel later the same
session -- are resolved once in __init__ and reused throughout, never
redrawn per candidate or per section. Only the OLS model's KNOWLEDGE of
demand changes between sections -- known for variance and identifiability,
a measurement-error proxy (at the client's supplied plausible quality) for
bias.
"""

from __future__ import annotations

import html
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import (
    _DEFAULT_MARGINAL_RETURNS,
    apply_adstock,
    calibrate_baseline,
    channel_contributions,
    simulate_demand,
)
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._identifiability import (
    IdentifiabilityDiagnostic,
    _per_channel_map,
)
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    _generate_phased_schedule,
    _get_month_labels,
)
from how_wrong_is_your_mmm._report import _channel_colors


# Default combo grid -- mirrors notebooks/11_phasing_strategy.ipynb's own
# LEVERS list verbatim. That notebook is where "edge+balanced beats
# Blackout on bias, variance, saturation AND adstock, at lower cost" was
# established -- this is the grid that finding was measured on, not a
# fresh choice made here. Each entry is
# (label, per-channel spec, nudge_shape, balance_signs).
def _default_levers(channels: list[str]) -> list[tuple[str, dict, str, bool]]:
    def all_channels(nominal: float) -> dict[str, float]:
        return {ch: nominal for ch in channels}

    return [
        ("unphased", all_channels(0.0), "uniform", False),
        ("+/-80% (uniform)", all_channels(80.0), "uniform", False),
        ("+/-40% (edge, balanced)", all_channels(40.0), "edge", True),
        ("+/-80% (edge, balanced)", all_channels(80.0), "edge", True),
        (
            "Blackout",
            {ch: Blackout(max_dark_weeks_per_month=1) for ch in channels},
            "uniform",
            False,
        ),
    ]


def _is_unphased(spec: dict) -> bool:
    return all(isinstance(v, float) and v == 0.0 for v in spec.values())


def _safe_improvement(before: float, after: float) -> float:
    """Fractional improvement of `after` over `before` (lower-is-better metric).

    Returns 0.0 rather than dividing by zero when `before` is already 0 --
    there is no room left to improve on, not an infinite gain.
    """
    if before == 0:
        return 0.0
    return (before - after) / before


def _svg_multiline(
    series: dict[str, np.ndarray],
    colors: dict[str, str],
    width: int = 680,
    height: int = 200,
    pad_left: int = 58,
    pad_bottom: int = 36,
    pad_top: int = 14,
    pad_right: int = 14,
    normalize: bool = True,
    y_fmt=None,
    y_label: str = "",
    x_label: str = "Week",
    x_tick_labels: list[str] | None = None,
    n_x_ticks: int = 5,
) -> str:
    """Render a small multi-series line chart as a self-contained inline SVG,
    with a real x/y axis -- gridlines, tick labels, axis titles -- the same
    visual language as docs/overview.html's time-series charts (e.g.
    chartCause) and this module's own _svg_forest.

    No JS, no external library -- every point is computed in Python and
    baked into the polyline points at render time, since none of this
    changes after the report-wide winner is picked.

    normalize:
        If True (default), each series is min-max scaled INDEPENDENTLY
        onto the same [0, 1] range, so shape (not absolute level) is what's
        compared, and the y-axis reads as a relative 0-100% scale rather
        than real units -- appropriate when channels sit on very different
        budgets and the week-to-week pattern, not the level, is the point.
        If False, every series shares ONE y-axis scaled to the combined
        min/max across all of them, formatted by `y_fmt` -- used by every
        appendix chart (spend, demand, the saturation response curve, the
        adstock decay-by-lag curve), since the actual level IS the point
        there, and independently-normalized curves would land on top of
        each other and hide the one thing worth seeing.
    y_fmt:
        Value formatter for y-axis tick labels when normalize=False.
        Defaults to `_fmt_gbp` (this report's most common y quantity);
        pass a different formatter for a non-£ axis (demand, a percentage).
    x_tick_labels:
        One label per data point (same length as each series), shown at a
        handful of evenly spaced ticks. Defaults to "Wk 1", "Wk 2", ... --
        pass real calendar labels, or spend-level labels for a response
        curve, to match what the x-axis actually represents.
    """
    n = len(next(iter(series.values())))
    if n < 2:
        return "<p><em>Not enough weeks to plot.</em></p>"
    fmt = y_fmt if y_fmt is not None else _fmt_gbp
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i: int) -> float:
        return pad_left + plot_w * i / (n - 1)

    if normalize:
        y_lo, y_hi = 0.0, 1.0
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

        def y_tick_fmt(v: float) -> str:
            return f"{v:.0%}"
    else:
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
        y_lo, y_hi, y_step = _nice_axis_bounds(
            float(all_vals.min()), float(all_vals.max())
        )
        y_ticks = list(np.arange(y_lo, y_hi + y_step / 2, y_step))
        y_tick_fmt = fmt

    def y_at(v: float) -> float:
        span = y_hi - y_lo if y_hi > y_lo else 1.0
        return pad_top + plot_h * (1 - (v - y_lo) / span)

    if x_tick_labels is None:
        x_tick_labels = [f"Wk {i + 1}" for i in range(n)]
    tick_idx = sorted(set(np.linspace(0, n - 1, min(n_x_ticks, n)).round().astype(int)))

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for v in y_ticks:
        y = y_at(v)
        is_zero_line = (not normalize) and y_lo < 0 < y_hi and abs(v) < 1e-9
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="{"#9ca3af" if is_zero_line else "#e5e7eb"}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'font-size="10" fill="#9ca3af">{y_tick_fmt(v)}</text>'
        )
    for i in tick_idx:
        x = x_at(i)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{pad_top + plot_h + 5:.1f}" stroke="#d1d5db" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 16:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#9ca3af">{html.escape(x_tick_labels[i])}</text>'
        )
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{pad_top + plot_h:.1f}" stroke="#d1d5db" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h:.1f}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h:.1f}" stroke="#d1d5db" stroke-width="1"/>'
    )
    if x_label:
        parts.append(
            f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle" '
            f'font-size="10" fill="#9ca3af">{html.escape(x_label)}</text>'
        )
    if y_label:
        cy = pad_top + plot_h / 2
        parts.append(
            f'<text x="12" y="{cy:.1f}" text-anchor="middle" font-size="10" fill="#9ca3af" '
            f'transform="rotate(-90 12 {cy:.1f})">{html.escape(y_label)}</text>'
        )

    for name, values in series.items():
        arr = np.asarray(values, dtype=float)
        if normalize:
            lo, hi = arr.min(), arr.max()
            span = hi - lo if hi > lo else 1.0
            points = " ".join(
                f"{x_at(i):.1f},{pad_top + plot_h * (1 - (v - lo) / span):.1f}"
                for i, v in enumerate(arr)
            )
        else:
            points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(arr))
        color = colors.get(name, "#111827")
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _corr_table_html(matrix: dict, channels: list[str]) -> str:
    """Static HTML table for a channel-by-channel correlation matrix, cells
    heat-shaded from the correlation value (no JS -- computed at render time,
    same reasoning as _svg_multiline)."""

    def cell_style(v: float) -> str:
        # White at 0, deepening red towards +1 (collinearity is the risk
        # this report cares about; negative correlation isn't the concern
        # here, so it gets a flat light shade rather than its own ramp).
        alpha = max(0.0, min(1.0, v))
        return f"background: rgba(220, 38, 38, {alpha * 0.65:.2f});"

    header = "".join(f"<th>{ch}</th>" for ch in channels)
    rows = []
    for a in channels:
        cells = "".join(
            f'<td style="{cell_style(matrix[a][b])}">{matrix[a][b]:.2f}</td>'
            for b in channels
        )
        rows.append(f"<tr><th>{a}</th>{cells}</tr>")
    return (
        f'<table class="corr-table"><thead><tr><th></th>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fmt_gbp(v: float) -> str:
    """£ formatting matching the JS fmtGBP() in docs/overview.html."""
    a = abs(v)
    if a < 1_000:
        return f"£{round(v):,.0f}"
    if a < 1_000_000:
        return f"£{round(v / 1_000):,.0f}k"
    return f"£{v / 1_000_000:.2f}m"


def _nice_tick_step(max_val: float, target_ticks: int = 6) -> float:
    """Round a chart's tick step up to a human "1/2/5 x 10^k" size, the way
    a plotting library would choose gridlines -- unlike docs/overview.html's
    charts, this report doesn't know its £ scale in advance (every user
    brings their own spend/marginal-return numbers), so the axis can't be
    hand-picked the way overview.html's fixed £m ticks are."""
    if max_val <= 0:
        return 1.0
    raw_step = max_val / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 5, 10):
        step = m * magnitude
        if step >= raw_step:
            return step
    return 10 * magnitude


def _nice_axis_bounds(
    lo: float, hi: float, target_ticks: int = 5
) -> tuple[float, float, float]:
    """Round an arbitrary [lo, hi] data range out to a human tick grid --
    same 1/2/5 x 10^k step selection as _nice_tick_step, but handling a
    negative lower bound too (needed for a standardised series like demand,
    which sits at mean 0 and can go either side of it)."""
    if hi <= lo:
        hi = lo + 1.0
    step = _nice_tick_step(hi - lo, target_ticks)
    nice_lo = math.floor(lo / step) * step
    nice_hi = math.ceil(hi / step) * step
    return nice_lo, nice_hi, step


def _hi(v: float | tuple[float, float]) -> float:
    """Upper end of a range mark, or the value itself for a point mark --
    lets _svg_forest's axis/label code treat both the same way."""
    return v[1] if isinstance(v, tuple) else v


def _svg_forest(
    data: list[dict],
    width: int = 700,
    row_h: int = 62,
    x_label: str = "Incremental revenue",
    fmt=_fmt_gbp,
) -> str:
    """Two-state horizontal chart: each channel gets a pale "before" mark
    and a solid "after" mark, plus a dashed line at the value implied by
    the truth -- the same visual language as docs/overview.html's
    chartProblem/chartImpact (JS there, static Python/SVG here since
    nothing in this report changes once the winner's picked, same
    reasoning as _svg_multiline).

    Each `data` entry: {name, color, before, after, truth: float | None}.
    `before`/`after` are each either a (p10, p90) tuple -- drawn as a
    rounded range bar, the variance section's shape -- or a single float
    -- drawn as a dot, for a mean-estimate quantity like the bias section
    that has no p10/p90 to show. The two states in one chart don't have to
    match shape (a point "before" against a range "after" renders fine),
    though every section built so far uses one shape throughout.

    Values are raw numbers -- tick step is picked at render time from
    whatever range this chart's own numbers span, via _nice_tick_step, and
    every axis/value label goes through `fmt` (defaults to £ formatting,
    the variance/bias sections' convention; pass a plain "{:.2f}".format
    for a non-£ quantity like the identifiability section's b/lambda).
    """
    m_top, m_right, m_bottom, m_left = 14, 96, 40, 100
    height = m_top + m_bottom + row_h * len(data)
    pw = width - m_left - m_right
    ph = height - m_top - m_bottom
    row = ph / len(data)

    raw_max = max(
        max(_hi(ch["before"]), _hi(ch["after"]), ch.get("truth") or 0.0) for ch in data
    )
    step = _nice_tick_step(raw_max)
    x_max = step * math.ceil(raw_max / step) if raw_max > 0 else step

    def sc_x(v: float) -> float:
        return m_left + (v / x_max) * pw

    def mark(
        v: float | tuple[float, float],
        cy: float,
        color: str,
        opacity: float | None = None,
    ) -> str:
        op = f' opacity="{opacity}"' if opacity is not None else ""
        if isinstance(v, tuple):
            lo, hi = v
            return (
                f'<line x1="{sc_x(lo):.1f}" y1="{cy:.1f}" x2="{sc_x(hi):.1f}" '
                f'y2="{cy:.1f}" stroke="{color}" stroke-width="5.5" '
                f'stroke-linecap="round"{op}/>'
            )
        return f'<circle cx="{sc_x(v):.1f}" cy="{cy:.1f}" r="5.5" fill="{color}"{op}/>'

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    for i in range(len(data)):
        if i % 2 == 1:
            y = m_top + row * i
            parts.append(
                f'<rect x="{m_left}" y="{y:.1f}" width="{pw}" height="{row:.1f}" '
                f'fill="#f9fafb"/>'
            )

    ticks = np.arange(0, x_max + step / 2, step)
    for v in ticks:
        x = sc_x(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="{m_top}" x2="{x:.1f}" y2="{m_top + ph:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{m_top + ph + 16:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#9ca3af">{fmt(v)}</text>'
        )
    parts.append(
        f'<text x="{m_left + pw / 2:.1f}" y="{m_top + ph + 30:.1f}" '
        f'text-anchor="middle" font-size="10" fill="#9ca3af">'
        f"{html.escape(x_label)}</text>"
    )
    parts.append(
        f'<rect x="{m_left}" y="{m_top}" width="{pw}" height="{ph:.1f}" '
        f'fill="none" stroke="#e5e7eb" stroke-width="1"/>'
    )

    for i, ch in enumerate(data):
        cy = m_top + row * i + row / 2
        step_y = row * 0.22
        cy_before, cy_after = cy - step_y, cy + step_y
        before, after = ch["before"], ch["after"]
        color = ch["color"]

        parts.append(mark(before, cy_before, color, opacity=0.35))
        parts.append(mark(after, cy_after, color))
        if ch.get("truth") is not None:
            tx = sc_x(ch["truth"])
            parts.append(
                f'<line x1="{tx:.1f}" y1="{cy_before - 5:.1f}" x2="{tx:.1f}" '
                f'y2="{cy_after + 5:.1f}" stroke="#111827" stroke-width="1.6" '
                f'stroke-dasharray="3,2"/>'
            )
        after_label = (
            f"{fmt(after[0])} &ndash; {fmt(after[1])}"
            if isinstance(after, tuple)
            else fmt(after)
        )
        parts.append(
            f'<text x="{sc_x(_hi(after)) + 8:.1f}" y="{cy_after + 3:.1f}" '
            f'font-size="11.5" fill="{color}" font-weight="700">'
            f"{after_label}</text>"
        )
        parts.append(
            f'<text x="{m_left - 10}" y="{cy + 4:.1f}" text-anchor="end" '
            f'font-size="12.5" font-weight="700" fill="{color}">'
            f"{html.escape(ch['name'])}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


def _svg_dotplot(
    data: list[dict],
    width: int = 700,
    row_h: int = 46,
    x_label: str = "",
    fmt=None,
) -> str:
    """Single-value-per-channel horizontal dot chart: one row per channel, a
    dot at that channel's value, against a value-axis with gridlines and
    tick labels -- the same axis language as _svg_forest, collapsed to one
    mark per row since this is a report INPUT (e.g. marginal return), not a
    before/after comparison.

    Each `data` entry: {name, color, value}.
    """
    fmt = fmt if fmt is not None else (lambda v: f"{v:.2f}")
    m_top, m_right, m_bottom, m_left = 14, 80, 40, 100
    height = m_top + m_bottom + row_h * len(data)
    pw = width - m_left - m_right
    ph = height - m_top - m_bottom
    row = ph / len(data)

    raw_max = max(ch["value"] for ch in data)
    step = _nice_tick_step(raw_max)
    x_max = step * math.ceil(raw_max / step) if raw_max > 0 else step

    def sc_x(v: float) -> float:
        return m_left + (v / x_max) * pw

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for i in range(len(data)):
        if i % 2 == 1:
            y = m_top + row * i
            parts.append(
                f'<rect x="{m_left}" y="{y:.1f}" width="{pw}" height="{row:.1f}" '
                f'fill="#f9fafb"/>'
            )

    ticks = np.arange(0, x_max + step / 2, step)
    for v in ticks:
        x = sc_x(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="{m_top}" x2="{x:.1f}" y2="{m_top + ph:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{m_top + ph + 16:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#9ca3af">{fmt(v)}</text>'
        )
    parts.append(
        f'<text x="{m_left + pw / 2:.1f}" y="{m_top + ph + 30:.1f}" '
        f'text-anchor="middle" font-size="10" fill="#9ca3af">'
        f"{html.escape(x_label)}</text>"
    )
    parts.append(
        f'<rect x="{m_left}" y="{m_top}" width="{pw}" height="{ph:.1f}" '
        f'fill="none" stroke="#e5e7eb" stroke-width="1"/>'
    )

    for i, ch in enumerate(data):
        cy = m_top + row * i + row / 2
        x = sc_x(ch["value"])
        color = ch["color"]
        parts.append(
            f'<line x1="{m_left}" y1="{cy:.1f}" x2="{x:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="3" opacity="0.35"/>'
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 10:.1f}" y="{cy + 4:.1f}" font-size="11.5" '
            f'fill="{color}" font-weight="700">{fmt(ch["value"])}</text>'
        )
        parts.append(
            f'<text x="{m_left - 10}" y="{cy + 4:.1f}" text-anchor="end" '
            f'font-size="12.5" font-weight="700" fill="{color}">'
            f"{html.escape(ch['name'])}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


class DiscoveryReport:
    """Sweep candidate phasing strategies and report their impact on all
    three reliability problems this package diagnoses.

    Parameters
    ----------
    history_df, plan_df:
        Same shape BudgetPhaser/CollinearityDiagnostic expect: weekly
        DatetimeIndex, one column per channel, matching columns in both.
    true_marginal_returns:
        Dict mapping channel to a plausible marginal return (see
        CollinearityDiagnostic's own docstring for why there's no safe
        universal default). Defaults to the package's illustrative
        0.5/1.0/1.5, which only matches history_df/plan_df named
        tv/meta/search.
    baseline_share, baseline_cv:
        Forwarded to calibrate_baseline() to derive base_sales and
        demand_coef from ROI + spend, rather than supplying either
        directly (see calibrate_baseline's own docstring). baseline_cv
        defaults to 0.05, NOT calibrate_baseline's own 0.0 -- a flat
        baseline creates no omitted-variable bias at all, which would
        make the bias section have nothing to show. Set it from your own
        MMM's baseline decomposition if you have one.
    demand_proxy_quality:
        The bias section's client-supplied plausible proxy quality, in
        (0, 1] -- forwarded to CollinearityDiagnostic's `controls` as a
        float (see that class's docstring: builds a measurement-error
        proxy at this quality from the report's own demand series).
        1.0 means an almost-perfect proxy (little bias left to show);
        lower values are more pessimistic about what you can actually
        control for in practice. Default 0.8.
    saturation, adstock:
        The plausible saturation exponent b in (0, 1] (1.0 = linear,
        default) and adstock decay lambda in [0, 1) (0.0 = no carryover,
        default) to assume PER CHANNEL -- either a dict covering every
        channel in history_df/plan_df, or a single float that broadcasts
        to every channel (the old shared-value behaviour). Each channel
        gets its own value because IdentifiabilityDiagnostic's grid search
        is now per-channel too: a channel's OWN spend pattern determines
        how well its OWN curvature can be pinned down, which differs by
        channel even when every channel shares the same assumed curvature
        (session 45, changed from the original single-shared-float design
        at Ryan's request -- see NOTES.md).
    revenue_noise_std:
        Forwarded to every internal CollinearityDiagnostic/IdentifiabilityDiagnostic.
        Set this from your own model's residual standard deviation -- see
        CollinearityDiagnostic's own docstring, same caveat applies here.
    levers:
        The candidate strategies to sweep, as a list of
        (label, per-channel max_weekly_deviation_pct spec, nudge_shape,
        balance_signs) tuples. Defaults to notebooks/11's own LEVERS list
        (unphased, +/-80% uniform, +/-40% and +/-80% edge+balanced,
        Blackout) -- see _default_levers. The first entry is expected to
        be the unphased baseline every other candidate is compared
        against; pass your own list to add or narrow candidates, keeping
        an unphased baseline entry first.
    client_name, plan_year:
        Free-text labels shown on the report cover, same convention as
        ReportBuilder.
    seed:
        Base random seed for demand and phasing draws.
    demand_process:
        Forwarded to simulate_demand() for the report's one shared demand
        series. Default "white_noise", matching the package's other
        defaults.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        true_marginal_returns: dict[str, float] | None = None,
        baseline_share: float = 0.7,
        baseline_cv: float = 0.05,
        demand_proxy_quality: float = 0.8,
        saturation: dict[str, float] | float = 1.0,
        adstock: dict[str, float] | float = 0.0,
        revenue_noise_std: float = 26_000.0,
        levers: list[tuple[str, dict, str, bool]] | None = None,
        client_name: str = "",
        plan_year: str = "",
        seed: int = 0,
        demand_process: str = "white_noise",
    ) -> None:
        if list(history_df.columns) != list(plan_df.columns):
            raise ValueError(
                "history_df and plan_df must have the same columns. "
                f"Got {list(history_df.columns)} vs {list(plan_df.columns)}."
            )
        if not 0.0 < demand_proxy_quality <= 1.0:
            raise ValueError("demand_proxy_quality must be in (0, 1]")

        _get_month_labels(plan_df)  # validates DatetimeIndex, fails fast

        self.history_df = history_df
        self.plan_df = plan_df
        self.true_marginal_returns = (
            true_marginal_returns
            if true_marginal_returns is not None
            else _DEFAULT_MARGINAL_RETURNS
        )
        self.baseline_share = baseline_share
        self.baseline_cv = baseline_cv
        self.demand_proxy_quality = demand_proxy_quality
        self.revenue_noise_std = revenue_noise_std
        self.client_name = client_name
        self.plan_year = plan_year
        self.seed = seed
        self.demand_process = demand_process

        self.channels_ = list(plan_df.columns)
        # Per-channel from here on (a bare float broadcasts) -- see
        # IdentifiabilityDiagnostic's per-channel profiling, the reason
        # this stopped being a single shared value.
        self.saturation = _per_channel_map(
            saturation,
            self.channels_,
            "saturation",
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
        )
        self.adstock = _per_channel_map(
            adstock,
            self.channels_,
            "adstock",
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=False,
        )
        self.levers_ = levers if levers is not None else _default_levers(self.channels_)
        self._plan_month_labels = _get_month_labels(plan_df)

        # Fixed across every candidate so every strategy is priced against
        # the SAME response curve -- same reasoning as BudgetPhaser's own
        # reference_spend default (see that class's docstring).
        self.reference_spend_ = {ch: float(plan_df[ch].mean()) for ch in self.channels_}

        # Total planned spend per channel over the plan period -- distinct
        # from reference_spend_ (mean weekly spend, anchors the saturation
        # curve). This is what "planned_spend" means to
        # CollinearityDiagnostic.summary()'s incremental-revenue range:
        # revenue = marginal-return draw x total spend for the period, the
        # same convention docs/overview.html's forest-plot charts use.
        self.planned_spend_ = {ch: float(plan_df[ch].sum()) for ch in self.channels_}

        self.calibration_ = calibrate_baseline(
            pd.concat([history_df, plan_df]),
            self.true_marginal_returns,
            baseline_share=baseline_share,
            baseline_cv=baseline_cv,
        )

        n_total = len(history_df) + len(plan_df)
        self.demand_ = simulate_demand(n_total, process=demand_process, seed=seed)

        self.results_: dict[str, dict] | None = None
        self.winner_: str | None = None
        self.winner_schedule_: pd.DataFrame | None = None
        self.report_data_: dict | None = None

    def _phase(
        self, spec: dict, nudge_shape: str, balance_signs: bool, seed: int
    ) -> pd.DataFrame:
        if _is_unphased(spec):
            return self.plan_df
        return _generate_phased_schedule(
            self.plan_df,
            self._plan_month_labels,
            alpha=1.0,
            max_weekly_deviation_pct=spec,
            seed=seed,
            nudge_shape=nudge_shape,
            balance_signs=balance_signs,
        )

    def fit(
        self,
        n_sims: int = 50,
        n_phasing_seeds: int = 5,
        id_n_sims: int = 20,
        id_b_candidates: np.ndarray | None = None,
        id_lam_candidates: np.ndarray | None = None,
        valley_tol: float = 0.01,
        proxy_seed: int = 0,
        fast_mode: bool = False,
    ) -> DiscoveryReport:
        """Sweep every candidate lever, run the three diagnostics on each,
        and pick the report-wide "highest impact" winner.

        Parameters
        ----------
        n_sims:
            Noise draws per CollinearityDiagnostic fit (variance and bias
            sections). Default 50.
        n_phasing_seeds:
            Independent phased-schedule draws averaged per lever (the
            unphased baseline always uses exactly 1 -- there's nothing
            random to average over). Default 5.
        id_n_sims:
            Noise draws per IdentifiabilityDiagnostic fit. Kept separate
            from n_sims and smaller by default -- profiling a (b, lambda)
            grid is far more expensive per draw than a single OLS fit.
            Default 20.
        id_b_candidates, id_lam_candidates:
            Forwarded to IdentifiabilityDiagnostic. Defaults to that
            class's own defaults (a wide 33x31 grid); shrink for a fast
            structural check.
        valley_tol:
            Forwarded to IdentifiabilityDiagnostic.summary()'s valley_pct.
        proxy_seed:
            Forwarded to CollinearityDiagnostic's bias-section fit as
            `proxy_seed` -- the demand-proxy draw's own seed.
        fast_mode:
            If True, uses cheap settings throughout (n_sims=10,
            n_phasing_seeds=2, id_n_sims=5) -- for iterating on the report
            itself, not for numbers to hand a client. to_html() watermarks
            a fast-mode report as a draft.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            n_phasing_seeds = 2
            id_n_sims = 5

        results: dict[str, dict] = {}
        schedules: dict[str, pd.DataFrame] = {}

        for label, spec, nudge_shape, balance_signs in self.levers_:
            unphased = _is_unphased(spec)
            seeds = (
                [self.seed]
                if unphased
                else [self.seed + j for j in range(n_phasing_seeds)]
            )

            variance_draws = []
            revenue_p10_draws = []
            revenue_p90_draws = []
            bias_draws = []
            id_draws = []
            id_b_p10_draws = []
            id_b_p90_draws = []
            id_lam_p10_draws = []
            id_lam_p90_draws = []
            corr_draws = []
            representative_schedule = None

            for j, sd in enumerate(seeds):
                phased_plan = self._phase(spec, nudge_shape, balance_signs, sd)
                if j == 0:
                    representative_schedule = phased_plan
                combined = pd.concat([self.history_df, phased_plan])

                diag_var = CollinearityDiagnostic(
                    spend_df=combined,
                    true_marginal_returns=self.true_marginal_returns,
                    base_sales=self.calibration_.baseline_level,
                    revenue_noise_std=self.revenue_noise_std,
                    demand=self.demand_,
                    demand_coef=self.calibration_.demand_coef,
                    saturation=self.saturation,
                    adstock=self.adstock,
                    reference_spend=self.reference_spend_,
                )
                diag_var.fit(n_sims=n_sims, controls=True)
                var_summary = diag_var.summary(planned_spend=phased_plan).set_index(
                    "channel"
                )
                variance_draws.append(var_summary["coef_of_variation"])
                revenue_p10_draws.append(var_summary["incremental_revenue_p10"])
                revenue_p90_draws.append(var_summary["incremental_revenue_p90"])
                corr_draws.append(diag_var.correlation_matrix)

                diag_bias = CollinearityDiagnostic(
                    spend_df=combined,
                    true_marginal_returns=self.true_marginal_returns,
                    base_sales=self.calibration_.baseline_level,
                    revenue_noise_std=self.revenue_noise_std,
                    demand=self.demand_,
                    demand_coef=self.calibration_.demand_coef,
                    saturation=self.saturation,
                    adstock=self.adstock,
                    reference_spend=self.reference_spend_,
                )
                diag_bias.fit(
                    n_sims=n_sims,
                    controls=self.demand_proxy_quality,
                    proxy_seed=proxy_seed,
                )
                bias_summary = diag_bias.summary().set_index("channel")
                bias_draws.append(bias_summary["mean_error_pct"])

                diag_id = IdentifiabilityDiagnostic(
                    spend_df=combined,
                    demand=self.demand_,
                    true_marginal_returns=self.true_marginal_returns,
                    true_saturation=self.saturation,
                    true_adstock=self.adstock,
                    demand_coef=self.calibration_.demand_coef,
                    base_sales=self.calibration_.baseline_level,
                    revenue_noise_std=self.revenue_noise_std,
                    reference_spend=self.reference_spend_,
                    b_candidates=id_b_candidates,
                    lam_candidates=id_lam_candidates,
                )
                diag_id.fit(n_sims=id_n_sims, noise_seed_offset=sd)
                id_draws.append(diag_id.summary(tol=valley_tol).set_index("channel"))
                # p10-p90 of each channel's own recovered b/lambda across
                # sims -- the range chart's real data (the summary's b_sd
                # is a spread too, but a p10-p90 range plots directly on
                # the same before/after forest chart sections 2/3 use).
                id_b_p10_draws.append(
                    pd.Series(
                        {
                            ch: diag_id.results_[ch]["recovered_b"].quantile(0.1)
                            for ch in self.channels_
                        }
                    )
                )
                id_b_p90_draws.append(
                    pd.Series(
                        {
                            ch: diag_id.results_[ch]["recovered_b"].quantile(0.9)
                            for ch in self.channels_
                        }
                    )
                )
                id_lam_p10_draws.append(
                    pd.Series(
                        {
                            ch: diag_id.results_[ch]["recovered_lam"].quantile(0.1)
                            for ch in self.channels_
                        }
                    )
                )
                id_lam_p90_draws.append(
                    pd.Series(
                        {
                            ch: diag_id.results_[ch]["recovered_lam"].quantile(0.9)
                            for ch in self.channels_
                        }
                    )
                )

            variance_cv = pd.concat(variance_draws, axis=1).mean(axis=1)
            revenue_p10 = pd.concat(revenue_p10_draws, axis=1).mean(axis=1)
            revenue_p90 = pd.concat(revenue_p90_draws, axis=1).mean(axis=1)
            bias_pct = pd.concat(bias_draws, axis=1).mean(axis=1)
            # Per-channel DataFrame (b_mean, b_sd, ..., valley_pct), each
            # draw already indexed by channel -- average across draws
            # channel-by-channel, column-by-column.
            identifiability = pd.concat(id_draws).groupby(level=0).mean()
            id_b_p10 = pd.concat(id_b_p10_draws, axis=1).mean(axis=1)
            id_b_p90 = pd.concat(id_b_p90_draws, axis=1).mean(axis=1)
            id_lam_p10 = pd.concat(id_lam_p10_draws, axis=1).mean(axis=1)
            id_lam_p90 = pd.concat(id_lam_p90_draws, axis=1).mean(axis=1)
            correlation = {
                a: {
                    b: float(np.mean([m.loc[a, b] for m in corr_draws]))
                    for b in self.channels_
                }
                for a in self.channels_
            }

            results[label] = {
                "variance_cv": variance_cv.to_dict(),
                "revenue_p10": revenue_p10.to_dict(),
                "revenue_p90": revenue_p90.to_dict(),
                "bias_pct": bias_pct.to_dict(),
                "identifiability": identifiability.to_dict(orient="index"),
                "b_p10": id_b_p10.to_dict(),
                "b_p90": id_b_p90.to_dict(),
                "lam_p10": id_lam_p10.to_dict(),
                "lam_p90": id_lam_p90.to_dict(),
                "correlation": correlation,
                "scores": {
                    "variance": float(variance_cv.mean()),
                    "bias": float(bias_pct.abs().mean()),
                    "identifiability": float(identifiability["valley_pct"].mean()),
                },
            }
            schedules[label] = representative_schedule

        self.results_ = results
        self.valley_tol_ = valley_tol
        self.winner_ = self._pick_winner(results)
        self.winner_schedule_ = schedules[self.winner_]
        self.report_data_ = self._build_report_data(fast_mode=fast_mode)
        return self

    def _pick_winner(self, results: dict[str, dict]) -> str:
        """Report-wide "highest impact" strategy: dominance check, else
        worst-axis (session 45 architecture decision -- see NOTES.md).

        Candidates are every lever except the unphased baseline (the first
        entry in self.levers_, by convention -- there is nothing to pick
        an unphased baseline "over"). Each candidate's improvement over
        unphased is computed on three axes (variance, bias,
        identifiability); a candidate that has the single best improvement
        on ALL THREE axes wins outright, else the candidate with the best
        WORST-axis improvement wins -- so a strategy can't win by tanking
        one problem to excel at the others.
        """
        baseline_label = self.levers_[0][0]
        baseline = results[baseline_label]["scores"]
        candidates = [label for label, *_ in self.levers_ if label != baseline_label]
        if not candidates:
            return baseline_label

        improvements = {}
        for label in candidates:
            scores = results[label]["scores"]
            improvements[label] = {
                "variance": _safe_improvement(baseline["variance"], scores["variance"]),
                "bias": _safe_improvement(baseline["bias"], scores["bias"]),
                "identifiability": _safe_improvement(
                    baseline["identifiability"], scores["identifiability"]
                ),
            }

        axes = ["variance", "bias", "identifiability"]
        best_per_axis = {
            axis: max(candidates, key=lambda lbl: improvements[lbl][axis])
            for axis in axes
        }
        dominant = (
            {best_per_axis[axes[0]]}
            & {best_per_axis[axes[1]]}
            & {best_per_axis[axes[2]]}
        )
        if dominant:
            return next(iter(dominant))

        return max(candidates, key=lambda lbl: min(improvements[lbl].values()))

    def _build_report_data(self, fast_mode: bool) -> dict:
        baseline_label = self.levers_[0][0]
        return {
            "meta": {
                "client_name": self.client_name,
                "plan_year": self.plan_year,
                "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
                "channels": self.channels_,
                "n_weeks_history": len(self.history_df),
                "n_weeks_plan": len(self.plan_df),
                "true_marginal_returns": self.true_marginal_returns,
                "demand_proxy_quality": self.demand_proxy_quality,
                "saturation": self.saturation,
                "adstock": self.adstock,
                "baseline_share": self.baseline_share,
                "baseline_cv": self.baseline_cv,
                "winner": self.winner_,
                "baseline_label": baseline_label,
                "fast_mode": fast_mode,
            },
            "levers": self.results_,
        }

    def summary(self) -> pd.DataFrame:
        """Cross-strategy table: one row per candidate lever, mean variance
        CV, mean |bias| %, and identifiability valley_pct.

        Raises
        ------
        RuntimeError if fit() hasn't been called yet.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")
        rows = []
        for label, *_ in self.levers_:
            scores = self.results_[label]["scores"]
            rows.append(
                {
                    "lever": label,
                    "mean_variance_cv": round(scores["variance"], 4),
                    "mean_abs_bias_pct": round(scores["bias"], 4),
                    "identifiability_valley_pct": round(scores["identifiability"], 4),
                    "is_winner": label == self.winner_,
                }
            )
        return pd.DataFrame(rows)

    def to_html(self, path: str | None = None) -> str:
        """Render the report as a single self-contained HTML document.

        Parameters
        ----------
        path:
            If given, also writes the HTML to this file path.

        Returns
        -------
        The rendered HTML as a string.
        """
        if self.report_data_ is None:
            raise RuntimeError("Call fit() before to_html().")

        html = _render_html(self)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def _render_html(report: DiscoveryReport) -> str:
    meta = report.report_data_["meta"]
    channels = report.channels_
    colors = _channel_colors(channels)
    legend = "".join(
        f'<div class="li"><span class="sw" style="background:{colors[ch]}"></span>{ch}</div>'
        for ch in channels
    )
    baseline_label = meta["baseline_label"]
    winner = meta["winner"]
    baseline = report.results_[baseline_label]
    best = report.results_[winner]

    draft_banner = (
        '<div class="draft-banner">DRAFT -- fast_mode was used, numbers are '
        "for iterating on the report, not for a client.</div>"
        if meta["fast_mode"]
        else ""
    )

    # Actual curved revenue over the plan period, per channel -- the
    # dashed "truth" reference on the variance/bias forest charts. Uses
    # the unphased plan (history + plan_df as given) since the true
    # revenue a plan generates doesn't depend on which lever is being
    # scored against it. channel_contributions is the same per-week
    # formula simulate_sales uses internally to generate the truth,
    # exposed directly -- a notebook could call this line for line.
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

    # Variance section: incremental-revenue forest chart (before/after,
    # £ p10-p90 range per channel) rather than a raw CV bar -- CV is the
    # metric the diagnostic optimizes, but £ revenue range is the number a
    # client actually feels, and matches docs/overview.html's own framing
    # of this same problem.
    variance_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": (baseline["revenue_p10"][ch], baseline["revenue_p90"][ch]),
            "after": (best["revenue_p10"][ch], best["revenue_p90"][ch]),
            "truth": true_revenue[ch],
        }
        for ch in channels
    ]
    variance_svg = _svg_forest(variance_forest_data)

    def _range_width(bounds: tuple[float, float]) -> float:
        return bounds[1] - bounds[0]

    variance_narrowing = {
        ch: _safe_improvement(
            _range_width((baseline["revenue_p10"][ch], baseline["revenue_p90"][ch])),
            _range_width((best["revenue_p10"][ch], best["revenue_p90"][ch])),
        )
        for ch in channels
    }
    variance_narrowing_text = ", ".join(
        f"{ch} {variance_narrowing[ch]:.0%}" for ch in channels
    )

    # Bias section: same chart family as variance (£ revenue, dashed true
    # line) but a point per state, not a range -- bias_pct is a mean error
    # averaged across draws, there's no p10/p90 to show. "Believed revenue"
    # is what a client would think they got if they trusted the biased
    # estimate: true revenue inflated/deflated by that mean error %.
    def _believed_revenue(ch: str, results: dict) -> float:
        return true_revenue[ch] * (1.0 + results["bias_pct"][ch] / 100.0)

    bias_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": _believed_revenue(ch, baseline),
            "after": _believed_revenue(ch, best),
            "truth": true_revenue[ch],
        }
        for ch in channels
    ]
    bias_svg = _svg_forest(bias_forest_data)
    bias_narrowing_text = ", ".join(
        f"{ch} {abs(baseline['bias_pct'][ch]):.0f}% &rarr; {abs(best['bias_pct'][ch]):.0f}%"
        for ch in channels
    )

    # Identifiability section: same forest-chart family as sections 2/3,
    # not the abandoned RSS "valley" picture (that showed the RSS surface
    # itself, which read as confusing rather than illuminating -- see
    # NOTES.md). What the client actually wants to know: given a plausible
    # saturation/adstock per channel, how wide is the range of estimates
    # recovered when fitting unphased vs {winner}? One chart per parameter,
    # one row per channel -- IdentifiabilityDiagnostic profiles each
    # channel's own curvature separately, holding every other channel at
    # its own true value (session 45's per-channel redesign).
    def _fmt_plain(v: float) -> str:
        return f"{v:.2f}"

    b_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": (baseline["b_p10"][ch], baseline["b_p90"][ch]),
            "after": (best["b_p10"][ch], best["b_p90"][ch]),
            "truth": report.saturation[ch],
        }
        for ch in channels
    ]
    b_svg = _svg_forest(
        b_forest_data, x_label="Saturation exponent (b)", fmt=_fmt_plain
    )

    lam_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": (baseline["lam_p10"][ch], baseline["lam_p90"][ch]),
            "after": (best["lam_p10"][ch], best["lam_p90"][ch]),
            "truth": report.adstock[ch],
        }
        for ch in channels
    ]
    lam_svg = _svg_forest(
        lam_forest_data, x_label="Adstock decay (lambda)", fmt=_fmt_plain
    )

    id_valley_before = baseline["scores"]["identifiability"]
    id_valley_after = best["scores"]["identifiability"]
    tol_pct = report.valley_tol_ * 100
    b_narrowing = {
        ch: _safe_improvement(
            _range_width((baseline["b_p10"][ch], baseline["b_p90"][ch])),
            _range_width((best["b_p10"][ch], best["b_p90"][ch])),
        )
        for ch in channels
    }
    lam_narrowing = {
        ch: _safe_improvement(
            _range_width((baseline["lam_p10"][ch], baseline["lam_p90"][ch])),
            _range_width((best["lam_p10"][ch], best["lam_p90"][ch])),
        )
        for ch in channels
    }
    b_narrowing_text = ", ".join(f"{ch} {b_narrowing[ch]:.0%}" for ch in channels)
    lam_narrowing_text = ", ".join(f"{ch} {lam_narrowing[ch]:.0%}" for ch in channels)

    corr_before_html = _corr_table_html(baseline["correlation"], channels)
    corr_after_html = _corr_table_html(best["correlation"], channels)

    # Cross-strategy table + dropdown -- the only part of the page driven
    # by JS, since every other section is fixed once the winner is picked.
    # Rows carry the lever label in a data- attribute (not an id -- several
    # rows share a lever, one per channel, so an id would collide) and the
    # JS below matches on it directly rather than via a CSS selector, which
    # sidesteps having to CSS-escape labels containing "+/-", "%", "()".
    lever_labels = [label for label, *_ in report.levers_]
    options_html = "".join(
        f'<option value="{html.escape(lbl)}"{" selected" if lbl == winner else ""}>'
        f"{html.escape(lbl)}{' (recommended)' if lbl == winner else ''}</option>"
        for lbl in lever_labels
    )
    table_rows_html = "".join(
        f'<tr data-lever="{html.escape(lbl)}"><td>{html.escape(ch)}</td>'
        f'<td class="v-var">{100 * report.results_[lbl]["variance_cv"][ch]:.1f}%</td>'
        f'<td class="v-bias">{report.results_[lbl]["bias_pct"][ch]:.1f}%</td></tr>'
        for lbl in lever_labels
        for ch in channels
    )

    # Appendix: everything the client supplied as an input to this report,
    # in the order they'd recognise supplying it -- marginal return,
    # adstock, saturation (each per channel), the actual spend series, and
    # the one shared latent demand series. Every chart below reads straight
    # off report's own public attributes -- nothing here is re-derived.
    week_labels = [d.strftime("%b '%y") for d in combined_baseline.index]

    mroi_data = [
        {"name": ch, "color": colors[ch], "value": report.true_marginal_returns[ch]}
        for ch in channels
    ]
    mroi_svg = _svg_dotplot(
        mroi_data,
        x_label="Marginal return (£ revenue per £1 of spend)",
        fmt=lambda v: f"£{v:.2f}",
    )

    adstock_values_text = ", ".join(
        f"{ch} {meta['adstock'][ch]:.2f}" for ch in channels
    )
    if any(lam != 0.0 for lam in meta["adstock"].values()):
        max_lag = min(
            52,
            max(
                [1]
                + [
                    int(np.ceil(np.log(0.02) / np.log(lam)))
                    for lam in meta["adstock"].values()
                    if lam > 0
                ]
            ),
        )
        impulse = np.zeros(max_lag + 1)
        impulse[0] = 1.0
        lag_curves = {
            ch: apply_adstock(impulse, meta["adstock"][ch]) for ch in channels
        }
        adstock_svg = _svg_multiline(
            lag_curves,
            colors,
            normalize=False,
            y_fmt=lambda v: f"{v:.0%}",
            y_label="Share of effect remaining",
            x_label="Weeks after spend",
            x_tick_labels=[f"+{i}" for i in range(max_lag + 1)],
        )
        adstock_fig_html = f"""
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Adstock, by channel</div>
      <div class="fig-sub">Share of a week's spend still driving sales, by weeks since it ran &middot; decay: {adstock_values_text}</div>
    </div>
    <div class="fig-body">
      <div class="legend">{legend}</div>
      {adstock_svg}
    </div>
  </div>"""
    else:
        adstock_fig_html = (
            "<p><em>Adstock was assumed instantaneous (decay = 0) for "
            "every channel in this report -- no carryover to plot.</em></p>"
        )

    saturation_values_text = ", ".join(
        f"{ch} {meta['saturation'][ch]:.2f}" for ch in channels
    )
    if any(b != 1.0 for b in meta["saturation"].values()):
        xs = np.linspace(0, max(report.reference_spend_.values()) * 1.5, 60)
        curves = {}
        for ch in channels:
            b = meta["saturation"][ch]
            x0 = report.reference_spend_[ch]
            mr0 = report.true_marginal_returns[ch]
            k = mr0 / (b * x0 ** (b - 1.0)) if x0 > 0 else 0.0
            curves[ch] = k * xs**b
        saturation_svg = _svg_multiline(
            curves,
            colors,
            normalize=False,
            y_label="Weekly revenue",
            x_label="Weekly spend",
            x_tick_labels=[_fmt_gbp(x) for x in xs],
        )
        saturation_fig_html = f"""
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Saturation, by channel</div>
      <div class="fig-sub">Assumed response curve, spend to revenue &middot; exponent (b): {saturation_values_text}</div>
    </div>
    <div class="fig-body">
      <div class="legend">{legend}</div>
      {saturation_svg}
    </div>
  </div>"""
    else:
        saturation_fig_html = (
            "<p><em>Saturation was assumed linear (b = 1.0) for every "
            "channel in this report -- no curvature to plot.</em></p>"
        )

    spend_series = {ch: combined_baseline[ch].to_numpy() for ch in channels}
    appendix_spend_svg = _svg_multiline(
        spend_series,
        colors,
        normalize=False,
        y_label="Weekly spend",
        x_label="Week",
        x_tick_labels=week_labels,
    )

    demand_series_svg = _svg_multiline(
        {"demand": report.demand_},
        {"demand": "#111827"},
        normalize=False,
        y_fmt=lambda v: f"{v:.1f}",
        y_label="Demand (standardised)",
        x_label="Week",
        x_tick_labels=week_labels,
    )

    weekly_sales = (
        report.calibration_.baseline_level
        + report.calibration_.demand_coef * report.demand_
        + true_contributions.sum(axis=1).to_numpy()
    )
    sales_series_svg = _svg_multiline(
        {"sales": weekly_sales},
        {"sales": "#111827"},
        normalize=False,
        y_label="Weekly sales / revenue",
        x_label="Week",
        x_tick_labels=week_labels,
    )

    def stat_box(label: str, value: str) -> str:
        return (
            f'<div class="meta-box"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MMM Discovery Report</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">
{draft_banner}
<header class="cover">
  <div class="cover-top">
    <div class="brand">how_wrong_is_your_mmm <span>Discovery Report</span></div>
    <div class="gen-date">Generated {meta["generated"]}</div>
  </div>
  <div class="report-title-eyebrow">Discovery</div>
  <h1 class="report-title">Which phasing strategy is worth pursuing?</h1>
  <p class="report-sub">One strategy applied identically to every channel, swept
  across {len(lever_labels)} candidates and scored on the three reliability
  problems this package diagnoses: ROI-interval variance, omitted-variable
  bias, and saturation/adstock identifiability.</p>
  <div class="cover-meta">
    {stat_box("Client", meta["client_name"] or "not set")}
    {stat_box("Plan year", meta["plan_year"] or "not set")}
    {stat_box("Plan weeks", str(meta["n_weeks_plan"]))}
    {stat_box("Recommended", winner)}
  </div>
</header>

<div class="headline">Recommended strategy: <b>{winner}</b> &mdash; the
candidate that most improves on the unphased plan without tanking any of
the three problems below (dominance check, else worst-axis).</div>

<main>

<section>
  <div class="s-label">Section 1</div>
  <h2>Spend correlation, before vs. after</h2>
  <p>How entangled each channel's spend is with every other channel's, across
  history + plan. The more correlated a pair, the harder it is for a model to
  tell their individual contributions apart -- phasing under <b>{winner}</b>
  is the only change made between the two matrices below.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Channel correlation, before vs. after</div>
      <div class="fig-sub">Pearson correlation, weekly spend by channel &middot; plan year only, monthly totals identical in both</div>
    </div>
    <div class="fig-body corr-cols">
      <div class="corr-col">
        <div class="corr-col-hdr">Before phasing</div>
        {corr_before_html}
      </div>
      <div class="corr-col">
        <div class="corr-col-hdr">After phasing</div>
        {corr_after_html}
      </div>
    </div>
    <p class="fig-cap">Monthly totals are identical on both sides -- only the
    within-month weekly pattern changes, which is what breaks the
    collinearity.</p>
  </div>
</section>

<section>
  <div class="s-label">Section 2</div>
  <h2>The variance problem</h2>
  <p><b>The problem:</b> spend is locked to a single plan, so channels move
  together and the model can't unpick which one actually earned the
  result.</p>
  <p><b>The fix:</b> every candidate phasing strategy was swept against
  this same history and plan, and <b>{winner}</b> won.</p>
  <p><b>The impact:</b> incremental-revenue ranges tighten by
  {variance_narrowing_text}.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">The range tightens</div>
      <div class="fig-sub">Incremental revenue, the model-estimated range: unphased vs {
        winner
    }</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today)</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Revenue at the true marginal return</span>
      </div>
      {variance_svg}
    </div>
    <p class="fig-cap">The dashed line marks the revenue implied by the
    true marginal return -- the centre barely moves, because the centre
    was never the problem. What changes is the width.</p>
  </div>
</section>

<section>
  <div class="s-label">Section 3</div>
  <h2>The bias problem</h2>
  <p><b>The problem:</b> even once phasing fixes the collinearity, demand
  is never measured perfectly -- working from a proxy of quality
  {meta["demand_proxy_quality"]:.0%} (not the true series) still pulls the
  model's estimate off the true marginal return.</p>
  <p><b>The fix:</b> the same phasing strategy was scored on this bias too,
  and <b>{winner}</b> won here as well.</p>
  <p><b>The impact:</b> mean estimation error narrows: {bias_narrowing_text}.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">The estimate moves toward the truth</div>
      <div class="fig-sub">Revenue implied by the biased estimate: unphased vs {
        winner
    }</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#9ca3af" opacity="0.35"/></svg> Believed today (unphased)</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#9ca3af"/></svg> Believed, {
        html.escape(winner)
    }</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> True revenue</span>
      </div>
      {bias_svg}
    </div>
    <p class="fig-cap">"Believed" is the revenue a client would expect if
    they trusted the biased estimate -- the dot moves toward the dashed
    true-revenue line as the proxy's remaining confound shrinks.</p>
  </div>
</section>

<section>
  <div class="s-label">Section 4</div>
  <h2>The identifiability problem</h2>
  <p><b>The problem:</b> the client supplies a plausible saturation and
  adstock per channel, but with demand known and the spend pattern locked
  to a single plan, many other curvature values fit the data about equally
  well -- so what the model recovers can range far from that plausible
  value.</p>
  <p><b>The fix:</b> the same phasing strategy was scored on this too, and
  <b>{winner}</b> won here as well.</p>
  <p><b>The impact:</b> saturation ranges narrow by {b_narrowing_text};
  adstock ranges narrow by {lam_narrowing_text}. The RSS valley shrinks from
  {id_valley_before:.0f}% to {id_valley_after:.0f}% of the (b, lambda) grid
  within {tol_pct:.0f}% of the best fit, averaged across channels.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Saturation range tightens, by channel</div>
      <div class="fig-sub">Recovered saturation exponent (b): unphased vs {winner}</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today)</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {b_svg}
    </div>
    <p class="fig-cap">Each row is the p10&ndash;p90 range of that
    channel's OWN recovered saturation across sims, holding every other
    channel at its own supplied curvature -- a wide range means this
    channel's spend pattern doesn't pin down HOW MUCH it saturates.</p>
  </div>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Adstock range tightens, by channel</div>
      <div class="fig-sub">Recovered adstock decay (lambda): unphased vs {winner}</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today)</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {lam_svg}
    </div>
    <p class="fig-cap">Same idea, for how long each channel's effect carries
    over -- a wide range means this channel's spend pattern doesn't pin down
    HOW LONG the effect lasts.</p>
  </div>
</section>

<section>
  <div class="s-label">Section 5</div>
  <h2>Every candidate strategy</h2>
  <p>The table above defaults to {winner}, the report's recommended
  strategy. Use the dropdown to see any other candidate's numbers -- the
  best strategy on your own data may not match the winner here.</p>
  <label for="lever-select"><b>Strategy:</b></label>
  <select id="lever-select">{options_html}</select>
  <table class="cross-table">
    <thead><tr><th>Channel</th><th>Variance CV</th><th>Bias %</th></tr></thead>
    <tbody id="cross-table-body"></tbody>
  </table>
  <table class="hidden" id="cross-table-source">{table_rows_html}</table>
</section>

<section>
  <div class="s-label">Section 6</div>
  <h2>Appendix: what you supplied</h2>
  <p>Every input this report is built on, laid out plainly -- so it can be
  checked against what you actually told us, and so anyone could reproduce
  every chart above from these inputs alone, in a notebook, without this
  report class.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Channel marginal return</div>
      <div class="fig-sub">The plausible ROI supplied per channel -- ground truth throughout this report</div>
    </div>
    <div class="fig-body">{mroi_svg}</div>
  </div>

  {adstock_fig_html}

  {saturation_fig_html}

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Spend, history + plan</div>
      <div class="fig-sub">Actual weekly spend by channel, as supplied -- before any phasing</div>
    </div>
    <div class="fig-body">
      <div class="legend">{legend}</div>
      {appendix_spend_svg}
    </div>
  </div>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Demand</div>
      <div class="fig-sub">The one latent demand series driving every simulated sales column in this report</div>
    </div>
    <div class="fig-body">{demand_series_svg}</div>
  </div>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Sales / revenue, weekly</div>
      <div class="fig-sub">Baseline + demand + channel contributions, no noise -- what these inputs imply</div>
    </div>
    <div class="fig-body">{sales_series_svg}</div>
  </div>
</section>

</main>
</div>
<script>
(function () {{
  var source = document.getElementById('cross-table-source');
  var body = document.getElementById('cross-table-body');
  var select = document.getElementById('lever-select');
  var allRows = Array.prototype.slice.call(source.querySelectorAll('tr[data-lever]'));
  function render(lever) {{
    body.innerHTML = '';
    allRows
      .filter(function (r) {{ return r.getAttribute('data-lever') === lever; }})
      .forEach(function (r) {{ body.appendChild(r.cloneNode(true)); }});
  }}
  select.addEventListener('change', function () {{ render(this.value); }});
  render(select.value);
}})();
</script>
</body>
</html>
"""


_CSS = r"""
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --text: #111827; --muted: #6b7280; --light: #d1d5db; --border: #e5e7eb;
  --bg: #f9fafb; --good: #059669; --w: 760px;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  color: var(--text); background: #f3f4f6; line-height: 1.7; font-size: 17px;
}
.page { max-width: var(--w); margin: 0 auto; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.draft-banner { background: #fef3c7; color: #92400e; text-align: center; padding: .5rem; font-weight: 700; font-size: .85rem; }
.cover { padding: 2.75rem 2rem 2.25rem; border-bottom: 3px solid var(--text); }
.cover-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem; }
.brand { font-size: .8rem; font-weight: 700; letter-spacing: .02em; }
.brand span { color: var(--muted); font-weight: 500; }
.gen-date { font-size: .78rem; color: var(--muted); text-align: right; }
.report-title-eyebrow { font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: .6rem; }
.report-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; margin-bottom: .5rem; }
.report-sub { color: var(--muted); font-size: 1.02rem; max-width: 56ch; }
.cover-meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin-top: 2rem; }
@media (max-width: 620px) { .cover-meta { grid-template-columns: repeat(2, 1fr); } }
.meta-box { border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; background: var(--bg); }
.meta-box .lbl { font-size: .66rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: .25rem; }
.meta-box .val { font-size: 1.1rem; font-weight: 700; }
.headline { margin: 2rem 2rem 0; padding: 1.1rem 1.4rem; border-left: 3px solid var(--good); background: var(--bg); border-radius: 0 6px 6px 0; font-size: 1.05rem; font-weight: 600; color: #1f2937; }
.headline b { color: var(--good); }
main { padding: 0 2rem 3rem; }
p { margin-bottom: 1.05rem; max-width: 64ch; font-size: .98rem; }
section { margin-top: 2.75rem; padding-top: 2rem; border-top: 1px solid var(--border); }
.s-label { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
h2 { font-size: 1.3rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.25; margin-bottom: .9rem; }
.fig { margin: 1.5rem 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.fig-hdr { padding: .75rem 1.1rem; background: var(--bg); border-bottom: 1px solid var(--border); }
.fig-title { font-size: .95rem; font-weight: 700; }
.fig-sub { font-size: .8rem; color: var(--muted); margin-top: .15rem; }
.fig-body { padding: 1.1rem 1.1rem .3rem; }
.fig-cap { padding: .6rem 1.1rem .95rem; font-size: .82rem; color: var(--muted); }
svg.chart { display: block; width: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: .85rem; padding: .1rem 1.1rem .95rem; font-size: .78rem; }
.li { display: flex; align-items: center; gap: .35rem; }
.sw { width: .7rem; height: .7rem; border-radius: 2px; display: inline-block; }
.corr-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
@media (max-width: 620px) { .corr-cols { grid-template-columns: 1fr; } }
.corr-col-hdr { font-size: .8rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: .5rem; text-align: center; }
table.corr-table, table.cross-table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .85rem; }
table.corr-table th, table.corr-table td, table.cross-table th, table.cross-table td { border: 1px solid var(--border); padding: .4rem .6rem; text-align: center; }
table.cross-table th { background: var(--bg); }
table.hidden { display: none; }
select { font-size: .95rem; padding: .3rem .5rem; margin: .5rem 0 1rem; }
"""
