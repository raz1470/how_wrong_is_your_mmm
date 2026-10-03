"""DiscoveryReport: sweep candidate phasing strategies, or pin one
directly with per-channel overrides, and report the impact against each
of the three reliability problems this package diagnoses.

Pass strategy_pct (optionally with channel_constraints) to pin a strategy
directly instead of sweeping for one, and call schedule_csv() for the
exportable weekly table.

Every DEFAULT candidate strategy is applied identically to every channel
-- there is no per-channel number to report for the swept grid, only a
single report-wide choice per candidate, so an unpinned report picks ONE
"highest impact" strategy (dominance check, else worst-axis -- see
_pick_winner) and uses it for every "impact from best lever" callout. A
pinned strategy (strategy_pct) skips that selection: it is added to the
sweep as one more candidate, so it still scores in the Appendix's
comparison table, and is used directly as the winner, with
channel_constraints letting specific channels override it individually --
shown in their own small Appendix table.

Report structure: problem, then impact, then what to do about it, with no
dropdown and no JS anywhere on the page. Sections 1-4 are
all about ONE strategy in play -- the swept winner, or the pinned strategy
when one is given -- but nothing in their construction assumes a sweep
happened. Only the appendix is sweep-specific:

1. Scenario inputs -- everything the report is built on, in one section:
   a plain per-channel input table (spend, ROI, saturation, adstock); the
   actual weekly spend behind the plan and the assumed response curves
   (adstock decay, saturation) that explain those numbers; and an
   "implied contribution" stacked-area chart (baseline, incl. demand, +
   each channel's modelled contribution, summing to weekly sales) -- the
   package's own synthetic outcome from the assumptions above, explicitly
   not something supplied. There is no standalone demand chart: it is a
   zero-mean synthetic series with no interpretive hook on its own, and
   its effect is already visible in Implied Contribution's Baseline band.
   Reproducible from these inputs alone, in a notebook,
   without this report class.
2. Diagnostics -- how bad the unphased problem is, full stop, before any
   fix is shown: spend correlation, variance, bias and identifiability
   (saturation + adstock), each as the unphased state only, via
   _svg_forest's single=True mode.
3. Impact -- the same four charts, now before vs after: what phasing
   under the winning strategy does to each problem Section 2 just showed.
   It does not restate the problem, and ends with the growth readout (the
   return at higher spend) and the cost.
4. Phased spend -- small-multiples "recommended pacing" chart, one per
   channel, as-supplied vs the winning strategy's own phased schedule.
   Kept apart from the strategy-impact table: the pacing chart is about
   the ONE winning strategy, the table about comparing every candidate.

Appendix: every strategy compared -- the strategy-impact table, one row
per candidate lever, variance/bias/identifiability improvement over
unphased plus phasing's revenue cost, winning row highlighted. The only
sweep-specific content in the report; sections 1-4 are all built from
this table's winning row alone.

The commentary is worded from the results (see _change_sentence and
_bias_change_sentence), never asserted: a sentence says a range narrows
only for the channels where it does, so the report stays true on any data
and for any pinned strategy.

Follows the package's shared-DGP design: one demand series drives every
simulated sales column in this report, and saturation/adstock (per
channel) are resolved once in __init__ and reused throughout, never
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
    DEFAULT_DEMAND_SPEND_CORR,
    apply_adstock,
    calibrate_baseline,
    channel_contributions,
    link_demand_to_spend,
)
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._identifiability import (
    IdentifiabilityDiagnostic,
    _per_channel_map,
)
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    MonthStep,
    Redistribute,
    _generate_phased_schedule,
    _get_month_labels,
)

# Categorical channel palette. First three slots match docs/overview.html's
# brand colours (tv/meta/search); slot 4 (violet) has been checked for
# colour-vision-deficiency accessibility and passes, with one WARN band
# requiring direct labels, which every chart here already has. Slots 5+
# are a reasonable extension, not yet checked the same way -- fine for
# now, but if a real report regularly needs more than ~5 channels it's
# worth re-validating the fuller set rather than assuming it holds.
_PALETTE = [
    "#2563eb",  # blue
    "#d97706",  # amber
    "#059669",  # green
    "#7c3aed",  # violet
    "#dc2626",  # red
    "#0891b2",  # cyan
    "#65a30d",  # olive
    "#c026d3",  # magenta
]


_WEEKS_PER_YEAR = 52
# Seed offset between the bias section's demand draws, kept well clear of
# the phasing seeds (self.seed + j for small j).
_BIAS_DRAW_SEED_STEP = 7919
# History that must stay un-phased ahead of a back-phased window. Zero:
# the diagnostics need spend to fit against, not un-phased spend, so a
# window covering all of history is valid. It is the "phasing had run the
# whole time" case, e.g. year 3 of time-to-benefit on 2 years of history.
_MIN_HEAD_WEEKS = 0


def _channel_colors(channels: list[str]) -> dict[str, str]:
    return {ch: _PALETTE[i % len(_PALETTE)] for i, ch in enumerate(channels)}


# _pick_winner scores rigor only, never operational feasibility, so a
# report's winner_ is not automatically the recommendation: callers that
# care about deployability should check the Cost and Peak week columns.
# Each sweep entry is (label, per-channel spec, nudge_shape, balance_signs).
# The default sweep's high month (Redistribute.high_month_pct): 2.5x plan.
# 3x gives a narrower saturation range, but costs about 3.7% of revenue
# and puts the peak week at about 3.7x plan.
_DEFAULT_HIGH_MONTH_PCT = 150.0

# Spend levels, as multiples of each channel's planned weekly spend, that the
# growth readout quotes a marginal return at: today, +50% and +100%.
_GROWTH_MULTIPLIERS = (1.0, 1.5, 2.0)
# The default sweep's nudge size, in percent.
_DEFAULT_NUDGE_PCT = 20.0


def _default_levers(channels: list[str]) -> list[tuple[str, dict, str, bool]]:
    """The default sweep: unphased, then six strategies. Three are building
    blocks, each aimed at one problem, and the fourth combines them:

    - "Weekly nudge": every week moves by exactly +/-20%, balanced within
      the month (the "edge" nudge shape). Aimed at variance.
    - "Dark month": four dark weeks in a row once a year, the freed budget
      moved into one other month (Redistribute with no edge layer). Aimed
      at bias and adstock.
    - "Peak month": one month a year at 2.5x plan, funded by a small equal
      cut to the other months (Redistribute with only its high month).
      Aimed at saturation.
    - "Combined": all three together. The weekly nudge runs only in the
      months the other two leave alone (edge_quiet_months_only), which
      keeps the peak week at about 2.5x plan instead of 3x.

    The other two are alternatives: "Month step" (month-level Hadamard
    steps at +/-20%) and "Dark week" (one dark week a quarter).

    Nudges are capped at 20% because smaller moves are more deployable and
    less likely to trigger ad-platform learning resets. The uniform and
    seesaw nudge shapes are not swept (the edge shape matched or beat both
    on every measure but cost) and stay available via levers=. Stronger
    intensities (40/60/80%) and Blackout's stronger settings are not swept
    either; pin them with strategy_pct / channel_constraints, or pass your
    own levers=."""

    def all_channels(spec) -> dict:
        return {ch: spec for ch in channels}

    pct = _DEFAULT_NUDGE_PCT
    high = _DEFAULT_HIGH_MONTH_PCT
    return [
        ("unphased", all_channels(0.0), "uniform", False),
        ("Weekly nudge", all_channels(pct), "edge", True),
        # A Redistribute's edge layer is always edge, balanced, and
        # MonthStep/Blackout have no weekly nudge, so nudge_shape and
        # balance_signs are unused from here on.
        ("Dark month", all_channels(Redistribute(edge_cap_pct=0.0)), "edge", True),
        (
            "Peak month",
            all_channels(
                Redistribute(dark_weeks=0, edge_cap_pct=0.0, high_month_pct=high)
            ),
            "edge",
            True,
        ),
        (
            "Combined",
            all_channels(
                Redistribute(
                    edge_cap_pct=pct, high_month_pct=high, edge_quiet_months_only=True
                )
            ),
            "edge",
            True,
        ),
        ("Month step", all_channels(MonthStep(step_pct=pct)), "uniform", False),
        (
            "Dark week",
            all_channels(Blackout(prob=1.0, max_dark_weeks_per_quarter=1)),
            "uniform",
            False,
        ),
    ]


def _moves_budget_between_months(spec: dict) -> bool:
    """True if any channel's spec (Redistribute, MonthStep) shifts budget
    between months -- every other spec preserves each month's total
    exactly."""
    return any(isinstance(v, Redistribute | MonthStep) for v in spec.values())


def _peak_week_multiple(plan_df: pd.DataFrame, schedule: pd.DataFrame) -> float:
    """Largest single-week spend in `schedule`, as a multiple of that same
    week's as-supplied plan, across every channel (1.0 = no week is ever
    above plan). The deployability read on a strategy: a strategy whose
    peak week is 4x plan needs a media buyer to quadruple one week's
    spend. Weeks with zero planned spend are skipped."""
    plan = plan_df.to_numpy(dtype=float)
    sched = schedule.to_numpy(dtype=float)
    planned = plan > 0
    if not planned.any():
        return 1.0
    return float(max(1.0, (sched[planned] / plan[planned]).max()))


def _redistribute_label(spec: Redistribute) -> str:
    """Label for a pinned or per-channel Redistribute spec, built from the
    pieces it has switched on: "dark month", "peak month" and a "+/-X%
    weekly nudge", e.g. "Dark month + peak month + 20% weekly nudge"."""
    parts = []
    if spec.dark_weeks > 0:
        parts.append("dark month")
    if spec.high_month_pct > 0:
        parts.append("peak month")
    if spec.edge_cap_pct > 0:
        quiet = " in quiet months" if spec.edge_quiet_months_only else ""
        parts.append(f"{spec.edge_cap_pct:.0f}% weekly nudge{quiet}")
    if not parts:
        return "Unphased"
    label = " + ".join(parts)
    return label[0].upper() + label[1:]


def _is_unphased(spec: dict) -> bool:
    return all(isinstance(v, float) and v == 0.0 for v in spec.values())


def _pinned_strategy_label(
    strategy_pct: float | Blackout | Redistribute | MonthStep,
    nudge_shape: str,
    balanced: bool,
    channel_constraints: dict[str, float | Blackout | Redistribute | MonthStep] | None,
) -> str:
    """Label for a user-pinned strategy (DiscoveryReport's strategy_pct),
    formatted like _default_levers' own candidates (e.g. "+/-80% (edge,
    balanced)") but prefixed "Pinned: " so it can never collide with a
    swept candidate that happens to share the same numbers, and suffixed
    with a channel-override count when channel_constraints narrows it
    further for specific channels."""
    if isinstance(strategy_pct, Blackout):
        base = "Blackout"
    elif isinstance(strategy_pct, Redistribute):
        base = _redistribute_label(strategy_pct)
    elif isinstance(strategy_pct, MonthStep):
        base = f"{strategy_pct.step_pct:.0f}% month step"
    else:
        shape_bits = nudge_shape + (", balanced" if balanced else "")
        base = f"+/-{strategy_pct:.0f}% ({shape_bits})"
    if channel_constraints:
        n = len(channel_constraints)
        base += f", {n} channel override" + ("" if n == 1 else "s")
    return f"Pinned: {base}"


def _safe_improvement(before: float, after: float) -> float:
    """Fractional improvement of `after` over `before` (lower-is-better metric).

    Returns 0.0 rather than dividing by zero when `before` is already 0 --
    there is no room left to improve on, not an infinite gain.
    """
    if before == 0:
        return 0.0
    return (before - after) / before


def _describe_strategy(
    spec: dict, nudge_shape: str, balance_signs: bool
) -> tuple[str, str]:
    """(what it does, which budget totals it keeps) for one lever, in plain
    words for the Appendix's glossary table. Described from the first
    channel's spec; a lever whose channels differ says so."""
    first = next(iter(spec.values()))
    mixed = any(v != first for v in spec.values())
    if isinstance(first, Redistribute):
        sentences = []
        if first.dark_weeks > 0:
            sentences.append(
                f"Each channel goes dark for {first.dark_weeks} consecutive "
                "weeks once a year, in a different month per channel. The "
                "freed budget is moved into one other month."
            )
        if first.high_month_pct > 0:
            sentences.append(
                f"One month a year runs at {1 + first.high_month_pct / 100:.1f}x "
                "plan, in a different month per channel, paid for by a small "
                "equal cut to the channel's other months."
            )
        if first.edge_cap_pct > 0:
            which = (
                "Every week outside those months"
                if first.edge_quiet_months_only
                else "Every week"
            )
            sentences.append(
                f"{which} is nudged up or down by exactly "
                f"{first.edge_cap_pct:.0f}% (half up, half down)."
            )
        what = " ".join(sentences)
        keeps = "Annual only"
    elif isinstance(first, MonthStep):
        what = (
            f"Each month's whole budget is stepped up or down by {first.step_pct:.0f}%, "
            "in a balanced pattern that is unrelated between channels. The "
            "weekly shape inside a month is unchanged, so budgets change at "
            "most 12 times a year."
        )
        keeps = "Annual only"
    elif isinstance(first, Blackout):
        if first.max_dark_weeks_per_quarter is not None:
            n = first.max_dark_weeks_per_quarter
            unit = "week" if n == 1 else "consecutive weeks"
            what = (
                f"Once a quarter a channel goes dark for {n} {unit} (zero "
                "spend) in one month. That month's other weeks absorb the budget."
            )
            if first.prob < 1.0:
                what += f" A quarter is affected with probability {first.prob:.0%}."
        elif first.max_dark_weeks_per_month is None:
            what = (
                "Each week is independently either dark (zero spend) or on; "
                "the weeks left on absorb the month's budget."
            )
        else:
            n = first.max_dark_weeks_per_month
            unit = "week" if n == 1 else "consecutive weeks"
            what = (
                f"In some months a channel goes dark for {n} {unit} (zero "
                "spend) and that month's other weeks absorb the budget."
            )
            if first.prob < 1.0:
                what += f" A month is affected with probability {first.prob:.0%}."
        keeps = "Monthly and annual"
    elif isinstance(first, (int, float)) and float(first) == 0.0:
        what = "The plan exactly as supplied. Every other row is measured against this one."
        keeps = "Monthly and annual"
    else:
        pct = f"{float(first):.0f}%"
        if nudge_shape == "seesaw":
            what = (
                f"Every week moves by exactly {pct}, alternating up and down "
                "week to week, then the month is rescaled to its planned total."
            )
        elif nudge_shape == "edge":
            what = (
                f"Every week moves by exactly {pct}, half of a month's weeks "
                "up and half down in a random order, then the month is "
                "rescaled to its planned total."
                if balance_signs
                else f"Every week moves by exactly {pct}, direction chosen at "
                "random, then the month is rescaled to its planned total."
            )
        else:
            what = (
                f"Each week moves by a random amount up to {pct}, direction "
                "chosen at random (a typical week moves about half that), "
                "then the month is rescaled to its planned total."
            )
        keeps = "Monthly and annual"
    if mixed:
        what += (
            " Some channels use a different setting; see the channel constraints below."
        )
    return what, keeps


def _strategy_glossary_html(report: DiscoveryReport) -> str:
    rows = ""
    for label, spec, nudge_shape, balance_signs in report.levers_:
        what, keeps = _describe_strategy(spec, nudge_shape, balance_signs)
        rows += (
            f'<tr><td class="strat-name">{html.escape(label)}</td>'
            f'<td class="strat-what">{html.escape(what)}</td>'
            f"<td>{keeps}</td></tr>"
        )
    return f"""
  <h3>What each strategy does</h3>
  <div class="table-scroll">
  <table class="cross-table glossary-table">
    <thead><tr><th>Strategy</th><th>What it does to the plan</th><th>Budget totals kept</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p class="fig-cap">Every strategy leaves each channel's annual budget and the
  split across channels untouched; only the timing changes. "Monthly and
  annual" means each calendar month still gets its planned budget, so
  changes stay inside the month. "Annual only" means budget can move between
  months.</p>
  <h3>How they compare</h3>"""


# Above this many channels a per-channel sentence stops being readable, so
# the report's commentary switches to a count and an average.
_MAX_CHANNELS_IN_A_SENTENCE = 6
# A change smaller than this (as a fraction) reads as "unchanged".
_NO_CHANGE = 0.005


def _join_and(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _change_sentence(
    changes: dict[str, float], subject: str, better_verb: str, worse_verb: str
) -> str:
    """One sentence on how a per-channel measure changed under phasing,
    worded from the numbers so it stays true when some channels get worse.

    `changes` maps channel to its improvement as a fraction (positive is
    better). `subject`, `better_verb` and `worse_verb` supply the words,
    e.g. "The range", "narrows", "widens".
    """
    better = {ch: v for ch, v in changes.items() if v > _NO_CHANGE}
    worse = {ch: -v for ch, v in changes.items() if v < -_NO_CHANGE}
    same = [ch for ch, v in changes.items() if abs(v) <= _NO_CHANGE]
    n = len(changes)
    if n > _MAX_CHANNELS_IN_A_SENTENCE:
        mean = float(np.mean(list(changes.values())))
        verb = better_verb if mean >= 0 else worse_verb
        return (
            f"{subject} {better_verb} for {len(better)} of {n} channels. "
            f"On average it {verb} by {abs(mean):.0%}."
        )

    def listed(group: dict[str, float]) -> str:
        return _join_and([f"{html.escape(ch)} by {v:.0%}" for ch, v in group.items()])

    if len(better) == n:
        return f"{subject} {better_verb} for every channel: {listed(better)}."
    if len(worse) == n:
        return f"{subject} {worse_verb} for every channel: {listed(worse)}."
    parts = []
    if better:
        parts.append(f"{better_verb} for {listed(better)}")
    if worse:
        parts.append(f"{worse_verb} for {listed(worse)}")
    if same:
        parts.append(f"is unchanged for {_join_and([html.escape(c) for c in same])}")
    return f"{subject} " + ". It ".join(parts) + "."


def _bias_change_sentence(before: dict[str, float], after: dict[str, float]) -> str:
    """One sentence on how each channel's absolute bias (%) moved."""
    n = len(before)
    fell = [ch for ch in before if after[ch] < before[ch] - 0.5]
    rose = [ch for ch in before if after[ch] > before[ch] + 0.5]
    if n > _MAX_CHANNELS_IN_A_SENTENCE:
        return (
            f"Bias falls for {len(fell)} of {n} channels. Mean absolute bias "
            f"goes from {np.mean(list(before.values())):.0f}% to "
            f"{np.mean(list(after.values())):.0f}%."
        )

    def listed(group: list[str]) -> str:
        return _join_and(
            [
                f"{html.escape(ch)} from {before[ch]:.0f}% to {after[ch]:.0f}%"
                for ch in group
            ]
        )

    if len(fell) == n:
        return f"Every point estimate moves toward the truth: {listed(fell)}."
    parts = []
    if fell:
        parts.append(f"falls for {listed(fell)}")
    if rose:
        parts.append(f"rises for {listed(rose)}")
    same = [ch for ch in before if ch not in fell and ch not in rose]
    if same:
        parts.append(f"is unchanged for {_join_and([html.escape(c) for c in same])}")
    return "Bias " + ". It ".join(parts) + "."


def _improvement_phrase(value: float, better: str, worse: str) -> str:
    """'45% better', '3% worse' or 'unchanged', from an improvement as a
    fraction."""
    if abs(value) < _NO_CHANGE:
        return "unchanged"
    return f"{abs(value):.0%} {better if value > 0 else worse}"


def _mean_pairwise(matrix: dict, channels: list[str]) -> float:
    pairs = [matrix[a][b] for i, a in enumerate(channels) for b in channels[i + 1 :]]
    return float(np.mean(pairs)) if pairs else 0.0


_BENEFIT_SERIES = (
    ("variance", "Variance", "#111827"),
    ("saturation", "Saturation", "#0d9488"),
    ("adstock", "Adstock", "#92400e"),
    ("bias", "Bias", "#7c3aed"),
)


def _svg_benefit_lines(improvement: dict[str, list[float]], years: list[int]) -> str:
    """One chart of how much each measure improves on the unphased plan as
    more years are phased, every line starting from zero at "Unphased".
    The same chart as docs/overview.html's "How the benefit builds over
    time". `improvement` maps each measure to one fraction per year.
    """
    width, height = 700, 300
    m_top, m_right, m_bottom, m_left = 16, 150, 52, 70
    pw, ph = width - m_left - m_right, height - m_top - m_bottom
    series = [
        (name, color, [0.0] + [100 * v for v in improvement[key]])
        for key, name, color in _BENEFIT_SERIES
    ]
    lowest = min(min(v) for _, _, v in series)
    y_min = 25 * math.floor(lowest / 25) if lowest < 0 else 0
    y_max = 100
    n_pts = len(years) + 1

    def sc_x(i: int) -> float:
        return m_left + (i / (n_pts - 1)) * pw

    def sc_y(v: float) -> float:
        return m_top + ph - ((v - y_min) / (y_max - y_min)) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'preserveAspectRatio="xMinYMin meet" role="img">'
    ]
    for v in range(y_min, y_max + 1, 25):
        stroke = "#9ca3af" if v == 0 and y_min < 0 else "#e5e7eb"
        parts.append(
            f'<line x1="{m_left}" y1="{sc_y(v):.1f}" x2="{m_left + pw}" '
            f'y2="{sc_y(v):.1f}" stroke="{stroke}" stroke-width="1"/>'
            f'<text x="{m_left - 8}" y="{sc_y(v) + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#9ca3af">{v}%</text>'
        )
    x_labels = ["Unphased"] + [f"{y} year{'s' if y > 1 else ''}" for y in years]
    for i, label in enumerate(x_labels):
        anchor = "start" if i == 0 else "end" if i == n_pts - 1 else "middle"
        parts.append(
            f'<text x="{sc_x(i):.1f}" y="{m_top + ph + 18}" text-anchor="{anchor}" '
            f'font-size="11" fill="#9ca3af">{label}</text>'
        )
    parts.append(
        f'<text x="{m_left + pw / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        'font-size="11" fill="#9ca3af">Years phased</text>'
    )
    y_mid = m_top + ph / 2
    parts.append(
        f'<text x="12" y="{y_mid:.1f}" text-anchor="middle" font-size="11" '
        f'fill="#9ca3af" transform="rotate(-90 12 {y_mid:.1f})">'
        "Improvement on the unphased plan</text>"
    )
    # End labels, highest first, pushed apart so they never overlap.
    order = sorted(range(len(series)), key=lambda k: -series[k][2][-1])
    label_y: dict[int, float] = {}
    previous = None
    for k in order:
        y = sc_y(series[k][2][-1])
        if previous is not None and y - previous < 14:
            y = previous + 14
        label_y[k] = previous = y
    for k, (name, color, values) in enumerate(series):
        points = " ".join(f"{sc_x(i):.1f},{sc_y(v):.1f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round"/>'
        )
        parts.extend(
            f'<circle cx="{sc_x(i):.1f}" cy="{sc_y(v):.1f}" r="3" fill="{color}"/>'
            for i, v in enumerate(values)
        )
        parts.append(
            f'<text x="{sc_x(n_pts - 1) + 8:.1f}" y="{label_y[k] + 4:.1f}" '
            f'font-size="12" font-weight="700" fill="{color}">'
            f"{name} {values[-1]:.0f}%</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _time_to_benefit_html(report: DiscoveryReport) -> str:
    """Appendix subsection: the recommended strategy's benefit at year 1,
    2, 3... of repeating the plan, against the unphased plan measured over
    the same data. Empty string when there is nothing to project (horizon
    of 1, or an unphased winner)."""
    ttb = report.time_to_benefit_
    if not ttb:
        return ""
    years = ttb["years"]
    n_years = len(years)
    winner = html.escape(ttb["label"])
    imp = ttb["improvement"]

    def at(i: int) -> str:
        return (
            f"variance {_improvement_phrase(imp['variance'][i], 'better', 'worse')}, "
            f"bias {_improvement_phrase(imp['bias'][i], 'better', 'worse')}, "
            f"saturation {_improvement_phrase(imp['saturation'][i], 'better', 'worse')} "
            f"and adstock {_improvement_phrase(imp['adstock'][i], 'better', 'worse')}"
        )

    return f"""
  <h3>How the benefit builds over time</h3>
  <p>After one year of <b>{winner}</b>: {at(0)}. After {n_years} years:
  {at(n_years - 1)}.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">How the benefit builds over time</div>
    </div>
    <div class="fig-body">
      {_svg_benefit_lines(imp, years)}
    </div>
    <p class="fig-cap">How much each measure improves on the unphased plan
    as more of the data is phased. Year 1 phases the plan year only. Each
    later year also phases one more year of your history, as if phasing
    had started then. It is a counterfactual on your own spend, not a
    forecast.</p>
  </div>"""


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
    markers: bool = False,
    highlight_from: int | None = None,
    highlight_label: str = "",
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
    highlight_from:
        Index of the first point of a span to shade behind the lines, from
        there to the right edge -- used to mark the plan year on a chart
        that also shows history. `highlight_label` is written in the
        shaded band's top corner.
    markers:
        If True, draw a small circle at every data point -- for a series
        with few, categorical x points (a handful of candidate levers)
        where the line alone doesn't make clear where the real values are.
        Left off (default) for dense weekly series, where a circle per
        week would just be clutter.
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
    if highlight_from is not None and 0 <= highlight_from < n:
        hx = x_at(highlight_from)
        parts.append(
            f'<rect x="{hx:.1f}" y="{pad_top}" width="{width - pad_right - hx:.1f}" '
            f'height="{plot_h:.1f}" fill="#e5e7eb" opacity="0.6"/>'
        )
        if highlight_label:
            parts.append(
                f'<text x="{hx + 4:.1f}" y="{pad_top + 10}" font-size="9" '
                f'font-weight="600" fill="#6b7280">{html.escape(highlight_label)}</text>'
            )
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
    # First/last tick sit exactly on the plot's own left/right edge --
    # text-anchor="middle" there overflows the SVG's own viewBox with
    # longer x-tick labels. Edge ticks anchor inward instead; interior
    # ticks keep
    # centring on their own gridline.
    first_tick, last_tick = tick_idx[0], tick_idx[-1]
    for i in tick_idx:
        x = x_at(i)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{pad_top + plot_h + 5:.1f}" stroke="#d1d5db" stroke-width="1"/>'
        )
        anchor = "start" if i == first_tick else "end" if i == last_tick else "middle"
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 16:.1f}" text-anchor="{anchor}" '
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
            xy = [
                (x_at(i), pad_top + plot_h * (1 - (v - lo) / span))
                for i, v in enumerate(arr)
            ]
        else:
            xy = [(x_at(i), y_at(v)) for i, v in enumerate(arr)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        color = colors.get(name, "#111827")
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if markers:
            for x, y in xy:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
                )
    parts.append("</svg>")
    return "".join(parts)


def _svg_stacked_area(
    band_order: list[str],
    series: dict[str, np.ndarray],
    colors: dict[str, str],
    width: int = 680,
    height: int = 240,
    pad_left: int = 58,
    pad_bottom: int = 36,
    pad_top: int = 14,
    pad_right: int = 14,
    y_fmt=None,
    y_label: str = "",
    x_label: str = "Week",
    x_tick_labels: list[str] | None = None,
    n_x_ticks: int = 6,
) -> str:
    """Render a stacked-area time series as a self-contained inline SVG --
    same axis/gridline/tick visual language as _svg_multiline, but filled
    cumulative bands (each series stacked on top of the last) rather than
    independent lines, for a "what does this add up to" decomposition
    chart. band_order fixes the stacking order bottom-to-top; series must
    have one array per name in band_order, all the same length.

    Every band is assumed non-negative (true for a baseline level plus
    channel contributions built off positive marginal returns under
    concave saturation) -- the y-axis always starts at 0, unlike
    _svg_multiline's own min/max scaling, since a stacked area only reads
    correctly anchored at zero.
    """
    n = len(next(iter(series.values())))
    if n < 2:
        return "<p><em>Not enough weeks to plot.</em></p>"
    fmt = y_fmt if y_fmt is not None else _fmt_gbp
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i: int) -> float:
        return pad_left + plot_w * i / (n - 1)

    arrays = {name: np.asarray(series[name], dtype=float) for name in band_order}
    cum = np.zeros((len(band_order) + 1, n))
    for i, name in enumerate(band_order):
        cum[i + 1] = cum[i] + arrays[name]
    y_lo, y_hi, y_step = _nice_axis_bounds(0.0, float(cum[-1].max()))
    y_ticks = list(np.arange(y_lo, y_hi + y_step / 2, y_step))

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
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'font-size="10" fill="#9ca3af">{fmt(v)}</text>'
        )
    # First/last tick sit exactly on the plot's own left/right edge --
    # text-anchor="middle" there overflows the SVG's own viewBox with
    # longer x-tick labels. Edge ticks anchor inward instead; interior
    # ticks keep
    # centring on their own gridline.
    first_tick, last_tick = tick_idx[0], tick_idx[-1]
    for i in tick_idx:
        x = x_at(i)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{pad_top + plot_h + 5:.1f}" stroke="#d1d5db" stroke-width="1"/>'
        )
        anchor = "start" if i == first_tick else "end" if i == last_tick else "middle"
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 16:.1f}" text-anchor="{anchor}" '
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

    for i, name in enumerate(band_order):
        top = [(x_at(j), y_at(cum[i + 1, j])) for j in range(n)]
        bottom = [(x_at(j), y_at(cum[i, j])) for j in range(n)][::-1]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in top + bottom)
        color = colors.get(name, "#9ca3af")
        parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.85" '
            f'stroke="#fff" stroke-width="1"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _demand_link_sentence(meta: dict) -> str:
    """One sentence stating how the report's demand series relates to spend.

    The link strength (demand_spend_corr) cannot be read off spend data, so
    the report states it as the assumption it is.
    """
    corrs = list(meta["realised_demand_corr"].values())
    span = f"{min(corrs):.2f} to {max(corrs):.2f}"
    n = meta["n_bias_draws"]
    if meta["demand_supplied"]:
        return (
            "The demand series was supplied with the spend. Its correlation "
            f"with each channel's unphased spend is {span}. Bias is averaged "
            f"over {n} draws of the proxy."
        )
    return (
        "The demand series is built to track your unphased spend at an assumed "
        f"correlation of {meta['demand_spend_corr']:.2f} ({span} by channel). "
        "Spend data cannot confirm that value, so treat it as an assumption. "
        f"Bias grows steeply with it. Bias is averaged over {n} draws of "
        "demand and its proxy, because any single draw can line up with one "
        "channel by chance."
    )


def _corr_table_html(matrix: dict, channels: list[str]) -> str:
    """Static HTML table for a channel-by-channel correlation matrix, cells
    heat-shaded from the correlation value (no JS -- computed at render time,
    same reasoning as _svg_multiline).

    Column headers render vertically (the .corr-table CSS) so more or
    longer channel names don't force the table wider than the page.
    """

    def cell_style(v: float) -> str:
        # White at 0, deepening red towards +1 (collinearity is the risk
        # this report cares about; negative correlation isn't the concern
        # here, so it gets a flat light shade rather than its own ramp).
        alpha = max(0.0, min(1.0, v))
        return f"background: rgba(220, 38, 38, {alpha * 0.65:.2f});"

    header = "".join(
        f'<th class="col-hdr"><span class="col-hdr-label">{ch}</span></th>'
        for ch in channels
    )
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


def _lighten_hex(hex_color: str, amount: float = 0.65) -> str:
    """Blend a #rrggbb colour toward white by `amount` (0 = unchanged, 1 =
    white) -- a pale-vs-solid pair in the SAME hue, matching the
    opacity=0.35 "before" convention _svg_forest uses, for a chart type
    (_svg_multiline) whose polylines have no opacity knob of their own.
    Used by the recommended-pacing chart so "as supplied" and the winning
    schedule read as two shades of that channel's own colour rather than
    a channel-blind grey/black pair."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


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


def _hi(v: float | tuple[float, float] | dict) -> float:
    """Upper end of a range mark, the value itself for a point mark, or
    the larger of the range/point for a combined mark --
    lets _svg_forest's axis/label code treat all three the same way."""
    if isinstance(v, dict):
        return max(v["range"][1], v["point"])
    return v[1] if isinstance(v, tuple) else v


def _label_source(v: float | tuple[float, float] | dict) -> float | tuple[float, float]:
    """The (lo, hi) or point value a mark's text label is built from.
    For a combined range+point mark that's the range: the ring drawn on
    the chart carries the point visually, so the label doesn't also spell
    it out."""
    return v["range"] if isinstance(v, dict) else v


def _svg_forest(
    data: list[dict],
    width: int = 700,
    row_h: int = 62,
    x_label: str = "Incremental revenue",
    fmt=_fmt_gbp,
    single: bool = False,
) -> str:
    """Two-state horizontal chart: each channel gets a pale "before" mark
    and a solid "after" mark, plus a dashed line at the value implied by
    the truth -- the same visual language as docs/overview.html's
    chartProblem/chartImpact (JS there, static Python/SVG here since
    nothing in this report changes once the winner's picked, same
    reasoning as _svg_multiline).

    Each `data` entry: {name, color, before, after, truth: float | None}.
    `before`/`after` are each one of three shapes: a (p10, p90) tuple --
    drawn as a rounded range bar; a single float -- drawn as a dot, for a
    mean-estimate quantity with no p10/p90 to show; or a dict
    {"range": (p10, p90), "point": float} -- drawn as both together, a
    ring marking the point estimate on top of the range bar
    (variance/saturation/adstock show a point estimate alongside their
    range, bias shows a band around its point). A combined mark's text
    label reads off its range only -- the ring carries the point visually
    rather than in the label text. The
    two states in one chart don't have to match shape, though every
    section built so far uses one shape throughout.

    `single=True` switches to a one-state-per-row rendering, for the
    Diagnostics section, which only ever shows the unphased problem (no
    before/after comparison -- that's what the future Impact section is
    for). Each `data` entry then needs {name, color, value, truth}
    instead of {before, after}: one mark per row, drawn at the row's
    centre, with a shorter dashed truth line either side of it.

    Values are raw numbers -- tick step is picked at render time from
    whatever range this chart's own numbers span, via _nice_tick_step, and
    every axis/value label goes through `fmt` (defaults to £ formatting,
    the variance/bias sections' convention; pass a plain "{:.2f}".format
    for a non-£ quantity like the identifiability section's b/lambda).
    """
    # m_left scales with the longest channel name -- the fixed 100px
    # default (sized for "search") clipped longer real-world names like
    # "search_generic" against the SVG's own left edge.
    longest_name = max(len(ch["name"]) for ch in data)
    m_top, m_right, m_bottom, m_left = 14, 96, 40, max(100, 20 + 9 * longest_name)
    height = m_top + m_bottom + row_h * len(data)
    pw = width - m_left - m_right
    ph = height - m_top - m_bottom
    row = ph / len(data)

    if single:
        raw_max = max(max(_hi(ch["value"]), ch.get("truth") or 0.0) for ch in data)
    else:
        raw_max = max(
            max(_hi(ch["before"]), _hi(ch["after"]), ch.get("truth") or 0.0)
            for ch in data
        )
    step = _nice_tick_step(raw_max)
    x_max = step * math.ceil(raw_max / step) if raw_max > 0 else step

    def sc_x(v: float) -> float:
        return m_left + (v / x_max) * pw

    def mark(
        v: float | tuple[float, float] | dict,
        cy: float,
        color: str,
        opacity: float | None = None,
    ) -> str:
        op = f' opacity="{opacity}"' if opacity is not None else ""
        if isinstance(v, dict):
            lo, hi = v["range"]
            bar = (
                f'<line x1="{sc_x(lo):.1f}" y1="{cy:.1f}" x2="{sc_x(hi):.1f}" '
                f'y2="{cy:.1f}" stroke="{color}" stroke-width="5.5" '
                f'stroke-linecap="round"{op}/>'
            )
            # White-fill ring rather than a solid dot, so the point
            # estimate reads as its own mark sitting ON the range bar
            # instead of being swallowed by it.
            point = (
                f'<circle cx="{sc_x(v["point"]):.1f}" cy="{cy:.1f}" r="3.5" '
                f'fill="#fff" stroke="{color}" stroke-width="1.8"{op}/>'
            )
            return bar + point
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
        color = ch["color"]

        if single:
            value = ch["value"]
            parts.append(mark(value, cy, color))
            if ch.get("truth") is not None:
                tx = sc_x(ch["truth"])
                parts.append(
                    f'<line x1="{tx:.1f}" y1="{cy - row * 0.3:.1f}" x2="{tx:.1f}" '
                    f'y2="{cy + row * 0.3:.1f}" stroke="#111827" stroke-width="1.6" '
                    f'stroke-dasharray="3,2"/>'
                )
            label_src = _label_source(value)
            value_label = (
                f"{fmt(label_src[0])} &ndash; {fmt(label_src[1])}"
                if isinstance(label_src, tuple)
                else fmt(label_src)
            )
            parts.append(
                f'<text x="{sc_x(_hi(value)) + 8:.1f}" y="{cy + 4:.1f}" '
                f'font-size="11.5" fill="{color}" font-weight="700">'
                f"{value_label}</text>"
            )
            parts.append(
                f'<text x="{m_left - 10}" y="{cy + 4:.1f}" text-anchor="end" '
                f'font-size="12.5" font-weight="700" fill="{color}">'
                f"{html.escape(ch['name'])}</text>"
            )
            continue

        step_y = row * 0.22
        cy_before, cy_after = cy - step_y, cy + step_y
        before, after = ch["before"], ch["after"]

        parts.append(mark(before, cy_before, color, opacity=0.35))
        parts.append(mark(after, cy_after, color))
        if ch.get("truth") is not None:
            tx = sc_x(ch["truth"])
            parts.append(
                f'<line x1="{tx:.1f}" y1="{cy_before - 5:.1f}" x2="{tx:.1f}" '
                f'y2="{cy_after + 5:.1f}" stroke="#111827" stroke-width="1.6" '
                f'stroke-dasharray="3,2"/>'
            )
        after_label_src = _label_source(after)
        after_label = (
            f"{fmt(after_label_src[0])} &ndash; {fmt(after_label_src[1])}"
            if isinstance(after_label_src, tuple)
            else fmt(after_label_src)
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
        to every channel. Each channel
        gets its own value because IdentifiabilityDiagnostic's grid search
        is now per-channel too: a channel's OWN spend pattern determines
        how well its OWN curvature can be pinned down, which differs by
        channel even when every channel shares the same assumed curvature.
    revenue_noise_pct:
        Weekly sales noise sd as a fraction of average weekly sales (the
        calibrated total from calibrate_baseline). Default 0.02 (2%).
        This is pure noise: week-to-week variation that neither the media
        nor demand explains. Demand, including the part the proxy misses,
        is modelled separately (baseline_cv, demand_proxy_quality), so
        this should be smaller than a real MMM's residual sd, which
        contains both. Resolved to GBP once, in __init__, and stored on
        revenue_noise_std. Variance CV scales about linearly with it. Bias
        barely depends on it.
    revenue_noise_std:
        Optional GBP override for the weekly noise sd. When set,
        revenue_noise_pct is ignored. Either way the resolved GBP value is
        forwarded to every internal
        CollinearityDiagnostic/IdentifiabilityDiagnostic.
    levers:
        The candidate strategies to sweep, as a list of
        (label, per-channel max_weekly_deviation_pct spec, nudge_shape,
        balance_signs) tuples. Defaults to unphased, then +/-20% at two
        nudge shapes (uniform, unbalanced -- vs edge, balanced), then the
        Redistribute family (round-robin blackout + recipient month + edge
        layer), plain and with a 2.5x high month, and the MonthStep family
        (month-level Hadamard steps), each at +/-20%, then the original
        single-week Blackout once a quarter -- 7 candidates total, see
        _default_levers. Higher intensities and Blackout's stronger
        settings are pinnable (strategy_pct) or passable here, but are not
        swept by default. notebooks/05_strategy_comparison.ipynb
        is the live one now). The first entry is expected to be the
        unphased baseline every other candidate is compared against; pass
        your own list to add or narrow candidates, keeping an unphased
        baseline entry first.
    strategy_pct, strategy_nudge_shape, strategy_balanced:
        Pin a single strategy instead of sweeping for one.
        strategy_pct is the same per-channel spec _default_levers uses for
        one candidate: a float (symmetric +/-X% for every channel), a
        Blackout, a Redistribute (its edge_cap_pct is the intensity;
        strategy_nudge_shape and strategy_balanced are ignored for it --
        the edge layer is always edge, balanced) or a MonthStep (its
        step_pct is the intensity; the nudge shape and balance are ignored
        too). Redistribute and MonthStep move budget between months, so
        only annual totals are preserved. When set, this exact strategy is
        added to the sweep as one more candidate (labelled "Pinned: ...") and used directly as
        self.winner_ -- _pick_winner's dominance check never runs, so
        Sections 1-4, the Appendix's headline callouts and schedule_csv()
        are all built from the strategy you chose, not one the sweep
        picked. Left at the default (None), behaviour is unchanged: the
        sweep runs and picks a winner exactly as before.
        strategy_nudge_shape ("uniform" or "edge") and strategy_balanced
        control the pinned strategy's own shape, same meaning as every
        other lever's nudge_shape/balance_signs. Ignored when strategy_pct
        is None.
    channel_constraints:
        Per-channel overrides applied on top of strategy_pct for specific
        channels (a float, Blackout, Redistribute or MonthStep) -- e.g.
        {"meta": 20.0} pins meta to +/-20% regardless of what strategy_pct
        says for every other channel, or {"meta":
        Blackout(max_dark_weeks_per_month=1)} switches meta to blackout-mode
        while the rest of the plan follows strategy_pct. Requires
        strategy_pct to be set (there is nothing to override otherwise);
        raises ValueError on an unknown channel name. Shown in the
        Appendix as its own small table, alongside the pinned strategy's
        row in the main comparison table.
    client_name, plan_year:
        Free-text labels shown on the report cover.
    seed:
        Base random seed for demand and phasing draws.
    backphase_years:
        Default 0. If N > 0, the last N years (52 weeks each) of
        history_df are phased too, as if phasing had started N years
        before the plan: the report's "plan" becomes that history tail plus
        plan_df, so every section (impact, ranges, phased spend) shows the
        N+1-year effect instead of one year. A counterfactual on your own
        spend: nothing is simulated beyond what the report already
        simulates from spend. N can cover all of history_df, in which case
        the whole supplied window is phased.
    demand_process:
        Shape of the unlinked part of the report's demand series. Forwarded
        to link_demand_to_spend(). Default "trend" (a random walk with
        drift). The linked part takes the shape of the supplied spend
        itself. At the default demand_spend_corr and a pairwise channel
        correlation of 0.7, this setting shapes about 45% of demand's
        variance. Ignored when `demand` is supplied.
    demand_spend_corr:
        Assumed corr(spend, demand), averaged over channels, in [0, 1].
        Sets how strongly the report's demand series tracks the unphased
        spend. Default 0.65, the middle of a 0.5 to 0.8 range judged
        plausible for media plans that partly follow demand. Spend data
        cannot reveal the true value, so the report states it as an
        assumption. Bias grows steeply with it. Ignored when `demand` is
        supplied.
    demand:
        Optional demand series of length len(history_df) + len(plan_df),
        e.g. the series passed to simulate_spend(demand=...) when the
        spend is simulated. Used instead of building one with
        link_demand_to_spend(). Standardised to mean 0 / sd 1 on the way
        in, since calibrate_baseline's demand_coef assumes that scale.
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
        revenue_noise_pct: float = 0.02,
        revenue_noise_std: float | None = None,
        levers: list[tuple[str, dict, str, bool]] | None = None,
        strategy_pct: float | Blackout | Redistribute | MonthStep | None = None,
        strategy_nudge_shape: str = "edge",
        strategy_balanced: bool = True,
        channel_constraints: dict[str, float | Blackout | Redistribute | MonthStep]
        | None = None,
        client_name: str = "",
        plan_year: str = "",
        seed: int = 0,
        demand_process: str = "trend",
        backphase_years: int = 0,
        demand_spend_corr: float = DEFAULT_DEMAND_SPEND_CORR,
        demand: np.ndarray | pd.Series | None = None,
    ) -> None:
        if list(history_df.columns) != list(plan_df.columns):
            raise ValueError(
                "history_df and plan_df must have the same columns. "
                f"Got {list(history_df.columns)} vs {list(plan_df.columns)}."
            )
        if not 0.0 < demand_proxy_quality <= 1.0:
            raise ValueError("demand_proxy_quality must be in (0, 1]")
        if revenue_noise_std is None and revenue_noise_pct < 0.0:
            raise ValueError("revenue_noise_pct must be non-negative")
        if revenue_noise_std is not None and revenue_noise_std < 0.0:
            raise ValueError("revenue_noise_std must be non-negative")

        _get_month_labels(plan_df)  # validates DatetimeIndex, fails fast

        if not 0.0 <= demand_spend_corr <= 1.0:
            raise ValueError("demand_spend_corr must be between 0 and 1 inclusive")
        if backphase_years < 0 or int(backphase_years) != backphase_years:
            raise ValueError("backphase_years must be a non-negative integer")
        self.supplied_history_df = history_df
        self.supplied_plan_df = plan_df
        self.backphase_years = int(backphase_years)
        if backphase_years > 0:
            # Counterfactual on the client's own history: treat the last
            # `backphase_years` years of history as if phasing had started
            # then, so the phased window is history's tail plus the plan.
            # Everything downstream just sees a longer plan.
            n_back = _WEEKS_PER_YEAR * self.backphase_years
            if len(history_df) - n_back < _MIN_HEAD_WEEKS:
                raise ValueError(
                    f"backphase_years={backphase_years} needs at least "
                    f"{n_back + _MIN_HEAD_WEEKS} weeks of history; got "
                    f"{len(history_df)}."
                )
            plan_df = pd.concat([history_df.iloc[-n_back:], plan_df])
            history_df = history_df.iloc[:-n_back]

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
        base_levers = levers if levers is not None else _default_levers(self.channels_)
        self.pinned_label_: str | None = None
        self.channel_constraints_: dict[
            str, float | Blackout | Redistribute | MonthStep
        ] = {}
        if strategy_pct is not None:
            pinned_spec: dict[str, float | Blackout | Redistribute | MonthStep] = {
                ch: strategy_pct for ch in self.channels_
            }
            if channel_constraints:
                unknown = set(channel_constraints) - set(self.channels_)
                if unknown:
                    raise ValueError(
                        "channel_constraints has unknown channel(s): "
                        f"{sorted(unknown)}. Expected a subset of {self.channels_}."
                    )
                pinned_spec.update(channel_constraints)
                self.channel_constraints_ = dict(channel_constraints)
            self.pinned_label_ = _pinned_strategy_label(
                strategy_pct,
                strategy_nudge_shape,
                strategy_balanced,
                channel_constraints,
            )
            base_levers = [
                *base_levers,
                (
                    self.pinned_label_,
                    pinned_spec,
                    strategy_nudge_shape,
                    strategy_balanced,
                ),
            ]
        elif channel_constraints is not None:
            raise ValueError(
                "channel_constraints requires strategy_pct to be set -- "
                "there is nothing to override on a swept, unpinned report."
            )
        self.levers_ = base_levers
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
        # Noise is set as a share of average weekly sales unless a GBP
        # override is given. Stored in GBP so every diagnostic below takes
        # it unchanged.
        if revenue_noise_std is None:
            self.revenue_noise_pct = float(revenue_noise_pct)
            self.revenue_noise_std = float(
                revenue_noise_pct * self.calibration_.total_sales
            )
        else:
            self.revenue_noise_std = float(revenue_noise_std)
            self.revenue_noise_pct = (
                self.revenue_noise_std / self.calibration_.total_sales
            )

        # Demand is built once, from the UNPHASED spend, and held fixed for
        # every candidate. Phasing can then only change how closely spend
        # tracks demand, never demand itself. Drawing demand independently
        # of spend (the old behaviour) left the two linked only when the
        # report seed happened to replay simulate_spend's own draws.
        unphased = pd.concat([history_df, plan_df])
        n_total = len(unphased)
        self.demand_spend_corr = demand_spend_corr
        if demand is not None:
            demand_arr = np.asarray(demand, dtype=float)
            if demand_arr.shape != (n_total,):
                raise ValueError(
                    "demand must be a 1-D series of length "
                    f"len(history_df) + len(plan_df) = {n_total}, "
                    f"got shape {demand_arr.shape}"
                )
            sd = demand_arr.std()
            if sd == 0:
                raise ValueError("demand has zero variance")
            self.demand_ = (demand_arr - demand_arr.mean()) / sd
            self.demand_link_ = None
        else:
            self.demand_link_ = link_demand_to_spend(
                unphased,
                demand_spend_corr=demand_spend_corr,
                process=demand_process,
                seed=seed,
            )
            self.demand_ = self.demand_link_.demand.to_numpy()
        self._unphased_spend = unphased
        self._bias_demand_cache: dict[int, np.ndarray] = {0: self.demand_}
        self.realised_demand_corr_ = {
            ch: float(np.corrcoef(unphased[ch].to_numpy(), self.demand_)[0, 1])
            for ch in self.channels_
        }

        self.results_: dict[str, dict] | None = None
        self.winner_: str | None = None
        self.winner_schedule_: pd.DataFrame | None = None
        self.schedules_: dict[str, pd.DataFrame] | None = None
        self.report_data_: dict | None = None
        self.time_to_benefit_: dict | None = None

    def _bias_demand(self, k: int) -> np.ndarray:
        """Demand series for bias draw k. Draw 0 is `demand_` itself.

        One demand path is one draw of luck: how a single trend path lines up
        with each channel's own slow movements moves that channel's bias by
        tens of points from seed to seed. The bias section therefore averages
        over several draws of the unlinked part of demand, all linked to the
        same unphased spend at the same demand_spend_corr. A supplied
        `demand` cannot be redrawn, so every draw reuses it (the proxy is
        still redrawn per draw).
        """
        if k not in self._bias_demand_cache:
            if self.demand_link_ is None:
                self._bias_demand_cache[k] = self.demand_
            else:
                self._bias_demand_cache[k] = link_demand_to_spend(
                    self._unphased_spend,
                    demand_spend_corr=self.demand_spend_corr,
                    process=self.demand_process,
                    seed=self.seed + _BIAS_DRAW_SEED_STEP * k,
                ).demand.to_numpy()
        return self._bias_demand_cache[k]

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
        n_phasing_seeds: int = 15,
        n_bias_draws: int = 100,
        id_n_sims: int = 50,
        id_b_candidates: np.ndarray | None = None,
        id_lam_candidates: np.ndarray | None = None,
        valley_tol: float = 0.01,
        proxy_seed: int = 0,
        fast_mode: bool = False,
        horizon_years: int = 3,
    ) -> DiscoveryReport:
        """Sweep every candidate lever, run the three diagnostics on each,
        and pick the report-wide "highest impact" winner.

        Parameters
        ----------
        n_sims:
            Noise draws per CollinearityDiagnostic fit (variance and bias
            sections). Default 50.
        n_phasing_seeds:
            Independent phased-schedule draws averaged per lever for the
            variance and identifiability sections (the unphased baseline
            always uses exactly 1 -- there's nothing random to average
            over). Default 15. Each draw costs one variance fit and one
            identifiability grid, so this is the expensive setting.
        n_bias_draws:
            Demand draws the bias section averages over. Draw k pairs
            demand draw k and proxy seed proxy_seed + k with phased
            schedule k (the same schedule the variance section uses for
            k < n_phasing_seeds); the unphased baseline runs one bias fit
            per draw on its single schedule. Default 100. Bias needs far
            more draws than variance: any single demand draw can line up
            with one channel by chance, and with 10 or more channels 15
            draws leaves enough noise to flip a strategy's bias result
            from better to worse than unphased, and with it the winner.
            A bias fit is one cheap regression per noise draw, so 100
            draws adds little to the run time.
        id_n_sims:
            Noise draws per IdentifiabilityDiagnostic fit. Default 50,
            matching n_sims. The grid solves every draw in one lstsq per
            grid point, so the cost grows well under linearly with it.
        id_b_candidates, id_lam_candidates:
            Forwarded to IdentifiabilityDiagnostic. Defaults to that
            class's own defaults (a wide 33x31 grid); shrink for a fast
            structural check.
        valley_tol:
            Forwarded to IdentifiabilityDiagnostic.summary()'s valley_pct.
        proxy_seed:
            Base seed for the bias section's demand-proxy draws. Draw j
            uses proxy_seed + j.
        fast_mode:
            If True, uses cheap settings throughout (n_sims=10,
            n_phasing_seeds=2, n_bias_draws=2, id_n_sims=5) -- for iterating on the report
            itself, not for numbers to hand a client. to_html() watermarks
            a fast-mode report as a draft.
        horizon_years:
            How many years of phasing to show in the Appendix's
            time-to-benefit view (default 3). Year 1 phases the plan year
            only (the main sweep's own result). Year 2 also phases the last
            year of the supplied history, year 3 the last two, each time as
            if phasing had started then -- a counterfactual on your own
            spend, no new data is simulated. The number of years is capped
            by how much history there is. Set to 1 to skip the projection.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10
            n_phasing_seeds = 2
            n_bias_draws = 2
            id_n_sims = 5
        if n_bias_draws < 1:
            raise ValueError("n_bias_draws must be at least 1")

        results: dict[str, dict] = {}
        schedules: dict[str, pd.DataFrame] = {}
        self.n_bias_draws_ = n_bias_draws

        for label, spec, nudge_shape, balance_signs in self.levers_:
            unphased = _is_unphased(spec)
            seeds = (
                [self.seed]
                if unphased
                else [self.seed + j for j in range(n_phasing_seeds)]
            )

            variance_draws = []
            revenue_mean_draws = []
            revenue_p10_draws = []
            revenue_p90_draws = []
            bias_draws = []
            bias_sims = []
            id_draws = []
            id_b_p10_draws = []
            id_b_p90_draws = []
            id_lam_p10_draws = []
            id_lam_p90_draws = []
            growth_draws = []
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
                revenue_mean_draws.append(var_summary["incremental_revenue_mean"])
                revenue_p10_draws.append(var_summary["incremental_revenue_p10"])
                revenue_p90_draws.append(var_summary["incremental_revenue_p90"])
                corr_draws.append(diag_var.correlation_matrix)

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

                # Recovered marginal return at each growth multiple: p10,
                # median and p90 across sims, as a (channel x quantile)
                # frame per multiple.
                marginal = diag_id.marginal_return_at(_GROWTH_MULTIPLIERS)
                growth_draws.append(
                    {
                        m: pd.DataFrame(
                            {
                                ch: marginal[ch][m].quantile([0.1, 0.5, 0.9])
                                for ch in self.channels_
                            }
                        ).T
                        for m in _GROWTH_MULTIPLIERS
                    }
                )

            # Bias runs on its own, larger set of draws. Demand draw k is
            # paired with phased schedule k, the same schedule the loop
            # above used while k < n_phasing_seeds. The unphased schedule
            # has no phasing randomness, so every draw reuses it.
            for k in range(n_bias_draws):
                phased_plan = (
                    representative_schedule
                    if unphased
                    else self._phase(spec, nudge_shape, balance_signs, self.seed + k)
                )
                mean_err, sims = self._bias_fit(
                    pd.concat([self.history_df, phased_plan]),
                    k,
                    n_sims,
                    proxy_seed + k,
                )
                bias_draws.append(mean_err)
                bias_sims.append(sims)

            variance_cv = pd.concat(variance_draws, axis=1).mean(axis=1)
            revenue_mean = pd.concat(revenue_mean_draws, axis=1).mean(axis=1)
            revenue_p10 = pd.concat(revenue_p10_draws, axis=1).mean(axis=1)
            revenue_p90 = pd.concat(revenue_p90_draws, axis=1).mean(axis=1)
            bias_pct = pd.concat(bias_draws, axis=1).mean(axis=1)
            # p10-p90 over every simulation from every draw pooled, so the
            # range includes the spread between demand draws, not only the
            # noise within one.
            pooled = pd.concat(bias_sims).groupby("channel")["error_pct"]
            bias_pct_p10 = pooled.quantile(0.1)
            bias_pct_p90 = pooled.quantile(0.9)
            # Per-channel DataFrame (b_mean, b_sd, ..., valley_pct), each
            # draw already indexed by channel -- average across draws
            # channel-by-channel, column-by-column.
            identifiability = pd.concat(id_draws).groupby(level=0).mean()
            id_b_p10 = pd.concat(id_b_p10_draws, axis=1).mean(axis=1)
            id_b_p90 = pd.concat(id_b_p90_draws, axis=1).mean(axis=1)
            id_lam_p10 = pd.concat(id_lam_p10_draws, axis=1).mean(axis=1)
            id_lam_p90 = pd.concat(id_lam_p90_draws, axis=1).mean(axis=1)
            growth = {}
            for m in _GROWTH_MULTIPLIERS:
                mean_q = sum(d[m] for d in growth_draws) / len(growth_draws)
                growth[f"{m:g}"] = {
                    "p10": mean_q[0.1].to_dict(),
                    "p50": mean_q[0.5].to_dict(),
                    "p90": mean_q[0.9].to_dict(),
                }
            correlation = {
                a: {
                    b: float(np.mean([m.loc[a, b] for m in corr_draws]))
                    for b in self.channels_
                }
                for a in self.channels_
            }

            results[label] = {
                "variance_cv": variance_cv.to_dict(),
                "revenue_mean": revenue_mean.to_dict(),
                "revenue_p10": revenue_p10.to_dict(),
                "revenue_p90": revenue_p90.to_dict(),
                "bias_pct": bias_pct.to_dict(),
                "bias_pct_p10": bias_pct_p10.to_dict(),
                "bias_pct_p90": bias_pct_p90.to_dict(),
                "identifiability": identifiability.to_dict(orient="index"),
                "b_p10": id_b_p10.to_dict(),
                "b_p90": id_b_p90.to_dict(),
                "lam_p10": id_lam_p10.to_dict(),
                "lam_p90": id_lam_p90.to_dict(),
                "growth": growth,
                "correlation": correlation,
                "scores": {
                    "variance": float(variance_cv.mean()),
                    "bias": float(bias_pct.abs().mean()),
                    "identifiability": float(identifiability["valley_pct"].mean()),
                    # Width of each channel's recovered p10-p90 range for the
                    # saturation exponent and adstock decay, averaged across
                    # channels: identifiability split into its two parts.
                    "saturation_range": float((id_b_p90 - id_b_p10).mean()),
                    "adstock_range": float((id_lam_p90 - id_lam_p10).mean()),
                },
            }
            schedules[label] = representative_schedule

        self.results_ = results
        self.valley_tol_ = valley_tol
        # A pinned strategy (self.pinned_label_) skips _pick_winner's
        # dominance check entirely: the strategy was chosen by the caller,
        # so there is nothing left to choose between. The sweep
        # still ran above (so the pinned candidate scores alongside every
        # other lever for the Appendix comparison table), only the
        # winner-SELECTION step is bypassed.
        self.winner_ = self.pinned_label_ or self._pick_winner(results)
        self.winner_schedule_ = schedules[self.winner_]
        self.schedules_ = schedules
        self.time_to_benefit_ = None
        if horizon_years > 1 and self.winner_ != self.levers_[0][0]:
            self.time_to_benefit_ = self._time_to_benefit(
                horizon_years=horizon_years,
                n_sims=n_sims,
                n_phasing_seeds=n_phasing_seeds,
                n_bias_draws=n_bias_draws,
                id_n_sims=id_n_sims,
                id_b_candidates=id_b_candidates,
                id_lam_candidates=id_lam_candidates,
                valley_tol=valley_tol,
                proxy_seed=proxy_seed,
            )
        self.report_data_ = self._build_report_data(fast_mode=fast_mode)
        return self

    def schedule_csv(self, path: str | None = None) -> pd.DataFrame:
        """Return the winning (or pinned) strategy's weekly schedule as a
        tidy, exportable table.

        One row per week, three columns per channel (the original plan
        figure, the recommended figure, and whether that week is a
        Blackout dark week), both £ columns rounded to the nearest penny.

        Parameters
        ----------
        path:
            If given, also writes the table to this path as a CSV.

        Returns
        -------
        pd.DataFrame indexed by week (same DatetimeIndex as plan_df), with
        columns "{channel}_original_plan", "{channel}_recommended", and
        "{channel}_dark_week" for every channel.
        """
        if self.winner_schedule_ is None:
            raise RuntimeError("Call fit() before schedule_csv().")

        recommended = self.winner_schedule_
        table = pd.DataFrame(index=self.plan_df.index)
        table.index.name = "week"
        for ch in self.channels_:
            table[f"{ch}_original_plan"] = self.plan_df[ch].round(2)
            table[f"{ch}_recommended"] = recommended[ch].round(2)
            table[f"{ch}_dark_week"] = recommended[ch].to_numpy() == 0.0

        if path is not None:
            table.to_csv(path)

        return table

    def _bias_fit(
        self, combined: pd.DataFrame, k: int, n_sims: int, proxy_seed: int
    ) -> tuple[pd.Series, pd.DataFrame]:
        """One bias-section fit on demand draw k.

        Returns the per-channel mean error % and the per-simulation
        (channel, error_pct) rows for pooled quantiles.
        """
        diag_bias = CollinearityDiagnostic(
            spend_df=combined,
            true_marginal_returns=self.true_marginal_returns,
            base_sales=self.calibration_.baseline_level,
            revenue_noise_std=self.revenue_noise_std,
            demand=self._bias_demand(k),
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
        mean_err = diag_bias.summary().set_index("channel")["mean_error_pct"]
        return mean_err, diag_bias.results_[["channel", "error_pct"]]

    def _scores_without_bias(
        self,
        combined: pd.DataFrame,
        demand: np.ndarray,
        n_sims: int,
        id_n_sims: int,
        id_b_candidates: np.ndarray | None,
        id_lam_candidates: np.ndarray | None,
        valley_tol: float,
        noise_seed: int,
    ) -> dict[str, float]:
        """Report-wide variance and identifiability scores for one
        history+plan spend frame -- the measures fit() scores every lever
        on (mean CV, mean valley %, mean range widths), on an
        arbitrary-length frame. Bias is scored separately because it runs
        on its own number of draws."""
        diag_var = CollinearityDiagnostic(
            spend_df=combined,
            true_marginal_returns=self.true_marginal_returns,
            base_sales=self.calibration_.baseline_level,
            revenue_noise_std=self.revenue_noise_std,
            demand=demand,
            demand_coef=self.calibration_.demand_coef,
            saturation=self.saturation,
            adstock=self.adstock,
            reference_spend=self.reference_spend_,
        )
        diag_var.fit(n_sims=n_sims, controls=True)
        variance = float(diag_var.summary()["coef_of_variation"].mean())

        diag_id = IdentifiabilityDiagnostic(
            spend_df=combined,
            demand=demand,
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
        diag_id.fit(n_sims=id_n_sims, noise_seed_offset=noise_seed)
        identifiability = float(diag_id.summary(tol=valley_tol)["valley_pct"].mean())
        b_range = np.mean(
            [
                diag_id.results_[ch]["recovered_b"].quantile(0.9)
                - diag_id.results_[ch]["recovered_b"].quantile(0.1)
                for ch in self.channels_
            ]
        )
        lam_range = np.mean(
            [
                diag_id.results_[ch]["recovered_lam"].quantile(0.9)
                - diag_id.results_[ch]["recovered_lam"].quantile(0.1)
                for ch in self.channels_
            ]
        )
        return {
            "variance": variance,
            "identifiability": identifiability,
            "saturation": float(b_range),
            "adstock": float(lam_range),
        }

    def _time_to_benefit(
        self,
        horizon_years: int,
        n_sims: int,
        n_phasing_seeds: int,
        n_bias_draws: int,
        id_n_sims: int,
        id_b_candidates: np.ndarray | None,
        id_lam_candidates: np.ndarray | None,
        valley_tol: float,
        proxy_seed: int,
    ) -> dict | None:
        """Winner's benefit if phasing had run for 1, 2, ... years.

        Year k phases the last k-1 years of the SUPPLIED history plus the
        plan (as if phasing had started k-1 years ago) and leaves the rest
        of the history as supplied. The unphased line is the same data
        unphased, so it is flat by construction: what changes across years
        is how much of the series has been phased. Nothing is simulated
        beyond what the report already simulates from spend. Year
        `backphase_years + 1` is the main sweep's own result and is reused.
        """
        baseline_label = self.levers_[0][0]
        spec, nudge_shape, balance_signs = next(
            (sp, ns, bs) for lbl, sp, ns, bs in self.levers_ if lbl == self.winner_
        )
        history = self.supplied_history_df
        plan = self.supplied_plan_df
        max_years = 1 + max(0, (len(history) - _MIN_HEAD_WEEKS) // _WEEKS_PER_YEAR)
        years = list(range(1, min(horizon_years, max_years) + 1))
        if len(years) < 2:
            return None

        axes = ("variance", "bias", "saturation", "adstock")
        # results_ score key for each axis
        key = {
            "variance": "variance",
            "bias": "bias",
            "saturation": "saturation_range",
            "adstock": "adstock_range",
        }
        base_scores = self.results_[baseline_label]["scores"]
        unphased = {ax: [base_scores[key[ax]]] * len(years) for ax in axes}
        phased: dict[str, list[float]] = {ax: [] for ax in axes}

        for k in years:
            if k - 1 == self.backphase_years:
                won = self.results_[self.winner_]["scores"]
                for ax in axes:
                    phased[ax].append(won[key[ax]])
                continue
            n_back = _WEEKS_PER_YEAR * (k - 1)
            head = history.iloc[: len(history) - n_back]
            window = pd.concat([history.iloc[len(history) - n_back :], plan])
            labels = _get_month_labels(window)

            def phased_frame(j: int) -> pd.DataFrame:
                phased_window = _generate_phased_schedule(
                    window,
                    labels,
                    alpha=1.0,
                    max_weekly_deviation_pct=spec,
                    seed=self.seed + j,
                    nudge_shape=nudge_shape,
                    balance_signs=balance_signs,
                )
                return pd.concat([head, phased_window])

            draws = [
                self._scores_without_bias(
                    phased_frame(j),
                    self.demand_,
                    n_sims,
                    id_n_sims,
                    id_b_candidates,
                    id_lam_candidates,
                    valley_tol,
                    self.seed + j,
                )
                for j in range(n_phasing_seeds)
            ]
            # Same pairing as fit(): demand draw j with phased schedule j,
            # then the mean error per channel across draws.
            bias_errs = [
                self._bias_fit(phased_frame(j), j, n_sims, proxy_seed + j)[0]
                for j in range(n_bias_draws)
            ]
            bias = float(pd.concat(bias_errs, axis=1).mean(axis=1).abs().mean())
            for ax in axes:
                phased[ax].append(
                    bias if ax == "bias" else float(np.mean([d[ax] for d in draws]))
                )

        improvement = {
            ax: [_safe_improvement(u, p) for u, p in zip(unphased[ax], phased[ax])]
            for ax in axes
        }
        return {
            "label": self.winner_,
            "years": years,
            "unphased": unphased,
            "phased": phased,
            "improvement": improvement,
        }

    def _pick_winner(self, results: dict[str, dict]) -> str:
        """Report-wide "highest impact" strategy: dominance check, else
        worst-axis.

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
                "demand_spend_corr": self.demand_spend_corr,
                "demand_supplied": self.demand_link_ is None,
                "realised_demand_corr": self.realised_demand_corr_,
                "n_bias_draws": self.n_bias_draws_,
                "saturation": self.saturation,
                "adstock": self.adstock,
                "baseline_share": self.baseline_share,
                "baseline_cv": self.baseline_cv,
                "revenue_noise_pct": self.revenue_noise_pct,
                "revenue_noise_std": self.revenue_noise_std,
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

    def growth_readout(self) -> pd.DataFrame:
        """Where can you grow: each channel's marginal return at higher
        spend, as the model would recover it unphased and under the winner.

        One row per channel and spend level (today, +50%, +100% of planned
        weekly spend). `true` is the return on the next GBP 1 at that level
        under the supplied curve. The p10-p90 columns are the range of
        recovered values across simulations, averaged across phasing draws.
        The ranges hold every other channel at its supplied curve and
        control for true demand, so they show curve uncertainty alone.

        Raises
        ------
        RuntimeError if fit() hasn't been called yet.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before growth_readout().")
        baseline = self.results_[self.levers_[0][0]]["growth"]
        best = self.results_[self.winner_]["growth"]
        rows = []
        for ch in self.channels_:
            for m in _GROWTH_MULTIPLIERS:
                key = f"{m:g}"
                rows.append(
                    {
                        "channel": ch,
                        "spend_change_pct": round(100 * (m - 1.0)),
                        "true": self.true_marginal_returns[ch]
                        * m ** (self.saturation[ch] - 1.0),
                        "unphased_p10": baseline[key]["p10"][ch],
                        "unphased_p90": baseline[key]["p90"][ch],
                        "phased_p10": best[key]["p10"][ch],
                        "phased_p90": best[key]["p90"][ch],
                    }
                )
        return pd.DataFrame(rows).round(4)

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
    # A pinned strategy (report.pinned_label_) skips
    # _pick_winner entirely, so the headline needs different wording --
    # there was no dominance check to describe.
    pinned = report.pinned_label_ is not None

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
    week_labels = [d.strftime("%b '%y") for d in combined_baseline.index]

    # Variance section: incremental-revenue forest chart (before/after,
    # £ p10-p90 range per channel) rather than a raw CV bar -- CV is the
    # metric the diagnostic optimizes, but £ revenue range is the number a
    # client actually feels, and matches docs/overview.html's own framing
    # of this same problem. Each mark also carries its mean incremental
    # revenue as a point estimate, drawn as a ring on the range bar.
    variance_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": {
                "range": (baseline["revenue_p10"][ch], baseline["revenue_p90"][ch]),
                "point": baseline["revenue_mean"][ch],
            },
            "after": {
                "range": (best["revenue_p10"][ch], best["revenue_p90"][ch]),
                "point": best["revenue_mean"][ch],
            },
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
    variance_change_text = _change_sentence(
        variance_narrowing, "The range", "narrows", "widens"
    )
    # The channel whose unphased range is widest relative to its true
    # revenue, quoted in the Diagnostics section.
    variance_widest = max(
        channels,
        key=lambda ch: (
            (baseline["revenue_p90"][ch] - baseline["revenue_p10"][ch])
            / max(true_revenue[ch], 1e-9)
        ),
    )
    variance_widest_text = (
        f"The widest range is {html.escape(variance_widest)}'s: "
        f"{_fmt_gbp(baseline['revenue_p10'][variance_widest])} to "
        f"{_fmt_gbp(baseline['revenue_p90'][variance_widest])}, against a true "
        f"{_fmt_gbp(true_revenue[variance_widest])}."
    )

    # Bias section: same chart family as variance (£ revenue, dashed true
    # line). "Believed revenue" is what a client would think they got if
    # they trusted the biased estimate: true revenue inflated/deflated by
    # the mean error %. bias_pct_p10/p90 give a band around that point,
    # the same combined range+point mark the variance section uses.
    def _believed_revenue(ch: str, results: dict) -> float:
        return true_revenue[ch] * (1.0 + results["bias_pct"][ch] / 100.0)

    def _believed_revenue_range(ch: str, results: dict) -> tuple[float, float]:
        return (
            true_revenue[ch] * (1.0 + results["bias_pct_p10"][ch] / 100.0),
            true_revenue[ch] * (1.0 + results["bias_pct_p90"][ch] / 100.0),
        )

    bias_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": {
                "range": _believed_revenue_range(ch, baseline),
                "point": _believed_revenue(ch, baseline),
            },
            "after": {
                "range": _believed_revenue_range(ch, best),
                "point": _believed_revenue(ch, best),
            },
            "truth": true_revenue[ch],
        }
        for ch in channels
    ]
    bias_svg = _svg_forest(bias_forest_data)
    bias_change_text = _bias_change_sentence(
        {ch: abs(baseline["bias_pct"][ch]) for ch in channels},
        {ch: abs(best["bias_pct"][ch]) for ch in channels},
    )
    signed_bias = {ch: baseline["bias_pct"][ch] for ch in channels}
    if len(channels) > _MAX_CHANNELS_IN_A_SENTENCE:
        bias_level_text = (
            "Mean absolute bias across channels is "
            f"{np.mean([abs(v) for v in signed_bias.values()]):.0f}%."
        )
    elif all(v > 0 for v in signed_bias.values()):
        bias_level_text = (
            "Every point estimate sits above the truth: "
            + _join_and(
                [f"{html.escape(ch)} by {v:.0f}%" for ch, v in signed_bias.items()]
            )
            + "."
        )
    else:
        bias_level_text = (
            "The point estimates miss the truth by "
            + _join_and(
                [f"{html.escape(ch)} {v:+.0f}%" for ch, v in signed_bias.items()]
            )
            + "."
        )
    if meta["demand_proxy_quality"] >= 1.0:
        bias_setup_text = (
            "Here the model is given a perfect demand proxy, so little bias "
            "is expected."
        )
    else:
        bias_setup_text = (
            "Here the model sees demand only through a proxy "
            f"({meta['demand_proxy_quality']:.0%} quality), as a real MMM would. "
            "Whatever the proxy misses gets credited to the channels."
        )

    # Identifiability section: same forest-chart family as sections 2/3,
    # not a picture of the RSS surface itself, which read as confusing
    # rather than illuminating. What the client wants to know: given a plausible
    # saturation/adstock per channel, how wide is the range of estimates
    # recovered when fitting unphased vs {winner}? One chart per parameter,
    # one row per channel -- IdentifiabilityDiagnostic profiles each
    # channel's own curvature separately, holding every other channel at
    # its own true value.
    def _fmt_plain(v: float) -> str:
        return f"{v:.2f}"

    # Point estimates: IdentifiabilityDiagnostic's own summary already
    # carries b_mean/lam_mean averaged across sims (see `identifiability`
    # above), so no new draws are needed.
    b_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": {
                "range": (baseline["b_p10"][ch], baseline["b_p90"][ch]),
                "point": baseline["identifiability"][ch]["b_mean"],
            },
            "after": {
                "range": (best["b_p10"][ch], best["b_p90"][ch]),
                "point": best["identifiability"][ch]["b_mean"],
            },
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
            "before": {
                "range": (baseline["lam_p10"][ch], baseline["lam_p90"][ch]),
                "point": baseline["identifiability"][ch]["lam_mean"],
            },
            "after": {
                "range": (best["lam_p10"][ch], best["lam_p90"][ch]),
                "point": best["identifiability"][ch]["lam_mean"],
            },
            "truth": report.adstock[ch],
        }
        for ch in channels
    ]
    lam_svg = _svg_forest(
        lam_forest_data, x_label="Adstock decay (lambda)", fmt=_fmt_plain
    )

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
    b_change_text = _change_sentence(
        b_narrowing, "The saturation range", "narrows", "widens"
    )
    lam_change_text = _change_sentence(
        lam_narrowing, "The adstock range", "narrows", "widens"
    )

    # Where can you grow: the recovered marginal return at double today's
    # spend, unphased vs the winner. The same curve uncertainty the
    # saturation chart shows, restated as the decision it feeds.
    def _fmt_return(v: float) -> str:
        return f"£{v:.2f}"

    def _growth_truth(ch: str, m: float) -> float:
        return report.true_marginal_returns[ch] * m ** (report.saturation[ch] - 1.0)

    def _growth_range(results: dict, ch: str, key: str) -> tuple[float, float]:
        return (results["growth"][key]["p10"][ch], results["growth"][key]["p90"][ch])

    growth_forest_data = [
        {
            "name": ch,
            "color": colors[ch],
            "before": {
                "range": _growth_range(baseline, ch, "2"),
                "point": baseline["growth"]["2"]["p50"][ch],
            },
            "after": {
                "range": _growth_range(best, ch, "2"),
                "point": best["growth"]["2"]["p50"][ch],
            },
            "truth": _growth_truth(ch, 2.0),
        }
        for ch in channels
    ]
    growth_svg = _svg_forest(
        growth_forest_data,
        x_label="Return on the next £1 at double today's spend",
        fmt=_fmt_return,
    )
    growth_narrowing = float(
        np.mean(
            [
                _safe_improvement(
                    _range_width(_growth_range(baseline, ch, "2")),
                    _range_width(_growth_range(best, ch, "2")),
                )
                for ch in channels
            ]
        )
    )
    growth_change_text = (
        f"{growth_narrowing:.0%} narrower"
        if growth_narrowing >= 0
        else f"{-growth_narrowing:.0%} wider"
    )
    # The channel whose unphased range at double spend is widest relative
    # to its true value: the clearest case of "the data cannot say".
    growth_widest = max(
        channels,
        key=lambda ch: (
            _range_width(_growth_range(baseline, ch, "2")) / _growth_truth(ch, 2.0)
        ),
    )
    growth_widest_lo, growth_widest_hi = _growth_range(baseline, growth_widest, "2")

    def _growth_cell(results: dict, ch: str, key: str) -> str:
        lo, hi = _growth_range(results, ch, key)
        return f"{_fmt_return(lo)} to {_fmt_return(hi)}"

    growth_rows_html = "".join(
        f"<tr><td>{html.escape(ch)}</td>"
        f"<td>{_fmt_return(_growth_truth(ch, 1.0))}</td>"
        f"<td>{_fmt_return(_growth_truth(ch, 1.5))}</td>"
        f"<td>{_growth_cell(baseline, ch, '1.5')}</td>"
        f"<td>{_growth_cell(best, ch, '1.5')}</td>"
        f"<td>{_fmt_return(_growth_truth(ch, 2.0))}</td>"
        f"<td>{_growth_cell(baseline, ch, '2')}</td>"
        f"<td>{_growth_cell(best, ch, '2')}</td></tr>"
        for ch in channels
    )

    corr_before_mean = _mean_pairwise(baseline["correlation"], channels)
    corr_after_mean = _mean_pairwise(best["correlation"], channels)
    corr_move = (
        "falls"
        if corr_after_mean < corr_before_mean - _NO_CHANGE
        else "rises"
        if corr_after_mean > corr_before_mean + _NO_CHANGE
        else "stays"
    )
    corr_change_text = (
        f"Mean pairwise correlation stays at {corr_before_mean:.2f}."
        if corr_move == "stays"
        else f"Mean pairwise correlation {corr_move} from {corr_before_mean:.2f} "
        f"to {corr_after_mean:.2f}."
    )
    corr_before_html = _corr_table_html(baseline["correlation"], channels)
    corr_after_html = _corr_table_html(best["correlation"], channels)

    # Diagnostics section: the unphased "before" half of each of the four
    # problem charts above, shown on its own ahead of any phasing solution
    # -- "how bad is the problem", before the reader has seen a fix. Built
    # by re-shaping the baseline-only fields already computed for the
    # before/after charts (no new numbers), via _svg_forest's single=True
    # mode.
    diag_variance_data = [
        {k: v for k, v in ch.items() if k != "after"} | {"value": ch["before"]}
        for ch in variance_forest_data
    ]
    diag_variance_svg = _svg_forest(diag_variance_data, single=True)

    diag_bias_data = [
        {k: v for k, v in ch.items() if k != "after"} | {"value": ch["before"]}
        for ch in bias_forest_data
    ]
    diag_bias_svg = _svg_forest(diag_bias_data, single=True)

    diag_b_data = [
        {k: v for k, v in ch.items() if k != "after"} | {"value": ch["before"]}
        for ch in b_forest_data
    ]
    diag_b_svg = _svg_forest(
        diag_b_data, x_label="Saturation exponent (b)", fmt=_fmt_plain, single=True
    )

    diag_lam_data = [
        {k: v for k, v in ch.items() if k != "after"} | {"value": ch["before"]}
        for ch in lam_forest_data
    ]
    diag_lam_svg = _svg_forest(
        diag_lam_data, x_label="Adstock decay (lambda)", fmt=_fmt_plain, single=True
    )

    lever_labels = [label for label, *_ in report.levers_]

    # Channel summary, part (a): plain per-channel inputs -- spend, ROI,
    # saturation, adstock, nothing modelled and nothing lever-dependent, so
    # no dropdown, no JS. Kept apart from what the model estimates under a
    # strategy, so "what you gave us" is never mixed with it. ROI
    # is true_marginal_returns[ch], the same £-per-£1-at-the-margin figure
    # the appendix dot-plot already shows -- no new computation.
    channel_summary_rows_html = "".join(
        f"<tr><td>{html.escape(ch)}</td>"
        f"<td>{_fmt_gbp(report.planned_spend_[ch])}</td>"
        f"<td>£{report.true_marginal_returns[ch]:.2f}</td>"
        f"<td>{report.saturation[ch]:.2f}</td>"
        f"<td>{report.adstock[ch]:.2f}</td></tr>"
        for ch in channels
    )

    # Scenario inputs, part (b): implied contribution -- the synthetic
    # OUTCOME these inputs produce, not something supplied (putting it in
    # the input table above would misrepresent a modelled number as client
    # data). Stacked area,
    # baseline at the bottom, each channel on top, summing to weekly
    # sales/revenue -- same terms channel_contributions/simulate_sales use
    # internally, exposed directly (true_contributions and
    # combined_baseline are already computed above for the variance/bias
    # truth lines). Baseline and demand are combined into one band rather
    # than split -- ties directly to the baseline_share sentence in the
    # table's own intro paragraph, and demand still gets its own dedicated
    # chart just above this one, so nothing is lost by not splitting it
    # here.
    contribution_band_order = ["Baseline", *channels]
    contribution_series = {
        "Baseline": (
            report.calibration_.baseline_level
            + report.calibration_.demand_coef * report.demand_
        ),
        **{ch: true_contributions[ch].to_numpy() for ch in channels},
    }
    contribution_colors = {"Baseline": "#9ca3af", **colors}
    contribution_svg = _svg_stacked_area(
        contribution_band_order,
        contribution_series,
        contribution_colors,
        y_label="Weekly sales / revenue",
        x_label="Week",
        x_tick_labels=week_labels,
    )
    contribution_legend = (
        '<div class="li"><span class="sw" style="background:#9ca3af"></span>Baseline</div>'
        + legend
    )

    # Channel summary, part (b): strategy-impact table, one row per
    # candidate lever -- replaces the four per-channel line charts a
    # single table reads faster for "which lever should I pick" than four
    # charts did. Variance/bias/identifiability columns are the exact same
    # report-wide (mean-across-channel) % improvement over unphased that
    # _pick_winner itself scores levers on, so the highlighted winning row
    # is provably consistent with the criterion that picked it -- just
    # evaluated for every lever, not only the winner, the same broadening
    # sections 2-4's own two-point comparisons already get elsewhere.
    #
    # Cost: % of true plan-period revenue given up by phasing, under the
    # assumed (possibly concave) response curve. Zero whenever a channel's
    # saturation is linear (b = 1) -- Jensen's inequality has nothing to
    # bite on. true_revenue (already computed above) is the unphased
    # baseline; each lever's own figure is built the exact same way
    # (channel_contributions on history + that lever's schedule, sliced to
    # the plan-only weeks) so the two are directly comparable. Shown here
    # as the mean across channels, alongside the three reliability columns
    # it trades off against.
    lever_cost_pct = {}
    for lbl in lever_labels:
        lever_contributions = channel_contributions(
            pd.concat([report.history_df, report.schedules_[lbl]]),
            report.true_marginal_returns,
            report.saturation,
            report.adstock,
            report.reference_spend_,
        )
        lever_revenue = {
            ch: float(lever_contributions[ch].iloc[-len(report.plan_df) :].sum())
            for ch in channels
        }
        lever_cost_pct[lbl] = {
            ch: 100 * _safe_improvement(true_revenue[ch], lever_revenue[ch])
            for ch in channels
        }

    # Winning strategy's own Cost, as a one-line callout at the end of
    # Section 3, so it is not visible only to someone who scrolls to the
    # appendix.
    winner_cost_pct = float(np.mean(list(lever_cost_pct[winner].values())))

    # Redistribute and MonthStep strategies move budget BETWEEN months (a
    # blackout run's budget lands in a recipient month; a step scales a
    # whole month); only each channel's 12-month block total is preserved. Every other strategy preserves each
    # month's total exactly, which is what the copy below normally says.
    winner_spec = next(spec for lbl, spec, *_ in report.levers_ if lbl == winner)
    if _moves_budget_between_months(winner_spec):
        totals_sub = (
            "annual totals unchanged from unphased; budget moves between months"
        )
        totals_kept = (
            "Each channel's annual budget is unchanged. Some budget moves "
            "between months."
        )
        red = next(
            (v for v in winner_spec.values() if isinstance(v, Redistribute)), None
        )
        if red is None:
            how_months_move = "each month is stepped up or down"
        else:
            moves = []
            if red.dark_weeks > 0:
                moves.append("a blackout run's budget lands in a recipient month")
            if red.high_month_pct > 0:
                moves.append("a peak month is funded by a small cut to the others")
            how_months_move = (
                ", and ".join(moves) or "weeks are nudged within the month"
            )
        totals_pacing = (
            "Each channel's annual total is identical on both sides, but "
            f"this strategy also moves budget between months: {how_months_move}, "
            "so individual monthly totals differ."
        )
    else:
        totals_sub = "monthly totals unchanged from unphased"
        totals_kept = (
            "Every month's total is unchanged. Only the timing within each month moves."
        )
        totals_pacing = (
            "The monthly totals are identical on both sides; only the "
            "timing within each month has moved."
        )
    corr_after_sub = (
        "Pearson correlation, weekly spend by channel &middot; plan year "
        f"only, {totals_sub}"
    )

    # Channel constraints table: only rendered when the pinned strategy
    # actually overrides specific channels -- an unpinned, swept report has no
    # per-channel overrides to show, and a pinned strategy with none set
    # doesn't need an empty table either.
    channel_constraints_html = ""
    if report.channel_constraints_:
        rows = ""
        for ch, override in report.channel_constraints_.items():
            if isinstance(override, Blackout):
                value_text = "Blackout"
            elif isinstance(override, Redistribute):
                value_text = _redistribute_label(override)
            elif isinstance(override, MonthStep):
                value_text = f"{override.step_pct:.0f}% month step"
            else:
                value_text = f"+/-{override:.0f}%"
            rows += (
                f"<tr><td>{html.escape(ch)}</td><td>{html.escape(value_text)}</td></tr>"
            )
        channel_constraints_html = f"""
  <h3>Channel constraints</h3>
  <p>These channels were pinned to their own value, overriding what
  <b>{winner}</b> would otherwise apply. Every other channel follows that
  strategy as supplied.</p>
  <div class="table-scroll">
  <table class="cross-table">
    <thead><tr><th>Channel</th><th>Override</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>"""

    time_to_benefit_html = _time_to_benefit_html(report)
    strategy_glossary_html = _strategy_glossary_html(report)

    baseline_scores = baseline["scores"]
    impact_table_rows_html = ""
    for lbl in lever_labels:
        scores = report.results_[lbl]["scores"]
        variance_impact = 100 * _safe_improvement(
            baseline_scores["variance"], scores["variance"]
        )
        bias_impact = 100 * _safe_improvement(baseline_scores["bias"], scores["bias"])
        sat_impact = 100 * _safe_improvement(
            baseline_scores["saturation_range"], scores["saturation_range"]
        )
        adstock_impact = 100 * _safe_improvement(
            baseline_scores["adstock_range"], scores["adstock_range"]
        )
        cost_impact = float(np.mean(list(lever_cost_pct[lbl].values())))
        peak_week = _peak_week_multiple(report.plan_df, report.schedules_[lbl])
        is_winner = lbl == winner
        row_class = ' class="winner-row"' if is_winner else ""
        winner_tag = ' <span class="winner-tag">Recommended</span>' if is_winner else ""
        impact_table_rows_html += (
            f'<tr data-lever="{html.escape(lbl)}"{row_class}>'
            f"<td>{html.escape(lbl)}{winner_tag}</td>"
            f"<td>{variance_impact:.0f}%</td>"
            f"<td>{bias_impact:.0f}%</td>"
            f"<td>{sat_impact:.0f}%</td>"
            f"<td>{adstock_impact:.0f}%</td>"
            f"<td>{cost_impact:.2f}%</td>"
            f"<td>{peak_week:.1f}x</td></tr>"
        )

    # Channel summary, part (c): recommended pacing -- as-supplied vs the
    # winner's own phased schedule, one small chart per channel. Both
    # series already share plan_df's shape (every lever's schedule is
    # plan-period only), so no history slicing needed here the way the
    # appendix spend chart needs it. Each channel's OWN colour, pale vs
    # solid (via _lighten_hex), not a channel-blind grey/black pair --
    # grey in particular read as too faint against the page background.
    # Only the plan year is shown, even when back-phasing (see the
    # constructor's backphase_years) phased earlier weeks of history too.
    n_plan_weeks = len(report.supplied_plan_df)
    backphase_note = (
        ""
        if report.backphase_years == 0
        else (
            f"Only the plan year is shown; the {report.backphase_years} "
            "earlier year(s) of history in the phased window were phased the "
            "same way."
        )
    )
    pacing_plan = report.plan_df.iloc[-n_plan_weeks:]
    pacing_schedule = report.winner_schedule_.iloc[-n_plan_weeks:]
    plan_week_labels = [d.strftime("%b '%y") for d in pacing_plan.index]
    pacing_cells_html = "".join(
        f'<div class="pacing-cell"><div class="pacing-title">{html.escape(ch)}</div>'
        + _svg_multiline(
            {
                "Plan": pacing_plan[ch].to_numpy(),
                "Recommended": pacing_schedule[ch].to_numpy(),
            },
            {"Plan": _lighten_hex(colors[ch]), "Recommended": colors[ch]},
            width=320,
            height=170,
            normalize=False,
            y_label="Weekly spend",
            x_label="Plan week",
            x_tick_labels=plan_week_labels,
            n_x_ticks=4,
        )
        + "</div>"
        for ch in channels
    )

    # Scenario inputs, part (c): the assumed response curves (adstock
    # decay, saturation), then the actual weekly spend and demand series
    # behind the plan -- everything this report is built on, so anyone
    # could reproduce every chart in this report from these inputs alone,
    # in a notebook, without this report class. Marginal return doesn't
    # get its own chart here -- it's already the ROI column in the
    # channel-summary table above, a dot-plot of the same numbers would be
    # a duplicate, not a new fact.
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
            "every channel in this report, so there is no carryover to "
            "plot.</em></p>"
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
            "channel in this report, so there is no curvature to plot."
            "</em></p>"
        )

    # Scenario inputs spend: one small chart per channel, own axis and
    # colour, same pacing-grid/pacing-cell markup Section 4 uses for its
    # before/after pacing charts -- rather than one combined multi-line
    # chart needing a channel legend to read.
    spend_cells_html = "".join(
        f'<div class="pacing-cell"><div class="pacing-title">{html.escape(ch)}</div>'
        + _svg_multiline(
            {ch: combined_baseline[ch].to_numpy()},
            {ch: colors[ch]},
            width=320,
            height=170,
            normalize=False,
            y_label="Weekly spend",
            x_label="Week",
            x_tick_labels=week_labels,
            n_x_ticks=4,
            highlight_from=len(combined_baseline) - len(report.supplied_plan_df),
            highlight_label="Plan year",
        )
        + "</div>"
        for ch in channels
    )

    # No standalone demand chart -- it's a zero-mean synthetic series with
    # no interpretive hook on its own, and its actual effect on sales is
    # already visible in Implied Contribution's Baseline band below
    # (baseline level + demand fluctuation combined).
    #
    # No separate "Sales / revenue, weekly" total-line chart any more --
    # the Implied Contribution stacked-area chart above already shows
    # this same total (it's the top edge of the stack), broken down by
    # source rather than flattened into one line, so a second chart here
    # would just be a strictly-worse duplicate.

    # Headline numbers and the cost sentence, worded from the scores so
    # they stay true on any data.
    base_scores, best_scores = baseline["scores"], best["scores"]

    def gain(key: str) -> float:
        return _safe_improvement(base_scores[key], best_scores[key])

    headline_numbers = (
        "Against the unphased plan: variance "
        f"{_improvement_phrase(gain('variance'), 'better', 'worse')}, bias "
        f"{_improvement_phrase(gain('bias'), 'better', 'worse')}, saturation range "
        f"{_improvement_phrase(gain('saturation_range'), 'narrower', 'wider')} and "
        "adstock range "
        f"{_improvement_phrase(gain('adstock_range'), 'narrower', 'wider')}."
    )
    if abs(winner_cost_pct) < 0.005:
        cost_text = (
            f"<b>{winner}</b> gives up no plan-year revenue here: the timing "
            "changes cost nothing under the supplied response curves."
        )
    elif winner_cost_pct > 0:
        cost_text = (
            f"<b>{winner}</b> gives up {winner_cost_pct:.2f}% of plan-year "
            "revenue, averaged across channels. Spend that is bunched up "
            "runs further into each channel's saturation curve."
        )
    else:
        cost_text = (
            f"<b>{winner}</b> adds {-winner_cost_pct:.2f}% to plan-year "
            "revenue under the supplied response curves, averaged across "
            "channels."
        )
    n_candidates = len(lever_labels) - 1
    picked_text = (
        "This strategy was set directly, not picked by the comparison."
        if pinned
        else "It is picked on variance, bias and identifiability. Cost is "
        "shown beside it and is not part of the pick."
    )
    picked_caption = (
        "The highlighted strategy was set directly, not picked."
        if pinned
        else "The highlighted strategy is picked on variance, bias and "
        "identifiability, not on cost."
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
  <p class="report-sub">A phasing strategy changes when the plan's spend
  lands. Each channel's annual budget stays the same. This report scores
  {n_candidates} {"strategy" if n_candidates == 1 else "strategies"} against
  the unphased plan on three problems: variance, bias, and whether
  saturation and adstock can be recovered.</p>
  <div class="cover-meta">
    {stat_box("Client", meta["client_name"] or "not set")}
    {stat_box("Plan year", meta["plan_year"] or "not set")}
    {stat_box("Plan weeks", str(meta["n_weeks_plan"]))}
    {stat_box("Recommended", winner)}
  </div>
</header>

<nav class="toc" aria-label="Report sections">
  <a href="#scenario-inputs">1 &middot; Scenario inputs</a>
  <a href="#diagnostics">2 &middot; Diagnostics</a>
  <a href="#impact">3 &middot; Impact</a>
  <a href="#phased-spend">4 &middot; Phased spend</a>
  <a href="#appendix">5 &middot; Appendix</a>
</nav>

<div class="headline">{"Pinned" if pinned else "Recommended"} strategy:
<b>{winner}</b>. {headline_numbers} {picked_text}</div>

<main>

<section id="scenario-inputs">
  <div class="s-label">Section 1</div>
  <h2>Scenario inputs</h2>
  <p>Everything the report is built on: each channel's planned spend,
  marginal return (ROI), saturation and adstock, the weekly spend behind
  the plan, and what those imply for weekly sales. Background demand
  accounts for {meta["baseline_share"]:.0%} of sales, leaving
  {1 - meta["baseline_share"]:.0%} for the channels.</p>
  <div class="table-scroll">
  <table class="cross-table">
    <thead><tr><th>Channel</th><th>Spend</th><th>ROI</th><th>Saturation</th><th>Adstock</th></tr></thead>
    <tbody>{channel_summary_rows_html}</tbody>
  </table>
  </div>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Spend, history + plan</div>
      <div class="fig-sub">Actual weekly spend by channel, as supplied, before any phasing</div>
    </div>
    <div class="fig-body">
      <div class="pacing-grid">{spend_cells_html}</div>
    </div>
  </div>

  {adstock_fig_html}

  {saturation_fig_html}

  <h3>Implied contribution</h3>
  <p>Background demand plus each channel's contribution, stacked into
  weekly sales. Nothing here was supplied directly. It follows from the
  inputs above, which makes it a check on those assumptions.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Sales / revenue, weekly, by source</div>
      <div class="fig-sub">Baseline (incl. demand) plus each channel's true contribution, history + plan</div>
    </div>
    <div class="fig-body">
      <div class="legend">{contribution_legend}</div>
      {contribution_svg}
    </div>
  </div>
</section>

<section id="diagnostics">
  <div class="s-label">Section 2</div>
  <h2>Diagnostics</h2>
  <p>The unphased plan, exactly as supplied: the spend correlation behind
  the problem, then variance, bias and identifiability. Section 3 shows
  the same charts after phasing.</p>

  <h3>Spend correlation</h3>
  <p>Channels whose spend moves together are hard for a model to tell
  apart. Mean pairwise correlation across the plan year is
  {corr_before_mean:.2f}.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Channel correlation, unphased</div>
      <div class="fig-sub">Pearson correlation, weekly spend by channel &middot; plan year only</div>
    </div>
    <div class="fig-body">
      {corr_before_html}
    </div>
  </div>

  <h3>Variance</h3>
  <p>If the model were refit on a slightly different version of the same
  history, how far would its answer move? The model is given the true
  demand and curve shapes, so this is a best case.
  {variance_widest_text}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Incremental revenue, unphased</div>
      <div class="fig-sub">Model-estimated range per channel, no phasing applied</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Revenue at the true marginal return</span>
      </div>
      {diag_variance_svg}
    </div>
    <p class="fig-cap">Each channel's estimated incremental revenue on the
    plan: the bar is the p10 to p90 range across simulations and the ring
    its mean. The dashed line is the revenue implied by the marginal
    return you supplied.</p>
  </div>

  <h3>Bias</h3>
  <p>{bias_setup_text} {bias_level_text}
  {_demand_link_sentence(meta)}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Revenue implied by the biased estimate, unphased</div>
      <div class="fig-sub">What a client would believe they got, per channel</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> Believed today (unphased), range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> True revenue</span>
      </div>
      {diag_bias_svg}
    </div>
    <p class="fig-cap">The ring is the revenue the biased estimate implies
    and the bar its p10 to p90 range across simulations. The gap to the
    dashed line is the bias.</p>
  </div>

  <h3>Identifiability</h3>
  <p>Can the model recover each channel's saturation and adstock? One
  channel at a time, every combination of saturation exponent and adstock
  decay is tried and the best fit kept. A range that covers most of the
  search means the data cannot tell the curves apart.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Recovered saturation, by channel, unphased</div>
      <div class="fig-sub">Saturation exponent (b): model-recovered range, no phasing applied</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {diag_b_svg}
    </div>
    <p class="fig-cap">The range of saturation exponents the model
    recovers for each channel (p10 to p90 across simulations) and its
    mean, with every other channel held at its supplied curve.</p>
  </div>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Recovered adstock, by channel, unphased</div>
      <div class="fig-sub">Adstock decay (lambda): model-recovered range, no phasing applied</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {diag_lam_svg}
    </div>
    <p class="fig-cap">The same for adstock decay: how long a week's
    spend keeps working.</p>
  </div>
</section>

<section id="impact">
  <div class="s-label">Section 3</div>
  <h2>Impact</h2>
  <p>The same charts as Section 2, after phasing under <b>{winner}</b>.</p>

  <h3>Spend correlation</h3>
  <p>{corr_change_text} {totals_kept}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Channel correlation, after phasing</div>
      <div class="fig-sub">{corr_after_sub}</div>
    </div>
    <div class="fig-body">
      {corr_after_html}
    </div>
  </div>

  <h3>Variance</h3>
  <p>{variance_change_text}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Incremental revenue range, before and after</div>
      <div class="fig-sub">Incremental revenue, the model-estimated range: unphased vs {
        winner
    }</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }, range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Revenue at the true marginal return</span>
      </div>
      {variance_svg}
    </div>
    <p class="fig-cap">The model's estimated range for incremental
    revenue on the unphased plan (faded) and the phased plan (solid),
    against the truth.</p>
  </div>

  <h3>Bias</h3>
  <p>{bias_change_text}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Point estimates, before and after</div>
      <div class="fig-sub">Revenue implied by the biased estimate: unphased vs {
        winner
    }</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Believed today (unphased), range</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> Believed, {
        html.escape(winner)
    }, range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> True revenue</span>
      </div>
      {bias_svg}
    </div>
    <p class="fig-cap">The revenue the biased estimate implies, before
    and after phasing, when demand is only seen through a proxy.</p>
  </div>

  <h3>Identifiability</h3>
  <p>{b_change_text} {lam_change_text}</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Saturation range, before and after</div>
      <div class="fig-sub">Recovered saturation exponent (b): unphased vs {winner}</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }, range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {b_svg}
    </div>
    <p class="fig-cap">The range of saturation exponents the model
    recovers for each channel (p10 to p90 across simulations) and its
    mean, with every other channel held at its supplied curve.</p>
  </div>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Adstock range, before and after</div>
      <div class="fig-sub">Recovered adstock decay (lambda): unphased vs {winner}</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }, range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Point estimate</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Plausible value supplied</span>
      </div>
      {lam_svg}
    </div>
    <p class="fig-cap">The same for adstock decay: how long a week's
    spend keeps working.</p>
  </div>

  <h3>Where you can grow</h3>
  <p>Whether a channel can take more budget depends on what the next £1
  returns at a higher spend level, and that depends on its saturation
  curve. Unphased, the model's answer for {html.escape(growth_widest)} at
  double today's spend runs from {_fmt_return(growth_widest_lo)} to
  {_fmt_return(growth_widest_hi)}, against
  {_fmt_return(_growth_truth(growth_widest, 2.0))} under the supplied
  curve. Under <b>{winner}</b> the ranges at double spend are
  {growth_change_text} on average.</p>
  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Return at double today's spend, by channel</div>
      <div class="fig-sub">Recovered return on the next £1 if weekly spend doubled: unphased vs {
        winner
    }</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.35"/></svg> Unphased (today), range</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af"/></svg> {
        html.escape(winner)
    }, range</span>
        <span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#fff" stroke="#9ca3af" stroke-width="1.8"/></svg> Median</span>
        <span class="li"><svg width="12" height="14"><line x1="6" y1="1" x2="6" y2="13" stroke="#111827" stroke-width="1.6" stroke-dasharray="3,2"/></svg> Value under the supplied curve</span>
      </div>
      {growth_svg}
    </div>
    <p class="fig-cap">The return the fitted curve gives at double the
    planned weekly spend: p10 to p90 across simulations, with the median.
    Other channels are held at their supplied curves, so this is curve
    uncertainty alone. A real model carries the variance and bias above
    on top of it.</p>
  </div>
  <div class="table-scroll">
  <table class="cross-table">
    <thead>
      <tr><th rowspan="2">Channel</th><th rowspan="2">Today, supplied</th>
      <th colspan="3">At +50% spend</th><th colspan="3">At +100% spend</th></tr>
      <tr><th>Supplied curve</th><th>Unphased</th><th>{html.escape(winner)}</th>
      <th>Supplied curve</th><th>Unphased</th><th>{html.escape(winner)}</th></tr>
    </thead>
    <tbody>{growth_rows_html}</tbody>
  </table>
  </div>

  <h3>Cost</h3>
  <p>{cost_text} Each channel's annual budget is unchanged and no extra
  spend is needed.</p>
</section>

<section id="phased-spend">
  <div class="s-label">Section 4</div>
  <h2>Phased spend</h2>
  <p>Each panel shows one channel's planned weekly spend (pale) and its
  phased schedule under <b>{winner}</b> (solid). {totals_pacing}
  {backphase_note}</p>
  <div class="pacing-grid">{pacing_cells_html}</div>
</section>

<section id="appendix">
  <div class="s-label">Section 5</div>
  <h2>Appendix: every strategy compared</h2>
  <p>What each strategy does to the plan, then how each one scores
  against the unphased plan. <b>{winner}</b> is highlighted. Sections 2 to
  4 are built from that row.</p>
  {strategy_glossary_html}
  <div class="table-scroll">
  <table class="cross-table">
    <thead><tr><th>Strategy</th><th>Variance impact</th><th>Bias impact</th><th>Saturation impact</th><th>Adstock impact</th><th>Cost</th><th>Peak week</th></tr></thead>
    <tbody>{impact_table_rows_html}</tbody>
  </table>
  </div>
  <p class="fig-cap">Impact is the improvement on the unphased plan,
  averaged across channels. Saturation and adstock are how much narrower
  the recovered range gets. Cost is the share of plan-year revenue given
  up under the supplied response curves. Peak week is the biggest week of
  spend as a multiple of that week's plan: the check on whether a media
  buyer can book it. {picked_caption}</p>
  {time_to_benefit_html}
  {channel_constraints_html}
</section>

</main>
</div>
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
.toc {
  position: sticky; top: 0; z-index: 20; background: #fff;
  padding: 0 2rem; border-bottom: 1px solid var(--border);
  display: flex; overflow-x: auto; gap: 0;
}
.toc a {
  flex-shrink: 0; padding: .75rem 0; margin-right: 1.75rem;
  text-decoration: none; font-size: .82rem; font-weight: 500;
  color: var(--muted); border-bottom: 2px solid transparent;
  white-space: nowrap; transition: color .15s, border-color .15s;
}
.toc a:hover { color: var(--text); border-bottom-color: var(--light); }
.headline { margin: 2rem 2rem 0; padding: 1.1rem 1.4rem; border-left: 3px solid var(--good); background: var(--bg); border-radius: 0 6px 6px 0; font-size: 1.05rem; font-weight: 600; color: #1f2937; }
.headline b { color: var(--good); }
main { padding: 0 2rem 3rem; }
p { margin-bottom: 1.05rem; max-width: 64ch; font-size: .98rem; }
section { margin-top: 2.75rem; padding-top: 2rem; border-top: 1px solid var(--border); }
.s-label { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
h2 { font-size: 1.3rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.25; margin-bottom: .9rem; }
h3 { font-size: 1.05rem; font-weight: 700; letter-spacing: -.01em; margin: 1.75rem 0 .5rem; }
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
.table-scroll { overflow-x: auto; }
table.corr-table, table.cross-table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .85rem; }
table.corr-table th, table.corr-table td, table.cross-table th, table.cross-table td { border: 1px solid var(--border); padding: .4rem .6rem; text-align: center; }
table.corr-table { table-layout: fixed; }
table.corr-table tr > th:first-child { width: 9rem; }
table.corr-table th.col-hdr {
  height: 7.75rem; width: 2.1rem; min-width: 2.1rem; max-width: 2.1rem;
  padding: 0 0 .5rem 0; vertical-align: bottom; text-align: center;
}
table.corr-table td { width: 2.1rem; }
table.corr-table th.col-hdr .col-hdr-label {
  display: inline-block; writing-mode: vertical-rl; transform: rotate(180deg);
  white-space: nowrap; font-weight: 600;
}
table.cross-table th { background: var(--bg); }
tr.winner-row td { background: #ecfdf5; font-weight: 700; }
.winner-tag { display: inline-block; font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; color: var(--good); background: #d1fae5; border-radius: 4px; padding: .1rem .4rem; margin-left: .35rem; }
.pacing-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0; }
@media (max-width: 620px) { .pacing-grid { grid-template-columns: 1fr; } }
.pacing-cell { border: 1px solid var(--border); border-radius: 8px; padding: .75rem .75rem .3rem; }
.pacing-title { font-size: .82rem; font-weight: 700; margin-bottom: .3rem; }
table.glossary-table td { text-align: left; vertical-align: top; }
table.glossary-table td.strat-name { font-weight: 600; white-space: nowrap; }
"""
