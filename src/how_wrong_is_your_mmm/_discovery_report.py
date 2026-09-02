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

Report structure: cover -> correlation (before) -> variance problem +
impact -> bias problem + impact -> identifiability problem + impact ->
correlation (after) -> cross-strategy table (every candidate, with a
dropdown) -> appendix (spend/demand time series, saturation curve).

Follows the package's shared-DGP design (session 44): one demand series
and one (saturation, adstock) assumption drive every simulated sales
column in this report. Only the OLS model's KNOWLEDGE of demand changes
between sections -- known for variance and identifiability, a
measurement-error proxy (at the client's supplied plausible quality) for
bias -- so a single demand draw and a single curvature assumption are
resolved once in __init__ and reused throughout, never redrawn per
candidate or per section.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import (
    _DEFAULT_MARGINAL_RETURNS,
    calibrate_baseline,
    simulate_demand,
)
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._identifiability import IdentifiabilityDiagnostic
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
    height: int = 170,
    pad_left: int = 40,
    pad_bottom: int = 20,
    pad_top: int = 10,
    pad_right: int = 10,
    normalize: bool = True,
) -> str:
    """Render a small multi-series line chart as a self-contained inline SVG.

    No JS, no external library -- every point is computed in Python and
    baked into the polyline points at render time, since none of this
    changes after the report-wide winner is picked.

    normalize:
        If True (default), each series is min-max scaled INDEPENDENTLY
        onto the same y-axis, so shape (not absolute level) is what's
        compared -- appropriate for the spend time series, where channels
        can sit on very different budgets and the week-to-week pattern is
        the point. If False, every series shares ONE y-axis scaled to the
        combined min/max across all of them -- required for the
        saturation curve: channels' response curves are the same power-law
        SHAPE up to a scale factor, so independently-normalized curves for
        different channels would land on top of each other and hide the
        one thing worth seeing (their different levels).
    """
    n = len(next(iter(series.values())))
    if n < 2:
        return "<p><em>Not enough weeks to plot.</em></p>"
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i: int) -> float:
        return pad_left + plot_w * i / (n - 1)

    shared_lo, shared_hi = None, None
    if not normalize:
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
        shared_lo, shared_hi = float(all_vals.min()), float(all_vals.max())

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    # Baseline axis.
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" '
        f'x2="{width - pad_right}" y2="{height - pad_bottom}" '
        f'stroke="#d1d5db" stroke-width="1"/>'
    )
    for name, values in series.items():
        arr = np.asarray(values, dtype=float)
        if normalize:
            lo, hi = arr.min(), arr.max()
        else:
            lo, hi = shared_lo, shared_hi
        span = hi - lo if hi > lo else 1.0
        points = " ".join(
            f"{x_at(i):.1f},{pad_top + plot_h * (1 - (v - lo) / span):.1f}"
            for i, v in enumerate(arr)
        )
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


def _bar_pair_html(
    label: str, before: float, after: float, color: str, fmt: str = "{:.1f}%"
) -> str:
    """One labelled before/after horizontal-bar comparison row."""
    scale = max(abs(before), abs(after), 1e-9)
    before_w = 100 * abs(before) / scale
    after_w = 100 * abs(after) / scale
    return f"""
    <div class="bar-row">
      <div class="bar-label">{label}</div>
      <div class="bar-track">
        <div class="bar bar-before" style="width:{before_w:.1f}%"></div>
        <span class="bar-val">{fmt.format(before)} &rarr; </span>
      </div>
      <div class="bar-track">
        <div class="bar bar-after" style="width:{after_w:.1f}%; background:{color}"></div>
        <span class="bar-val">{fmt.format(after)}</span>
      </div>
    </div>"""


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
        The report's single shared plausible saturation exponent b in
        (0, 1] (1.0 = linear, default) and adstock decay lambda in
        [0, 1) (0.0 = no carryover, default), applied uniformly to every
        channel. Unlike CollinearityDiagnostic/BudgetPhaser, these are
        plain floats here, not per-channel dicts -- IdentifiabilityDiagnostic's
        grid search (used in the identifiability section) assumes one
        shared curvature for the whole plan, consistent with the
        package's shared-DGP design (session 44); giving variance/bias a
        per-channel curvature while identifiability only understands one
        shared value would make the "one true world" claim false.
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
        saturation: float = 1.0,
        adstock: float = 0.0,
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
        if not 0.0 < saturation <= 1.0:
            raise ValueError("saturation must be in (0, 1]")
        if not 0.0 <= adstock < 1.0:
            raise ValueError("adstock must be in [0, 1)")

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
        self.saturation = saturation
        self.adstock = adstock
        self.revenue_noise_std = revenue_noise_std
        self.client_name = client_name
        self.plan_year = plan_year
        self.seed = seed
        self.demand_process = demand_process

        self.channels_ = list(plan_df.columns)
        self.levers_ = levers if levers is not None else _default_levers(self.channels_)
        self._plan_month_labels = _get_month_labels(plan_df)

        # Fixed across every candidate so every strategy is priced against
        # the SAME response curve -- same reasoning as BudgetPhaser's own
        # reference_spend default (see that class's docstring).
        self.reference_spend_ = {ch: float(plan_df[ch].mean()) for ch in self.channels_}

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
            bias_draws = []
            id_draws = []
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
                var_summary = diag_var.summary().set_index("channel")
                variance_draws.append(var_summary["coef_of_variation"])
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
                id_draws.append(diag_id.summary(tol=valley_tol))

            variance_cv = pd.concat(variance_draws, axis=1).mean(axis=1)
            bias_pct = pd.concat(bias_draws, axis=1).mean(axis=1)
            identifiability = pd.concat(id_draws, axis=1).mean(axis=1)
            correlation = {
                a: {
                    b: float(np.mean([m.loc[a, b] for m in corr_draws]))
                    for b in self.channels_
                }
                for a in self.channels_
            }

            results[label] = {
                "variance_cv": variance_cv.to_dict(),
                "bias_pct": bias_pct.to_dict(),
                "identifiability": identifiability.to_dict(),
                "correlation": correlation,
                "scores": {
                    "variance": float(variance_cv.mean()),
                    "bias": float(bias_pct.abs().mean()),
                    "identifiability": float(identifiability["valley_pct"]),
                },
            }
            schedules[label] = representative_schedule

        self.results_ = results
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

    variance_rows = "".join(
        _bar_pair_html(
            ch,
            100 * baseline["variance_cv"][ch],
            100 * best["variance_cv"][ch],
            colors[ch],
        )
        for ch in channels
    )
    bias_rows = "".join(
        _bar_pair_html(ch, baseline["bias_pct"][ch], best["bias_pct"][ch], colors[ch])
        for ch in channels
    )

    id_before = baseline["identifiability"]
    id_after = best["identifiability"]
    id_rows = "".join(
        _bar_pair_html(lbl, id_before[key], id_after[key], "#7c3aed", fmt="{:.3f}")
        for lbl, key in [
            ("b (saturation) sd", "b_sd"),
            ("lambda (adstock) sd", "lam_sd"),
        ]
    ) + _bar_pair_html(
        "RSS valley width", id_before["valley_pct"], id_after["valley_pct"], "#7c3aed"
    )

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

    # Appendix: winner schedule spend time series + demand, and a
    # saturation curve (only meaningful when saturation isn't linear).
    winner_combined = pd.concat([report.history_df, report.winner_schedule_])
    spend_series = {ch: winner_combined[ch].to_numpy() for ch in channels}
    appendix_spend_svg = _svg_multiline(spend_series, colors)

    saturation_note = ""
    saturation_svg = ""
    if meta["saturation"] < 1.0:
        b = meta["saturation"]
        xs = np.linspace(0, max(report.reference_spend_.values()) * 1.5, 60)
        curves = {}
        for ch in channels:
            x0 = report.reference_spend_[ch]
            mr0 = report.true_marginal_returns[ch]
            k = mr0 / (b * x0 ** (b - 1.0)) if x0 > 0 else 0.0
            curves[ch] = k * xs**b
        saturation_svg = _svg_multiline(curves, colors, normalize=False)
    else:
        saturation_note = (
            "<p><em>Saturation was assumed linear (b = 1.0) for this report "
            "-- no curvature to plot.</em></p>"
        )

    def stat_box(label: str, value: str) -> str:
        return (
            f'<div class="meta-box"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div></div>'
        )

    legend = "".join(
        f'<div class="li"><span class="sw" style="background:{colors[ch]}"></span>{ch}</div>'
        for ch in channels
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
  <h2>Spend correlation, before</h2>
  <p>How entangled each channel's spend is with every other channel's, across
  history + plan. The more correlated a pair, the harder it is for a model to
  tell their individual contributions apart.</p>
  {corr_before_html}
</section>

<section>
  <div class="s-label">Section 2</div>
  <h2>The variance problem</h2>
  <p>Coefficient of variation of each channel's model-estimated marginal
  return, with demand known -- how wide a range of "equally plausible" ROI
  estimates your spend history alone can support. Lower is better.</p>
  <div class="fig"><div class="fig-body">{variance_rows}</div></div>
  <p class="fig-cap">Before (unphased) &rarr; after ({winner}).</p>
</section>

<section>
  <div class="s-label">Section 3</div>
  <h2>The bias problem</h2>
  <p>Mean estimation error against the true marginal return, with demand
  controlled for only via a proxy of quality {meta["demand_proxy_quality"]:.0%}
  (not the true series) -- this is the omitted-variable bias a model sees
  when demand is imperfectly measured, as it usually is in practice.</p>
  <div class="fig"><div class="fig-body">{bias_rows}</div></div>
  <p class="fig-cap">Before (unphased) &rarr; after ({winner}).</p>
</section>

<section>
  <div class="s-label">Section 4</div>
  <h2>The identifiability problem</h2>
  <p>Can this spend pattern pin down the saturation exponent (b) and the
  adstock decay (lambda), even with demand known? Measured by the spread of
  the recovered value across draws (sd) and the width of the RSS valley
  (valley_pct) -- a flat valley means many curvatures fit about equally
  well, i.e. unmeasurable from this design. Lower is better throughout.</p>
  <div class="fig"><div class="fig-body">{id_rows}</div></div>
  <p class="fig-cap">Before (unphased) &rarr; after ({winner}).</p>
</section>

<section>
  <div class="s-label">Section 5</div>
  <h2>Spend correlation, after</h2>
  <p>The same correlation matrix, recomputed on the recommended
  ({winner}) schedule.</p>
  {corr_after_html}
</section>

<section>
  <div class="s-label">Section 6</div>
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
  <div class="s-label">Section 7</div>
  <h2>Appendix</h2>
  <div class="fig">
    <div class="fig-hdr"><div class="fig-title">Spend, history + plan (recommended schedule)</div></div>
    <div class="fig-body">{appendix_spend_svg}</div>
    <div class="legend">{legend}</div>
  </div>
  {
        ""
        if not saturation_svg
        else f'''
  <div class="fig">
    <div class="fig-hdr"><div class="fig-title">Assumed saturation curve (b = {meta["saturation"]:.2f})</div></div>
    <div class="fig-body">{saturation_svg}</div>
    <div class="legend">{legend}</div>
  </div>'''
    }
  {saturation_note}
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
.fig-body { padding: 1.1rem 1.1rem .3rem; }
.fig-cap { padding: .6rem 1.1rem .95rem; font-size: .82rem; color: var(--muted); }
svg.chart { display: block; width: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: .85rem; padding: .1rem 1.1rem .95rem; font-size: .78rem; }
.li { display: flex; align-items: center; gap: .35rem; }
.sw { width: .7rem; height: .7rem; border-radius: 2px; display: inline-block; }
.bar-row { margin-bottom: .9rem; }
.bar-label { font-size: .82rem; font-weight: 700; margin-bottom: .25rem; }
.bar-track { position: relative; background: var(--bg); border-radius: 4px; height: 1.4rem; margin-bottom: .2rem; display: flex; align-items: center; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: var(--light); }
.bar-before { background: var(--light); }
.bar-val { position: relative; margin-left: .5rem; font-size: .78rem; color: var(--text); z-index: 1; }
table.corr-table, table.cross-table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .85rem; }
table.corr-table th, table.corr-table td, table.cross-table th, table.cross-table td { border: 1px solid var(--border); padding: .4rem .6rem; text-align: center; }
table.cross-table th { background: var(--bg); }
table.hidden { display: none; }
select { font-size: .95rem; padding: .3rem .5rem; margin: .5rem 0 1rem; }
"""
