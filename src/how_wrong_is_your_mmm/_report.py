"""ReportBuilder: render a client-facing HTML report from Diagnose + Phase output.

Orchestrates CollinearityDiagnostic and BudgetPhaser end to end and renders
the result as a single self-contained HTML document (two "pages": a
client-facing headline page, and a "Scenario exploration" supporting-detail
page). All the chart-drawing logic lives in client-side JS embedded in the
template — this module's job is to compute real numbers with
CollinearityDiagnostic/BudgetPhaser and hand them to that JS as a single
JSON blob, not to re-derive the design.

Report structure (page 1): cover -> headline callout -> Diagnose (spend
correlation matrix + honest-range forest plot + table) -> Phase (recommended
weekly schedule + before/after correlation + lever table) -> Retrain/Impact
(before/after forest plot + kbox summary + marginal-return caveat) ->
Methodology appendix.

Report structure (page 2, "Scenario exploration"): channel sensitivity
(CV-vs-amplitude per channel, via BudgetPhaser.channel_sensitivity — decides
which lever a channel needs) then time-to-benefit (CV reduction over time,
by phasing intensity, for the least reliable channel).
"""

from __future__ import annotations

import json
import warnings
from datetime import date

import pandas as pd

from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    BudgetPhaser,
    DeviationSpec,
    _tile_plan,
)

# Categorical channel palette. First three slots match the existing
# overview.html/collinearity_research.html brand colours (tv/meta/search); slot 4
# (violet) has been checked for colour-vision-deficiency accessibility and
# passes, with one WARN band requiring direct labels, which every chart
# here already has. Slots 5+ are a reasonable extension, not yet checked
# the same way — fine for now, but if a real report regularly needs more
# than ~5 channels it's worth re-validating the fuller set rather than
# assuming it holds.
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


def _channel_colors(channels: list[str]) -> dict[str, str]:
    return {ch: _PALETTE[i % len(_PALETTE)] for i, ch in enumerate(channels)}


# ReportBuilder's own default lever, matching BudgetPhaser's. Kept as a named
# constant (rather than just the literal 40.0 in the signature) because
# fit()'s auto_lever logic checks identity against this exact value to decide
# whether the caller left max_weekly_deviation_pct untouched — see fit().
_DEFAULT_LEVER = 40.0


def _horizon_label(h: int) -> str:
    """Human-readable label for a horizon in weeks (e.g. 52 -> '1 year')."""
    if h % 52 == 0 and h >= 52:
        years = h // 52
        return f"{years} year" + ("s" if years > 1 else "")
    if h >= 52:
        return f"{h} weeks"
    months = max(1, round(h / 4.33))
    return f"{months} month" + ("s" if months > 1 else "")


def _resolve_channel_lever(
    channel: str, spec: DeviationSpec | dict[str, DeviationSpec]
) -> dict:
    """Format one channel's configured lever for the Phase table/schedule chart."""
    value = spec[channel] if isinstance(spec, dict) else spec
    if isinstance(value, Blackout):
        return {
            "lever": "blackout",
            "amplitude_pct": None,
            "dark_cap": value.max_dark_weeks_per_month,
        }
    return {"lever": "continuous", "amplitude_pct": float(value), "dark_cap": None}


class ReportBuilder:
    """Build a client-facing HTML report from a history + plan spend dataset.

    Runs CollinearityDiagnostic (today's model-estimated range + spend correlation)
    and BudgetPhaser (the recommended schedule, its per-channel sensitivity,
    and the before/after impact over several horizons), then renders the
    result as a single self-contained HTML file.

    Parameters
    ----------
    history_df:
        Multi-year spend history with a weekly DatetimeIndex. Fixed — not
        modified by phasing. Same shape BudgetPhaser expects.
    plan_df:
        The upcoming plan-year spend, same columns as history_df.
    true_marginal_returns:
        Dict mapping channel to a plausible (not proven) marginal return —
        £ revenue per £ spend, a.k.a. mROAS, NOT an economic elasticity
        (see _dgp.py) — used to simulate the sales column the diagnostic
        reasons about. Defaults to {"tv": 0.5, "meta": 1.0, "search": 1.5}
        ONLY when history_df/plan_df use exactly those three channel names;
        for any other channel naming you must supply this yourself, or
        CollinearityDiagnostic raises ValueError deep inside fit(). There
        is no safe universal default — for your own data, supply your own
        per-channel values (from a prior model, a plausible ROI range from
        finance, or a held-out incrementality test). This is the single
        most important input to get right: every CV and every £ range in
        the report is anchored to it, and CV is exactly inversely
        proportional to whatever value you supply here (see
        CollinearityDiagnostic.summary).
    base_sales, revenue_noise_std:
        Forwarded to every internal CollinearityDiagnostic/BudgetPhaser.
        revenue_noise_std (£) is the second most important input to get
        right — CV scales ~linearly with it. Set it from your own model's
        residual standard deviation (e.g. the residual std of a simple OLS
        fit of your actual sales on your actual spend history), not from
        this package's default of 26,000, which is an arbitrary
        placeholder with no relationship to your business. Defaults:
        base_sales=1_000.0, revenue_noise_std=26_000.0, matching
        CollinearityDiagnostic's own defaults.
    max_weekly_deviation_pct:
        Per-channel phasing lever, same shape BudgetPhaser accepts: a single
        float (symmetric +/-X% for every channel), a single Blackout, or a
        dict mixing both per channel. Default 40.0 — but see fit()'s
        auto_lever parameter: left at this default, fit() will by default
        replace it with a per-channel-differentiated spec chosen via
        BudgetPhaser.recommend_levers() rather than using a flat 40% for
        every channel. Pass an explicit float, Blackout, or dict here to
        opt out and pin the lever yourself.
    client_name, plan_year:
        Free-text labels shown on the report cover. Both optional — an
        empty client_name reads as "Prepared for [not set]" in the cover,
        a reasonable signal something was left blank rather than silently
        blank.
    seed:
        Base random seed, forwarded to BudgetPhaser.
    true_elasticities:
        Deprecated alias for `true_marginal_returns`, kept for backward
        compatibility. Raises ValueError if both are supplied. Emits a
        FutureWarning -- migrate to `true_marginal_returns`.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        true_marginal_returns: dict[str, float] | None = None,
        max_weekly_deviation_pct: DeviationSpec
        | dict[str, DeviationSpec] = _DEFAULT_LEVER,
        client_name: str = "",
        plan_year: str = "",
        seed: int = 0,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 26_000.0,
        true_elasticities: dict[str, float] | None = None,
    ) -> None:
        if true_elasticities is not None:
            if true_marginal_returns is not None:
                raise ValueError(
                    "Pass only one of true_marginal_returns or the "
                    "deprecated true_elasticities, not both."
                )
            warnings.warn(
                "true_elasticities is deprecated and will be removed in a "
                "future release -- these are marginal returns (£ revenue "
                "per £ spend), not elasticities. Use true_marginal_returns "
                "instead.",
                FutureWarning,
                stacklevel=2,
            )
            true_marginal_returns = true_elasticities

        self.history_df = history_df
        self.plan_df = plan_df
        self.true_marginal_returns = true_marginal_returns
        self.max_weekly_deviation_pct = max_weekly_deviation_pct
        self.client_name = client_name
        self.plan_year = plan_year
        self.seed = seed
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std

        self.phaser_: BudgetPhaser | None = None
        self.diagnostic_today_: CollinearityDiagnostic | None = None
        self.report_data_: dict | None = None

    @property
    def true_elasticities(self) -> dict[str, float] | None:
        """Deprecated alias for `true_marginal_returns`. See __init__."""
        warnings.warn(
            "ReportBuilder.true_elasticities is deprecated, use "
            "true_marginal_returns instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.true_marginal_returns

    def fit(
        self,
        n_sims: int = 50,
        grid_steps: int = 25,
        n_phasing_seeds: int = 10,
        horizons_weeks: list[int] | None = None,
        sensitivity_magnitude_pct: float = 80.0,
        sensitivity_alphas: list[float] | None = None,
        sensitivity_blackout: Blackout | None = Blackout(max_dark_weeks_per_month=1),
        ttb_horizons_weeks: list[int] | None = None,
        fast_mode: bool = False,
        auto_lever: bool = True,
        lever_improvement_threshold_pct: float = 10.0,
    ) -> ReportBuilder:
        """Run the full Diagnose -> Phase pipeline and build the report data.

        Parameters
        ----------
        n_sims, grid_steps, n_phasing_seeds:
            Forwarded to BudgetPhaser.fit() (which also runs its own
            selection-bias confirmation pass — see that method's docstring).
            Defaults (25, 10) match notebook 02's own full-mode settings on
            raz1470.github.io/how_wrong_is_your_mmm, not just an arbitrary
            choice — a larger n_phasing_seeds also reduces the noise the
            confirmation pass exists to correct for in the first place.
        horizons_weeks:
            Horizons for the Impact section's before/after comparison.
            Defaults to a quarter, the plan's own length, and double the
            plan's own length (e.g. [13, 52, 104] for a 52-week plan).
        sensitivity_magnitude_pct, sensitivity_alphas, sensitivity_blackout:
            Forwarded to BudgetPhaser.channel_sensitivity() for every
            channel (page 2, Section 4), and sensitivity_magnitude_pct is
            also forwarded to recommend_levers() when auto_lever is active
            (see below) — the page-2 sensitivity chart and the actual
            page-1 lever choice are computed from the same magnitude/
            blackout options, so they can't disagree. sensitivity_alphas
            defaults to [0.0, 0.125, 0.25, 0.5, 1.0], which combined with
            the default sensitivity_magnitude_pct=80.0 gives checkpoints at
            0/10/20/40/80% — matching the approved report design (session
            28: "CV reduction vs. phasing amplitude, 0% -> 100% continuous
            -> Blackout") rather than the flat 0-40% even-step sweep this
            shipped with initially. Pass your own list of alphas (each in
            [0, 1], multiplied by sensitivity_magnitude_pct to get a %) for
            different checkpoints.
        ttb_horizons_weeks:
            Horizons for the time-to-benefit chart (page 2, Section 5).
            Defaults to a quarter, half, the plan's own length, and double
            the plan's own length.
        fast_mode:
            If True, uses cheap settings throughout — for iterating on the
            report itself, not for numbers to actually hand a client. The
            rendered report is watermarked as a draft whenever this is True
            (see to_html()), so a fast-mode report can't be mistaken for a
            final one further down the line.
        auto_lever:
            If True (default) and self.max_weekly_deviation_pct was left at
            its default (a flat 40% for every channel), automatically picks
            a per-channel lever via BudgetPhaser.recommend_levers() instead
            — using Blackout for whichever channels actually benefit
            meaningfully more from it than the strongest continuous option
            (+/-sensitivity_magnitude_pct, 80% by default) offers, plain
            continuous otherwise. Exists because a single shared lever can
            make the wrong channel look like phasing barely helps at all,
            purely because that channel happens to be the one the flat
            lever suits worst (see recommend_levers()'s own docstring for
            the real report this happened on). Has no effect if
            self.max_weekly_deviation_pct was set explicitly (a
            caller-chosen lever is always respected as-is) or if
            sensitivity_blackout is None (nothing to compare against).
        lever_improvement_threshold_pct:
            Forwarded to recommend_levers() as improvement_threshold_pct
            when auto_lever is active. Default 10.0.

        Returns
        -------
        self
        """
        channels = list(self.plan_df.columns)
        n_weeks = len(self.plan_df)

        if sensitivity_alphas is None:
            sensitivity_alphas = [0.0, 0.125, 0.25, 0.5, 1.0]

        if horizons_weeks is None:
            horizons_weeks = sorted({max(1, n_weeks // 4), n_weeks, n_weeks * 2})
        if ttb_horizons_weeks is None:
            ttb_horizons_weeks = sorted(
                {max(1, n_weeks // 4), max(1, n_weeks // 2), n_weeks, n_weeks * 2}
            )

        # --- Today: correlation + model-estimated range, unphased -----------
        today_combined = pd.concat([self.history_df, self.plan_df])
        diag_today = CollinearityDiagnostic(
            spend_df=today_combined,
            true_marginal_returns=self.true_marginal_returns,
            base_sales=self.base_sales,
            revenue_noise_std=self.revenue_noise_std,
        )
        diag_today.fit(n_sims=10 if fast_mode else n_sims)
        planned_spend = self.plan_df.sum().to_dict()
        today_summary = diag_today.summary(planned_spend=planned_spend).set_index(
            "channel"
        )
        self.diagnostic_today_ = diag_today

        resolved_marginal_returns = diag_today.true_marginal_returns
        correlation_matrix = diag_today.correlation_matrix.round(4)

        least_reliable = str(today_summary["coef_of_variation"].idxmax())

        # --- Auto-pick a per-channel lever, unless the caller set one -------
        # Only engages when max_weekly_deviation_pct was left at its default
        # (an explicit caller choice, of any shape, is always respected) and
        # there's a Blackout option to compare against. See fit()'s own
        # auto_lever docstring and recommend_levers() for why this exists.
        auto_lever_applied = (
            auto_lever
            and self.max_weekly_deviation_pct == _DEFAULT_LEVER
            and sensitivity_blackout is not None
        )
        if auto_lever_applied:
            lever_scout = BudgetPhaser(
                history_df=self.history_df,
                plan_df=self.plan_df,
                true_marginal_returns=resolved_marginal_returns,
                seed=self.seed,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
            )
            resolved_lever_spec = lever_scout.recommend_levers(
                magnitude_pct=sensitivity_magnitude_pct,
                blackout=sensitivity_blackout,
                improvement_threshold_pct=lever_improvement_threshold_pct,
                n_sims=n_sims,
                n_phasing_seeds=n_phasing_seeds,
                fast_mode=fast_mode,
            )
        else:
            resolved_lever_spec = self.max_weekly_deviation_pct

        # --- Phase: recommended schedule ------------------------------------
        phaser = BudgetPhaser(
            history_df=self.history_df,
            plan_df=self.plan_df,
            true_marginal_returns=resolved_marginal_returns,
            max_weekly_deviation_pct=resolved_lever_spec,
            seed=self.seed,
            base_sales=self.base_sales,
            revenue_noise_std=self.revenue_noise_std,
        )
        phaser.fit(
            n_sims=n_sims,
            grid_steps=grid_steps,
            n_phasing_seeds=n_phasing_seeds,
            fast_mode=fast_mode,
        )
        self.phaser_ = phaser

        # --- Retrain: before/after impact over horizons ---------------------
        impact = phaser.impact_over_horizons(
            horizons_weeks=horizons_weeks,
            n_sims=n_sims,
            n_phasing_seeds=n_phasing_seeds,
            fast_mode=fast_mode,
            include_revenue=True,
        )

        # --- Channel sensitivity (page 2, section 4) -------------------------
        sensitivity = {
            ch: phaser.channel_sensitivity(
                ch,
                alphas=sensitivity_alphas,
                magnitude_pct=sensitivity_magnitude_pct,
                blackout=sensitivity_blackout,
                n_sims=n_sims,
                n_phasing_seeds=n_phasing_seeds,
                fast_mode=fast_mode,
            )
            for ch in channels
        }

        # --- Time to benefit, isolated to the least reliable channel ---------
        # (page 2, section 5). "Today" (x=0) is 0% reduction by definition for
        # every series; each later point is this channel's CV reduction at
        # that horizon, isolated the same way channel_sensitivity() isolates
        # its curve (every other channel locked at 0), so this answers "how
        # fast would phasing THIS channel alone pay off," not the joint
        # recommended schedule's own (usually faster, since other channels
        # help too) trajectory — the Impact section above is where the joint
        # trajectory lives.
        ttb_today_cv: dict[int, float] = {}
        for h in ttb_horizons_weeks:
            tiled = _tile_plan(self.plan_df, h)
            combined = pd.concat([self.history_df, tiled])
            d = CollinearityDiagnostic(
                spend_df=combined,
                true_marginal_returns=resolved_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
            )
            d.fit(n_sims=10 if fast_mode else n_sims)
            ttb_today_cv[h] = float(
                d.summary()
                .set_index("channel")
                .loc[least_reliable, "coef_of_variation"]
            )

        ttb_intensities: list[tuple[str, float | Blackout]] = [
            ("±20% continuous", 20.0),
            ("±40% continuous", 40.0),
            ("±80% continuous", 80.0),
        ]
        if sensitivity_blackout is not None:
            ttb_intensities.append(("Blackout", sensitivity_blackout))

        locked = dict.fromkeys(channels, 0.0)
        ttb_series: dict[str, list[float]] = {}
        for label, spec in ttb_intensities:
            pts = [0.0]
            for h in ttb_horizons_weeks:
                tiled = _tile_plan(self.plan_df, h)
                tmp = BudgetPhaser(
                    history_df=self.history_df,
                    plan_df=tiled,
                    true_marginal_returns=resolved_marginal_returns,
                    seed=self.seed,
                    base_sales=self.base_sales,
                    revenue_noise_std=self.revenue_noise_std,
                )
                full_spec = {**locked, least_reliable: spec}
                # Private helper, same package -- reused deliberately rather
                # than adding a public BudgetPhaser method for a computation
                # that's specific to this report's own narrative.
                result = tmp._evaluate_spec_at_alpha(
                    full_spec,
                    1.0,
                    10 if fast_mode else n_sims,
                    1 if fast_mode else n_phasing_seeds,
                    seed_offset=0,
                )
                cv_after = result[f"cv_{least_reliable}"]
                cv_today = ttb_today_cv[h]
                reduction = 100 * (cv_today - cv_after) / cv_today
                pts.append(round(reduction, 1))
            ttb_series[label] = pts

        # --- Assemble everything into one JSON-serialisable structure --------
        colors = _channel_colors(channels)
        primary_horizon = min(horizons_weeks, key=lambda h: abs(h - n_weeks))

        channel_rows = []
        for ch in channels:
            lever = _resolve_channel_lever(ch, resolved_lever_spec)
            channel_rows.append(
                {
                    "key": ch,
                    "name": ch.upper() if len(ch) <= 4 else ch.title(),
                    "hex": colors[ch],
                    "spend": float(self.plan_df[ch].sum()),
                    **lever,
                }
            )

        diagnose_today = {}
        for ch in channels:
            row = today_summary.loc[ch]
            diagnose_today[ch] = {
                "spend": float(planned_spend[ch]),
                "central": float(row["mean_estimated"] * planned_spend[ch]),
                "p10": float(row["incremental_revenue_p10"]),
                "p90": float(row["incremental_revenue_p90"]),
                "cv": float(row["coef_of_variation"]),
            }

        recommended = phaser.recommended_schedule_

        # Before/after correlation, same plan-year weeks in both -- unlike
        # correlation_matrix above (today's full history + plan), this pair
        # isolates the effect of phasing itself: same weeks, same monthly
        # totals, only the within-month timing differs.
        plan_corr = self.plan_df.corr().round(4)
        after_corr = recommended.corr().round(4)

        schedule = {
            "plan": {ch: self.plan_df[ch].round(2).tolist() for ch in channels},
            "recommended": {ch: recommended[ch].round(2).tolist() for ch in channels},
            "dark": {
                ch: (recommended[ch].to_numpy() == 0.0).tolist() for ch in channels
            },
        }

        horizon_labels = {h: _horizon_label(h) for h in horizons_weeks}

        impact_horizons = []
        for h in horizons_weeks:
            sub = impact[impact["horizon_weeks"] == h].set_index("channel")
            impact_horizons.append(
                {
                    "weeks": int(h),
                    "label": horizon_labels[h],
                    "channels": {
                        ch: {
                            "cv_today": float(sub.loc[ch, "cv_today"]),
                            "cv_after": float(sub.loc[ch, "cv_after"]),
                            "reduction_pct": float(sub.loc[ch, "cv_reduction_pct"]),
                            "revenue_today_p10": float(
                                sub.loc[ch, "revenue_today_p10"]
                            ),
                            "revenue_today_p90": float(
                                sub.loc[ch, "revenue_today_p90"]
                            ),
                            "revenue_after_p10": float(
                                sub.loc[ch, "revenue_after_p10"]
                            ),
                            "revenue_after_p90": float(
                                sub.loc[ch, "revenue_after_p90"]
                            ),
                        }
                        for ch in channels
                    },
                }
            )

        sensitivity_json = {
            ch: sensitivity[ch].to_dict(orient="records") for ch in channels
        }

        ttb_month_labels = ["Today"] + [_horizon_label(h) for h in ttb_horizons_weeks]

        from how_wrong_is_your_mmm import __version__ as package_version

        self.report_data_ = {
            "meta": {
                "client_name": self.client_name or "(not set)",
                "plan_year": self.plan_year or "(not set)",
                "generated_date": date.today().strftime("%-d %b %Y"),
                "package_version": package_version,
                "n_weeks": n_weeks,
                "fast_mode": bool(fast_mode),
                "auto_lever_applied": bool(auto_lever_applied),
            },
            "channels": channel_rows,
            "annual_plan_total": float(self.plan_df.sum().sum()),
            "least_reliable_channel": least_reliable,
            "correlation_matrix": {
                a: {b: float(correlation_matrix.loc[a, b]) for b in channels}
                for a in channels
            },
            "correlation_before_after": {
                "before": {
                    a: {b: float(plan_corr.loc[a, b]) for b in channels}
                    for a in channels
                },
                "after": {
                    a: {b: float(after_corr.loc[a, b]) for b in channels}
                    for a in channels
                },
            },
            "diagnose_today": diagnose_today,
            "schedule": schedule,
            "primary_horizon_weeks": int(primary_horizon),
            "impact_horizons": impact_horizons,
            "sensitivity": sensitivity_json,
            "ttb": {
                "channel": least_reliable,
                "month_labels": ttb_month_labels,
                "series": ttb_series,
            },
        }

        return self

    def schedule_csv(self, path: str | None = None) -> pd.DataFrame:
        """Return the recommended weekly schedule as a tidy, exportable table.

        The HTML report's Phase section only shows the recommended schedule
        as a chart plus a per-channel lever summary — there's no way to get
        the actual week-by-week numbers back out of it, which matters given
        the package's own stated purpose: "a concrete plan-year weekly
        spend schedule the practitioner can hand to their media agency"
        (see BudgetPhaser's module docstring). This is that table.

        One row per week, three columns per channel: the original plan
        figure, the recommended figure, and whether that week is a
        Blackout dark week (always False for a channel on a continuous
        lever). Both £ columns rounded to the nearest penny — this is
        meant to be opened in a spreadsheet and handed over, not used for
        further computation (use self.phaser_.recommended_schedule_
        directly for that, it's the unrounded source of truth).

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
        if self.phaser_ is None:
            raise RuntimeError("Call fit() before schedule_csv().")

        recommended = self.phaser_.recommended_schedule_
        channels = list(self.plan_df.columns)

        table = pd.DataFrame(index=recommended.index)
        table.index.name = "week"
        for ch in channels:
            table[f"{ch}_original_plan"] = self.plan_df[ch].round(2)
            table[f"{ch}_recommended"] = recommended[ch].round(2)
            table[f"{ch}_dark_week"] = recommended[ch].to_numpy() == 0.0

        if path is not None:
            table.to_csv(path)

        return table

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

        html = _HTML_TEMPLATE.replace(
            "/*__REPORT_JSON__*/", json.dumps(self.report_data_)
        )

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

        return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MMM Reliability &amp; Phasing Report</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --text:   #111827;
  --muted:  #6b7280;
  --light:  #d1d5db;
  --border: #e5e7eb;
  --bg:     #f9fafb;
  --good:   #059669;
  --w:      760px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  color: var(--text);
  background: #f3f4f6;
  line-height: 1.7;
  font-size: 17px;
}

.page {
  max-width: var(--w);
  margin: 0 auto;
  background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}

/* -- Header -- */
.cover {
  padding: 2.75rem 2rem 2.25rem;
  border-bottom: 3px solid var(--text);
}
.cover-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2rem;
}
.brand { font-size: .8rem; font-weight: 700; letter-spacing: .02em; }
.brand span { color: var(--muted); font-weight: 500; }
.gen-date { font-size: .78rem; color: var(--muted); text-align: right; }
.report-title-eyebrow {
  font-size: .75rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .6rem;
}
.report-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; margin-bottom: .5rem; }
.report-sub { color: var(--muted); font-size: 1.02rem; max-width: 56ch; }

.cover-meta {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: .75rem;
  margin-top: 2rem;
}
@media (max-width: 620px) { .cover-meta { grid-template-columns: repeat(2, 1fr); } }
.meta-box { border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; background: var(--bg); }
.meta-box .lbl { font-size: .66rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: .25rem; }
.meta-box .val { font-size: 1.15rem; font-weight: 700; }

/* -- Headline callout -- */
.headline {
  margin: 2rem 2rem 0;
  padding: 1.1rem 1.4rem;
  border-left: 3px solid var(--good);
  background: var(--bg);
  border-radius: 0 6px 6px 0;
  font-size: 1.05rem;
  font-weight: 600;
  color: #1f2937;
}
.headline b { color: var(--good); }

main { padding: 0 2rem 3rem; }

p { margin-bottom: 1.05rem; max-width: 64ch; font-size: .98rem; }

section { margin-top: 2.75rem; padding-top: 2rem; border-top: 1px solid var(--border); }

.s-label { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
h2 { font-size: 1.3rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.25; margin-bottom: .9rem; }

/* -- Figure -- */
.fig { margin: 1.5rem 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; page-break-inside: avoid; }
.fig-hdr { padding: .75rem 1.1rem; background: var(--bg); border-bottom: 1px solid var(--border); }
.fig-title { font-size: .95rem; font-weight: 700; }
.fig-sub   { font-size: .8rem; color: var(--muted); margin-top: .15rem; }
.fig-body  { padding: 1.1rem 1.1rem .3rem; }
.fig-cap   { padding: .6rem 1.1rem .95rem; font-size: .82rem; color: var(--muted); line-height: 1.55; max-width: none; }

svg.chart { display: block; width: 100%; }

.legend { display: flex; flex-wrap: wrap; gap: .85rem; padding: .1rem 0 .5rem; font-size: .78rem; color: var(--text); }
.li { display: flex; align-items: center; gap: .35rem; }

/* -- Table -- */
table.rpt { width: 100%; border-collapse: collapse; font-size: .88rem; margin: 1.25rem 0; }
table.rpt th, table.rpt td { padding: .55rem .7rem; text-align: left; border-bottom: 1px solid var(--border); }
table.rpt th { font-size: .68rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 700; }
table.rpt td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.rpt th.num { text-align: right; }
table.rpt .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: .45rem; }
table.rpt tr:last-child td { border-bottom: none; }

/* -- Key numbers -- */
.knums { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin: 1.5rem 0 .5rem; }
@media (max-width: 560px) { .knums { grid-template-columns: 1fr; } }
.kbox { border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.1rem; background: var(--bg); text-align: center; }
.kbox .lbl { font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: .3rem; }
.kbox .num { font-size: 1.6rem; font-weight: 800; line-height: 1; margin-bottom: .3rem; color: var(--text); }
.kbox .lab { font-size: .74rem; color: var(--muted); line-height: 1.4; }

.corr-wrap { display: flex; justify-content: center; padding: 1.3rem 1.1rem 1.5rem; overflow-x: auto; }
table.corr { border-collapse: collapse; }
table.corr th { font-size: .76rem; font-weight: 700; color: var(--muted); padding: 0 .5rem .5rem; text-align: center; }
table.corr th.rh { text-align: right; padding-right: .7rem; }
table.corr td { width: 58px; height: 40px; text-align: center; font-size: .84rem; font-weight: 700; font-variant-numeric: tabular-nums; border-radius: 4px; }
table.corr td.diag { background: repeating-linear-gradient(45deg, #f3f4f6, #f3f4f6 4px, #e5e7eb 4px, #e5e7eb 8px); color: var(--muted); font-weight: 500; }

.caveat { display: flex; gap: .65rem; align-items: flex-start; border: 1px solid #fde68a; background: #fffbeb; border-radius: 8px; padding: .85rem 1.1rem; margin: 1.5rem 0 0; font-size: .84rem; color: #78350f; line-height: 1.55; }
.caveat .icon { flex: none; font-size: 1rem; color: #b45309; line-height: 1.5; }
.caveat b { color: #78350f; }
.caveat p { margin: 0; max-width: none; font-size: inherit; color: inherit; }

/* -- Draft/fast-mode watermark -- shown only when the report was built with
   fast_mode=True, so a quick preview run can never be mistaken for a real
   one further down the line (this is what happened before this banner
   existed — see fit()'s fast_mode docstring). Deliberately alarming (red,
   not the amber used for the marginal-return caveat above) and kept visible in
   print, since the whole point is that it must survive a PDF export too. */
.draft-banner { display: none; }
.draft-banner.show {
  display: flex; gap: .6rem; align-items: center;
  background: #dc2626; color: #fff; font-weight: 700; font-size: .82rem;
  letter-spacing: .02em; padding: .65rem 1.1rem; text-align: center;
  justify-content: center;
}
@media print { .draft-banner.show { display: flex; -webkit-print-color-adjust: exact; print-color-adjust: exact; } }

.appendix { font-size: .84rem; color: var(--muted); }
.appendix code { background: var(--bg); padding: .1rem .35rem; border-radius: 4px; font-size: .82rem; }

footer { margin-top: 2.5rem; padding: 1.5rem 2rem 2rem; border-top: 1px solid var(--border); font-size: .8rem; color: var(--muted); }

/* -- Page 2 -- */
.page2 { margin-top: 1.75rem; padding: 2.25rem 2rem 1rem; }
.page2-hdr { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid var(--text); padding-bottom: 1.1rem; margin-bottom: .5rem; }
.page2-hdr h1 { font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; }
.page2-hdr .page-tag { font-size: .74rem; color: var(--muted); font-weight: 600; }
.page2-sub { color: var(--muted); font-size: .95rem; max-width: 60ch; margin: .9rem 0 0; }
.insight { border: 1px solid var(--border); border-radius: 8px; padding: .9rem 1.1rem; margin: 1.25rem 0; background: var(--bg); }
.insight-row { display: flex; gap: .7rem; align-items: baseline; padding: .35rem 0; border-bottom: 1px dashed var(--border); font-size: .87rem; }
.insight-row:last-child { border-bottom: none; }
.insight-row .swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; margin-top: .35rem; }
.insight-row b { font-size: .87rem; }
.insight-row .verdict { color: var(--muted); }

@media print {
  .page2 { break-before: page; }
  body { background: #fff; }
  .page { box-shadow: none; }
  section { break-inside: avoid-page; }
}
</style>
</head>
<body>

<div class="page">

<div class="draft-banner" id="draftBanner">&#9888; DRAFT &mdash; generated in fast mode. Not for client use; re-run with fast_mode=False for real numbers.</div>

<div class="cover">
  <div class="cover-top">
    <div class="brand"><span>how_wrong_is_your_mmm</span> &middot; MMM Reliability &amp; Phasing Report</div>
    <div class="gen-date" id="genDate"></div>
  </div>
  <p class="report-title-eyebrow">Prepared for</p>
  <h1 class="report-title" id="clientName"></h1>
  <p class="report-sub" id="reportSub"></p>

  <div class="cover-meta">
    <div class="meta-box"><div class="lbl">Plan year</div><div class="val" id="metaPlanYear"></div></div>
    <div class="meta-box"><div class="lbl">Channels</div><div class="val" id="metaChannels"></div></div>
    <div class="meta-box"><div class="lbl">Annual plan</div><div class="val" id="metaAnnualPlan"></div></div>
    <div class="meta-box"><div class="lbl">Extra spend needed</div><div class="val">&pound;0</div></div>
  </div>
</div>

<div class="headline" id="headline"></div>

<main>

<!-- SECTION 1: DIAGNOSE -->
<section id="diagnose">
  <p class="s-label">1 &middot; Diagnose</p>
  <h2>How reliable is your current plan?</h2>
  <p>Here's why: how correlated your channels have actually been over your supplied spend history. The higher the number, the more two channels move together week to week &mdash; and the harder it is for the model to tell their effects apart.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Historical spend correlation</div>
      <div class="fig-sub">Pearson correlation, weekly spend by channel</div>
    </div>
    <div class="fig-body corr-wrap">
      <div id="corrMatrix"></div>
    </div>
    <p class="fig-cap" id="corrCaption"></p>
  </div>

  <p>We simulated many equally plausible versions of your spend history consistent with this correlation structure, refitting the model on each one. Here's the model-estimated range of incremental revenue each channel could support as a result.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Today's model-estimated range, before any change</div>
      <div class="fig-sub">Incremental revenue, 10th&ndash;90th percentile across simulated histories</div>
    </div>
    <div class="fig-body">
      <svg class="chart" id="chartDiagnose" viewBox="0 0 700 240" preserveAspectRatio="xMinYMin meet"></svg>
    </div>
    <p class="fig-cap">Each channel's model-estimated range, given how your channels have actually moved together. Wider bars mean a less identifiable marginal-return estimate for that channel &mdash; not necessarily a less accurate one; see the Methodology note on scope.</p>
  </div>

  <table class="rpt">
    <thead><tr><th>Channel</th><th class="num">Spend</th><th class="num">Central estimate</th><th class="num">Model-estimated range</th><th class="num">CV</th></tr></thead>
    <tbody id="diagnoseTable"></tbody>
  </table>
</section>

<!-- SECTION 2: PHASE -->
<section id="phase">
  <p class="s-label">2 &middot; Phase</p>
  <h2>Recommended weekly schedule</h2>
  <p>A recommended weekly pacing for your plan, with monthly totals preserved exactly per channel &mdash; only which week the money lands in changes.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Original plan vs. recommended pacing</div>
      <div class="fig-sub" id="phaseSub"></div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="20" height="3"><line x1="0" y1="1.5" x2="20" y2="1.5" stroke="#9ca3af" stroke-width="2.5"/></svg> Original plan</span>
        <span class="li"><svg width="20" height="3"><line x1="0" y1="1.5" x2="20" y2="1.5" stroke="#111827" stroke-width="2.5"/></svg> Recommended pacing</span>
        <span class="li"><svg width="10" height="10"><circle cx="5" cy="5" r="3.6" fill="#111827"/></svg> Dark week (&pound;0)</span>
      </div>
      <svg class="chart" id="chartPhase" viewBox="0 0 700 700" preserveAspectRatio="xMinYMin meet"></svg>
    </div>
    <p class="fig-cap">Channels on a continuous nudge vary smoothly and never hit &pound;0. Channels on <code>Blackout</code> go fully dark on the weeks marked, with the remaining weeks in that month absorbing the difference so the monthly total is untouched.</p>
  </div>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Channel correlation, before vs. after</div>
      <div class="fig-sub">Pearson correlation, weekly spend by channel &middot; plan year only, monthly totals identical in both</div>
    </div>
    <div class="fig-body corr-wrap" style="gap:2.5rem;">
      <div>
        <div style="text-align:center;font-size:.78rem;font-weight:700;color:var(--muted);margin-bottom:.5rem;">Before phasing</div>
        <div id="corrBeforeMatrix"></div>
      </div>
      <div>
        <div style="text-align:center;font-size:.78rem;font-weight:700;color:var(--muted);margin-bottom:.5rem;">After phasing</div>
        <div id="corrAfterMatrix"></div>
      </div>
    </div>
    <p class="fig-cap" id="corrBeforeAfterCaption"></p>
  </div>

  <table class="rpt">
    <thead><tr><th>Channel</th><th>Lever</th><th class="num">Amplitude</th><th class="num">Max dark weeks / month</th></tr></thead>
    <tbody id="phaseTable"></tbody>
  </table>
</section>

<!-- SECTION 3: RETRAIN / IMPACT -->
<section id="impact">
  <p class="s-label">3 &middot; Retrain</p>
  <h2>What this buys you</h2>
  <p>Refitting on the phased schedule instead of the original plan, averaged across many equally-valid ways the schedule could land, not just one draw.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title" id="impactFigTitle"></div>
      <div class="fig-sub">Incremental revenue, model-estimated range</div>
    </div>
    <div class="fig-body">
      <div class="legend">
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#9ca3af" opacity="0.4"/></svg> Today</span>
        <span class="li"><svg width="16" height="8"><rect width="16" height="8" fill="#111827"/></svg> <span id="impactAfterLegend">After phasing</span></span>
      </div>
      <svg class="chart" id="chartImpact" viewBox="0 0 700 240" preserveAspectRatio="xMinYMin meet"></svg>
    </div>
    <p class="fig-cap">Every channel narrows. The channel that was least reliable today typically sees the biggest gain when it's on the stronger <code>Blackout</code> lever.</p>
  </div>

  <div class="knums" id="impactKnums"></div>

  <div class="caveat">
    <span class="icon">&#9888;</span>
    <p><b>These &pound; figures depend on the marginal returns you supplied</b> &mdash; plausible, not proven. The specific pound amounts above, and where the range sits, will move if your assumed marginal returns turn out to be off &mdash; and this diagnostic can't tell you whether they're right, only how precisely they'd be estimated if they are. The percentage CV reduction from phasing is unaffected by an across-the-board misjudgement of scale (say, if every channel's true return is actually double what you assumed) &mdash; but it can shift if what's off is the returns' proportions <i>relative to each other</i>, since that changes which channel looks least identified and therefore which phasing intensity gets recommended for it. See <a href="#method">Methodology</a>.</p>
  </div>
</section>

<!-- APPENDIX -->
<section id="method">
  <p class="s-label">Appendix</p>
  <h2>Methodology</h2>
  <p class="appendix" id="methodologyText"></p>
</section>

</main>

<footer>
  <p>This report is a simulation-based diagnostic, not a guarantee &mdash; treat the ranges as your model's honest uncertainty, not a forecast. Source: <a href="https://github.com/raz1470/how_wrong_is_your_mmm">github.com/raz1470/how_wrong_is_your_mmm</a></p>
</footer>

</div>

<div class="page page2">

  <div class="page2-hdr">
    <h1>Scenario exploration</h1>
    <div class="page-tag">Page 2 of 2</div>
  </div>
  <p class="page2-sub">Supporting detail for the decisions behind page 1's recommendation &mdash; which channels actually need the stronger lever, and how fast it pays off once you've picked one. Background for the DS/marketing conversation, not headline numbers.</p>

<main style="padding: 0;">

<!-- SECTION 4: WHICH CHANNELS NEED MORE RANDOMNESS -->
<section id="sensitivity" style="margin-top: 2rem;">
  <p class="s-label">4 &middot; Channel sensitivity</p>
  <h2>Which channels need more randomness?</h2>
  <p>This is what decides the lever on page 1. For each channel, how much reliability keeps improving as you widen the amplitude, all the way through to <code>Blackout</code>, with every other channel left untouched. A curve that flattens early means that channel doesn't need much perturbation. A curve still climbing at the far right means the channel is more tightly locked to the planning cycle, and needs the strongest lever available to break it.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">CV reduction vs. phasing amplitude</div>
      <div class="fig-sub">Continuous nudge 0% &rarr; 100%, then Blackout &mdash; every other channel locked</div>
    </div>
    <div class="fig-body">
      <div class="legend" id="sensitivityLegend"></div>
      <svg class="chart" id="chartSensitivity" viewBox="0 0 700 260" preserveAspectRatio="xMinYMin meet"></svg>
    </div>
    <p class="fig-cap">Compare the shape of each curve, not just its endpoint: an early plateau means a continuous nudge already captures most of the benefit; a curve still rising at the right needs <code>Blackout</code> to reach its full potential.</p>
  </div>

  <div class="insight" id="sensitivityInsight"></div>
</section>

<!-- SECTION 5: TIME TO BENEFIT -->
<section id="ttb">
  <p class="s-label">5 &middot; Time to benefit</p>
  <h2>How fast does each intensity pay off?</h2>
  <p id="ttbIntro"></p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title" id="ttbFigTitle"></div>
      <div class="fig-sub" id="ttbFigSub"></div>
    </div>
    <div class="fig-body">
      <div class="legend" id="ttbLegend"></div>
      <svg class="chart" id="chartTTB" viewBox="0 0 700 260" preserveAspectRatio="xMinYMin meet"></svg>
    </div>
    <p class="fig-cap">Every intensity keeps improving the longer it runs; the strongest lever gets there both higher and faster. This isolates the channel above from the rest of the schedule &mdash; the joint recommended schedule on page 1 (which phases every channel together) typically pays off faster still.</p>
  </div>

  <table class="rpt" id="ttbTableWrap">
    <thead><tr><th>Intensity</th></tr></thead>
    <tbody id="ttbTable"></tbody>
  </table>
</section>

</main>

<footer>
  <p>Page 2 is exploratory detail generated alongside the main report &mdash; same simulation basis as page 1, run at additional amplitude/horizon settings.</p>
</footer>

</div>

<script>
const REPORT = /*__REPORT_JSON__*/;

function el(tag, attrs, text) {
  const ns = 'http://www.w3.org/2000/svg';
  const e = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function fmtGBP(v) {
  const abs = Math.abs(v);
  if (abs < 1000) return `£${Math.round(v)}`;
  if (abs < 1e6) return `£${Math.round(v / 1000)}k`;
  return `£${(v / 1e6).toFixed(2)}m`;
}
function fmtPct(v) { return `${Math.round(v * 100)}%`; }
function chName(ch) {
  const found = REPORT.channels.find((c) => c.key === ch);
  return found ? found.name : ch;
}
function chColor(ch) {
  const found = REPORT.channels.find((c) => c.key === ch);
  return found ? found.hex : '#6b7280';
}

// -- Cover / static text ------------------------------------------------------
if (REPORT.meta.fast_mode) {
  document.getElementById('draftBanner').classList.add('show');
}
document.getElementById('genDate').innerHTML = `Generated ${REPORT.meta.generated_date}<br>v${REPORT.meta.package_version}`;
document.getElementById('clientName').textContent = REPORT.meta.client_name;
document.getElementById('reportSub').textContent =
  `A diagnosis of how precisely your ${REPORT.meta.plan_year} marketing mix model can identify each channel's marginal return, and a recommended weekly spend schedule to tighten that — same budget, same monthly totals.`;
document.getElementById('metaPlanYear').textContent = REPORT.meta.plan_year;
document.getElementById('metaChannels').textContent = REPORT.channels.length;
document.getElementById('metaAnnualPlan').textContent = fmtGBP(REPORT.annual_plan_total);

const leastReliable = REPORT.least_reliable_channel;
const todayCvLeast = REPORT.diagnose_today[leastReliable].cv;
const primaryImpact = REPORT.impact_horizons.find((h) => h.weeks === REPORT.primary_horizon_weeks);
const afterCvLeast = primaryImpact.channels[leastReliable].cv_after;
const reductionLeast = primaryImpact.channels[leastReliable].reduction_pct;

document.getElementById('headline').innerHTML =
  `Today, your least reliable channel (<b>${chName(leastReliable)}</b>) could be off by <b>±${Math.round(todayCvLeast * 100)}%</b>. ` +
  `Phasing your ${REPORT.meta.plan_year} schedule as recommended below could narrow that to <b>±${Math.round(afterCvLeast * 100)}%</b> within ${primaryImpact.label} — without changing a single monthly total.`;

document.getElementById('methodologyText').innerHTML =
  `Generated with <code>how_wrong_is_your_mmm v${REPORT.meta.package_version}</code>. ` +
  `Diagnostic: OLS refit across simulated sales consistent with your supplied spend history, using marginal returns (£ revenue per £ spend) you supplied. ` +
  `Phasing: grid search over amplitude per lever, minimising the worst-case channel CV, then re-confirmed independently on a larger sample before being recommended, to avoid favouring a setting that only looked best by chance. ` +
  (REPORT.meta.auto_lever_applied
    ? `Lever choice: each channel's own lever (continuous range vs. a harder on/off Blackout) was picked automatically from how much that channel individually benefits from each option. `
    : ``) +
  `<br><br><b>Scope:</b> the ranges above measure how precisely your spend design can identify each channel's marginal return under a model that is correctly specified by construction &mdash; sampling variance, not model error. They are silent on misspecification: an omitted driver (seasonality, adstock, saturation, a competitor event) can leave this diagnostic looking healthy while the underlying point estimate is badly biased, because the same omission that inflates the estimate typically inflates it more than it widens the range. Read a narrow range as "well-identified by this design," not as "correct." ` +
  `Full method: <a href="https://raz1470.github.io/how_wrong_is_your_mmm/collinearity_research.html">raz1470.github.io/how_wrong_is_your_mmm/collinearity_research.html</a>.`;

document.getElementById('phaseSub').textContent =
  `${REPORT.meta.n_weeks}-week schedule, monthly totals unchanged per channel`;

// -- Correlation matrix -------------------------------------------------------
function corrColor(v) {
  const t = Math.max(0, Math.min(1, v));
  const lo = [249, 250, 251], hi = [17, 24, 39];
  const mix = lo.map((c, i) => Math.round(c + (hi[i] - c) * (t * 0.92)));
  return `rgb(${mix.join(',')})`;
}
function renderCorrTable(targetId, matrix, channels) {
  let html = '<table class="corr"><tr><th></th>';
  for (const ch of channels) html += `<th>${ch.name}</th>`;
  html += '</tr>';
  for (const rowCh of channels) {
    html += `<tr><th class="rh" style="color:${rowCh.hex}">${rowCh.name}</th>`;
    for (const colCh of channels) {
      const v = matrix[rowCh.key][colCh.key];
      if (rowCh.key === colCh.key) {
        html += `<td class="diag">${v.toFixed(2)}</td>`;
      } else {
        const textColor = v >= 0.6 ? '#fff' : '#111827';
        html += `<td style="background:${corrColor(v)};color:${textColor}">${v.toFixed(2)}</td>`;
      }
    }
    html += '</tr>';
  }
  html += '</table>';
  document.getElementById(targetId).innerHTML = html;
}

(function drawCorrMatrix() {
  renderCorrTable('corrMatrix', REPORT.correlation_matrix, REPORT.channels);
})();

(function drawCorrBeforeAfter() {
  if (!REPORT.correlation_before_after) return;
  const channels = REPORT.channels;
  renderCorrTable('corrBeforeMatrix', REPORT.correlation_before_after.before, channels);
  renderCorrTable('corrAfterMatrix', REPORT.correlation_before_after.after, channels);

  function meanPairwise(m) {
    let sum = 0, n = 0;
    for (let i = 0; i < channels.length; i++) {
      for (let j = i + 1; j < channels.length; j++) {
        sum += m[channels[i].key][channels[j].key];
        n++;
      }
    }
    return n ? sum / n : 0;
  }
  const beforeMean = meanPairwise(REPORT.correlation_before_after.before);
  const afterMean = meanPairwise(REPORT.correlation_before_after.after);
  document.getElementById('corrBeforeAfterCaption').textContent =
    `Mean pairwise correlation drops from ${beforeMean.toFixed(2)} to ${afterMean.toFixed(2)} once the recommended schedule replaces the original plan, across the ${REPORT.meta.n_weeks}-week plan year — monthly totals are identical in both.`;
})();

(function corrCaption() {
  const entries = [];
  const channels = REPORT.channels.map((c) => c.key);
  for (let i = 0; i < channels.length; i++) {
    for (let j = i + 1; j < channels.length; j++) {
      entries.push({ pair: [channels[i], channels[j]], v: REPORT.correlation_matrix[channels[i]][channels[j]] });
    }
  }
  entries.sort((a, b) => b.v - a.v);
  const top = entries[0];
  document.getElementById('corrCaption').textContent = top
    ? `${chName(top.pair[0])} and ${chName(top.pair[1])} are the most entangled in your spend history, at ${top.v.toFixed(2)} correlation — the harder that pair is to tell apart, the more phasing helps.`
    : '';
})();

// -- Diagnose table + forest chart --------------------------------------------
const diagBody = document.getElementById('diagnoseTable');
for (const ch of REPORT.channels) {
  const d = REPORT.diagnose_today[ch.key];
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="swatch" style="background:${ch.hex}"></span>${ch.name}</td>
    <td class="num">${fmtGBP(d.spend)}</td>
    <td class="num">${fmtGBP(d.central)}</td>
    <td class="num">${fmtGBP(d.p10)} &ndash; ${fmtGBP(d.p90)}</td>
    <td class="num">${fmtPct(d.cv)}</td>`;
  diagBody.appendChild(tr);
}

const phaseBody = document.getElementById('phaseTable');
for (const ch of REPORT.channels) {
  const tr = document.createElement('tr');
  const leverLabel = ch.lever === 'continuous' ? 'Continuous nudge' : 'Blackout';
  const amp = ch.lever === 'continuous' ? `±${Math.round(ch.amplitude_pct)}%` : '—';
  const darkCap = ch.lever === 'blackout' ? (ch.dark_cap ?? 'unlimited') : '—';
  tr.innerHTML = `<td><span class="swatch" style="background:${ch.hex}"></span>${ch.name}</td>
    <td>${leverLabel}</td>
    <td class="num">${amp}</td>
    <td class="num">${darkCap}</td>`;
  phaseBody.appendChild(tr);
}

function drawForest(svgId, rows, xMax, twoState) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  const W = 700, H = 240;
  const m = { top: 14, right: 98, bottom: 42, left: 66 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const scX = (v) => m.left + (v / xMax) * pw;
  const rowH = ph / rows.length;
  const rowCy = (i) => m.top + rowH * i + rowH / 2;
  const g = el('g', {});

  for (let i = 0; i < rows.length; i++) {
    if (i % 2 === 1) g.appendChild(el('rect', { x: m.left, y: m.top + rowH * i, width: pw, height: rowH, fill: '#f9fafb' }));
  }

  const step = xMax <= 1.2 ? xMax / 6 : Math.ceil(xMax / 6 * 2) / 2;
  for (let v = 0; v <= xMax + 1e-6; v += step) {
    const x = scX(v);
    g.appendChild(el('line', { x1: x, y1: m.top, x2: x, y2: m.top + ph, stroke: '#e5e7eb', 'stroke-width': 1 }));
    g.appendChild(el('text', { x, y: m.top + ph + 16, 'text-anchor': 'middle', 'font-size': 10, fill: '#9ca3af' }, fmtGBP(v)));
  }
  g.appendChild(el('text', { x: m.left + pw / 2, y: m.top + ph + 30, 'text-anchor': 'middle', 'font-size': 10, fill: '#9ca3af' }, 'Incremental revenue'));
  g.appendChild(el('rect', { x: m.left, y: m.top, width: pw, height: ph, fill: 'none', stroke: '#e5e7eb', 'stroke-width': 1 }));

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const cy = rowCy(i);
    if (twoState) {
      const s2 = rowH * 0.22;
      g.appendChild(el('line', { x1: scX(r.today[0]), y1: cy - s2, x2: scX(r.today[1]), y2: cy - s2, stroke: r.color, 'stroke-width': 6, 'stroke-linecap': 'round', opacity: 0.4 }));
      g.appendChild(el('line', { x1: scX(r.after[0]), y1: cy + s2, x2: scX(r.after[1]), y2: cy + s2, stroke: r.color, 'stroke-width': 6, 'stroke-linecap': 'round' }));
      g.appendChild(el('text', { x: scX(r.after[1]) + 8, y: cy + s2 + 3, 'font-size': 9.5, fill: r.color, 'font-weight': '700' }, `${fmtGBP(r.after[0])}–${fmtGBP(r.after[1])}`));
    } else {
      g.appendChild(el('line', { x1: scX(r.p10), y1: cy, x2: scX(r.p90), y2: cy, stroke: r.color, 'stroke-width': 7, 'stroke-linecap': 'round' }));
      g.appendChild(el('text', { x: scX(r.p90) + 8, y: cy + 4, 'font-size': 11.5, fill: r.color, 'font-weight': '700' }, `${fmtGBP(r.p10)} – ${fmtGBP(r.p90)}`));
    }
    g.appendChild(el('text', { x: m.left - 10, y: cy - 2, 'text-anchor': 'end', 'font-size': 12.5, 'font-weight': '700', fill: r.color }, r.name));
    g.appendChild(el('text', { x: m.left - 10, y: cy + 12, 'text-anchor': 'end', 'font-size': 9.5, fill: '#9ca3af' }, `${fmtGBP(r.spend)} spend`));
  }
  svg.appendChild(g);
}

(function () {
  const rows = REPORT.channels.map((ch) => {
    const d = REPORT.diagnose_today[ch.key];
    return { name: ch.name, color: ch.hex, spend: d.spend, p10: d.p10, p90: d.p90 };
  });
  const xMax = Math.max(...rows.map((r) => r.p90)) * 1.15;
  drawForest('chartDiagnose', rows, xMax);
})();

document.getElementById('impactFigTitle').textContent = `Before vs. after ${primaryImpact.label} of phasing`;
document.getElementById('impactAfterLegend').textContent = `After ${primaryImpact.label} of phasing`;

(function () {
  const rows = REPORT.channels.map((ch) => {
    const d = REPORT.diagnose_today[ch.key];
    const imp = primaryImpact.channels[ch.key];
    return {
      name: ch.name, color: ch.hex, spend: d.spend,
      today: [imp.revenue_today_p10, imp.revenue_today_p90],
      after: [imp.revenue_after_p10, imp.revenue_after_p90],
    };
  });
  const xMax = Math.max(...rows.map((r) => Math.max(r.today[1], r.after[1]))) * 1.15;
  drawForest('chartImpact', rows, xMax, true);
})();

(function () {
  const box = document.getElementById('impactKnums');
  box.innerHTML = `
    <div class="kbox">
      <div class="lbl">Annual plan</div>
      <div class="num">${fmtGBP(REPORT.annual_plan_total)}</div>
      <div class="lab">Unchanged — same monthly totals throughout</div>
    </div>
    <div class="kbox">
      <div class="lbl">Least reliable channel today</div>
      <div class="num">±${Math.round(todayCvLeast * 100)}%</div>
      <div class="lab">${chName(leastReliable)}, 10th&ndash;90th percentile CV</div>
    </div>
    <div class="kbox">
      <div class="lbl">Least reliable channel, after ${primaryImpact.label}</div>
      <div class="num" style="color:var(--good)">±${Math.round(afterCvLeast * 100)}%</div>
      <div class="lab">${chName(leastReliable)} again — ${Math.round(reductionLeast)}% narrower</div>
    </div>`;
})();

// -- Phase: stacked weekly schedule, one row per channel ----------------------
(function drawPhase() {
  const svg = document.getElementById('chartPhase');
  const channels = REPORT.channels;
  const nWeeks = REPORT.meta.n_weeks;
  const W = 700, H = 175 * channels.length;
  svg.setAttribute('viewBox', `0 0 700 ${H}`);
  const rowH = H / channels.length;
  const m = { top: 26, right: 16, bottom: 14, left: 58 };
  const pw = W - m.left - m.right;
  const ph = rowH - m.top - m.bottom;
  const g = el('g', {});

  channels.forEach((ch, idx) => {
    const offsetY = idx * rowH;
    const pg = el('g', { transform: `translate(0,${offsetY})` });
    const plan = REPORT.schedule.plan[ch.key];
    const rec = REPORT.schedule.recommended[ch.key];
    const dark = REPORT.schedule.dark[ch.key];
    const yMax = Math.max(...plan, ...rec) * 1.15;
    const scX = (i) => m.left + (i / (nWeeks - 1)) * pw;
    const scY = (v) => m.top + ph - (v / yMax) * ph;
    const pathD = (data) => data.map((v, i) => `${i === 0 ? 'M' : 'L'}${scX(i).toFixed(1)},${scY(v).toFixed(1)}`).join(' ');

    const leverLabel = ch.lever === 'continuous' ? `continuous ±${Math.round(ch.amplitude_pct)}%` : `Blackout, cap ${ch.dark_cap ?? 'unlimited'}/mo`;
    pg.appendChild(el('text', { x: m.left, y: 14, 'font-size': 12.5, 'font-weight': '700', fill: ch.hex }, `${ch.name} — ${leverLabel}`));

    [0, yMax].forEach((v) => {
      const y = scY(v);
      pg.appendChild(el('line', { x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: '#e5e7eb', 'stroke-width': 1 }));
      pg.appendChild(el('text', { x: m.left - 8, y: y + 3, 'text-anchor': 'end', 'font-size': 9.5, fill: '#9ca3af' }, fmtGBP(v)));
    });
    pg.appendChild(el('line', { x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));
    pg.appendChild(el('line', { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));
    pg.appendChild(el('text', { x: m.left, y: m.top + ph + 13, 'font-size': 9.5, fill: '#9ca3af' }, 'Wk 1'));
    pg.appendChild(el('text', { x: m.left + pw, y: m.top + ph + 13, 'text-anchor': 'end', 'font-size': 9.5, fill: '#9ca3af' }, `Wk ${nWeeks}`));

    pg.appendChild(el('path', { d: pathD(plan), fill: 'none', stroke: '#9ca3af', 'stroke-width': 1.75, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: 0.9 }));
    pg.appendChild(el('path', { d: pathD(rec), fill: 'none', stroke: '#111827', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));

    rec.forEach((v, i) => {
      if (dark[i]) pg.appendChild(el('circle', { cx: scX(i), cy: scY(0), r: 3, fill: '#111827' }));
    });

    g.appendChild(pg);
  });

  svg.appendChild(g);
})();

// -- Page 2, Section 4: channel sensitivity -----------------------------------
(function () {
  const legend = document.getElementById('sensitivityLegend');
  legend.innerHTML = REPORT.channels.map((ch) =>
    `<span class="li"><svg width="20" height="3"><line x1="0" y1="1.5" x2="20" y2="1.5" stroke="${ch.hex}" stroke-width="2.5"/></svg> ${ch.name}</span>`
  ).join('');
})();

(function drawSensitivity() {
  const svg = document.getElementById('chartSensitivity');
  const channels = REPORT.channels;
  const first = REPORT.sensitivity[channels[0].key];
  const levels = first.map((r) => r.label);
  const W = 700, H = 260;
  const m = { top: 16, right: 96, bottom: 34, left: 46 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const allCv = channels.flatMap((ch) => REPORT.sensitivity[ch.key].map((r) => r.cv * 100));
  const yMax = Math.max(...allCv) * 1.15;
  const scX = (i) => m.left + (i / (levels.length - 1)) * pw;
  const scY = (v) => m.top + ph - (v / yMax) * ph;
  const g = el('g', {});

  const yStep = Math.ceil(yMax / 4 / 5) * 5;
  for (let v = 0; v <= yMax; v += yStep) {
    const y = scY(v);
    g.appendChild(el('line', { x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: '#e5e7eb', 'stroke-width': 1 }));
    g.appendChild(el('text', { x: m.left - 8, y: y + 3, 'text-anchor': 'end', 'font-size': 10, fill: '#9ca3af' }, `${Math.round(v)}%`));
  }
  levels.forEach((label, i) => {
    g.appendChild(el('text', { x: scX(i), y: m.top + ph + 18, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#9ca3af' }, label));
  });
  const dividerX = scX(levels.length - 1.5);
  g.appendChild(el('line', { x1: dividerX, y1: m.top, x2: dividerX, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1, 'stroke-dasharray': '3,3' }));
  g.appendChild(el('line', { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));

  const endLabels = [];
  for (const ch of channels) {
    const rows = REPORT.sensitivity[ch.key];
    const d = rows.map((r, i) => `${i === 0 ? 'M' : 'L'}${scX(i).toFixed(1)},${scY(r.cv * 100).toFixed(1)}`).join(' ');
    g.appendChild(el('path', { d, fill: 'none', stroke: ch.hex, 'stroke-width': 2.25, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    rows.forEach((r, i) => {
      const isBlackout = r.is_blackout;
      g.appendChild(el(isBlackout ? 'rect' : 'circle', isBlackout
        ? { x: scX(i) - 3.2, y: scY(r.cv * 100) - 3.2, width: 6.4, height: 6.4, transform: `rotate(45 ${scX(i)} ${scY(r.cv * 100)})`, fill: ch.hex }
        : { cx: scX(i), cy: scY(r.cv * 100), r: 2.75, fill: ch.hex }));
    });
    const last = rows[rows.length - 1];
    endLabels.push({ y: scY(last.cv * 100), x: scX(rows.length - 1) + 9, text: `${ch.name} · ${Math.round(last.cv * 100)}%`, color: ch.hex });
  }
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    if (endLabels[i].y - endLabels[i - 1].y < 13) endLabels[i].y = endLabels[i - 1].y + 13;
  }
  endLabels.forEach((l) => g.appendChild(el('text', { x: l.x, y: l.y + 3.5, 'font-size': 10.5, 'font-weight': '700', fill: l.color }, l.text)));
  svg.appendChild(g);
})();

(function fillSensitivityInsight() {
  const box = document.getElementById('sensitivityInsight');
  for (const ch of REPORT.channels) {
    const rows = REPORT.sensitivity[ch.key];
    const continuous = rows.filter((r) => !r.is_blackout);
    const blackoutRow = rows.find((r) => r.is_blackout);
    const midIdx = Math.floor(continuous.length / 2);
    const lateGain = continuous[continuous.length - 1].cv > 0
      ? (continuous[midIdx].cv - continuous[continuous.length - 1].cv) / continuous[continuous.length - 1].cv
      : 0;
    let verdict;
    if (blackoutRow && blackoutRow.cv < continuous[continuous.length - 1].cv * 0.85) {
      verdict = `Still improving well past the midpoint of the continuous range — <code>Blackout</code> reaches ${Math.round(blackoutRow.cv * 100)}% CV, clearly lower than the continuous ceiling.`;
    } else if (Math.abs(lateGain) < 0.1) {
      verdict = `Plateaus roughly halfway through the continuous range — a moderate nudge captures most of the benefit; the harder lever adds little.`;
    } else {
      verdict = `Keeps improving through the continuous range; check whether ${blackoutRow ? 'Blackout adds a further gain worth the disruption' : 'a wider continuous amplitude is worth considering'}.`;
    }
    const row = document.createElement('div');
    row.className = 'insight-row';
    row.innerHTML = `<span class="swatch" style="background:${ch.hex}"></span><div><b>${ch.name}:</b> <span class="verdict">${verdict}</span></div>`;
    box.appendChild(row);
  }
})();

// -- Page 2, Section 5: time to benefit ---------------------------------------
const TTB_COLORS = ['#bfdbfe', '#60a5fa', '#1d4ed8', '#111827'];

document.getElementById('ttbIntro').innerHTML =
  `Once the lever's picked, this is the follow-up question: how long before it shows. Shown for <b>${chName(REPORT.ttb.channel)}</b>, today's least reliable channel, isolated from the other channels.`;
document.getElementById('ttbFigTitle').textContent = `CV reduction over time, by phasing intensity — ${chName(REPORT.ttb.channel)}`;
document.getElementById('ttbFigSub').textContent = `${REPORT.ttb.month_labels.join(' vs ')} of phasing`;

(function () {
  const legend = document.getElementById('ttbLegend');
  legend.innerHTML = Object.keys(REPORT.ttb.series).map((name, i) =>
    `<span class="li"><svg width="20" height="3"><line x1="0" y1="1.5" x2="20" y2="1.5" stroke="${TTB_COLORS[i % TTB_COLORS.length]}" stroke-width="3"/></svg> ${name}</span>`
  ).join('');
})();

(function drawTTB() {
  const svg = document.getElementById('chartTTB');
  const months = REPORT.ttb.month_labels;
  const series = REPORT.ttb.series;
  const W = 700, H = 260;
  const m = { top: 16, right: 92, bottom: 34, left: 46 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const allPts = Object.values(series).flat();
  const yMax = Math.max(...allPts) * 1.15;
  const scX = (i) => m.left + (i / (months.length - 1)) * pw;
  const scY = (v) => m.top + ph - (v / yMax) * ph;
  const g = el('g', {});

  const yStep = Math.ceil(yMax / 4 / 5) * 5;
  for (let v = 0; v <= yMax; v += yStep) {
    const y = scY(v);
    g.appendChild(el('line', { x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: '#e5e7eb', 'stroke-width': 1 }));
    g.appendChild(el('text', { x: m.left - 8, y: y + 3, 'text-anchor': 'end', 'font-size': 10, fill: '#9ca3af' }, `${Math.round(v)}%`));
  }
  months.forEach((label, i) => {
    g.appendChild(el('text', { x: scX(i), y: m.top + ph + 18, 'text-anchor': 'middle', 'font-size': 10, fill: '#9ca3af' }, label));
  });
  g.appendChild(el('line', { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));

  const endLabels = [];
  Object.entries(series).forEach(([name, pts], si) => {
    const color = TTB_COLORS[si % TTB_COLORS.length];
    const d = pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${scX(i).toFixed(1)},${scY(v).toFixed(1)}`).join(' ');
    g.appendChild(el('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.25, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    pts.forEach((v, i) => g.appendChild(el('circle', { cx: scX(i), cy: scY(v), r: 2.75, fill: color })));
    const lastI = pts.length - 1;
    endLabels.push({ y: scY(pts[lastI]), x: scX(lastI) + 8, text: `${Math.round(pts[lastI])}%`, color });
  });
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    if (endLabels[i].y - endLabels[i - 1].y < 13) endLabels[i].y = endLabels[i - 1].y + 13;
  }
  endLabels.forEach((l) => g.appendChild(el('text', { x: l.x, y: l.y + 3.5, 'font-size': 10.5, 'font-weight': '700', fill: l.color }, l.text)));
  svg.appendChild(g);
})();

(function fillTTBTable() {
  const thead = document.querySelector('#ttbTableWrap thead tr');
  const months = REPORT.ttb.month_labels.slice(1); // drop "Today" (always 0%)
  for (const label of months) thead.innerHTML += `<th class="num">${label}</th>`;

  const body = document.getElementById('ttbTable');
  Object.entries(REPORT.ttb.series).forEach(([name, pts], si) => {
    const color = TTB_COLORS[si % TTB_COLORS.length];
    const tr = document.createElement('tr');
    const cells = pts.slice(1).map((v) => `<td class="num">${Math.round(v)}%</td>`).join('');
    tr.innerHTML = `<td><span class="swatch" style="background:${color}"></span>${name}</td>${cells}`;
    body.appendChild(tr);
  });
})();
</script>

</body>
</html>
"""
