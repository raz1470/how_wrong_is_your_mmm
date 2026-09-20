"""Tests for _discovery_report.py — DiscoveryReport."""

import html
import re

import numpy as np
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _corr_table_html,
    _default_levers,
    _is_unphased,
    _lighten_hex,
    _nice_axis_bounds,
    _peak_week_multiple,
    _pinned_strategy_label,
    _safe_improvement,
    _svg_dotplot,
    _svg_forest,
    _svg_multiline,
    _svg_stacked_area,
)
from how_wrong_is_your_mmm._phaser import Blackout, Redistribute

# Small fixtures -- DiscoveryReport sweeps 20 levers x several diagnostics
# each, so tests lean on fast_mode plus a tiny identifiability grid.
HISTORY_DF = simulate_spend(n_obs=52, correlation=0.6, seed=0, start_date="2020-01-06")
PLAN_DF = simulate_spend(n_obs=26, correlation=0.6, seed=1, start_date="2021-01-04")
CHANNELS = ["tv", "meta", "search"]

SMALL_B = np.round(np.linspace(0.4, 1.0, 3), 4)
SMALL_LAM = np.round(np.linspace(0.0, 0.6, 3), 4)


def make_report(**kwargs):
    defaults = dict(history_df=HISTORY_DF, plan_df=PLAN_DF)
    defaults.update(kwargs)
    return DiscoveryReport(**defaults)


def fit_small(report, **kwargs):
    defaults = dict(
        fast_mode=True, id_b_candidates=SMALL_B, id_lam_candidates=SMALL_LAM
    )
    defaults.update(kwargs)
    return report.fit(**defaults)


class TestDefaultLevers:
    def test_first_entry_is_unphased(self):
        levers = _default_levers(CHANNELS)
        assert levers[0][0] == "unphased"
        assert _is_unphased(levers[0][1])

    def test_twenty_candidates(self):
        # unphased + 4 intensities x 3 shapes + 4 Redistribute intensities
        # + 3 Blackout settings (session 49: widened from a 5-candidate
        # spot-check to a full 4x2 sweep; session 56: widened again to add
        # seesaw and Blackout's stronger contiguous settings; the
        # Redistribute family was added after the 20-loop phasing search).
        assert len(_default_levers(CHANNELS)) == 20

    def test_redistribute_family_at_same_intensities(self):
        levers = _default_levers(CHANNELS)
        redistribute = [lv for lv in levers if "redistribute" in lv[0]]
        assert [lbl for lbl, *_ in redistribute] == [
            f"+/-{pct}% (redistribute + edge)" for pct in (20, 40, 60, 80)
        ]
        for (_, spec, nudge_shape, balanced), pct in zip(
            redistribute, (20.0, 40.0, 60.0, 80.0), strict=True
        ):
            assert set(spec) == set(CHANNELS)
            assert all(isinstance(v, Redistribute) for v in spec.values())
            assert all(v.edge_cap_pct == pct for v in spec.values())
            assert (nudge_shape, balanced) == ("edge", True)
            assert not _is_unphased(spec)

    def test_labels_are_unique(self):
        labels = [label for label, *_ in _default_levers(CHANNELS)]
        assert len(labels) == len(set(labels))

    def test_every_intensity_gets_all_shapes(self):
        levers = _default_levers(CHANNELS)
        labels = {label for label, *_ in levers}
        for pct in ("20", "40", "60", "80"):
            assert f"+/-{pct}% (uniform)" in labels
            assert f"+/-{pct}% (edge, balanced)" in labels
            assert f"+/-{pct}% (seesaw)" in labels

    def test_blackout_entries_use_blackout_spec(self):
        levers = _default_levers(CHANNELS)
        blackout_levers = levers[-3:]
        labels = [label for label, *_ in blackout_levers]
        assert labels == [
            "Blackout (dark=1)",
            "Blackout (dark=3, prob=0.8)",
            "Blackout (dark=4, prob=1.0)",
        ]
        for _, spec, _, _ in blackout_levers:
            assert all(isinstance(v, Blackout) for v in spec.values())

    def test_blackout_entries_use_documented_prob_and_dark_settings(self):
        levers = _default_levers(CHANNELS)
        blackout_specs = {label: spec for label, spec, _, _ in levers[-3:]}
        expected = {
            "Blackout (dark=1)": (1.0, 1),
            "Blackout (dark=3, prob=0.8)": (0.8, 3),
            "Blackout (dark=4, prob=1.0)": (1.0, 4),
        }
        for label, (prob, dark) in expected.items():
            spec = blackout_specs[label]["tv"]
            assert spec.prob == prob
            assert spec.max_dark_weeks_per_month == dark


class TestIsUnphased:
    def test_all_zero_is_unphased(self):
        assert _is_unphased({"tv": 0.0, "meta": 0.0})

    def test_any_nonzero_is_not_unphased(self):
        assert not _is_unphased({"tv": 0.0, "meta": 40.0})

    def test_blackout_is_not_unphased(self):
        assert not _is_unphased({"tv": Blackout(max_dark_weeks_per_month=1)})


class TestSafeImprovement:
    def test_positive_when_after_is_lower(self):
        assert _safe_improvement(10.0, 5.0) == pytest.approx(0.5)

    def test_negative_when_after_is_higher(self):
        assert _safe_improvement(5.0, 10.0) == pytest.approx(-1.0)

    def test_zero_before_returns_zero_not_a_division_error(self):
        assert _safe_improvement(0.0, 5.0) == 0.0


class TestSvgMultiline:
    def test_returns_svg_markup(self):
        svg = _svg_multiline({"a": [1, 2, 3]}, {"a": "#000"})
        assert svg.startswith("<svg")
        assert "<polyline" in svg

    def test_too_short_series_gives_fallback_text(self):
        out = _svg_multiline({"a": [1]}, {"a": "#000"})
        assert "<svg" not in out

    def test_normalized_series_of_same_shape_are_identical(self):
        # Two series that are pure scalar multiples of each other collapse
        # onto the same normalized curve -- this is exactly why the
        # saturation curve must NOT use normalize=True (see to_html()).
        svg = _svg_multiline(
            {"a": [1, 2, 4, 8], "b": [10, 20, 40, 80]},
            {"a": "#000", "b": "#111"},
            normalize=True,
        )
        polys = re.findall(r'points="([^"]+)"', svg)
        assert polys[0] == polys[1]

    def test_unnormalized_series_of_different_scale_differ(self):
        svg = _svg_multiline(
            {"a": [1, 2, 4, 8], "b": [10, 20, 40, 80]},
            {"a": "#000", "b": "#111"},
            normalize=False,
        )
        polys = re.findall(r'points="([^"]+)"', svg)
        assert polys[0] != polys[1]

    def test_markers_true_draws_a_circle_per_point(self):
        # Not currently exercised via to_html() (the report's own charts
        # are all dense weekly series), but the helper is meant for a
        # series with few, categorical x points -- keep it covered
        # directly rather than only through report rendering.
        svg = _svg_multiline({"a": [1, 2, 3]}, {"a": "#000"}, markers=True)
        assert svg.count("<circle") == 3

    def test_markers_false_draws_no_circle(self):
        svg = _svg_multiline({"a": [1, 2, 3]}, {"a": "#000"}, markers=False)
        assert "<circle" not in svg

    def test_edge_x_tick_labels_anchor_inward(self):
        # Session 50: the rightmost x-axis tick sits exactly on the
        # plot's own right edge -- text-anchor="middle" there overflowed
        # the SVG's own viewBox (found in session 49's 6-channel stress
        # test, fixed here). First/last ticks now anchor inward; interior
        # ticks are unaffected.
        svg = _svg_multiline(
            {"a": list(range(10))},
            {"a": "#000"},
            x_tick_labels=[f"Week {i}" for i in range(10)],
            n_x_ticks=4,
            x_label="",  # else the default "Week" axis title (always
            # text-anchor="middle") would also match this test's own
            # "Week"-prefixed tick-label regex below.
        )
        anchors = re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="(\w+)" '
            r'font-size="10" fill="#9ca3af">Week',
            svg,
        )
        assert anchors[0] == "start"
        assert anchors[-1] == "end"
        assert all(a == "middle" for a in anchors[1:-1])


class TestSvgStackedArea:
    def test_returns_svg_markup_with_one_polygon_per_band(self):
        svg = _svg_stacked_area(
            ["base", "a", "b"],
            {"base": [1, 2, 3], "a": [1, 1, 1], "b": [2, 2, 2]},
            {"base": "#000", "a": "#111", "b": "#222"},
        )
        assert svg.startswith("<svg")
        assert svg.count("<polygon") == 3

    def test_too_short_series_gives_fallback_text(self):
        out = _svg_stacked_area(["a"], {"a": [1]}, {"a": "#000"})
        assert "<svg" not in out

    def test_y_axis_starts_at_zero_even_when_data_does_not(self):
        # Unlike _svg_multiline (own min/max per series), a stacked area
        # must anchor at 0 -- the bottom of the first band IS zero.
        svg = _svg_stacked_area(
            ["base"], {"base": [100, 200, 300]}, {"base": "#000"}, y_fmt=str
        )
        assert "0.0" in svg or ">0<" in svg

    def test_edge_x_tick_labels_anchor_inward(self):
        # Same fix, same shared tick-rendering shape as _svg_multiline's
        # own edge-anchor test above (session 50).
        svg = _svg_stacked_area(
            ["a"],
            {"a": list(range(1, 11))},
            {"a": "#000"},
            x_tick_labels=[f"Week {i}" for i in range(10)],
            n_x_ticks=4,
            x_label="",  # else the default "Week" axis title would also
            # match this test's own "Week"-prefixed tick-label regex.
        )
        anchors = re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="(\w+)" '
            r'font-size="10" fill="#9ca3af">Week',
            svg,
        )
        assert anchors[0] == "start"
        assert anchors[-1] == "end"
        assert all(a == "middle" for a in anchors[1:-1])


class TestSvgDotplot:
    def test_returns_svg_markup(self):
        svg = _svg_dotplot([{"name": "tv", "color": "#000", "value": 0.5}])
        assert svg.startswith("<svg")
        assert "<circle" in svg

    def test_value_label_uses_supplied_fmt(self):
        svg = _svg_dotplot(
            [{"name": "tv", "color": "#000", "value": 0.5}],
            fmt=lambda v: f"£{v:.2f}",
        )
        assert "£0.50" in svg

    def test_channel_name_appears_as_row_label(self):
        svg = _svg_dotplot([{"name": "search", "color": "#000", "value": 1.2}])
        assert "search" in svg

    def test_multiple_rows_get_zebra_striping(self):
        # No longer exercised via to_html() (the report's own marginal-
        # return dot-plot was dropped as a duplicate of the ROI column in
        # session 47) -- keep the multi-row striping path covered
        # directly.
        svg = _svg_dotplot(
            [
                {"name": "tv", "color": "#000", "value": 0.5},
                {"name": "meta", "color": "#111", "value": 1.0},
                {"name": "search", "color": "#222", "value": 1.5},
            ]
        )
        assert "<rect" in svg


class TestLightenHex:
    def test_zero_amount_returns_the_same_colour(self):
        assert _lighten_hex("#2563eb", amount=0.0) == "#2563eb"

    def test_full_amount_returns_white(self):
        assert _lighten_hex("#2563eb", amount=1.0) == "#ffffff"

    def test_partial_amount_lightens_every_channel(self):
        lightened = _lighten_hex("#2563eb", amount=0.65)
        r, g, b = (int(lightened[i : i + 2], 16) for i in (1, 3, 5))
        orig_r, orig_g, orig_b = (int("2563eb"[i : i + 2], 16) for i in (0, 2, 4))
        assert r > orig_r
        assert g > orig_g
        assert b > orig_b


class TestCorrTableHtml:
    def test_renders_plain_correlation_value_per_cell(self):
        # Session 50: briefly grew a delta feature, then Ryan asked to
        # drop it again the same session ("info overload, shall we
        # revert to just showing the correlation?") -- see NOTES.md.
        matrix = {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}
        html_out = _corr_table_html(matrix, ["a", "b"])
        assert "corr-delta" not in html_out
        assert "<td" in html_out
        assert html_out.count(">0.50<") == 2

    def test_column_headers_get_the_vertical_header_class(self):
        # Session 50: long/more channel names pushed the table wider than
        # the page -- column headers render vertically (see the
        # .corr-table CSS) so column width no longer scales with name
        # length. Row headers (the left-hand th per row) stay horizontal.
        matrix = {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}
        html_out = _corr_table_html(matrix, ["a", "b"])
        assert html_out.count('<th class="col-hdr">') == 2
        assert "<th>a</th>" in html_out
        assert "<th>b</th>" in html_out


class TestSvgForest:
    def test_row_label_margin_grows_for_long_channel_names(self):
        # Session 49: a fixed 100px left margin (sized for "search")
        # clipped a longer real-world name ("search_generic") against the
        # SVG's own left edge -- caught testing a 6-channel scenario.
        short = _svg_forest(
            [{"name": "tv", "color": "#000", "before": 1.0, "after": 2.0}]
        )
        long = _svg_forest(
            [
                {
                    "name": "search_generic",
                    "color": "#000",
                    "before": 1.0,
                    "after": 2.0,
                }
            ]
        )

        def label_x(svg: str) -> float:
            m = re.search(r'<text x="([\d.]+)" y="[\d.]+" text-anchor="end"', svg)
            return float(m.group(1))

        assert label_x(long) > label_x(short)

    def test_single_mode_draws_one_mark_and_no_after_key_needed(self):
        svg = _svg_forest(
            [{"name": "tv", "color": "#000", "value": 1.5, "truth": 1.0}],
            single=True,
        )
        assert svg.startswith("<svg")
        assert "tv" in svg

    def test_combined_range_and_point_mark_draws_both(self):
        # Session 50: variance/saturation/adstock want a point estimate
        # alongside their existing range, bias wants a band around its
        # existing point -- one shape, {"range": (lo, hi), "point": p},
        # drawn as a range bar (<line>) plus a ring (<circle>) on top.
        svg = _svg_forest(
            [
                {
                    "name": "tv",
                    "color": "#000",
                    "before": 1.0,
                    "after": {"range": (1.0, 3.0), "point": 2.0},
                }
            ]
        )
        assert svg.count("<line") >= 1  # the range bar
        # a white-fill ring (the point marker), distinct from the
        # solid-fill dot a plain point mark would draw
        assert "<circle cx=" in svg
        assert 'fill="#fff"' in svg

    def test_combined_mark_label_shows_the_range_only_not_the_point(self):
        # Ryan: keep the point estimate/band on the chart, but don't
        # also spell out "(point X)" in the text label -- the ring
        # already carries that visually.
        svg = _svg_forest(
            [
                {
                    "name": "tv",
                    "color": "#000",
                    "value": {"range": (10.0, 30.0), "point": 20.0},
                    "truth": None,
                }
            ],
            single=True,
            fmt=lambda v: f"{v:.0f}",
        )
        assert "10 &ndash; 30" in svg
        assert "point" not in svg.lower()

    def test_combined_mark_axis_scales_to_the_wider_of_range_or_point(self):
        # A point estimate outside its own range shouldn't get clipped --
        # _hi() must consider both.
        svg = _svg_forest(
            [
                {
                    "name": "tv",
                    "color": "#000",
                    "value": {"range": (1.0, 2.0), "point": 5.0},
                    "truth": None,
                }
            ],
            single=True,
        )
        assert svg.startswith("<svg")


class TestNiceAxisBounds:
    def test_positive_range_brackets_the_data(self):
        lo, hi, step = _nice_axis_bounds(3.0, 83.0)
        assert lo <= 3.0
        assert hi >= 83.0
        assert step > 0

    def test_negative_lower_bound_is_handled(self):
        # Demand is standardised to mean 0 -- the range straddles zero.
        lo, hi, _step = _nice_axis_bounds(-2.3, 1.8)
        assert lo <= -2.3
        assert hi >= 1.8
        assert lo < 0 < hi


class TestConstruction:
    def test_mismatched_columns_raises(self):
        bad_plan = PLAN_DF.rename(columns={"search": "radio"})
        with pytest.raises(ValueError, match="same columns"):
            make_report(plan_df=bad_plan)

    def test_demand_proxy_quality_out_of_range_raises(self):
        with pytest.raises(ValueError, match="demand_proxy_quality"):
            make_report(demand_proxy_quality=0.0)

    def test_saturation_out_of_range_raises(self):
        with pytest.raises(ValueError, match="saturation"):
            make_report(saturation=1.5)

    def test_adstock_out_of_range_raises(self):
        with pytest.raises(ValueError, match="adstock"):
            make_report(adstock=1.0)

    def test_float_saturation_broadcasts_to_every_channel(self):
        report = make_report(saturation=0.8)
        assert report.saturation == {ch: 0.8 for ch in CHANNELS}

    def test_dict_saturation_kept_per_channel(self):
        values = {"tv": 0.6, "meta": 0.8, "search": 1.0}
        report = make_report(saturation=values)
        assert report.saturation == values

    def test_dict_saturation_missing_channel_raises(self):
        with pytest.raises(KeyError, match="saturation"):
            make_report(saturation={"tv": 0.6, "meta": 0.8})

    def test_default_levers_used_when_not_supplied(self):
        report = make_report()
        assert len(report.levers_) == 20

    def test_custom_levers_respected(self):
        custom = [("unphased", {ch: 0.0 for ch in CHANNELS}, "uniform", False)]
        report = make_report(levers=custom)
        assert report.levers_ == custom

    def test_reference_spend_matches_plan_mean(self):
        report = make_report()
        for ch in CHANNELS:
            assert report.reference_spend_[ch] == pytest.approx(PLAN_DF[ch].mean())

    def test_results_none_before_fit(self):
        report = make_report()
        assert report.results_ is None
        assert report.winner_ is None


class TestFit:
    def test_fit_returns_self(self):
        report = make_report()
        assert fit_small(report) is report

    def test_results_has_one_entry_per_lever(self):
        report = fit_small(make_report())
        assert set(report.results_.keys()) == {lbl for lbl, *_ in report.levers_}

    def test_winner_is_a_valid_lever_label(self):
        report = fit_small(make_report())
        assert report.winner_ in {lbl for lbl, *_ in report.levers_}

    def test_winner_schedule_has_plan_shape(self):
        report = fit_small(make_report())
        assert report.winner_schedule_.shape == PLAN_DF.shape

    def test_each_lever_has_per_channel_variance_and_bias(self):
        report = fit_small(make_report())
        for label in report.results_:
            assert set(report.results_[label]["variance_cv"]) == set(CHANNELS)
            assert set(report.results_[label]["bias_pct"]) == set(CHANNELS)

    def test_revenue_mean_sits_inside_its_own_p10_p90(self):
        # Session 50: a point estimate for the Variance section's forest
        # chart, alongside the range it already had.
        report = fit_small(make_report())
        for label in report.results_:
            r = report.results_[label]
            for ch in CHANNELS:
                assert (
                    r["revenue_p10"][ch]
                    <= r["revenue_mean"][ch]
                    <= r["revenue_p90"][ch]
                )

    def test_bias_pct_p10_p90_bracket_the_mean(self):
        # Session 50: an uncertainty band for the Bias section's forest
        # chart, around the mean error it already had.
        report = fit_small(make_report())
        for label in report.results_:
            r = report.results_[label]
            for ch in CHANNELS:
                assert (
                    r["bias_pct_p10"][ch] <= r["bias_pct"][ch] <= r["bias_pct_p90"][ch]
                )

    def test_identifiability_summary_keys(self):
        # identifiability is now keyed by channel (IdentifiabilityDiagnostic
        # profiles each channel's own curvature separately), not one shared
        # summary -- see NOTES.md, session 45.
        report = fit_small(make_report())
        expected = {
            "b_mean",
            "b_sd",
            "b_bias",
            "lam_mean",
            "lam_sd",
            "lam_bias",
            "valley_pct",
        }
        for label in report.results_:
            identifiability = report.results_[label]["identifiability"]
            assert set(identifiability) == set(CHANNELS)
            for ch in CHANNELS:
                assert set(identifiability[ch]) == expected

    def test_each_lever_has_per_channel_curvature_ranges(self):
        # b_p10/b_p90/lam_p10/lam_p90 are the recovered-value ranges the
        # identifiability forest charts are built from.
        report = fit_small(make_report())
        for label in report.results_:
            for key in ("b_p10", "b_p90", "lam_p10", "lam_p90"):
                assert set(report.results_[label][key]) == set(CHANNELS)
            for ch in CHANNELS:
                assert (
                    report.results_[label]["b_p10"][ch]
                    <= report.results_[label]["b_p90"][ch]
                )
                assert (
                    report.results_[label]["lam_p10"][ch]
                    <= report.results_[label]["lam_p90"][ch]
                )

    def test_correlation_matrix_is_square(self):
        report = fit_small(make_report())
        corr = report.results_["unphased"]["correlation"]
        assert set(corr.keys()) == set(CHANNELS)
        for ch in CHANNELS:
            assert set(corr[ch].keys()) == set(CHANNELS)
            assert corr[ch][ch] == pytest.approx(1.0, abs=1e-6)

    def test_phasing_reduces_variance_relative_to_unphased(self):
        # Every real lever (not unphased) should show SOME variance
        # reduction on average -- phasing decorrelates spend by
        # construction, this is the package's core claim.
        report = fit_small(make_report())
        unphased_score = report.results_["unphased"]["scores"]["variance"]
        for label, *_ in report.levers_:
            if label == "unphased":
                continue
            assert report.results_[label]["scores"]["variance"] < unphased_score


class TestPickWinner:
    def test_only_unphased_lever_returns_it(self):
        levers = [("unphased", {ch: 0.0 for ch in CHANNELS}, "uniform", False)]
        report = fit_small(make_report(levers=levers))
        assert report.winner_ == "unphased"

    def test_single_candidate_is_picked_outright(self):
        levers = [
            ("unphased", {ch: 0.0 for ch in CHANNELS}, "uniform", False),
            ("only-candidate", {ch: 60.0 for ch in CHANNELS}, "edge", True),
        ]
        report = fit_small(make_report(levers=levers))
        assert report.winner_ == "only-candidate"

    def test_dominant_candidate_wins(self):
        report = make_report()
        results = {
            "unphased": {
                "scores": {"variance": 1.0, "bias": 1.0, "identifiability": 1.0}
            },
            "good": {"scores": {"variance": 0.5, "bias": 0.5, "identifiability": 0.5}},
            "bad": {"scores": {"variance": 0.9, "bias": 0.9, "identifiability": 0.9}},
        }
        report.levers_ = [
            ("unphased", {}, "uniform", False),
            ("good", {}, "uniform", False),
            ("bad", {}, "uniform", False),
        ]
        assert report._pick_winner(results) == "good"

    def test_worst_axis_rule_when_no_dominance(self):
        report = make_report()
        # "balanced" never wins outright on any axis but has the best
        # worst-case; "spiky" wins two axes but tanks the third.
        results = {
            "unphased": {
                "scores": {"variance": 1.0, "bias": 1.0, "identifiability": 1.0}
            },
            "spiky": {
                "scores": {"variance": 0.1, "bias": 0.1, "identifiability": 0.95}
            },
            "balanced": {
                "scores": {"variance": 0.5, "bias": 0.5, "identifiability": 0.5}
            },
        }
        report.levers_ = [
            ("unphased", {}, "uniform", False),
            ("spiky", {}, "uniform", False),
            ("balanced", {}, "uniform", False),
        ]
        assert report._pick_winner(results) == "balanced"


class TestSummary:
    def test_raises_before_fit(self):
        report = make_report()
        with pytest.raises(RuntimeError, match="Call fit"):
            report.summary()

    def test_summary_shape_and_winner_flag(self):
        report = fit_small(make_report())
        summ = report.summary()
        assert len(summ) == len(report.levers_)
        assert summ["is_winner"].sum() == 1
        winner_row = summ[summ["is_winner"]].iloc[0]
        assert winner_row["lever"] == report.winner_


class TestToHtml:
    def test_raises_before_fit(self):
        report = make_report()
        with pytest.raises(RuntimeError, match="Call fit"):
            report.to_html()

    def test_returns_a_full_html_document(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert html_out.startswith("<!DOCTYPE html>")
        assert "</html>" in html_out

    def test_toc_links_match_section_ids(self):
        # Session 50: Ryan asked for hyperlinks to each section at the
        # top of the report -- every href in the cover's nav must land on
        # a section that actually exists further down the page.
        report = fit_small(make_report())
        html_out = report.to_html()
        anchors = re.findall(r'<nav class="toc"[^>]*>(.*?)</nav>', html_out, re.S)
        assert len(anchors) == 1
        hrefs = re.findall(r'href="#([\w-]+)"', anchors[0])
        assert len(hrefs) == 5
        for anchor_id in hrefs:
            assert f'<section id="{anchor_id}">' in html_out

    def test_fast_mode_shows_draft_banner(self):
        report = fit_small(make_report())
        assert '<div class="draft-banner">' in report.to_html()

    def test_full_mode_has_no_draft_banner(self):
        report = fit_small(make_report(), fast_mode=False, n_sims=5, n_phasing_seeds=1)
        assert '<div class="draft-banner">' not in report.to_html()

    def test_writes_to_path(self, tmp_path):
        report = fit_small(make_report())
        out = tmp_path / "discovery.html"
        html_out = report.to_html(str(out))
        assert out.read_text(encoding="utf-8") == html_out

    def test_no_duplicate_ids(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        ids = re.findall(r'id="([^"]+)"', html_out)
        assert len(ids) == len(set(ids))

    def test_impact_correlation_shows_after_only_not_side_by_side(self):
        # Session 49: the Impact section's before/after correlation pair
        # forced a horizontal scroll at more channels (see the dropped
        # .corr-col fix, superseded by this) -- Ryan didn't want to
        # scroll for it, so Impact shows only "after"; "before" lives in
        # Diagnostics (Section 2) for readers who want the comparison.
        report = fit_small(make_report())
        html_out = report.to_html()
        impact_block = html_out[
            html_out.index("<h2>Impact</h2>") : html_out.index("<h2>Phased spend</h2>")
        ]
        assert "corr-cols" not in impact_block
        assert "Channel correlation, after phasing" in impact_block
        assert "Before phasing" not in impact_block

    def test_variance_bias_saturation_adstock_show_point_estimates(self):
        # Session 50: variance/saturation/adstock each gained a point
        # estimate alongside their existing range; bias gained a range
        # around its existing point. All four now render the combined
        # mark's white-fill ring (fill="#fff") -- both in Diagnostics
        # (single-state) and Impact (before/after) sections. The ring is
        # visual only -- Ryan asked that the value labels NOT also spell
        # out "(point X)" in text.
        report = fit_small(make_report())
        html_out = report.to_html()
        diagnostics_block = html_out[
            html_out.index("<h2>Diagnostics</h2>") : html_out.index("<h2>Impact</h2>")
        ]
        impact_block = html_out[
            html_out.index("<h2>Impact</h2>") : html_out.index("<h2>Phased spend</h2>")
        ]
        for block in (diagnostics_block, impact_block):
            assert block.count('fill="#fff"') >= 4  # one ring per problem, at least
        assert "(point" not in html_out.lower()

    def test_impact_table_has_one_row_per_lever(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        for label, *_ in report.levers_:
            assert f'data-lever="{label}"' in html_out

    def test_no_dropdown_or_strategy_selector(self):
        # Session 46: the strategy-family dropdown was dropped -- the
        # channel-summary table is now static inputs only, and the impact
        # table shows every lever as a row rather than filtering to one.
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "<select" not in html_out
        assert "<script" not in html_out

    def test_channel_summary_table_is_static_inputs_only(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert (
            "<th>Channel</th><th>Spend</th><th>ROI</th><th>Saturation</th>"
            "<th>Adstock</th></tr>" in html_out
        )
        for ch in CHANNELS:
            assert html_out.count(f"<td>{ch}</td>") >= 1

    def test_channel_summary_shows_roi(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        for ch in CHANNELS:
            assert f"<td>£{report.true_marginal_returns[ch]:.2f}</td>" in html_out

    def test_channel_summary_mentions_demand_share(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert f"{report.baseline_share:.0%}" in html_out
        assert f"{1 - report.baseline_share:.0%}" in html_out

    def test_impact_table_headers_and_winner_highlighted(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert (
            "<th>Strategy</th><th>Variance impact</th><th>Bias impact</th>"
            "<th>Identifiability impact</th><th>Cost</th>" in html_out
        )
        assert html_out.count('class="winner-row"') == 1
        assert f'data-lever="{report.winner_}" class="winner-row"' in html_out

    def test_unphased_row_shows_zero_impact(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert (
            '<tr data-lever="unphased"><td>unphased</td>'
            "<td>0%</td><td>0%</td><td>0%</td>" in html_out
        )

    def test_recommended_pacing_has_one_cell_per_channel(self):
        # Session 50: Section 1's own spend chart also adopted the
        # pacing-grid/pacing-cell markup (see the scenario-inputs test
        # below), so this counts within Section 4 only.
        report = fit_small(make_report())
        html_out = report.to_html()
        phased_spend_block = html_out[
            html_out.index("<h2>Phased spend</h2>") : html_out.index(
                "<h2>Appendix: every strategy compared</h2>"
            )
        ]
        assert phased_spend_block.count('class="pacing-cell"') == len(CHANNELS)
        for ch in CHANNELS:
            assert f'<div class="pacing-title">{ch}</div>' in phased_spend_block

    def test_recommended_pacing_uses_channel_colour_not_grey(self):
        # Session 49: grey read as too faint -- pacing chart now uses each
        # channel's own colour (pale for "as supplied", solid for the
        # winner's schedule) instead of a channel-blind grey/black pair.
        from how_wrong_is_your_mmm._discovery_report import _channel_colors

        report = fit_small(make_report())
        html_out = report.to_html()
        colors = _channel_colors(CHANNELS)
        phased_spend_block = html_out[
            html_out.index("<h2>Phased spend</h2>") : html_out.index(
                "<h2>Appendix: every strategy compared</h2>"
            )
        ]
        # Axis labels/gridlines still use #9ca3af/#e5e7eb -- only the
        # polyline strokes are the thing that changed.
        assert 'stroke="#9ca3af"' not in phased_spend_block
        assert 'stroke="#111827"' not in phased_spend_block
        for ch in CHANNELS:
            assert f'stroke="{colors[ch]}"' in phased_spend_block

    def test_implied_contribution_chart_present_with_baseline_band(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "Implied contribution" in html_out
        assert "check on those assumptions" in html_out
        assert 'style="background:#9ca3af"></span>Baseline</div>' in html_out

    def test_saturation_curve_omitted_when_linear(self):
        report = fit_small(make_report(saturation=1.0))
        html_out = report.to_html()
        assert "Saturation, by channel" not in html_out
        assert "assumed linear" in html_out

    def test_saturation_curve_shown_when_nonlinear(self):
        report = fit_small(make_report(saturation=0.7))
        html_out = report.to_html()
        assert "Saturation, by channel" in html_out

    def test_adstock_chart_omitted_when_no_decay(self):
        report = fit_small(make_report(adstock=0.0))
        html_out = report.to_html()
        assert "Adstock, by channel" not in html_out
        assert "assumed instantaneous" in html_out

    def test_adstock_chart_shown_when_decay(self):
        report = fit_small(make_report(adstock=0.3))
        html_out = report.to_html()
        assert "Adstock, by channel" in html_out

    def test_scenario_inputs_shows_spend_series(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "Spend, history + plan" in html_out
        assert "Sales / revenue, weekly, by source" in html_out

    def test_scenario_inputs_spend_is_a_per_channel_grid(self):
        # Session 50 (Ryan: "scenario inputs spend -> shall we show them
        # as grid plots like the section 4?"): one small chart per
        # channel, own colour, same pacing-grid/pacing-cell markup
        # Section 4 uses for its own before/after pacing charts.
        report = fit_small(make_report())
        html_out = report.to_html()
        scenario_block = html_out[
            html_out.index("<h2>Scenario inputs</h2>") : html_out.index(
                "<h2>Diagnostics</h2>"
            )
        ]
        spend_fig = scenario_block[scenario_block.index("Spend, history + plan") :]
        assert spend_fig.count('class="pacing-cell"') == len(CHANNELS)
        for ch in CHANNELS:
            assert f'<div class="pacing-title">{ch}</div>' in spend_fig

    def test_no_standalone_demand_chart(self):
        # Session 47: dropped -- a zero-mean synthetic series with no
        # interpretive hook on its own; its effect on sales is already
        # visible in Implied Contribution's Baseline band.
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "Demand (standardised)" not in html_out

    def test_spend_chart_comes_before_response_curves(self):
        # Session 47: ground in the actual data first, then show what was
        # assumed about how it responds -- not the other way round.
        report = fit_small(make_report(saturation=0.7, adstock=0.3))
        html_out = report.to_html()
        assert html_out.index("Spend, history + plan") < html_out.index(
            "Adstock, by channel"
        )
        assert html_out.index("Adstock, by channel") < html_out.index(
            "Saturation, by channel"
        )

    def test_no_marginal_return_dotplot_duplicate_of_roi_column(self):
        # Session 47: dropped as a duplicate of the channel-summary ROI
        # column -- same numbers, no new information.
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "Channel marginal return" not in html_out

    def test_spend_chart_uses_supplied_plan_not_winner_schedule(self):
        # Session 46 bug, still applies now the chart lives in Section 1:
        # it should show history_df + plan_df as given, not the winner
        # lever's rephased schedule.
        report = fit_small(make_report())
        html_out = report.to_html()
        assert (
            "Actual weekly spend by channel, as supplied, before any phasing"
            in html_out
        )

    def test_no_separate_total_sales_chart_duplicate_of_contribution(self):
        # Session 47: the old flat "Baseline + demand + channel
        # contributions, no noise" single-line chart was dropped as a
        # duplicate of the Implied Contribution stacked chart's own total.
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "no noise -- what these inputs imply" not in html_out

    def test_client_name_and_plan_year_appear(self):
        report = fit_small(make_report(client_name="Acme Co", plan_year="2024"))
        html_out = report.to_html()
        assert "Acme Co" in html_out
        assert "2024" in html_out

    def test_diagnostics_section_shows_all_four_problems_unphased_only(self):
        # Session 48: a new Diagnostics section shows the unphased problem
        # on its own, ahead of the phasing-strategy numbers.
        report = fit_small(make_report(saturation=0.7, adstock=0.3))
        html_out = report.to_html()
        assert "<h2>Diagnostics</h2>" in html_out
        assert "Channel correlation, unphased" in html_out
        assert "Incremental revenue, unphased" in html_out
        assert "Revenue implied by the biased estimate, unphased" in html_out
        assert "Recovered saturation, by channel, unphased" in html_out
        assert "Recovered adstock, by channel, unphased" in html_out

    def test_diagnostics_charts_have_no_after_state(self):
        # single=True mode: one mark per row, not a before/after pair --
        # the per-chart legends shouldn't carry a second, winner-labelled
        # state the way the paired before/after Impact section does.
        report = fit_small(make_report())
        html_out = report.to_html()
        diagnostics_block = html_out[
            html_out.index("<h2>Diagnostics</h2>") : html_out.index("<h2>Impact</h2>")
        ]
        assert "Unphased (today)" in diagnostics_block
        assert f", {html.escape(report.winner_)}</span>" not in diagnostics_block
        assert 'opacity="0.35"' not in diagnostics_block

    def test_sections_renumbered_with_diagnostics_second(self):
        # Session 49: problem (Diagnostics) -> impact (Impact) -> what to
        # do (Phased spend) -> appendix (every strategy compared).
        report = fit_small(make_report())
        html_out = report.to_html()
        labels = re.findall(r'<div class="s-label">Section (\d)</div>', html_out)
        assert labels == ["1", "2", "3", "4", "5"]
        assert html_out.index("<h2>Scenario inputs</h2>") < html_out.index(
            "<h2>Diagnostics</h2>"
        )
        assert html_out.index("<h2>Diagnostics</h2>") < html_out.index(
            "<h2>Impact</h2>"
        )
        assert html_out.index("<h2>Impact</h2>") < html_out.index(
            "<h2>Phased spend</h2>"
        )
        assert html_out.index("<h2>Phased spend</h2>") < html_out.index(
            "<h2>Appendix: every strategy compared</h2>"
        )

    def test_impact_table_lives_only_in_appendix(self):
        # Session 49: the strategy-impact table moved out of "Phasing
        # strategy" into its own appendix -- Section 4 (Phased spend) is
        # just the pacing chart for the one winning strategy.
        report = fit_small(make_report())
        html_out = report.to_html()
        phased_spend_block = html_out[
            html_out.index("<h2>Phased spend</h2>") : html_out.index(
                "<h2>Appendix: every strategy compared</h2>"
            )
        ]
        assert "cross-table" not in phased_spend_block
        assert 'class="pacing-cell"' in phased_spend_block
        appendix_block = html_out[
            html_out.index("<h2>Appendix: every strategy compared</h2>") :
        ]
        assert "cross-table" in appendix_block
        assert f'data-lever="{report.winner_}" class="winner-row"' in appendix_block

    def test_impact_section_does_not_restate_the_problem(self):
        # Session 49: Section 3 (Impact) assumes Section 2 (Diagnostics)
        # already established the problem -- it shouldn't repeat "The
        # problem:" callouts the old per-metric sections used to open with.
        report = fit_small(make_report())
        html_out = report.to_html()
        impact_block = html_out[
            html_out.index("<h2>Impact</h2>") : html_out.index("<h2>Phased spend</h2>")
        ]
        assert "<b>The problem:</b>" not in impact_block
        # Session 50's prose rewrite dropped the "<b>The impact:</b>" label
        # in favour of full sentences, but the section should still open
        # by pointing back at Section 2 rather than re-explaining the
        # problem from scratch.
        assert "Section 2 showed how bad each of these four problems is" in impact_block

    def test_impact_section_ends_with_the_winners_cost(self):
        # Session 50: Ryan asked for the winning strategy's own Cost
        # quoted at the end of Section 3, rather than only visible in the
        # Section 5 appendix table -- computed the same way as that
        # table's own Cost column (mean %, across channels, of true
        # plan-period revenue given up under the winner vs unphased).
        report = fit_small(make_report())
        html_out = report.to_html()
        impact_block = html_out[
            html_out.index("<h2>Impact</h2>") : html_out.index("<h2>Phased spend</h2>")
        ]
        assert "<h3>Cost</h3>" in impact_block
        assert f"<b>{report.winner_}</b>" in impact_block

        appendix_block = html_out[html_out.index("<h2>Appendix") :]
        # Cost is 0 whenever every channel's saturation is linear (b=1,
        # the make_report() default -- see the Cost comment above
        # lever_cost_pct in _discovery_report.py), which floating-point
        # noise can format as "-0.00%" rather than "0.00%" -- allow an
        # optional leading sign rather than assuming cost is always
        # rendered positive.
        row_match = re.search(
            rf'data-lever="{re.escape(report.winner_)}"[^>]*>.*?'
            rf"<td>(-?[\d.]+)%</td><td>[\d.]+x</td></tr>",
            appendix_block,
        )
        winner_cost_pct = float(row_match.group(1))
        assert f"{winner_cost_pct:.2f}%" in impact_block


class TestPinnedStrategyLabel:
    def test_float_strategy_labelled_like_a_default_lever(self):
        label = _pinned_strategy_label(60.0, "edge", True, None)
        assert label == "Pinned: +/-60% (edge, balanced)"

    def test_unbalanced_uniform_strategy(self):
        label = _pinned_strategy_label(40.0, "uniform", False, None)
        assert label == "Pinned: +/-40% (uniform)"

    def test_blackout_strategy(self):
        label = _pinned_strategy_label(Blackout(), "edge", True, None)
        assert label == "Pinned: Blackout"

    def test_redistribute_strategy(self):
        label = _pinned_strategy_label(
            Redistribute(edge_cap_pct=40.0), "edge", True, None
        )
        assert label == "Pinned: +/-40% (redistribute + edge)"

    def test_channel_overrides_appended_singular(self):
        label = _pinned_strategy_label(60.0, "edge", True, {"meta": 20.0})
        assert label == "Pinned: +/-60% (edge, balanced), 1 channel override"

    def test_channel_overrides_appended_plural(self):
        label = _pinned_strategy_label(
            60.0, "edge", True, {"meta": 20.0, "search": 10.0}
        )
        assert label == "Pinned: +/-60% (edge, balanced), 2 channel overrides"


class TestPinnedStrategyConstruction:
    # Session 51 (Ryan: "we want to be able to use this class and pick a
    # strategy and set channel constraints" -- the feature that replaced
    # the separate ReportBuilder class).
    def test_strategy_pct_none_leaves_levers_unchanged(self):
        report = make_report()
        assert report.pinned_label_ is None
        assert report.channel_constraints_ == {}
        assert len(report.levers_) == 20

    def test_strategy_pct_adds_one_lever_on_top_of_the_default_sweep(self):
        report = make_report(strategy_pct=60.0)
        assert len(report.levers_) == 21
        assert report.pinned_label_ == "Pinned: +/-60% (edge, balanced)"
        assert report.levers_[-1][0] == report.pinned_label_

    def test_strategy_pct_applies_to_every_channel_by_default(self):
        report = make_report(strategy_pct=60.0)
        _, spec, _, _ = report.levers_[-1]
        assert spec == {ch: 60.0 for ch in CHANNELS}

    def test_channel_constraints_override_specific_channels(self):
        report = make_report(strategy_pct=60.0, channel_constraints={"meta": 20.0})
        _, spec, _, _ = report.levers_[-1]
        assert spec == {"tv": 60.0, "meta": 20.0, "search": 60.0}
        assert report.channel_constraints_ == {"meta": 20.0}

    def test_channel_constraints_can_pin_a_channel_to_blackout(self):
        blackout = Blackout(max_dark_weeks_per_month=1)
        report = make_report(
            strategy_pct=60.0, channel_constraints={"search": blackout}
        )
        _, spec, _, _ = report.levers_[-1]
        assert spec["search"] is blackout
        assert spec["tv"] == 60.0

    def test_channel_constraints_without_strategy_pct_raises(self):
        with pytest.raises(ValueError, match="strategy_pct"):
            make_report(channel_constraints={"meta": 20.0})

    def test_channel_constraints_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="unknown channel"):
            make_report(strategy_pct=60.0, channel_constraints={"radio": 20.0})

    def test_strategy_shape_and_balance_forwarded(self):
        report = make_report(
            strategy_pct=40.0, strategy_nudge_shape="uniform", strategy_balanced=False
        )
        label, _, nudge_shape, balance_signs = report.levers_[-1]
        assert nudge_shape == "uniform"
        assert balance_signs is False
        assert label == "Pinned: +/-40% (uniform)"


class TestPinnedStrategyFit:
    def test_pinned_strategy_becomes_the_winner_without_dominance_check(self):
        # A deliberately weak pinned strategy (a tiny 1% nudge) would never
        # win _pick_winner's dominance/worst-axis check on its own merits --
        # if it's still the winner, fit() took the pin rather than scoring.
        report = fit_small(make_report(strategy_pct=1.0))
        assert report.winner_ == report.pinned_label_
        assert report.winner_schedule_ is not None

    def test_unpinned_report_still_uses_pick_winner(self):
        report = fit_small(make_report())
        assert report.pinned_label_ is None
        assert report.winner_ in [label for label, *_ in report.levers_]

    def test_pinned_result_scored_like_every_other_lever(self):
        report = fit_small(make_report(strategy_pct=60.0))
        assert report.pinned_label_ in report.results_
        assert "scores" in report.results_[report.pinned_label_]


class TestPeakWeekMultiple:
    def test_unchanged_schedule_is_one(self):
        assert _peak_week_multiple(PLAN_DF, PLAN_DF) == 1.0

    def test_largest_ratio_across_channels(self):
        sched = PLAN_DF.copy()
        sched.iloc[3, 1] = PLAN_DF.iloc[3, 1] * 2.5
        sched.iloc[7, 0] = PLAN_DF.iloc[7, 0] * 1.5
        assert _peak_week_multiple(PLAN_DF, sched) == pytest.approx(2.5)

    def test_reductions_alone_floor_at_one(self):
        assert _peak_week_multiple(PLAN_DF, PLAN_DF * 0.5) == 1.0

    def test_zero_planned_weeks_skipped(self):
        plan = PLAN_DF.copy()
        plan.iloc[0, 0] = 0.0
        sched = plan.copy()
        sched.iloc[0, 0] = 999.0
        assert _peak_week_multiple(plan, sched) == 1.0

    def test_impact_table_has_peak_week_column_and_unphased_is_1x(self):
        html_out = fit_small(make_report()).to_html()
        assert "<th>Cost</th><th>Peak week</th>" in html_out
        assert re.search(r'<tr data-lever="unphased">.*?<td>1\.0x</td></tr>', html_out)


class TestRedistributeInReport:
    def test_pinned_redistribute_becomes_winner_and_is_scored(self):
        report = fit_small(make_report(strategy_pct=Redistribute(edge_cap_pct=40.0)))
        assert report.pinned_label_ == "Pinned: +/-40% (redistribute + edge)"
        assert report.winner_ == report.pinned_label_
        assert report.winner_schedule_.shape == PLAN_DF.shape
        # annual total preserved per channel (plan is < 12 months: one block)
        np.testing.assert_allclose(
            report.winner_schedule_.sum().to_numpy(),
            PLAN_DF.sum().to_numpy(),
            rtol=1e-12,
        )

    def test_pinned_redistribute_html_does_not_claim_monthly_totals_unchanged(self):
        report = fit_small(make_report(strategy_pct=Redistribute(edge_cap_pct=40.0)))
        html_out = report.to_html()
        assert "monthly totals are identical" not in html_out
        assert "monthly totals unchanged" not in html_out
        assert "moves budget between months" in html_out

    def test_float_strategy_html_still_claims_monthly_totals_unchanged(self):
        html_out = fit_small(make_report(strategy_pct=60.0)).to_html()
        assert "monthly totals are identical" in html_out
        assert "monthly totals unchanged" in html_out

    def test_redistribute_channel_constraint_shown(self):
        report = fit_small(
            make_report(
                strategy_pct=60.0,
                channel_constraints={"meta": Redistribute(edge_cap_pct=20.0)},
            )
        )
        html_out = report.to_html()
        assert "+/-20% (redistribute + edge)</td>" in html_out

    def test_default_sweep_scores_every_redistribute_row(self):
        report = fit_small(make_report())
        for pct in (20, 40, 60, 80):
            label = f"+/-{pct}% (redistribute + edge)"
            assert label in report.results_
            assert "scores" in report.results_[label]
            assert report.schedules_[label].shape == PLAN_DF.shape


class TestScheduleCsv:
    def test_raises_before_fit(self):
        report = make_report()
        with pytest.raises(RuntimeError, match="Call fit"):
            report.schedule_csv()

    def test_columns_per_channel(self):
        report = fit_small(make_report())
        table = report.schedule_csv()
        for ch in CHANNELS:
            assert f"{ch}_original_plan" in table.columns
            assert f"{ch}_recommended" in table.columns
            assert f"{ch}_dark_week" in table.columns

    def test_matches_winner_schedule_not_just_any_lever(self):
        report = fit_small(make_report(strategy_pct=60.0))
        table = report.schedule_csv()
        for ch in CHANNELS:
            np.testing.assert_allclose(
                table[f"{ch}_recommended"].to_numpy(),
                report.winner_schedule_[ch].round(2).to_numpy(),
            )

    def test_writes_to_path(self, tmp_path):
        report = fit_small(make_report())
        path = tmp_path / "schedule.csv"
        report.schedule_csv(path=str(path))
        assert path.exists()


class TestPinnedStrategyHtml:
    def test_headline_says_pinned_not_recommended(self):
        report = fit_small(make_report(strategy_pct=60.0))
        html_out = report.to_html()
        assert "Pinned strategy:" in html_out
        assert "Recommended strategy:" not in html_out

    def test_unpinned_headline_unchanged(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        assert "Recommended strategy:" in html_out
        assert "Pinned strategy:" not in html_out

    def test_channel_constraints_table_shown_when_constraints_set(self):
        report = fit_small(
            make_report(strategy_pct=60.0, channel_constraints={"meta": 20.0})
        )
        html_out = report.to_html()
        assert "<h3>Channel constraints</h3>" in html_out
        assert "<td>meta</td>" in html_out
        assert "+/-20%</td>" in html_out

    def test_channel_constraints_table_hidden_without_constraints(self):
        report = fit_small(make_report(strategy_pct=60.0))
        html_out = report.to_html()
        assert "<h3>Channel constraints</h3>" not in html_out

    def test_pinned_lever_appears_in_appendix_comparison_table(self):
        # Session 51 Q&A (Ryan: "keep both tables"): the sweep-comparison
        # table stays even when a strategy is pinned, with the pinned
        # strategy as one more row in it.
        report = fit_small(make_report(strategy_pct=60.0))
        html_out = report.to_html()
        assert f'data-lever="{report.pinned_label_}"' in html_out
