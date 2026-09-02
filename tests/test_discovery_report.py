"""Tests for _discovery_report.py — DiscoveryReport."""

import re

import numpy as np
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._discovery_report import (
    DiscoveryReport,
    _default_levers,
    _is_unphased,
    _safe_improvement,
    _svg_multiline,
)
from how_wrong_is_your_mmm._phaser import Blackout

# Small fixtures -- DiscoveryReport sweeps 5 levers x several diagnostics
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

    def test_five_candidates(self):
        assert len(_default_levers(CHANNELS)) == 5

    def test_blackout_entry_uses_blackout_spec(self):
        levers = _default_levers(CHANNELS)
        label, spec, _, _ = levers[-1]
        assert label == "Blackout"
        assert all(isinstance(v, Blackout) for v in spec.values())


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

    def test_default_levers_used_when_not_supplied(self):
        report = make_report()
        assert len(report.levers_) == 5

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

    def test_identifiability_summary_keys(self):
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
            assert set(report.results_[label]["identifiability"]) == expected

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

    def test_cross_strategy_table_has_one_row_group_per_lever(self):
        report = fit_small(make_report())
        html_out = report.to_html()
        for label, *_ in report.levers_:
            assert f'data-lever="{label}"' in html_out

    def test_saturation_curve_omitted_when_linear(self):
        report = fit_small(make_report(saturation=1.0))
        html_out = report.to_html()
        assert "Assumed saturation curve" not in html_out
        assert "assumed linear" in html_out

    def test_saturation_curve_shown_when_nonlinear(self):
        report = fit_small(make_report(saturation=0.7))
        html_out = report.to_html()
        assert "Assumed saturation curve" in html_out

    def test_client_name_and_plan_year_appear(self):
        report = fit_small(make_report(client_name="Acme Co", plan_year="2024"))
        html_out = report.to_html()
        assert "Acme Co" in html_out
        assert "2024" in html_out
