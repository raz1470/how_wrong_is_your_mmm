"""Tests for _report.py — ReportBuilder (end-to-end client report)."""

import json
import re

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._phaser import Blackout
from how_wrong_is_your_mmm._report import ReportBuilder, _horizon_label

MARGINAL_RETURNS = {"tv": 0.3, "meta": 0.5}
CHANNELS = ["tv", "meta"]

# Small dataset, chosen for test speed: 52 weeks of history, a 13-week plan.
HISTORY_DF = simulate_spend(
    n_obs=52, correlation=0.7, seed=0, channels=CHANNELS, start_date="2019-01-07"
)
PLAN_DF = simulate_spend(
    n_obs=13, correlation=0.7, seed=1, channels=CHANNELS, start_date="2023-01-09"
)

# Cheap kwargs shared by every fit() call in this file -- correctness of the
# underlying numbers is BudgetPhaser/CollinearityDiagnostic's job (covered in
# their own test files); these tests are about ReportBuilder's own wiring.
FIT_KWARGS = {
    "n_sims": 3,
    "grid_steps": 3,
    "n_phasing_seeds": 1,
    "horizons_weeks": [4],
    "ttb_horizons_weeks": [4],
}


def make_builder(**overrides) -> ReportBuilder:
    kwargs = {
        "history_df": HISTORY_DF,
        "plan_df": PLAN_DF,
        "true_marginal_returns": MARGINAL_RETURNS,
        "seed": 0,
    }
    kwargs.update(overrides)
    return ReportBuilder(**kwargs)


class TestHorizonLabel:
    def test_weeks_under_a_year(self):
        assert _horizon_label(4) == "1 month"

    def test_several_months(self):
        assert _horizon_label(13) == "3 months"

    def test_exact_year(self):
        assert _horizon_label(52) == "1 year"

    def test_multiple_exact_years(self):
        assert _horizon_label(104) == "2 years"

    def test_non_exact_year_multiple_falls_back_to_weeks(self):
        assert _horizon_label(60) == "60 weeks"


class TestReportBuilderFit:
    def test_fit_returns_self(self):
        rb = make_builder()
        assert rb.fit(**FIT_KWARGS) is rb

    def test_report_data_populated(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.report_data_ is not None

    def test_to_html_before_fit_raises(self):
        rb = make_builder()
        with pytest.raises(RuntimeError, match="Call fit"):
            rb.to_html()

    def test_report_data_top_level_keys(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert {
            "meta",
            "channels",
            "annual_plan_total",
            "least_reliable_channel",
            "correlation_matrix",
            "diagnose_today",
            "schedule",
            "primary_horizon_weeks",
            "impact_horizons",
            "sensitivity",
            "ttb",
        }.issubset(rb.report_data_.keys())

    def test_channels_match_plan_columns(self):
        rb = make_builder().fit(**FIT_KWARGS)
        keys = [c["key"] for c in rb.report_data_["channels"]]
        assert keys == CHANNELS

    def test_least_reliable_channel_is_a_real_channel(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.report_data_["least_reliable_channel"] in CHANNELS

    def test_correlation_matrix_is_symmetric_square(self):
        rb = make_builder().fit(**FIT_KWARGS)
        corr = rb.report_data_["correlation_matrix"]
        assert set(corr.keys()) == set(CHANNELS)
        for a in CHANNELS:
            for b in CHANNELS:
                assert corr[a][b] == pytest.approx(corr[b][a])
            assert corr[a][a] == pytest.approx(1.0)

    def test_diagnose_today_has_a_row_per_channel(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert set(rb.report_data_["diagnose_today"].keys()) == set(CHANNELS)

    def test_schedule_preserves_plan_length(self):
        rb = make_builder().fit(**FIT_KWARGS)
        for ch in CHANNELS:
            assert len(rb.report_data_["schedule"]["plan"][ch]) == len(PLAN_DF)
            assert len(rb.report_data_["schedule"]["recommended"][ch]) == len(PLAN_DF)

    def test_impact_horizons_match_requested_horizons(self):
        rb = make_builder().fit(**FIT_KWARGS)
        weeks = {h["weeks"] for h in rb.report_data_["impact_horizons"]}
        assert weeks == {4}

    def test_impact_horizons_include_revenue_fields(self):
        rb = make_builder().fit(**FIT_KWARGS)
        row = rb.report_data_["impact_horizons"][0]["channels"][CHANNELS[0]]
        for key in [
            "cv_today",
            "cv_after",
            "reduction_pct",
            "revenue_today_p10",
            "revenue_today_p90",
            "revenue_after_p10",
            "revenue_after_p90",
        ]:
            assert key in row

    def test_sensitivity_has_a_series_per_channel(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert set(rb.report_data_["sensitivity"].keys()) == set(CHANNELS)
        for ch in CHANNELS:
            assert len(rb.report_data_["sensitivity"][ch]) > 0

    def test_sensitivity_default_checkpoints_are_10_20_40_80_pct(self):
        """Default sensitivity_alphas=[0, .125, .25, .5, 1.0] combined with
        the default sensitivity_magnitude_pct=80.0 should give continuous
        checkpoints at 0/10/20/40/80%, matching the approved report design
        rather than the flat 0-40% sweep this originally shipped with."""
        rb = make_builder().fit(**FIT_KWARGS)
        for ch in CHANNELS:
            labels = {
                row["label"]
                for row in rb.report_data_["sensitivity"][ch]
                if not row["is_blackout"]
            }
            assert labels == {"0%", "10%", "20%", "40%", "80%"}

    def test_sensitivity_alphas_and_magnitude_are_overridable(self):
        rb = make_builder().fit(
            **FIT_KWARGS,
            sensitivity_alphas=[0.0, 1.0],
            sensitivity_magnitude_pct=50.0,
        )
        for ch in CHANNELS:
            labels = {
                row["label"]
                for row in rb.report_data_["sensitivity"][ch]
                if not row["is_blackout"]
            }
            assert labels == {"0%", "50%"}

    def test_ttb_series_include_today_zero_point(self):
        rb = make_builder().fit(**FIT_KWARGS)
        ttb = rb.report_data_["ttb"]
        assert ttb["month_labels"][0] == "Today"
        for series in ttb["series"].values():
            assert series[0] == 0.0

    def test_ttb_channel_is_least_reliable_channel(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert (
            rb.report_data_["ttb"]["channel"]
            == rb.report_data_["least_reliable_channel"]
        )

    def test_ttb_omits_blackout_series_when_disabled(self):
        rb = make_builder().fit(**FIT_KWARGS, sensitivity_blackout=None)
        assert "Blackout" not in rb.report_data_["ttb"]["series"]

    def test_ttb_includes_blackout_series_by_default(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert "Blackout" in rb.report_data_["ttb"]["series"]

    def test_client_name_and_plan_year_recorded(self):
        rb = make_builder(client_name="Monzo", plan_year="2027").fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["client_name"] == "Monzo"
        assert rb.report_data_["meta"]["plan_year"] == "2027"

    def test_blank_client_name_reads_as_not_set(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["client_name"] == "(not set)"

    def test_default_horizons_scale_with_plan_length(self):
        rb = make_builder().fit(
            n_sims=3, grid_steps=3, n_phasing_seeds=1, fast_mode=True
        )
        weeks = {h["weeks"] for h in rb.report_data_["impact_horizons"]}
        # 13-week plan -> quarter/plan/double-plan, deduplicated.
        assert weeks == {max(1, 13 // 4), 13, 26}

    def test_continuous_lever_reflected_in_channel_rows(self):
        rb = make_builder().fit(**FIT_KWARGS, sensitivity_blackout=None)
        for row in rb.report_data_["channels"]:
            assert row["lever"] == "continuous"
            assert row["amplitude_pct"] == pytest.approx(40.0)

    def test_blackout_lever_reflected_in_channel_rows(self):
        rb = make_builder(
            max_weekly_deviation_pct=Blackout(max_dark_weeks_per_month=1)
        ).fit(**FIT_KWARGS)
        for row in rb.report_data_["channels"]:
            assert row["lever"] == "blackout"
            assert row["dark_cap"] == 1

    def test_per_channel_mixed_levers(self):
        spec = {"tv": Blackout(max_dark_weeks_per_month=1), "meta": 40.0}
        rb = make_builder(max_weekly_deviation_pct=spec).fit(**FIT_KWARGS)
        by_key = {row["key"]: row for row in rb.report_data_["channels"]}
        assert by_key["tv"]["lever"] == "blackout"
        assert by_key["meta"]["lever"] == "continuous"


class TestReportBuilderFastModeWatermark:
    """fast_mode=True reports must be visibly marked as drafts (a fast_mode
    sample was once mistaken for real client numbers)."""

    def test_meta_records_fast_mode_true(self):
        rb = make_builder().fit(
            n_sims=3,
            grid_steps=3,
            n_phasing_seeds=1,
            horizons_weeks=[4],
            ttb_horizons_weeks=[4],
            fast_mode=True,
        )
        assert rb.report_data_["meta"]["fast_mode"] is True

    def test_meta_records_fast_mode_false_by_default(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["fast_mode"] is False

    def test_html_contains_draft_banner_markup_and_toggle(self):
        rb = make_builder().fit(**FIT_KWARGS)
        html = rb.to_html()
        assert 'id="draftBanner"' in html
        assert "REPORT.meta.fast_mode" in html
        assert "draftBanner" in html and "classList.add('show')" in html


class TestReportBuilderAutoLever:
    """recommend_levers() wiring — auto_lever picks a per-channel lever by
    default, but always defers to an explicit caller choice."""

    def test_auto_lever_applied_by_default(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["auto_lever_applied"] is True

    def test_auto_lever_not_applied_with_explicit_scalar_lever(self):
        rb = make_builder(max_weekly_deviation_pct=60.0).fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["auto_lever_applied"] is False
        for row in rb.report_data_["channels"]:
            assert row["lever"] == "continuous"
            assert row["amplitude_pct"] == pytest.approx(60.0)

    def test_auto_lever_not_applied_with_explicit_blackout(self):
        rb = make_builder(
            max_weekly_deviation_pct=Blackout(max_dark_weeks_per_month=1)
        ).fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["auto_lever_applied"] is False

    def test_auto_lever_not_applied_when_blackout_disabled(self):
        rb = make_builder().fit(**FIT_KWARGS, sensitivity_blackout=None)
        assert rb.report_data_["meta"]["auto_lever_applied"] is False

    def test_auto_lever_false_disables_even_at_default_lever(self):
        rb = make_builder().fit(**FIT_KWARGS, auto_lever=False)
        assert rb.report_data_["meta"]["auto_lever_applied"] is False
        for row in rb.report_data_["channels"]:
            assert row["lever"] == "continuous"
            assert row["amplitude_pct"] == pytest.approx(40.0)

    def test_auto_lever_channel_rows_are_valid_shapes(self):
        rb = make_builder().fit(**FIT_KWARGS)
        for row in rb.report_data_["channels"]:
            assert row["lever"] in {"continuous", "blackout"}
            if row["lever"] == "blackout":
                assert row["dark_cap"] is not None
                assert row["amplitude_pct"] is None
            else:
                assert row["amplitude_pct"] is not None

    def test_phaser_confirmation_populated(self):
        """The selection-bias confirmation pass (BudgetPhaser.fit()) is
        wired all the way through ReportBuilder, not bypassed."""
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.phaser_.confirmation_ is not None
        assert len(rb.phaser_.confirmation_) > 0


class TestReportBuilderScheduleCsv:
    def test_before_fit_raises(self):
        rb = make_builder()
        with pytest.raises(RuntimeError, match="Call fit"):
            rb.schedule_csv()

    def test_row_per_week(self):
        rb = make_builder().fit(**FIT_KWARGS)
        table = rb.schedule_csv()
        assert len(table) == len(PLAN_DF)

    def test_index_is_week_datetime(self):
        rb = make_builder().fit(**FIT_KWARGS)
        table = rb.schedule_csv()
        assert table.index.name == "week"
        pd.testing.assert_index_equal(table.index, PLAN_DF.index, check_names=False)

    def test_three_columns_per_channel(self):
        rb = make_builder().fit(**FIT_KWARGS)
        table = rb.schedule_csv()
        for ch in CHANNELS:
            assert f"{ch}_original_plan" in table.columns
            assert f"{ch}_recommended" in table.columns
            assert f"{ch}_dark_week" in table.columns

    def test_original_plan_column_matches_plan_df(self):
        rb = make_builder().fit(**FIT_KWARGS)
        table = rb.schedule_csv()
        for ch in CHANNELS:
            expected = PLAN_DF[ch].round(2).to_numpy()
            actual = table[f"{ch}_original_plan"].to_numpy()
            assert actual == pytest.approx(expected)

    def test_dark_week_flags_zero_spend_weeks(self):
        rb = make_builder(
            max_weekly_deviation_pct=Blackout(max_dark_weeks_per_month=1)
        ).fit(**FIT_KWARGS)
        table = rb.schedule_csv()
        for ch in CHANNELS:
            dark_rows = table[table[f"{ch}_dark_week"]]
            if len(dark_rows) > 0:
                assert (dark_rows[f"{ch}_recommended"] == 0.0).all()

    def test_continuous_lever_never_dark(self):
        # 55.0 deliberately != _DEFAULT_LEVER (40.0): an explicit lever that
        # happens to match the default sentinel would still trigger
        # auto_lever, per its documented "left at its default" check. A
        # distinct value is what actually exercises "explicit lever pinned,
        # auto_lever bypassed."
        rb = make_builder(max_weekly_deviation_pct=55.0).fit(**FIT_KWARGS)
        assert rb.report_data_["meta"]["auto_lever_applied"] is False
        table = rb.schedule_csv()
        for ch in CHANNELS:
            # A continuous +/-55% lever starting from real positive spend
            # should not land exactly on zero.
            assert not table[f"{ch}_dark_week"].any()

    def test_writes_to_path(self, tmp_path):
        rb = make_builder().fit(**FIT_KWARGS)
        out = tmp_path / "schedule.csv"
        table = rb.schedule_csv(path=str(out))
        assert out.exists()
        reloaded = pd.read_csv(out, index_col="week")
        assert len(reloaded) == len(table)

    def test_no_path_does_not_write_a_file(self, tmp_path):
        rb = make_builder().fit(**FIT_KWARGS)
        rb.schedule_csv()
        assert list(tmp_path.iterdir()) == []


class TestReportBuilderToHtml:
    def test_returns_html_document(self):
        rb = make_builder().fit(**FIT_KWARGS)
        html = rb.to_html()
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_embeds_valid_json_matching_report_data(self):
        rb = make_builder().fit(**FIT_KWARGS)
        html = rb.to_html()
        match = re.search(r"const REPORT = (\{.*?\});", html, re.DOTALL)
        assert match is not None
        embedded = json.loads(match.group(1))
        # Compare via re-serialised JSON rather than the parsed objects:
        # report_data_ legitimately contains NaN (the Blackout row's
        # magnitude_pct), and nan != nan under normal equality.
        assert json.dumps(embedded, sort_keys=True) == json.dumps(
            rb.report_data_, sort_keys=True
        )

    def test_no_leftover_template_placeholder(self):
        rb = make_builder().fit(**FIT_KWARGS)
        html = rb.to_html()
        assert "__REPORT_JSON__" not in html

    def test_client_name_appears_in_html(self):
        rb = make_builder(client_name="Monzo").fit(**FIT_KWARGS)
        html = rb.to_html()
        assert "Monzo" in html

    def test_writes_to_path(self, tmp_path):
        rb = make_builder().fit(**FIT_KWARGS)
        out = tmp_path / "report.html"
        html = rb.to_html(path=str(out))
        assert out.read_text(encoding="utf-8") == html

    def test_no_path_does_not_write_a_file(self, tmp_path):
        rb = make_builder().fit(**FIT_KWARGS)
        rb.to_html()
        assert list(tmp_path.iterdir()) == []


class TestReportBuilderNoiseAssumptionsForwarded:
    """P0.3: base_sales/revenue_noise_std must be settable on ReportBuilder
    and actually change the reported numbers, not just be accepted and
    silently ignored while every internal component uses the package
    default."""

    def test_higher_revenue_noise_std_widens_reported_cv(self):
        rb_low = make_builder(revenue_noise_std=5_000.0).fit(**FIT_KWARGS)
        rb_high = make_builder(revenue_noise_std=80_000.0).fit(**FIT_KWARGS)
        cv_low = rb_low.report_data_["diagnose_today"][CHANNELS[0]]["cv"]
        cv_high = rb_high.report_data_["diagnose_today"][CHANNELS[0]]["cv"]
        assert cv_high > cv_low

    def test_default_matches_collinearity_diagnostic_default(self):
        rb = make_builder().fit(**FIT_KWARGS)
        assert rb.revenue_noise_std == 26_000.0
        assert rb.base_sales == 1_000.0


class TestReportBuilderDeprecatedTrueElasticitiesAlias:
    def test_warns_and_still_works(self):
        with pytest.warns(FutureWarning, match="true_elasticities is deprecated"):
            rb = ReportBuilder(
                history_df=HISTORY_DF,
                plan_df=PLAN_DF,
                true_elasticities=MARGINAL_RETURNS,
                seed=0,
            )
        rb.fit(**FIT_KWARGS)
        assert rb.report_data_ is not None

    def test_both_given_raises(self):
        with pytest.raises(ValueError, match="only one of"):
            ReportBuilder(
                history_df=HISTORY_DF,
                plan_df=PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                true_elasticities=MARGINAL_RETURNS,
            )

    def test_attribute_access_warns(self):
        rb = make_builder()
        with pytest.warns(FutureWarning, match="deprecated"):
            assert rb.true_elasticities == MARGINAL_RETURNS
