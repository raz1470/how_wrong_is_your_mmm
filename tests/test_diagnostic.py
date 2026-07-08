"""Tests for _diagnostic.py — CollinearityDiagnostic."""

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic

ELASTICITIES = {"tv": 0.3, "meta": 0.5, "search": 0.4}
CHANNELS = ["tv", "meta", "search"]

SUMMARY_COLS = {
    "channel",
    "true_elasticity",
    "mean_estimated",
    "std_estimated",
    "mean_error_pct",
    "coef_of_variation",
}


class TestSyntheticSpendPath:
    def test_fit_returns_self(self):
        diag = CollinearityDiagnostic(correlation=0.7)
        assert diag.fit(n_sims=5) is diag

    def test_results_shape(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        # 3 channels x 5 sims
        assert diag.results_.shape == (15, 6)

    def test_summary_columns(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLS

    def test_summary_rows(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert len(diag.summary()) == 3

    def test_summary_channels(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert set(diag.summary()["channel"]) == set(CHANNELS)

    def test_cv_positive(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=10)
        assert (diag.summary()["coef_of_variation"] > 0).all()

    def test_actual_correlation(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert 0 < diag.actual_correlation < 1

    def test_correlation_matrix_shape(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert diag.correlation_matrix.shape == (3, 3)

    def test_fast_mode_overrides_n_sims(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=50, fast_mode=True)
        assert len(diag.results_) == 3 * 10

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CollinearityDiagnostic(correlation=0.7).summary()

    def test_actual_correlation_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            _ = CollinearityDiagnostic(correlation=0.7).actual_correlation

    def test_custom_channels(self):
        diag = CollinearityDiagnostic(
            correlation=0.5,
            channels=["tv", "meta"],
            true_elasticities={"tv": 0.3, "meta": 0.5},
        ).fit(n_sims=5)
        assert set(diag.summary()["channel"]) == {"tv", "meta"}


class TestRealSpendPath:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.6, seed=99)

    def test_fit_runs(self):
        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5)
        assert diag.results_ is not None

    def test_summary_columns(self):
        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLS

    def test_spend_df_not_mutated(self):
        original = self.spend_df.copy()
        CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5)
        pd.testing.assert_frame_equal(self.spend_df, original)

    def test_actual_correlation_matches_input(self):

        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5)
        corr = self.spend_df.corr().to_numpy()
        n = 3
        expected = float(sum(corr[i, j] for i in range(n) for j in range(i + 1, n)) / 3)
        assert abs(diag.actual_correlation - expected) < 1e-10

    def test_correlation_param_ignored_when_spend_df_supplied(self):
        diag_low = CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES, correlation=0.1
        ).fit(n_sims=5)
        diag_high = CollinearityDiagnostic(
            spend_df=self.spend_df, true_elasticities=ELASTICITIES, correlation=0.9
        ).fit(n_sims=5)
        assert abs(diag_low.actual_correlation - diag_high.actual_correlation) < 1e-10


class TestPlannedSpend:
    def setup_method(self):
        self.diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=20)

    def test_backward_compatible_when_none(self):
        # planned_spend=None must reproduce exactly the pre-feature columns/values.
        assert set(self.diag.summary().columns) == SUMMARY_COLS
        assert set(self.diag.summary(planned_spend=None).columns) == SUMMARY_COLS
        pd.testing.assert_frame_equal(
            self.diag.summary(), self.diag.summary(planned_spend=None)
        )

    def test_adds_revenue_columns(self):
        summary = self.diag.summary(
            planned_spend={"tv": 1_000_000, "meta": 800_000, "search": 600_000}
        )
        assert {"incremental_revenue_p10", "incremental_revenue_p90"}.issubset(
            summary.columns
        )
        # p10 should not exceed p90 for any channel.
        assert (
            summary["incremental_revenue_p10"] <= summary["incremental_revenue_p90"]
        ).all()

    def test_unit_spend_matches_elasticity_quantiles(self):
        # With planned_spend=1 per channel, incremental revenue equals the raw
        # elasticity distribution, so its quantiles must match a direct computation.
        planned_spend = {"tv": 1.0, "meta": 1.0, "search": 1.0}
        summary = self.diag.summary(planned_spend=planned_spend)
        direct = (
            self.diag.results_.groupby("channel")["estimated_elasticity"]
            .quantile([0.1, 0.9])
            .unstack()
        )
        for channel in CHANNELS:
            row = summary[summary["channel"] == channel].iloc[0]
            # summary() rounds to 4dp; compare against equally-rounded direct value.
            assert row["incremental_revenue_p10"] == round(direct.loc[channel, 0.1], 4)
            assert row["incremental_revenue_p90"] == round(direct.loc[channel, 0.9], 4)

    def test_scaling_is_linear(self):
        base = {"tv": 100_000, "meta": 100_000, "search": 100_000}
        scaled = {k: v * 3 for k, v in base.items()}
        summary_base = self.diag.summary(planned_spend=base)
        summary_scaled = self.diag.summary(planned_spend=scaled)
        for channel in CHANNELS:
            base_p90 = summary_base.loc[
                summary_base["channel"] == channel, "incremental_revenue_p90"
            ].iloc[0]
            scaled_p90 = summary_scaled.loc[
                summary_scaled["channel"] == channel, "incremental_revenue_p90"
            ].iloc[0]
            # summary() rounds to 4dp; allow rounding error compounded by the 3x scale.
            assert abs(scaled_p90 - 3 * base_p90) < 1e-3

    def test_missing_channel_key_raises(self):
        with pytest.raises(KeyError):
            self.diag.summary(planned_spend={"tv": 1_000_000, "meta": 800_000})

    def test_extra_keys_are_ignored(self):
        planned_spend = {
            "tv": 1_000_000,
            "meta": 800_000,
            "search": 600_000,
            "tiktok": 500_000,
        }
        summary = self.diag.summary(planned_spend=planned_spend)
        assert len(summary) == 3


class TestValuePerUnit:
    def setup_method(self):
        self.diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=20)

    def test_backward_compatible_when_none(self):
        assert set(self.diag.summary().columns) == SUMMARY_COLS
        assert set(self.diag.summary(value_per_unit=None).columns) == SUMMARY_COLS
        pd.testing.assert_frame_equal(
            self.diag.summary(), self.diag.summary(value_per_unit=None)
        )

    def test_adds_cac_and_roi_columns(self):
        summary = self.diag.summary(value_per_unit=150.0)
        assert {"cac_p10", "cac_p90", "roi_p10", "roi_p90"}.issubset(summary.columns)

    def test_cac_is_reciprocal_of_elasticity_quantiles(self):
        summary = self.diag.summary(value_per_unit=150.0)
        direct = self.diag.results_.copy()
        direct["cac"] = 1.0 / direct["estimated_elasticity"]
        direct_range = direct.groupby("channel")["cac"].quantile([0.1, 0.9]).unstack()
        for channel in CHANNELS:
            row = summary[summary["channel"] == channel].iloc[0]
            assert row["cac_p10"] == round(direct_range.loc[channel, 0.1], 4)
            assert row["cac_p90"] == round(direct_range.loc[channel, 0.9], 4)

    def test_roi_equals_elasticity_times_value_per_unit(self):
        value_per_unit = 150.0
        summary = self.diag.summary(value_per_unit=value_per_unit)
        direct = self.diag.results_.copy()
        direct["roi"] = direct["estimated_elasticity"] * value_per_unit
        direct_range = direct.groupby("channel")["roi"].quantile([0.1, 0.9]).unstack()
        for channel in CHANNELS:
            row = summary[summary["channel"] == channel].iloc[0]
            assert row["roi_p10"] == round(direct_range.loc[channel, 0.1], 4)
            assert row["roi_p90"] == round(direct_range.loc[channel, 0.9], 4)

    def test_cac_and_roi_independent_of_planned_spend(self):
        # Linear DGP: CAC and ROI are channel properties, not scaled by
        # how much you plan to spend.
        small = self.diag.summary(
            planned_spend={"tv": 1, "meta": 1, "search": 1}, value_per_unit=150.0
        )
        large = self.diag.summary(
            planned_spend={"tv": 10_000_000, "meta": 8_000_000, "search": 6_000_000},
            value_per_unit=150.0,
        )
        pd.testing.assert_frame_equal(
            small[["channel", "cac_p10", "cac_p90", "roi_p10", "roi_p90"]],
            large[["channel", "cac_p10", "cac_p90", "roi_p10", "roi_p90"]],
        )

    def test_value_per_unit_scales_incremental_revenue(self):
        planned_spend = {"tv": 1_000_000, "meta": 800_000, "search": 600_000}
        no_ltv = self.diag.summary(planned_spend=planned_spend)
        with_ltv = self.diag.summary(planned_spend=planned_spend, value_per_unit=150.0)
        for channel in CHANNELS:
            no_ltv_p90 = no_ltv.loc[
                no_ltv["channel"] == channel, "incremental_revenue_p90"
            ].iloc[0]
            with_ltv_p90 = with_ltv.loc[
                with_ltv["channel"] == channel, "incremental_revenue_p90"
            ].iloc[0]
            assert abs(with_ltv_p90 - 150.0 * no_ltv_p90) < 1
