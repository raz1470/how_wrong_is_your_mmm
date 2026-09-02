"""Tests for _diagnostic.py — CollinearityDiagnostic."""

import warnings

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic

MARGINAL_RETURNS = {"tv": 0.3, "meta": 0.5, "search": 0.4}
CHANNELS = ["tv", "meta", "search"]

SUMMARY_COLS = {
    "channel",
    "true_marginal_return",
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
            true_marginal_returns={"tv": 0.3, "meta": 0.5},
        ).fit(n_sims=5)
        assert set(diag.summary()["channel"]) == {"tv", "meta"}


class TestRealSpendPath:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.6, seed=99)

    def test_fit_runs(self):
        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5)
        assert diag.results_ is not None

    def test_summary_columns(self):
        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLS

    def test_spend_df_not_mutated(self):
        original = self.spend_df.copy()
        CollinearityDiagnostic(
            spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5)
        pd.testing.assert_frame_equal(self.spend_df, original)

    def test_actual_correlation_matches_input(self):

        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5)
        corr = self.spend_df.corr().to_numpy()
        n = 3
        expected = float(sum(corr[i, j] for i in range(n) for j in range(i + 1, n)) / 3)
        assert abs(diag.actual_correlation - expected) < 1e-10

    def test_correlation_param_ignored_when_spend_df_supplied(self):
        diag_low = CollinearityDiagnostic(
            spend_df=self.spend_df,
            true_marginal_returns=MARGINAL_RETURNS,
            correlation=0.1,
        ).fit(n_sims=5)
        diag_high = CollinearityDiagnostic(
            spend_df=self.spend_df,
            true_marginal_returns=MARGINAL_RETURNS,
            correlation=0.9,
        ).fit(n_sims=5)
        assert abs(diag_low.actual_correlation - diag_high.actual_correlation) < 1e-10


class TestValidateSpendData:
    """Real client spend is never as clean as the synthetic DGP output --
    NaN gaps, a paused (zero-spend) channel, or too little history are all
    realistic. fit() must fail with a clear, specific message rather than a
    raw LAPACK error or a silently meaningless result."""

    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.6, seed=99)

    def test_nan_raises_value_error(self):
        bad = self.spend_df.copy()
        bad.iloc[10, 0] = float("nan")
        with pytest.raises(ValueError, match="missing"):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_nan_error_names_the_column(self):
        bad = self.spend_df.copy()
        bad_col = bad.columns[1]
        bad.loc[bad.index[0], bad_col] = float("nan")
        with pytest.raises(ValueError, match=bad_col):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_inf_raises_value_error(self):
        bad = self.spend_df.copy()
        bad.iloc[0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_zero_variance_channel_raises_value_error(self):
        bad = self.spend_df.copy()
        zero_col = bad.columns[-1]
        bad[zero_col] = 0.0
        with pytest.raises(ValueError, match="zero variance"):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_zero_variance_error_names_the_column(self):
        bad = self.spend_df.copy()
        zero_col = bad.columns[-1]
        bad[zero_col] = 12_345.0  # constant, non-zero -- still zero variance
        with pytest.raises(ValueError, match=zero_col):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_non_numeric_column_raises_value_error(self):
        bad = self.spend_df.copy()
        bad[bad.columns[0]] = "not a number"
        with pytest.raises(ValueError, match="numeric"):
            CollinearityDiagnostic(
                spend_df=bad, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_too_few_observations_raises_value_error(self):
        # 3 channels: need len(channels) + 2 = 5 rows minimum.
        tiny = self.spend_df.iloc[:4]
        with pytest.raises(ValueError, match="observation"):
            CollinearityDiagnostic(
                spend_df=tiny, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_minimum_valid_length_does_not_raise(self):
        minimal = self.spend_df.iloc[:5]  # exactly len(channels) + 2
        # Also below the per-channel warning floor -- expected here, not a
        # stray warning, so capture it explicitly rather than let it float.
        with pytest.warns(UserWarning, match="observations"):
            diag = CollinearityDiagnostic(
                spend_df=minimal, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)
        assert diag.results_ is not None

    def test_low_obs_per_channel_warns_but_does_not_raise(self):
        # 3 channels x 10 = 30 is the warning floor; below it should warn.
        short = self.spend_df.iloc[:20]
        with pytest.warns(UserWarning, match="observations"):
            diag = CollinearityDiagnostic(
                spend_df=short, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)
        assert diag.results_ is not None

    def test_ample_data_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CollinearityDiagnostic(
                spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
            ).fit(n_sims=5)

    def test_synthetic_path_is_never_validated(self):
        # Synthetic spend is generated internally and always well-formed --
        # validation only applies to the real spend_df= path.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CollinearityDiagnostic(correlation=0.9, n_obs=6).fit(n_sims=5)


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

    def test_unit_spend_matches_marginal_return_quantiles(self):
        # With planned_spend=1 per channel, incremental revenue equals the
        # raw marginal-return distribution, so its quantiles must match a
        # direct computation.
        planned_spend = {"tv": 1.0, "meta": 1.0, "search": 1.0}
        summary = self.diag.summary(planned_spend=planned_spend)
        direct = (
            self.diag.results_.groupby("channel")["estimated_marginal_return"]
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

    def test_cac_is_reciprocal_of_marginal_return_quantiles(self):
        summary = self.diag.summary(value_per_unit=150.0)
        direct = self.diag.results_.copy()
        direct["cac"] = 1.0 / direct["estimated_marginal_return"]
        direct_range = direct.groupby("channel")["cac"].quantile([0.1, 0.9]).unstack()
        for channel in CHANNELS:
            row = summary[summary["channel"] == channel].iloc[0]
            assert row["cac_p10"] == round(direct_range.loc[channel, 0.1], 4)
            assert row["cac_p90"] == round(direct_range.loc[channel, 0.9], 4)

    def test_roi_equals_marginal_return_times_value_per_unit(self):
        value_per_unit = 150.0
        summary = self.diag.summary(value_per_unit=value_per_unit)
        direct = self.diag.results_.copy()
        direct["roi"] = direct["estimated_marginal_return"] * value_per_unit
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


class TestAnalyticCV:
    """analytic_cv(): closed-form CV, no simulation. See P1.8."""

    def test_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CollinearityDiagnostic(correlation=0.7).analytic_cv()

    def test_returns_series_indexed_by_channel(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        cv = diag.analytic_cv()
        assert set(cv.index) == set(CHANNELS)
        assert (cv > 0).all()

    def test_approximately_matches_large_sample_monte_carlo(self):
        # The Monte Carlo CV estimate should converge to the analytic value
        # as n_sims grows -- a large-n_sims run should land within a loose
        # tolerance (the MC estimate itself is still noisy).
        diag = CollinearityDiagnostic(correlation=0.7, spend_seed=1).fit(n_sims=400)
        analytic = diag.analytic_cv()
        mc = diag.summary().set_index("channel")["coef_of_variation"]
        for ch in CHANNELS:
            assert abs(analytic[ch] - mc[ch]) / analytic[ch] < 0.25

    def test_inversely_proportional_to_marginal_return(self):
        # CV_j = sigma * sqrt([(X'X)^-1]_jj) / beta_j -- exactly inversely
        # proportional to the assumed marginal return (P0.2's core claim).
        diag = CollinearityDiagnostic(
            correlation=0.7,
            true_marginal_returns={"tv": 1.0, "meta": 1.0, "search": 1.0},
        ).fit(n_sims=5)
        cv1 = diag.analytic_cv()
        diag2 = CollinearityDiagnostic(
            correlation=0.7,
            true_marginal_returns={"tv": 2.0, "meta": 2.0, "search": 2.0},
        ).fit(n_sims=5)
        cv2 = diag2.analytic_cv()
        for ch in CHANNELS:
            assert abs(cv2[ch] - cv1[ch] / 2) < 1e-9

    def test_scales_linearly_with_revenue_noise_std(self):
        diag_lo = CollinearityDiagnostic(correlation=0.7, revenue_noise_std=10_000).fit(
            n_sims=5
        )
        diag_hi = CollinearityDiagnostic(correlation=0.7, revenue_noise_std=20_000).fit(
            n_sims=5
        )
        for ch in CHANNELS:
            assert abs(diag_hi.analytic_cv()[ch] - 2 * diag_lo.analytic_cv()[ch]) < 1e-9


class TestNoiseSeedOffset:
    def test_offset_changes_estimates(self):
        diag1 = CollinearityDiagnostic(correlation=0.7).fit(
            n_sims=5, noise_seed_offset=0
        )
        diag2 = CollinearityDiagnostic(correlation=0.7).fit(
            n_sims=5, noise_seed_offset=1000
        )
        assert not diag1.results_["estimated_marginal_return"].equals(
            diag2.results_["estimated_marginal_return"]
        )

    def test_default_offset_reproduces_original_seeding(self):
        # noise_seed_offset defaults to 0, i.e. seeds 0..n_sims-1, matching
        # pre-P1.7 behaviour exactly.
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert list(diag.results_.loc[diag.results_["channel"] == "tv", "sim"]) == [
            0,
            1,
            2,
            3,
            4,
        ]


class TestDeprecatedTrueElasticitiesAlias:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.6, seed=99)

    def test_constructor_warns(self):
        with pytest.warns(FutureWarning, match="true_elasticities is deprecated"):
            CollinearityDiagnostic(
                spend_df=self.spend_df, true_elasticities=MARGINAL_RETURNS
            )

    def test_both_given_raises(self):
        with pytest.raises(ValueError, match="only one of"):
            CollinearityDiagnostic(
                spend_df=self.spend_df,
                true_marginal_returns=MARGINAL_RETURNS,
                true_elasticities=MARGINAL_RETURNS,
            )

    def test_attribute_access_warns(self):
        diag = CollinearityDiagnostic(
            spend_df=self.spend_df, true_marginal_returns=MARGINAL_RETURNS
        )
        with pytest.warns(FutureWarning, match="deprecated"):
            assert diag.true_elasticities == MARGINAL_RETURNS


class TestDemandAndControls:
    """demand_coef/demand + fit(controls=...): omitted-variable bias wiring."""

    def test_backward_compatible_no_demand_by_default(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=10)
        assert diag.demand_ is None
        assert diag.controls_ is None

    def test_demand_coef_creates_bias_when_uncontrolled(self):
        diag = CollinearityDiagnostic(
            correlation=0.7, spend_seed=1, demand_coef=2_000.0
        ).fit(n_sims=150, controls=False)
        tv_bias_pct = diag.summary().set_index("channel").loc["tv", "mean_error_pct"]
        assert abs(tv_bias_pct) > 5

    def test_controlling_with_true_demand_reduces_bias(self):
        omitted = CollinearityDiagnostic(
            correlation=0.7, spend_seed=1, demand_coef=2_000.0
        ).fit(n_sims=150, controls=False)
        controlled = CollinearityDiagnostic(
            correlation=0.7, spend_seed=1, demand_coef=2_000.0
        ).fit(n_sims=150, controls=True)
        bias_omitted = abs(
            omitted.summary().set_index("channel").loc["tv", "mean_error_pct"]
        )
        bias_controlled = abs(
            controlled.summary().set_index("channel").loc["tv", "mean_error_pct"]
        )
        assert bias_controlled < bias_omitted / 3

    def test_controls_true_requires_demand(self):
        with pytest.raises(ValueError, match="requires a demand series"):
            CollinearityDiagnostic(correlation=0.7).fit(n_sims=5, controls=True)

    def test_real_spend_path_demand_coef_without_demand_raises(self):
        spend_df = simulate_spend(n_obs=60, correlation=0.5, seed=2)
        with pytest.raises(ValueError, match="no demand series was supplied"):
            CollinearityDiagnostic(spend_df=spend_df, demand_coef=10.0).fit(n_sims=5)

    def test_real_spend_path_accepts_explicit_demand(self):
        spend_df = simulate_spend(n_obs=60, correlation=0.5, seed=2)
        demand = pd.Series(range(60), dtype=float)
        diag = CollinearityDiagnostic(
            spend_df=spend_df, demand=demand, demand_coef=1.0
        ).fit(n_sims=5)
        assert diag.demand_ is not None
        assert len(diag.demand_) == 60

    def test_explicit_controls_series_used_over_demand(self):
        diag = CollinearityDiagnostic(
            correlation=0.7, spend_seed=1, demand_coef=1_000.0
        )
        diag.fit(n_sims=5)  # draws demand_ via demand_coef, controls defaults to None
        proxy = pd.Series(
            diag.demand_ * 0.6 + 1.0, index=diag.spend_df_.index, name="proxy"
        )
        diag2 = CollinearityDiagnostic(
            correlation=0.7, spend_seed=1, demand_coef=1_000.0
        ).fit(n_sims=5, controls=proxy)
        assert diag2.controls_ is proxy

    def test_analytic_cv_matches_monte_carlo_with_controls(self):
        diag = CollinearityDiagnostic(
            correlation=0.7, spend_seed=3, demand_coef=1_500.0
        ).fit(n_sims=400, controls=True)
        analytic = diag.analytic_cv()
        mc = diag.summary().set_index("channel")["coef_of_variation"]
        for ch in CHANNELS:
            assert abs(analytic[ch] - mc[ch]) / analytic[ch] < 0.3

    def test_synthetic_path_explicit_demand_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="demand must be a 1-D series"):
            CollinearityDiagnostic(
                correlation=0.7, n_obs=50, demand=[1.0, 2.0, 3.0]
            ).fit(n_sims=5)

    def test_saturation_and_adstock_do_not_raise(self):
        diag = CollinearityDiagnostic(correlation=0.7, saturation=0.7, adstock=0.3).fit(
            n_sims=10
        )
        assert diag.results_.shape == (30, 6)
