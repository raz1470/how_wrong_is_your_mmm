"""Tests for _dgp.py — simulate_spend and simulate_sales."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import (
    DEMAND_PROCESSES,
    calibrate_baseline,
    simulate_demand,
    simulate_sales,
    simulate_spend,
)

CHANNELS = ["tv", "meta", "search"]
MARGINAL_RETURNS = {"tv": 0.3, "meta": 0.5, "search": 0.4}

# First row of simulate_spend(n_obs=12, correlation=0.7, seed=...), captured
# from the implementation BEFORE the demand/demand_share additions.
GOLDEN_FIRST_ROW = {
    0: [72072.80579934, 90757.8543783, 56372.38706309],
    7: [101404.79956818, 81557.71741405, 59759.27639805],
    42: [106958.88636585, 80364.66048063, 62761.45149999],
}


class TestSimulateSpend:
    def test_default_shape(self):
        df = simulate_spend()
        assert df.shape == (104, 3)
        assert list(df.columns) == CHANNELS

    def test_custom_channels(self):
        df = simulate_spend(channels=["tv", "meta"], n_obs=52)
        assert df.shape == (52, 2)
        assert list(df.columns) == ["tv", "meta"]

    def test_custom_n_obs(self):
        df = simulate_spend(n_obs=52)
        assert df.shape == (52, 3)

    def test_correlation_direction(self):
        """Higher target correlation should produce higher mean pairwise correlation."""

        def mean_corr(corr_val):
            df = simulate_spend(correlation=corr_val, seed=0)
            c = df.corr().to_numpy()
            n = len(df.columns)
            return np.mean([c[i, j] for i in range(n) for j in range(i + 1, n)])

        assert mean_corr(0.8) > mean_corr(0.2)

    def test_reproducibility(self):
        pd.testing.assert_frame_equal(simulate_spend(seed=42), simulate_spend(seed=42))

    def test_different_seeds_differ(self):
        assert not simulate_spend(seed=0).equals(simulate_spend(seed=1))

    def test_invalid_correlation(self):
        with pytest.raises(ValueError):
            simulate_spend(correlation=0.0)
        with pytest.raises(ValueError):
            simulate_spend(correlation=1.0)

    def test_start_date_gives_datetime_index(self):
        df = simulate_spend(n_obs=52, start_date="2023-01-02")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 52

    def test_no_start_date_gives_integer_index(self):
        df = simulate_spend(n_obs=52)
        assert not isinstance(df.index, pd.DatetimeIndex)


class TestSimulateSales:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.5, seed=0)

    def test_output_length(self):
        assert len(simulate_sales(self.spend_df, MARGINAL_RETURNS)) == len(
            self.spend_df
        )

    def test_output_name(self):
        assert simulate_sales(self.spend_df, MARGINAL_RETURNS).name == "sales"

    def test_reproducibility(self):
        s1 = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=7)
        s2 = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=7)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=0)
        s2 = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=1)
        assert not s1.equals(s2)

    def test_marginal_return_direction(self):
        """Higher marginal return should produce higher mean sales."""
        low = simulate_sales(
            self.spend_df, {"tv": 0.1, "meta": 0.1, "search": 0.1}, revenue_noise_std=0
        ).mean()
        high = simulate_sales(
            self.spend_df, {"tv": 0.9, "meta": 0.9, "search": 0.9}, revenue_noise_std=0
        ).mean()
        assert high > low

    def test_missing_channel_raises(self):
        with pytest.raises(ValueError, match="has no entry"):
            simulate_sales(self.spend_df, {"tv": 0.3})

    def test_default_marginal_returns(self):
        # should not raise when using default marginal returns on default channels
        sales = simulate_sales(self.spend_df)
        assert len(sales) == len(self.spend_df)

    def test_default_marginal_returns_are_not_stale_elasticity_values(self):
        # P0.1: the illustrative defaults were reset from the original
        # elasticity-scale numbers (0.3/0.5/0.4) to defensible marginal
        # returns -- guard against a silent regression back to those.
        from how_wrong_is_your_mmm._dgp import _DEFAULT_MARGINAL_RETURNS

        assert _DEFAULT_MARGINAL_RETURNS == {"tv": 0.5, "meta": 1.0, "search": 1.5}


class TestDeprecatedTrueElasticitiesAlias:
    """true_elasticities is a deprecated alias for true_marginal_returns --
    kept working (with a FutureWarning) for backward compatibility."""

    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.5, seed=0)

    def test_warns(self):
        with pytest.warns(FutureWarning, match="true_elasticities is deprecated"):
            simulate_sales(self.spend_df, true_elasticities=MARGINAL_RETURNS)

    def test_still_works(self):
        with pytest.warns(FutureWarning):
            s1 = simulate_sales(
                self.spend_df, true_elasticities=MARGINAL_RETURNS, seed=3
            )
        s2 = simulate_sales(
            self.spend_df, true_marginal_returns=MARGINAL_RETURNS, seed=3
        )
        pd.testing.assert_series_equal(s1, s2)

    def test_both_given_raises(self):
        with pytest.raises(ValueError, match="only one of"):
            simulate_sales(
                self.spend_df,
                true_marginal_returns=MARGINAL_RETURNS,
                true_elasticities=MARGINAL_RETURNS,
            )


class TestSimulateSpendUnchanged:
    """The demand/demand_share additions must not move any published number.

    Golden values captured from the pre-change implementation. Session 41's
    nudge_shape lesson: changing a generator's call sequence silently moves
    every seeded schedule and every figure published from one.
    """

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_defaults_byte_identical(self, seed):
        df = simulate_spend(n_obs=12, correlation=0.7, seed=seed)
        np.testing.assert_allclose(
            df.iloc[0].to_numpy(), GOLDEN_FIRST_ROW[seed], rtol=0, atol=1e-6
        )

    def test_supplying_the_drawn_demand_reproduces_output(self):
        """The internal draw always happens, so passing back the series the
        function would have drawn anyway must change nothing."""
        df_a, demand = simulate_spend(n_obs=12, seed=0, return_demand=True)
        df_b = simulate_spend(n_obs=12, seed=0, demand=demand.to_numpy())
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_demand_is_first_draw_from_seed(self):
        """Guarantees the extension is faithful to already-published spend."""
        _, demand = simulate_spend(n_obs=20, seed=3, return_demand=True)
        expected = np.random.default_rng(3).standard_normal(20)
        np.testing.assert_allclose(demand.to_numpy(), expected)


class TestSimulateSpendDemand:
    def test_return_demand_returns_tuple(self):
        result = simulate_spend(n_obs=20, seed=0, return_demand=True)
        assert isinstance(result, tuple)
        df, demand = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(demand, pd.Series)
        assert len(demand) == 20
        assert demand.name == "demand"

    def test_return_demand_shares_the_frame_index(self):
        df, demand = simulate_spend(
            n_obs=10, seed=0, start_date="2023-01-02", return_demand=True
        )
        pd.testing.assert_index_equal(df.index, demand.index)

    def test_supplied_demand_is_used(self):
        custom = np.linspace(-2.0, 2.0, 30)
        _, demand = simulate_spend(n_obs=30, seed=0, demand=custom, return_demand=True)
        np.testing.assert_allclose(demand.to_numpy(), custom)

    def test_supplied_demand_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length n_obs"):
            simulate_spend(n_obs=30, demand=np.zeros(29))

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_demand_share_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="demand_share"):
            simulate_spend(n_obs=20, demand_share=bad)

    @pytest.mark.parametrize("demand_share", [1.0, 0.75, 0.5, 0.25, 0.0])
    def test_pairwise_correlation_is_preserved(self, demand_share):
        """demand_share must not change what the package actually diagnoses."""
        df = simulate_spend(
            n_obs=20_000, correlation=0.7, seed=0, demand_share=demand_share
        )
        c = df.corr().to_numpy()
        pairs = [c[i, j] for i in range(3) for j in range(i + 1, 3)]
        assert np.mean(pairs) == pytest.approx(0.7, abs=0.02)

    @pytest.mark.parametrize("demand_share", [1.0, 0.5, 0.25, 0.0])
    def test_coupling_follows_sqrt_demand_share_times_correlation(self, demand_share):
        df, demand = simulate_spend(
            n_obs=20_000,
            correlation=0.7,
            seed=0,
            demand_share=demand_share,
            return_demand=True,
        )
        observed = np.corrcoef(df["tv"].to_numpy(), demand.to_numpy())[0, 1]
        assert observed == pytest.approx(np.sqrt(demand_share * 0.7), abs=0.02)

    def test_demand_share_default_matches_omitting_it(self):
        pd.testing.assert_frame_equal(
            simulate_spend(n_obs=50, seed=1),
            simulate_spend(n_obs=50, seed=1, demand_share=1.0),
        )

    def test_demand_share_leaves_channel_noise_untouched(self):
        """The planning factor comes from a separate stream, so two
        demand_share settings at one seed differ ONLY by the common factor --
        they are the same draw seen two ways, unlike nudge_shape."""
        n = 200
        base = simulate_spend(n_obs=n, correlation=0.7, seed=0, demand_share=1.0)
        mixed, demand = simulate_spend(
            n_obs=n, correlation=0.7, seed=0, demand_share=0.5, return_demand=True
        )
        planning = np.random.default_rng([0, 20_260_828]).standard_normal(n)
        expected_delta = 20_000 * (
            (np.sqrt(0.5) - 1) * demand.to_numpy() + np.sqrt(0.5) * planning
        )
        np.testing.assert_allclose(
            mixed["tv"].to_numpy() - base["tv"].to_numpy(), expected_delta
        )


class TestSimulateDemand:
    @pytest.mark.parametrize("process", DEMAND_PROCESSES)
    def test_standardised(self, process):
        """gamma must mean the same thing across processes, so every process
        is standardised -- otherwise comparing shapes compares amplitudes."""
        series = simulate_demand(200, process=process, seed=0)
        assert series.mean() == pytest.approx(0.0, abs=1e-12)
        assert series.std() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("process", DEMAND_PROCESSES)
    def test_length_and_reproducibility(self, process):
        a = simulate_demand(60, process=process, seed=5)
        b = simulate_demand(60, process=process, seed=5)
        assert a.shape == (60,)
        np.testing.assert_allclose(a, b)

    def test_ar1_is_more_persistent_than_white_noise(self):
        def lag1(series):
            return np.corrcoef(series[:-1], series[1:])[0, 1]

        assert lag1(simulate_demand(500, process="ar1", seed=0)) > lag1(
            simulate_demand(500, process="white_noise", seed=0)
        )

    def test_seasonal_repeats_at_its_period(self):
        series = simulate_demand(104, process="seasonal", seed=0, season_period=52.0)
        np.testing.assert_allclose(series[:52], series[52:], atol=1e-9)

    def test_seasonal_ar1_sits_between(self):
        def lag1(series):
            return np.corrcoef(series[:-1], series[1:])[0, 1]

        mixed = lag1(simulate_demand(500, process="seasonal_ar1", seed=0))
        assert lag1(simulate_demand(500, process="ar1", seed=0)) < mixed

    def test_invalid_process_raises(self):
        with pytest.raises(ValueError, match="process must be one of"):
            simulate_demand(50, process="brownian")

    @pytest.mark.parametrize("bad", [-1.0, 1.0, 1.5])
    def test_invalid_ar_coef_raises(self, bad):
        with pytest.raises(ValueError, match="ar_coef"):
            simulate_demand(50, process="ar1", ar_coef=bad)

    def test_invalid_season_weight_raises(self):
        with pytest.raises(ValueError, match="season_weight"):
            simulate_demand(50, process="seasonal_ar1", season_weight=1.5)

    def test_invalid_season_period_raises(self):
        with pytest.raises(ValueError, match="season_period"):
            simulate_demand(50, process="seasonal", season_period=0.0)


class TestCalibrateBaseline:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.7, seed=0)

    def test_shares_sum_to_one(self):
        cal = calibrate_baseline(self.spend_df, baseline_share=0.7)
        total = sum(cal.channel_shares.values()) + cal.baseline_level / cal.total_sales
        assert total == pytest.approx(1.0)

    def test_baseline_level_matches_requested_share(self):
        cal = calibrate_baseline(self.spend_df, baseline_share=0.72)
        assert cal.baseline_level / cal.total_sales == pytest.approx(0.72)

    def test_demand_coef_is_level_times_cv(self):
        cal = calibrate_baseline(self.spend_df, baseline_share=0.7, baseline_cv=0.05)
        assert cal.demand_coef == pytest.approx(cal.baseline_level * 0.05)

    def test_flat_baseline_gives_zero_demand_coef(self):
        """A flat baseline, however large, is absorbed by the intercept."""
        cal = calibrate_baseline(self.spend_df, baseline_share=0.9, baseline_cv=0.0)
        assert cal.demand_coef == 0.0

    def test_contributions_are_roi_times_mean_spend(self):
        returns = {"tv": 0.5, "meta": 1.0, "search": 1.5}
        cal = calibrate_baseline(self.spend_df, returns, baseline_share=0.7)
        for ch, roi in returns.items():
            assert cal.contributions[ch] == pytest.approx(
                roi * self.spend_df[ch].mean()
            )

    def test_self_consistent_shares_reconcile_exactly(self):
        """ROI and share are not independent inputs. Shares derived from the
        ROIs must imply one identical total for every channel."""
        cal = calibrate_baseline(self.spend_df, baseline_share=0.7)
        checked = calibrate_baseline(
            self.spend_df, baseline_share=0.7, reported_shares=cal.channel_shares
        )
        assert checked.share_spread_pct == pytest.approx(0.0, abs=1e-9)
        for implied in checked.implied_totals.values():
            assert implied == pytest.approx(cal.total_sales)

    def test_inconsistent_shares_produce_a_spread(self):
        checked = calibrate_baseline(
            self.spend_df,
            baseline_share=0.72,
            reported_shares={"tv": 0.08, "meta": 0.09, "search": 0.11},
        )
        assert checked.share_spread_pct > 10.0

    def test_spread_is_none_without_reported_shares(self):
        cal = calibrate_baseline(self.spend_df, baseline_share=0.7)
        assert cal.share_spread_pct is None
        assert cal.implied_totals == {}

    def test_to_frame_includes_baseline_row(self):
        frame = calibrate_baseline(self.spend_df, baseline_share=0.7).to_frame()
        assert "baseline" in frame["channel"].to_numpy()
        assert frame["share_of_sales"].sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_invalid_baseline_share_raises(self, bad):
        with pytest.raises(ValueError, match="baseline_share"):
            calibrate_baseline(self.spend_df, baseline_share=bad)

    def test_negative_cv_raises(self):
        with pytest.raises(ValueError, match="baseline_cv"):
            calibrate_baseline(self.spend_df, baseline_share=0.7, baseline_cv=-0.1)

    def test_missing_channel_raises(self):
        with pytest.raises(ValueError, match="no entry in true_marginal_returns"):
            calibrate_baseline(self.spend_df, {"tv": 0.5}, baseline_share=0.7)

    def test_missing_reported_share_raises(self):
        with pytest.raises(ValueError, match="no entry for channel"):
            calibrate_baseline(
                self.spend_df, baseline_share=0.7, reported_shares={"tv": 0.1}
            )


class TestSimulateSalesDemand:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=52, correlation=0.7, seed=0)

    def test_no_demand_is_unchanged(self):
        pd.testing.assert_series_equal(
            simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=1),
            simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=1, demand=None),
        )

    def test_zero_coefficient_is_unchanged(self):
        demand = simulate_demand(52, seed=0)
        pd.testing.assert_series_equal(
            simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=1),
            simulate_sales(
                self.spend_df, MARGINAL_RETURNS, seed=1, demand=demand, demand_coef=0.0
            ),
        )

    def test_demand_shifts_sales_by_coef_times_demand(self):
        demand = simulate_demand(52, seed=0)
        without = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=1)
        with_demand = simulate_sales(
            self.spend_df, MARGINAL_RETURNS, seed=1, demand=demand, demand_coef=250.0
        )
        np.testing.assert_allclose((with_demand - without).to_numpy(), 250.0 * demand)

    def test_wrong_length_demand_raises(self):
        with pytest.raises(ValueError, match="to match"):
            simulate_sales(
                self.spend_df, MARGINAL_RETURNS, demand=np.zeros(51), demand_coef=1.0
            )
