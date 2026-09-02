"""Tests for _dgp.py — simulate_spend and simulate_sales."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import (
    DEMAND_PROCESSES,
    apply_adstock,
    calibrate_baseline,
    simulate_demand,
    simulate_demand_proxy,
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

    def test_trend_differs_by_seed(self):
        """Stochastic, like "white_noise"/"ar1" -- not a fixed shape."""
        a = simulate_demand(60, process="trend", seed=0)
        b = simulate_demand(60, process="trend", seed=1)
        assert not np.allclose(a, b)

    def test_trend_is_more_persistent_than_ar1(self):
        """The property the adstock threat test leans on: integration
        concentrates a random walk's energy near zero frequency more than
        even a highly persistent stationary AR(1) does, because AR(1) must
        mean-revert (ar_coef < 1 is enforced) and a random walk does not."""

        def lag1(series):
            return np.corrcoef(series[:-1], series[1:])[0, 1]

        assert lag1(simulate_demand(500, process="trend", seed=0)) > lag1(
            simulate_demand(500, process="ar1", ar_coef=0.8, seed=0)
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_trend_drift_sign_sets_direction_on_average(self, seed):
        """Not guaranteed for any single draw (it's a random walk -- a
        single path can wander against its own drift), but the endpoint
        should exceed the start more often than not across draws, and
        flipping the drift's sign should flip that tendency."""
        up = simulate_demand(200, process="trend", trend_drift=0.15, seed=seed)
        down = simulate_demand(200, process="trend", trend_drift=-0.15, seed=seed)
        assert (up[-1] - up[0]) > (down[-1] - down[0])

    def test_trend_drift_zero_is_a_pure_random_walk(self):
        """trend_drift=0 must not raise or special-case -- it's just cumsum
        of the innovations, no drift term."""
        series = simulate_demand(104, process="trend", trend_drift=0.0, seed=0)
        assert series.shape == (104,)
        assert series.std() == pytest.approx(1.0, abs=1e-12)

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


class TestApplyAdstock:
    """Normalised geometric adstock: a[t] = (1-d)*x[t] + d*a[t-1]."""

    def test_zero_decay_returns_input_unchanged(self):
        x = np.array([1.0, 5.0, 3.0, 9.0])
        assert apply_adstock(x, 0.0) is x

    @pytest.mark.parametrize("decay", [0.0, 0.3, 0.6, 0.9])
    def test_constant_spend_passes_through_unchanged(self, decay):
        # This is what "normalised" buys: a channel's steady-state level does
        # not move with decay, so the supplied marginal return keeps meaning
        # the same thing and two decays are comparable.
        const = np.full(40, 7.0)
        np.testing.assert_allclose(apply_adstock(const, decay), 7.0)

    @pytest.mark.parametrize("decay", [0.3, 0.6, 0.9])
    def test_impulse_decays_at_exactly_the_decay_rate(self, decay):
        impulse = np.zeros(30)
        impulse[5] = 100.0
        out = apply_adstock(impulse, decay)
        np.testing.assert_allclose(out[5], 100.0 * (1.0 - decay))
        for t in range(6, 15):
            np.testing.assert_allclose(out[t] / out[t - 1], decay)

    @pytest.mark.parametrize("decay", [0.3, 0.6])
    def test_impulse_mass_is_conserved(self, decay):
        impulse = np.zeros(200)
        impulse[5] = 100.0
        np.testing.assert_allclose(apply_adstock(impulse, decay).sum(), 100.0)

    def test_seeded_from_first_observation_not_zero(self):
        x = np.array([10.0, 10.0, 10.0])
        # A zero seed would give 10*(1-d) here and read as a warm-up ramp that
        # is an artefact of where the array starts.
        assert apply_adstock(x, 0.5)[0] == 10.0

    def test_low_passes_weekly_variation(self):
        x = simulate_spend(n_obs=104, seed=0)["tv"].to_numpy()
        spreads = [np.diff(apply_adstock(x, d)).std() for d in (0.0, 0.3, 0.6, 0.9)]
        assert spreads == sorted(spreads, reverse=True)


class TestSaturationAndAdstockAreDefaultInert:
    spend_df = simulate_spend(n_obs=52, seed=0)

    def _base(self):
        return simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"saturation": None},
            {"saturation": 1.0},
            {"saturation": {"tv": 1.0}},
            {"adstock": None},
            {"adstock": 0.0},
            {"adstock": {"tv": 0.0}},
            {"saturation": 1.0, "adstock": 0.0},
            {"saturation": {}, "adstock": {}},
        ],
    )
    def test_inert_settings_reproduce_the_linear_output_exactly(self, kwargs):
        # Exact equality, not allclose: b=1 and decay=0 take the original
        # code path deliberately so published seeded numbers cannot move.
        pd.testing.assert_series_equal(
            simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=3, **kwargs),
            self._base(),
        )

    def test_channels_absent_from_the_dict_stay_linear(self):
        only_tv = simulate_sales(
            self.spend_df, MARGINAL_RETURNS, seed=3, saturation={"tv": 0.5}
        )
        base = self._base()
        # meta and search unchanged means the whole difference is TV's.
        tv = self.spend_df["tv"].to_numpy()
        k = MARGINAL_RETURNS["tv"] / (0.5 * tv.mean() ** (0.5 - 1.0))
        expected = base.to_numpy() - MARGINAL_RETURNS["tv"] * tv + k * tv**0.5
        np.testing.assert_allclose(only_tv.to_numpy(), expected)


class TestSaturation:
    spend_df = simulate_spend(n_obs=52, seed=0)

    @pytest.mark.parametrize("b", [0.3, 0.6, 0.9])
    def test_marginal_return_at_reference_equals_the_supplied_value(self, b):
        # The calibration promise: the supplied marginal return is the return
        # on the next pound at the reference level.
        ref = {ch: float(self.spend_df[ch].mean()) for ch in CHANNELS}
        eps = 1.0
        for ch in CHANNELS:
            lo = self.spend_df.copy()
            hi = self.spend_df.copy()
            for other in CHANNELS:
                lo[other] = 0.0 if other != ch else ref[ch] - eps / 2
                hi[other] = 0.0 if other != ch else ref[ch] + eps / 2
            kwargs = dict(
                seed=0,
                saturation=b,
                reference_spend=ref,
                revenue_noise_std=0.0,
                base_sales=0.0,
            )
            slope = (
                simulate_sales(hi, MARGINAL_RETURNS, **kwargs).iloc[0]
                - simulate_sales(lo, MARGINAL_RETURNS, **kwargs).iloc[0]
            )
            np.testing.assert_allclose(slope, MARGINAL_RETURNS[ch], rtol=1e-6)

    def test_doubling_spend_scales_contribution_by_two_to_the_b(self):
        ref = {ch: float(self.spend_df[ch].mean()) for ch in CHANNELS}
        kwargs = dict(
            seed=0,
            saturation=0.6,
            reference_spend=ref,
            revenue_noise_std=0.0,
            base_sales=0.0,
        )
        one = simulate_sales(self.spend_df, MARGINAL_RETURNS, **kwargs).sum()
        two = simulate_sales(self.spend_df * 2, MARGINAL_RETURNS, **kwargs).sum()
        np.testing.assert_allclose(two / one, 2.0**0.6)

    def test_explicit_reference_spend_overrides_the_column_mean(self):
        ref = {ch: float(self.spend_df[ch].mean()) for ch in CHANNELS}
        default = simulate_sales(
            self.spend_df, MARGINAL_RETURNS, seed=0, saturation=0.6
        )
        explicit = simulate_sales(
            self.spend_df, MARGINAL_RETURNS, seed=0, saturation=0.6, reference_spend=ref
        )
        pd.testing.assert_series_equal(default, explicit)
        moved = simulate_sales(
            self.spend_df,
            MARGINAL_RETURNS,
            seed=0,
            saturation=0.6,
            reference_spend={ch: v * 2 for ch, v in ref.items()},
        )
        assert not np.allclose(moved.to_numpy(), explicit.to_numpy())

    @pytest.mark.parametrize("bad", [0.0, -0.2, 1.5])
    def test_out_of_range_exponent_raises(self, bad):
        with pytest.raises(ValueError, match="must be in"):
            simulate_sales(self.spend_df, MARGINAL_RETURNS, saturation=bad)

    def test_non_positive_reference_spend_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            simulate_sales(
                self.spend_df,
                MARGINAL_RETURNS,
                saturation=0.6,
                reference_spend={ch: 0.0 for ch in CHANNELS},
            )

    def test_negative_spend_raises(self):
        negative = self.spend_df.copy()
        negative.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="negative spend"):
            simulate_sales(negative, MARGINAL_RETURNS, saturation=0.6)


class TestAdstockInSimulateSales:
    spend_df = simulate_spend(n_obs=52, seed=0)

    @pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
    def test_out_of_range_decay_raises(self, bad):
        with pytest.raises(ValueError, match="must be in"):
            simulate_sales(self.spend_df, MARGINAL_RETURNS, adstock=bad)

    def test_linear_with_adstock_is_marginal_return_times_adstocked_spend(self):
        out = simulate_sales(
            self.spend_df,
            MARGINAL_RETURNS,
            seed=0,
            adstock=0.5,
            revenue_noise_std=0.0,
            base_sales=0.0,
        )
        expected = sum(
            MARGINAL_RETURNS[ch] * apply_adstock(self.spend_df[ch].to_numpy(), 0.5)
            for ch in CHANNELS
        )
        np.testing.assert_allclose(out.to_numpy(), expected)

    def test_adstock_is_applied_before_saturation(self):
        # The Robyn / Meridian / PyMC order. Saturating first would give a
        # different series, which is what this pins down.
        ref = {ch: float(self.spend_df[ch].mean()) for ch in CHANNELS}
        out = simulate_sales(
            self.spend_df,
            MARGINAL_RETURNS,
            seed=0,
            adstock=0.5,
            saturation=0.6,
            reference_spend=ref,
            revenue_noise_std=0.0,
            base_sales=0.0,
        )
        expected = np.zeros(len(self.spend_df))
        for ch in CHANNELS:
            a = apply_adstock(self.spend_df[ch].to_numpy(), 0.5)
            k = MARGINAL_RETURNS[ch] / (0.6 * ref[ch] ** (0.6 - 1.0))
            expected = expected + k * a**0.6
        np.testing.assert_allclose(out.to_numpy(), expected)

        saturate_first = np.zeros(len(self.spend_df))
        for ch in CHANNELS:
            k = MARGINAL_RETURNS[ch] / (0.6 * ref[ch] ** (0.6 - 1.0))
            saturate_first = saturate_first + apply_adstock(
                k * self.spend_df[ch].to_numpy() ** 0.6, 0.5
            )
        assert not np.allclose(out.to_numpy(), saturate_first)

    def test_demand_term_is_unaffected_by_either_transform(self):
        demand = simulate_demand(52, seed=0)
        ref = {ch: float(self.spend_df[ch].mean()) for ch in CHANNELS}
        kwargs = dict(seed=1, saturation=0.6, adstock=0.4, reference_spend=ref)
        without = simulate_sales(self.spend_df, MARGINAL_RETURNS, **kwargs)
        with_demand = simulate_sales(
            self.spend_df, MARGINAL_RETURNS, demand=demand, demand_coef=250.0, **kwargs
        )
        np.testing.assert_allclose((with_demand - without).to_numpy(), 250.0 * demand)


class TestSimulateDemandProxy:
    def test_standardised(self):
        demand = simulate_demand(300, process="ar1", seed=0)
        proxy = simulate_demand_proxy(demand, quality=0.7, seed=1)
        assert proxy.mean() == pytest.approx(0.0, abs=1e-9)
        assert proxy.std() == pytest.approx(1.0, abs=1e-9)

    def test_quality_one_is_the_truth_standardised(self):
        demand = simulate_demand(300, process="ar1", seed=0)
        proxy = simulate_demand_proxy(demand, quality=1.0)
        np.testing.assert_allclose(proxy, demand, atol=1e-9)

    def test_correlation_lands_near_target_quality(self):
        demand = simulate_demand(2000, process="white_noise", seed=0)
        for quality in (0.5, 0.7, 0.9):
            proxy = simulate_demand_proxy(demand, quality=quality, seed=1)
            corr = np.corrcoef(proxy, demand)[0, 1]
            assert corr == pytest.approx(quality, abs=0.05)

    def test_lower_quality_is_less_correlated(self):
        demand = simulate_demand(500, process="ar1", seed=0)
        corr_hi = np.corrcoef(
            simulate_demand_proxy(demand, quality=0.9, seed=1), demand
        )[0, 1]
        corr_lo = np.corrcoef(
            simulate_demand_proxy(demand, quality=0.5, seed=1), demand
        )[0, 1]
        assert corr_hi > corr_lo

    def test_seed_changes_the_draw(self):
        demand = simulate_demand(300, process="ar1", seed=0)
        a = simulate_demand_proxy(demand, quality=0.7, seed=1)
        b = simulate_demand_proxy(demand, quality=0.7, seed=2)
        assert not np.allclose(a, b)

    def test_reproducible_at_the_same_seed(self):
        demand = simulate_demand(300, process="ar1", seed=0)
        a = simulate_demand_proxy(demand, quality=0.7, seed=1)
        b = simulate_demand_proxy(demand, quality=0.7, seed=1)
        np.testing.assert_allclose(a, b)

    def test_accepts_a_series(self):
        demand = pd.Series(simulate_demand(200, process="ar1", seed=0))
        proxy = simulate_demand_proxy(demand, quality=0.8, seed=1)
        assert proxy.shape == (200,)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, 2.0])
    def test_invalid_quality_raises(self, bad):
        demand = simulate_demand(100, process="ar1", seed=0)
        with pytest.raises(ValueError, match="quality must be in"):
            simulate_demand_proxy(demand, quality=bad)

    def test_proxy_noise_stream_does_not_collide_with_other_draws(self):
        """Different seed argument to simulate_demand_proxy must not
        reproduce the same noise a different seeded draw elsewhere in the
        module would produce at that same integer -- see _PROXY_STREAM."""
        demand = simulate_demand(300, process="white_noise", seed=7)
        proxy = simulate_demand_proxy(demand, quality=0.7, seed=7)
        # seed=7 reused deliberately: proxy's stream is offset from
        # simulate_demand's own, so this is not the same draw as demand.
        assert not np.allclose(proxy, demand)
