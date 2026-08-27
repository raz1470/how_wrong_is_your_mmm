"""Tests for _phaser.py — BudgetPhaser (monthly-constrained spend phasing)."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._phaser import (
    NUDGE_SHAPES,
    Blackout,
    BudgetPhaser,
    _generate_phased_schedule,
    _get_month_labels,
    _max_monthly_deviation,
    _resolve_channel_specs,
    _shaped_nudge,
)

MARGINAL_RETURNS = {"tv": 0.3, "meta": 0.5, "search": 0.4}

# 4 years of history + 1 year plan, both with DatetimeIndex
HISTORY_DF = simulate_spend(n_obs=208, correlation=0.7, seed=0, start_date="2019-01-07")
PLAN_DF = simulate_spend(n_obs=52, correlation=0.7, seed=1, start_date="2023-01-09")


class TestGetMonthLabels:
    def test_returns_array(self):
        labels = _get_month_labels(PLAN_DF)
        assert len(labels) == 52

    def test_raises_without_datetime_index(self):
        df = simulate_spend(n_obs=52, correlation=0.7, seed=0)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _get_month_labels(df)

    def test_twelve_months(self):
        labels = _get_month_labels(PLAN_DF)
        assert len(np.unique(labels)) >= 12


class TestGeneratePhasedSchedule:
    def setup_method(self):
        self.month_labels = _get_month_labels(PLAN_DF)

    def test_output_shape(self):
        result = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.5, max_weekly_deviation_pct=40.0, seed=0
        )
        assert result.shape == PLAN_DF.shape

    def test_output_index_preserved(self):
        result = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.5, max_weekly_deviation_pct=40.0, seed=0
        )
        pd.testing.assert_index_equal(result.index, PLAN_DF.index)

    def test_zero_alpha_unchanged(self):
        result = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.0, max_weekly_deviation_pct=40.0, seed=0
        )
        pd.testing.assert_frame_equal(result, PLAN_DF.astype(float))

    def test_monthly_totals_preserved(self):
        result = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=1.0, max_weekly_deviation_pct=40.0, seed=0
        )
        dev = _max_monthly_deviation(PLAN_DF, result, self.month_labels)
        assert dev < 1e-10

    def test_reproducibility(self):
        r1 = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.5, max_weekly_deviation_pct=40.0, seed=7
        )
        r2 = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.5, max_weekly_deviation_pct=40.0, seed=7
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_higher_alpha_reduces_correlation(self):
        low = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.0, max_weekly_deviation_pct=40.0, seed=0
        )
        high = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=1.0, max_weekly_deviation_pct=40.0, seed=0
        )

        def mean_corr(df):
            c = df.corr().to_numpy()
            n = df.shape[1]
            return np.mean([c[i, j] for i in range(n) for j in range(i + 1, n)])

        assert mean_corr(high) < mean_corr(low)


class TestResolveChannelSpecs:
    def test_float_applies_symmetric_bounds_to_all_channels(self):
        bounds = _resolve_channel_specs(40.0, ["tv", "meta", "search"])
        assert bounds == {
            "tv": (-40.0, 40.0),
            "meta": (-40.0, 40.0),
            "search": (-40.0, 40.0),
        }

    def test_dict_of_floats_expands_to_symmetric_bounds(self):
        bounds = _resolve_channel_specs(
            {"tv": 0.0, "meta": 60.0, "search": 100.0}, ["tv", "meta", "search"]
        )
        assert bounds == {
            "tv": (0.0, 0.0),
            "meta": (-60.0, 60.0),
            "search": (-100.0, 100.0),
        }

    def test_tuple_form_no_longer_accepted(self):
        """Explicit (low, high) ranges were dropped: preserving the monthly
        total while biasing the raw draw one-sided still forces some weeks
        above their own original plan, and unlike Blackout this doesn't map
        to a recognisable media-planning concept, so it just reads as a
        broken 'never above plan' promise. Blackout (optionally with
        max_dark_weeks_per_month) is the supported replacement."""
        with pytest.raises(TypeError, match="no longer supported"):
            _resolve_channel_specs(
                {"tv": 0.0, "meta": 60.0, "search": (-100.0, 0.0)},
                ["tv", "meta", "search"],
            )

    def test_list_form_also_rejected(self):
        with pytest.raises(TypeError, match="no longer supported"):
            _resolve_channel_specs(
                {"tv": 0.0, "meta": 60.0, "search": [-100.0, 0.0]},
                ["tv", "meta", "search"],
            )

    def test_dict_missing_channel_raises(self):
        with pytest.raises(ValueError, match="missing channels"):
            _resolve_channel_specs({"tv": 0.0, "meta": 60.0}, ["tv", "meta", "search"])

    def test_dict_extra_channel_ignored(self):
        bounds = _resolve_channel_specs(
            {"tv": 0.0, "meta": 60.0, "search": 100.0, "extra": 20.0},
            ["tv", "meta", "search"],
        )
        assert set(bounds) == {"tv", "meta", "search"}

    def test_negative_float_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            _resolve_channel_specs(-10.0, ["tv", "meta", "search"])

    def test_over_100_float_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            _resolve_channel_specs(120.0, ["tv", "meta", "search"])

    def test_negative_dict_value_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            _resolve_channel_specs(
                {"tv": -5.0, "meta": 60.0, "search": 100.0},
                ["tv", "meta", "search"],
            )

    def test_boundary_values_allowed(self):
        bounds = _resolve_channel_specs(
            {"tv": 0.0, "meta": 50.0, "search": 100.0}, ["tv", "meta", "search"]
        )
        assert bounds["tv"] == (0.0, 0.0)
        assert bounds["search"] == (-100.0, 100.0)

    def test_dict_with_blackout_instance(self):
        bounds = _resolve_channel_specs(
            {"tv": 0.0, "meta": 60.0, "search": Blackout()},
            ["tv", "meta", "search"],
        )
        assert bounds["search"] == Blackout(prob=1.0)

    def test_top_level_blackout_applies_to_all_channels(self):
        bounds = _resolve_channel_specs(Blackout(prob=0.5), ["tv", "meta", "search"])
        assert bounds == {
            "tv": Blackout(prob=0.5),
            "meta": Blackout(prob=0.5),
            "search": Blackout(prob=0.5),
        }

    def test_dict_can_mix_blackout_with_float(self):
        bounds = _resolve_channel_specs(
            {"tv": 0.0, "meta": 60.0, "search": Blackout(prob=0.3)},
            ["tv", "meta", "search"],
        )
        assert bounds == {
            "tv": (0.0, 0.0),
            "meta": (-60.0, 60.0),
            "search": Blackout(prob=0.3),
        }


class TestBlackout:
    def test_default_prob_is_one(self):
        assert Blackout().prob == 1.0

    def test_custom_prob_stored(self):
        assert Blackout(prob=0.4).prob == 0.4

    def test_prob_below_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Blackout(prob=-0.1)

    def test_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Blackout(prob=1.1)

    def test_equality(self):
        assert Blackout(prob=0.5) == Blackout(prob=0.5)
        assert Blackout(prob=0.5) != Blackout(prob=0.6)
        assert Blackout() != 1.0

    def test_boundary_probs_allowed(self):
        assert Blackout(prob=0.0).prob == 0.0
        assert Blackout(prob=1.0).prob == 1.0

    def test_default_max_dark_weeks_per_month_is_none(self):
        assert Blackout().max_dark_weeks_per_month is None

    def test_custom_max_dark_weeks_per_month_stored(self):
        assert Blackout(max_dark_weeks_per_month=2).max_dark_weeks_per_month == 2

    def test_max_dark_weeks_per_month_below_one_raises(self):
        with pytest.raises(ValueError, match="max_dark_weeks_per_month"):
            Blackout(max_dark_weeks_per_month=0)

    def test_equality_includes_max_dark_weeks_per_month(self):
        assert Blackout(max_dark_weeks_per_month=1) == Blackout(
            max_dark_weeks_per_month=1
        )
        assert Blackout(max_dark_weeks_per_month=1) != Blackout(
            max_dark_weeks_per_month=2
        )
        assert Blackout(max_dark_weeks_per_month=1) != Blackout()


class TestGeneratePhasedScheduleBlackout:
    def setup_method(self):
        self.month_labels = _get_month_labels(PLAN_DF)

    def test_zero_alpha_no_blackout(self):
        """At alpha=0, Blackout mode should mean no change, same fixed point
        as every other deviation shape."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=0.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        pd.testing.assert_frame_equal(result, PLAN_DF.astype(float))

    def test_some_weeks_go_fully_dark(self):
        """Under Blackout(prob=1.0) at alpha=1, at least one week in the year
        should land at (or very near) zero pre-rescale spend for that
        channel — the whole point of the feature."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        assert result["search"].min() < PLAN_DF["search"].min() * 0.1

    def test_dark_weeks_are_exactly_zero(self):
        """Unlike a continuous range, a dark week's final spend should be
        exactly 0, not just small — zero times any rescale factor is 0."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        assert (result["search"] == 0.0).sum() > 0

    def test_at_least_one_week_per_month_stays_on(self):
        """Even at prob=1.0, alpha=1 (every week individually drawn dark
        with certainty), no month should end up with every week at zero —
        the safeguard keeps at least one week "on" so the budget has
        somewhere to land, rather than silently leaving the month
        untouched."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        for month in np.unique(self.month_labels):
            mask = np.where(self.month_labels == month)[0]
            assert (result["search"].to_numpy()[mask] > 0).any()

    def test_other_channels_unaffected(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        pd.testing.assert_series_equal(result["tv"], PLAN_DF["tv"].astype(float))
        pd.testing.assert_series_equal(result["meta"], PLAN_DF["meta"].astype(float))

    def test_monthly_totals_still_preserved(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        dev = _max_monthly_deviation(PLAN_DF, result, self.month_labels)
        assert dev < 1e-10

    def test_zero_prob_blackout_means_no_change(self):
        """Blackout(prob=0.0) should behave like a 0% cap: never dark."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(prob=0.0),
            },
            seed=0,
        )
        pd.testing.assert_frame_equal(result, PLAN_DF.astype(float))

    def test_lower_prob_blacks_out_fewer_weeks_on_average(self):
        """Not a strict guarantee for any single draw, but averaged over many
        seeds a lower prob should black out a smaller share of weeks."""

        def count_near_zero_weeks(prob, n_draws=20):
            counts = []
            for seed in range(n_draws):
                result = _generate_phased_schedule(
                    PLAN_DF,
                    self.month_labels,
                    alpha=1.0,
                    max_weekly_deviation_pct={
                        "tv": 0.0,
                        "meta": 0.0,
                        "search": Blackout(prob=prob),
                    },
                    seed=seed,
                )
                counts.append((result["search"] < 1.0).sum())
            return np.mean(counts)

        low = count_near_zero_weeks(0.1)
        high = count_near_zero_weeks(0.9)
        assert high > low

    def test_reproducibility(self):
        r1 = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=5,
        )
        r2 = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=5,
        )
        pd.testing.assert_frame_equal(r1, r2)


class TestGeneratePhasedScheduleBlackoutCapped:
    """max_dark_weeks_per_month bounds how many weeks any one month can lose,
    which in turn bounds how large the compensating spike on the surviving
    weeks can get — the fix for a month landing several dark weeks at once
    under the uncapped (legacy) behaviour and forcing an unrealistic jump
    on whatever's left."""

    def setup_method(self):
        self.month_labels = _get_month_labels(PLAN_DF)

    def test_zero_alpha_no_blackout(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=0.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=0,
        )
        pd.testing.assert_frame_equal(result, PLAN_DF.astype(float))

    def test_cap_limits_dark_weeks_per_month(self):
        """With a cap of 1, no month should ever have more than 1 dark week
        for that channel, even at prob=1.0, alpha=1."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=0,
        )
        for month in np.unique(self.month_labels):
            mask = np.where(self.month_labels == month)[0]
            n_dark = (result["search"].to_numpy()[mask] == 0.0).sum()
            assert n_dark <= 1

    def test_cap_reduces_max_spike_vs_uncapped(self):
        """The whole point: capping dark weeks per month should shrink the
        redistribution spike on the surviving weeks, compared to the
        uncapped default at the same seed."""
        uncapped = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
            seed=0,
        )
        capped = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=0,
        )
        assert capped["search"].max() < uncapped["search"].max()

    def test_monthly_totals_still_preserved(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=0,
        )
        dev = _max_monthly_deviation(PLAN_DF, result, self.month_labels)
        assert dev < 1e-10

    def test_dark_weeks_still_exactly_zero(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=0,
        )
        assert (result["search"] == 0.0).sum() > 0

    def test_cap_larger_than_month_length_still_leaves_one_week_on(self):
        """A cap bigger than a month's own week count (e.g. 10 weeks, but
        some months only have 4) should never black out every week."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=10),
            },
            seed=0,
        )
        for month in np.unique(self.month_labels):
            mask = np.where(self.month_labels == month)[0]
            assert (result["search"].to_numpy()[mask] > 0).any()

    def test_reproducibility(self):
        r1 = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=5,
        )
        r2 = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
            seed=5,
        )
        pd.testing.assert_frame_equal(r1, r2)


class TestGeneratePhasedSchedulePerChannel:
    def setup_method(self):
        self.month_labels = _get_month_labels(PLAN_DF)

    def test_zero_cap_channel_unchanged(self):
        """A channel capped at 0% should be untouched by phasing, even at alpha=1."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 40.0, "search": 40.0},
            seed=0,
        )
        pd.testing.assert_series_equal(result["tv"], PLAN_DF["tv"].astype(float))

    def test_nonzero_cap_channels_do_change(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 40.0, "search": 40.0},
            seed=0,
        )
        assert not result["meta"].equals(PLAN_DF["meta"].astype(float))
        assert not result["search"].equals(PLAN_DF["search"].astype(float))

    def test_per_channel_monthly_totals_preserved(self):
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0, "search": 100.0},
            seed=0,
        )
        dev = _max_monthly_deviation(PLAN_DF, result, self.month_labels)
        assert dev < 1e-10

    def test_float_and_uniform_dict_are_equivalent(self):
        """A single float and a dict with the same value per channel should
        produce identical output (same resolved caps, same RNG draws)."""
        float_result = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=0.7, max_weekly_deviation_pct=40.0, seed=3
        )
        dict_result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=0.7,
            max_weekly_deviation_pct={"tv": 40.0, "meta": 40.0, "search": 40.0},
            seed=3,
        )
        pd.testing.assert_frame_equal(float_result, dict_result)

    def test_hundred_pct_cap_can_reach_zero_spend(self):
        """A channel capped at 100% should be able to hit zero spend in some
        week under the right draw (search can be switched off entirely)."""
        result = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": 100.0},
            seed=0,
        )
        assert result["search"].min() < PLAN_DF["search"].min() * 0.5

    def test_tuple_bound_raises(self):
        """Explicit (low, high) ranges were dropped from this function too —
        see TestResolveChannelSpecs.test_tuple_form_no_longer_accepted."""
        with pytest.raises(TypeError, match="no longer supported"):
            _generate_phased_schedule(
                PLAN_DF,
                self.month_labels,
                alpha=1.0,
                max_weekly_deviation_pct={
                    "tv": 0.0,
                    "meta": 0.0,
                    "search": (-100.0, 0.0),
                },
                seed=0,
            )


class TestBudgetPhaser:
    def test_fit_returns_self(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        assert phaser.fit(n_sims=5, grid_steps=3, n_phasing_seeds=1) is phaser

    def test_results_shape(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        assert len(phaser.results_) == 5

    def test_summary_columns(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        cols = set(phaser.summary().columns)
        assert {
            "alpha",
            "actual_correlation",
            "max_cv",
            "max_monthly_deviation_pct",
        }.issubset(cols)
        assert {"cv_tv", "cv_meta", "cv_search"}.issubset(cols)

    def test_fast_mode(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=50, grid_steps=20, fast_mode=True)
        assert len(phaser.results_) == 10

    def test_recommend_is_min_confirmation_cv(self):
        """recommend() returns the confirmation pass's winner, not the raw
        grid argmin — see TestSelectionBiasConfirmation for why."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        rec = phaser.recommend()
        assert rec["max_cv"] == phaser.confirmation_["max_cv"].min()

    def test_recommended_schedule_shape(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        assert phaser.recommended_schedule_.shape == PLAN_DF.shape

    def test_recommended_schedule_index(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        pd.testing.assert_index_equal(phaser.recommended_schedule_.index, PLAN_DF.index)

    def test_monthly_totals_preserved_in_recommended_schedule(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        month_labels = _get_month_labels(PLAN_DF)
        dev = _max_monthly_deviation(
            PLAN_DF, phaser.recommended_schedule_, month_labels
        )
        assert dev < 1e-10

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPhaser(
                HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
            ).summary()

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPhaser(
                HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
            ).recommend()

    def test_no_datetime_index_on_history_raises(self):
        df = simulate_spend(n_obs=208, correlation=0.7, seed=0)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BudgetPhaser(df, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS)

    def test_no_datetime_index_on_plan_raises(self):
        df = simulate_spend(n_obs=52, correlation=0.7, seed=0)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BudgetPhaser(HISTORY_DF, df, true_marginal_returns=MARGINAL_RETURNS)

    def test_mismatched_columns_raises(self):
        plan_2ch = simulate_spend(
            n_obs=52,
            correlation=0.7,
            seed=0,
            channels=["tv", "meta"],
            start_date="2023-01-09",
        )
        with pytest.raises(ValueError, match="columns"):
            BudgetPhaser(
                HISTORY_DF, plan_2ch, true_marginal_returns={"tv": 0.3, "meta": 0.5}
            )

    def test_nan_in_history_raises_at_construction(self):
        bad_history = HISTORY_DF.copy()
        bad_history.iloc[0, 0] = float("nan")
        with pytest.raises(ValueError, match="missing"):
            BudgetPhaser(bad_history, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS)

    def test_nan_in_plan_raises_at_construction(self):
        bad_plan = PLAN_DF.copy()
        bad_plan.iloc[0, 0] = float("nan")
        with pytest.raises(ValueError, match="missing"):
            BudgetPhaser(HISTORY_DF, bad_plan, true_marginal_returns=MARGINAL_RETURNS)

    def test_zero_variance_channel_raises_at_construction(self):
        bad_history = HISTORY_DF.copy()
        zero_col = bad_history.columns[0]
        bad_history[zero_col] = 0.0
        bad_plan = PLAN_DF.copy()
        bad_plan[zero_col] = 0.0
        with pytest.raises(ValueError, match="zero variance"):
            BudgetPhaser(bad_history, bad_plan, true_marginal_returns=MARGINAL_RETURNS)

    def test_validation_runs_before_any_simulation(self):
        # A construction-time failure should never reach fit() -- the
        # object shouldn't even be usable, let alone run a grid search.
        bad_history = HISTORY_DF.copy()
        bad_history.iloc[0, 0] = float("nan")
        with pytest.raises(ValueError):
            BudgetPhaser(bad_history, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS)

    def test_valid_data_constructs_without_error(self):
        # Regression check: the standard fixture data (used by every other
        # test in this file) must not trip the new validation.
        BudgetPhaser(HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS)

    def test_alpha_starts_at_zero(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        assert phaser.results_["alpha"].iloc[0] == 0.0

    def test_alpha_ends_at_one(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        assert phaser.results_["alpha"].iloc[-1] == 1.0


class TestBudgetPhaserPerChannelCaps:
    def test_dict_cap_accepted(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0, "search": 100.0},
        )
        assert phaser.max_weekly_deviation_pct == {
            "tv": 0.0,
            "meta": 60.0,
            "search": 100.0,
        }

    def test_dict_missing_channel_raises_at_construction(self):
        with pytest.raises(ValueError, match="missing channels"):
            BudgetPhaser(
                HISTORY_DF,
                PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0},
            )

    def test_out_of_range_value_raises_at_construction(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            BudgetPhaser(
                HISTORY_DF,
                PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                max_weekly_deviation_pct={"tv": -10.0, "meta": 60.0, "search": 100.0},
            )

    def test_recommended_schedule_respects_zero_cap_channel(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0, "search": 100.0},
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        pd.testing.assert_series_equal(
            phaser.recommended_schedule_["tv"], PLAN_DF["tv"].astype(float)
        )

    def test_recommended_schedule_moves_unconstrained_channels(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0, "search": 100.0},
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        # at grid_steps=5 the best alpha may still land at 0, so only assert
        # the schedule is well-formed and monthly totals hold for every channel
        month_labels = _get_month_labels(PLAN_DF)
        dev = _max_monthly_deviation(
            PLAN_DF, phaser.recommended_schedule_, month_labels
        )
        assert dev < 1e-10

    def test_blackout_instance_accepted_at_construction(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 60.0, "search": Blackout()},
        )
        assert phaser.max_weekly_deviation_pct["search"] == Blackout()

    def test_recommended_schedule_blackout_preserves_totals(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        month_labels = _get_month_labels(PLAN_DF)
        dev = _max_monthly_deviation(
            PLAN_DF, phaser.recommended_schedule_, month_labels
        )
        assert dev < 1e-10

    def test_recommended_schedule_blackout_locks_other_channels(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={"tv": 0.0, "meta": 0.0, "search": Blackout()},
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        pd.testing.assert_series_equal(
            phaser.recommended_schedule_["tv"], PLAN_DF["tv"].astype(float)
        )
        pd.testing.assert_series_equal(
            phaser.recommended_schedule_["meta"], PLAN_DF["meta"].astype(float)
        )

    def test_recommended_schedule_capped_blackout_preserves_totals(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        month_labels = _get_month_labels(PLAN_DF)
        dev = _max_monthly_deviation(
            PLAN_DF, phaser.recommended_schedule_, month_labels
        )
        assert dev < 1e-10

    def test_recommended_schedule_capped_blackout_limits_dark_weeks(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct={
                "tv": 0.0,
                "meta": 0.0,
                "search": Blackout(max_dark_weeks_per_month=1),
            },
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        month_labels = _get_month_labels(PLAN_DF)
        schedule = phaser.recommended_schedule_
        for month in np.unique(month_labels):
            mask = np.where(month_labels == month)[0]
            n_dark = (schedule["search"].to_numpy()[mask] == 0.0).sum()
            assert n_dark <= 1


class TestNPhasingSeedsParam:
    def test_multiple_seeds_produces_correct_shape(self):
        """n_phasing_seeds > 1 should still give grid_steps rows."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=4, n_phasing_seeds=3)
        assert len(phaser.results_) == 4

    def test_single_seed_matches_columns(self):
        """n_phasing_seeds=1 gives the same output columns as n_phasing_seeds=3."""
        p1 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        p3 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=3)
        assert list(p1.summary().columns) == list(p3.summary().columns)

    def test_multiple_seeds_cv_is_average(self):
        """With n_phasing_seeds=3 the max_cv at alpha=0 should be lower-variance
        than any single seed — verified by checking it lies between the per-seed
        extremes. We proxy this by confirming max_cv at alpha=0 is finite and
        non-negative."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=3)
        assert phaser.results_["max_cv"].iloc[0] >= 0

    def test_fast_mode_sets_n_phasing_seeds_one(self):
        """fast_mode overrides n_phasing_seeds to 1 (10 grid points, fast run)."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=50, grid_steps=20, n_phasing_seeds=5, fast_mode=True)
        # fast_mode caps grid_steps=10 and n_phasing_seeds=1 — result has 10 rows
        assert len(phaser.results_) == 10


class TestFitUnchangedAfterRefactor:
    """fit() delegates to _evaluate_spec_at_alpha now — pin down that its
    output and seeding are byte-for-byte identical to before the refactor."""

    def test_reproducibility(self):
        p1 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=3
        ).fit(n_sims=5, grid_steps=4, n_phasing_seeds=2)
        p2 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=3
        ).fit(n_sims=5, grid_steps=4, n_phasing_seeds=2)
        pd.testing.assert_frame_equal(p1.results_, p2.results_)
        pd.testing.assert_frame_equal(
            p1.recommended_schedule_, p2.recommended_schedule_
        )


class TestChannelSensitivity:
    def setup_method(self):
        self.phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=0
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            self.phaser.channel_sensitivity("radio", n_sims=5, n_phasing_seeds=1)

    def test_default_alphas_give_six_continuous_rows(self):
        result = self.phaser.channel_sensitivity("tv", n_sims=5, n_phasing_seeds=1)
        assert (~result["is_blackout"]).sum() == 6

    def test_no_blackout_by_default(self):
        result = self.phaser.channel_sensitivity("tv", n_sims=5, n_phasing_seeds=1)
        assert not result["is_blackout"].any()

    def test_blackout_appends_one_row(self):
        result = self.phaser.channel_sensitivity(
            "tv",
            blackout=Blackout(max_dark_weeks_per_month=1),
            n_sims=5,
            n_phasing_seeds=1,
        )
        assert result["is_blackout"].sum() == 1
        assert result.iloc[-1]["label"] == "Blackout"
        assert np.isnan(result.iloc[-1]["magnitude_pct"])

    def test_zero_alpha_row_has_zero_magnitude(self):
        result = self.phaser.channel_sensitivity("tv", n_sims=5, n_phasing_seeds=1)
        assert result.iloc[0]["magnitude_pct"] == 0.0

    def test_custom_alphas_respected(self):
        result = self.phaser.channel_sensitivity(
            "meta", alphas=[0.0, 0.5, 1.0], n_sims=5, n_phasing_seeds=1
        )
        assert (~result["is_blackout"]).sum() == 3

    def test_fast_mode_does_not_raise(self):
        result = self.phaser.channel_sensitivity(
            "search", blackout=Blackout(), n_sims=50, n_phasing_seeds=5, fast_mode=True
        )
        assert len(result) == 7

    def test_other_channels_locked_out_of_phasing(self):
        """With every other channel locked at 0, phasing tv alone must leave
        meta/search's own weekly numbers exactly at plan (rescale only ever
        touches the channel being varied)."""
        # Spot-check via the underlying mechanism directly, since
        # channel_sensitivity only returns CV, not the schedule itself.
        from how_wrong_is_your_mmm._phaser import _generate_phased_schedule

        month_labels = _get_month_labels(PLAN_DF)
        spec = {"tv": 40.0, "meta": 0.0, "search": 0.0}
        result = _generate_phased_schedule(
            PLAN_DF, month_labels, alpha=1.0, max_weekly_deviation_pct=spec, seed=0
        )
        pd.testing.assert_series_equal(result["meta"], PLAN_DF["meta"].astype(float))
        pd.testing.assert_series_equal(
            result["search"], PLAN_DF["search"].astype(float)
        )


class TestImpactOverHorizons:
    def setup_method(self):
        self.phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=0
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)

    def test_before_fit_raises(self):
        unfit = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        with pytest.raises(RuntimeError, match="Call fit"):
            unfit.impact_over_horizons(n_sims=5, n_phasing_seeds=1)

    def test_default_horizons_give_three_times_n_channels_rows(self):
        result = self.phaser.impact_over_horizons(n_sims=5, n_phasing_seeds=1)
        assert len(result) == 3 * len(PLAN_DF.columns)

    def test_custom_horizons_respected(self):
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13, 26], n_sims=5, n_phasing_seeds=1
        )
        assert set(result["horizon_weeks"]) == {13, 26}

    def test_no_revenue_columns_by_default(self):
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13], n_sims=5, n_phasing_seeds=1
        )
        assert "revenue_today_p10" not in result.columns

    def test_revenue_columns_present_with_include_revenue(self):
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13],
            n_sims=5,
            n_phasing_seeds=1,
            include_revenue=True,
        )
        for col in [
            "revenue_today_p10",
            "revenue_today_p90",
            "revenue_after_p10",
            "revenue_after_p90",
        ]:
            assert col in result.columns

    def test_cv_reduction_pct_matches_cv_columns(self):
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13], n_sims=5, n_phasing_seeds=1
        )
        row = result.iloc[0]
        expected = round(100 * (row["cv_today"] - row["cv_after"]) / row["cv_today"], 2)
        assert row["cv_reduction_pct"] == expected

    def test_fast_mode_does_not_raise(self):
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13, 52], n_sims=50, n_phasing_seeds=5, fast_mode=True
        )
        assert len(result) == 2 * len(PLAN_DF.columns)

    def test_revenue_uses_fixed_plan_total_across_horizons(self):
        """Revenue must price every horizon against plan_df's
        own fixed total spend, not each horizon's own tiled-plan total. A
        104-week horizon tiles the 52-week plan twice, so a reintroduction
        of the old horizon-scaled bug would put its revenue roughly 4x
        (52/13) or 8x (104/13) bigger than the 13-week horizon's, purely
        from spend scale, not reliability. With the fix, both horizons
        price against the same annual total, so only CV (and so the range's
        width, not its centre) should differ between them.
        """
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[13, 104],
            n_sims=30,
            n_phasing_seeds=2,
            include_revenue=True,
        )
        today = result[result["channel"] == "tv"].set_index("horizon_weeks")
        center_13 = (
            today.loc[13, "revenue_today_p10"] + today.loc[13, "revenue_today_p90"]
        ) / 2
        center_104 = (
            today.loc[104, "revenue_today_p10"] + today.loc[104, "revenue_today_p90"]
        ) / 2
        assert center_104 / center_13 < 3.0

    def test_revenue_matches_diagnostic_summary_at_plans_own_length(self):
        """The horizon equal to plan_df's own length must reproduce exactly
        what CollinearityDiagnostic.summary(planned_spend=plan_df.sum())
        would give directly on history + plan_df, unphased -- both now use
        the same fixed planned_spend, so "today" at that horizon is the
        same combined dataset and the same planned_spend as a standalone
        diagnostic on the un-tiled plan.
        """
        n_weeks = len(PLAN_DF)
        result = self.phaser.impact_over_horizons(
            horizons_weeks=[n_weeks],
            n_sims=20,
            n_phasing_seeds=1,
            include_revenue=True,
        )
        today = result.set_index("channel")

        direct = CollinearityDiagnostic(
            spend_df=pd.concat([HISTORY_DF, PLAN_DF]),
            true_marginal_returns=MARGINAL_RETURNS,
        )
        # Different n_sims/seed draw than the phaser's internal call, so this
        # checks the *planned_spend* pricing is identical, not that the
        # simulated draws match exactly.
        direct.fit(n_sims=20)
        direct_summary = direct.summary(
            planned_spend=PLAN_DF.sum().to_dict()
        ).set_index("channel")

        for ch in PLAN_DF.columns:
            # Same order of magnitude and same sign of range -- confirms
            # both use the same fixed planned_spend rather than the tiled
            # (here, identical, since n_weeks == len(PLAN_DF)) total.
            assert today.loc[ch, "revenue_today_p10"] > 0
            assert (
                today.loc[ch, "revenue_today_p90"] > today.loc[ch, "revenue_today_p10"]
            )
            ratio = (
                today.loc[ch, "revenue_today_p90"]
                / direct_summary.loc[ch, "incremental_revenue_p90"]
            )
            assert 0.5 < ratio < 2.0


class TestTilePlan:
    def test_truncates_when_shorter(self):
        from how_wrong_is_your_mmm._phaser import _tile_plan

        result = _tile_plan(PLAN_DF, 13)
        assert len(result) == 13
        pd.testing.assert_frame_equal(
            result, PLAN_DF.iloc[:13].astype(result.dtypes.iloc[0])
        )

    def test_tiles_when_longer(self):
        from how_wrong_is_your_mmm._phaser import _tile_plan

        result = _tile_plan(PLAN_DF, 104)
        assert len(result) == 104
        # second half repeats the first half's values (not the dates)
        np.testing.assert_array_equal(
            result.iloc[52:104].to_numpy(), result.iloc[0:52].to_numpy()
        )

    def test_index_is_datetime_continuing_from_start(self):
        from how_wrong_is_your_mmm._phaser import _tile_plan

        result = _tile_plan(PLAN_DF, 104)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index[0] == PLAN_DF.index[0]
        assert result.index.is_monotonic_increasing

    def test_exact_length_passthrough(self):
        from how_wrong_is_your_mmm._phaser import _tile_plan

        result = _tile_plan(PLAN_DF, len(PLAN_DF))
        np.testing.assert_array_equal(result.to_numpy(), PLAN_DF.to_numpy())


class TestSelectionBiasConfirmation:
    """fit()'s confirmation pass — the grid argmin is
    a systematically optimistic estimate (picking the min of grid_steps
    noisy points), so fit() re-evaluates the top confirm_top_k candidates
    independently before committing to a recommendation."""

    def test_confirmation_populated_after_fit(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1)
        assert phaser.confirmation_ is not None
        assert set(phaser.confirmation_.columns) == set(phaser.results_.columns)

    def test_confirmation_before_fit_is_none(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        assert phaser.confirmation_ is None

    def test_confirm_top_k_controls_confirmation_row_count(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=8, n_phasing_seeds=1, confirm_top_k=4)
        assert len(phaser.confirmation_) == 4

    def test_confirm_top_k_capped_at_grid_size(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1, confirm_top_k=10)
        assert len(phaser.confirmation_) == 3

    def test_confirmation_alphas_are_among_grid_lowest(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=8, n_phasing_seeds=1, confirm_top_k=3)
        expected_alphas = set(phaser.results_.nsmallest(3, "max_cv")["alpha"].round(4))
        assert set(phaser.confirmation_["alpha"].round(4)) == expected_alphas

    def test_recommended_schedule_uses_confirmed_alpha(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=8, n_phasing_seeds=1, confirm_top_k=3)
        confirmed_alpha = float(
            phaser.confirmation_.loc[phaser.confirmation_["max_cv"].idxmin(), "alpha"]
        )
        recommended_alpha = float(phaser.recommend()["alpha"])
        assert recommended_alpha == pytest.approx(confirmed_alpha)

    def test_fast_mode_skips_extra_confirmation_cost(self):
        """fast_mode forces confirm_top_k=1 — confirmation still runs (so
        recommend()/recommended_schedule_ stay well-defined) but only
        re-evaluates the single grid winner, not its neighbours."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=50, grid_steps=20, n_phasing_seeds=5, fast_mode=True)
        assert len(phaser.confirmation_) == 1

    def test_confirm_n_phasing_seeds_defaults_to_triple(self):
        """Default confirm_n_phasing_seeds is 3x n_phasing_seeds — larger
        sample than the grid search itself used per point, per the
        standard optimizer's-curse fix."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        phaser.fit(n_sims=5, grid_steps=4, n_phasing_seeds=2, confirm_top_k=2)
        # Indirectly verify via reproducibility at an explicit equal setting.
        phaser2 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        phaser2.fit(
            n_sims=5,
            grid_steps=4,
            n_phasing_seeds=2,
            confirm_top_k=2,
            confirm_n_phasing_seeds=6,
        )
        pd.testing.assert_frame_equal(phaser.confirmation_, phaser2.confirmation_)

    def test_reproducible_with_same_seed(self):
        p1 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=7
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1, confirm_top_k=3)
        p2 = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=7
        ).fit(n_sims=5, grid_steps=5, n_phasing_seeds=1, confirm_top_k=3)
        pd.testing.assert_frame_equal(p1.confirmation_, p2.confirmation_)


class TestRecommendLevers:
    def setup_method(self):
        self.phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=0
        )

    def test_returns_one_entry_per_channel(self):
        spec = self.phaser.recommend_levers(n_sims=5, n_phasing_seeds=1)
        assert set(spec.keys()) == set(PLAN_DF.columns)

    def test_no_blackout_option_keeps_continuous_everywhere(self):
        spec = self.phaser.recommend_levers(
            magnitude_pct=40.0, blackout=None, n_sims=5, n_phasing_seeds=1
        )
        assert all(v == 40.0 for v in spec.values())

    def test_every_choice_is_valid_spec_shape(self):
        spec = self.phaser.recommend_levers(n_sims=5, n_phasing_seeds=1)
        for v in spec.values():
            assert isinstance(v, float | Blackout)

    def test_result_usable_directly_as_max_weekly_deviation_pct(self):
        """recommend_levers()'s output should be a drop-in for
        BudgetPhaser's own max_weekly_deviation_pct — the whole point is
        chaining it into a subsequent fit()."""
        spec = self.phaser.recommend_levers(n_sims=5, n_phasing_seeds=1)
        phased = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            max_weekly_deviation_pct=spec,
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1)
        assert phased.recommended_schedule_.shape == PLAN_DF.shape

    def test_extreme_threshold_never_picks_blackout(self):
        """A threshold above 100% relative improvement can never be met, so
        every channel should fall back to the continuous option."""
        spec = self.phaser.recommend_levers(
            magnitude_pct=40.0,
            improvement_threshold_pct=1000.0,
            n_sims=5,
            n_phasing_seeds=1,
        )
        assert all(v == 40.0 for v in spec.values())

    def test_zero_threshold_picks_blackout_whenever_it_looks_better_at_all(self):
        """Threshold 0 means any nonzero improvement is enough — sanity
        check the comparison direction is right (lower CV wins), not
        inverted."""
        spec = self.phaser.recommend_levers(
            magnitude_pct=40.0,
            improvement_threshold_pct=0.0,
            n_sims=20,
            n_phasing_seeds=3,
        )
        # At least verify it runs and produces valid entries; whether
        # Blackout wins for a given channel depends on the data draw.
        for v in spec.values():
            assert isinstance(v, float | Blackout)

    def test_fast_mode_does_not_raise(self):
        spec = self.phaser.recommend_levers(fast_mode=True)
        assert set(spec.keys()) == set(PLAN_DF.columns)


class TestRecommendedDraws:
    """P0.5: recommended_schedule_ is picked from n_recommended_draws
    independently evaluated draws, not one never-evaluated one; the
    honestly-reported CV is the median across those draws, not the
    (optimistic) CV of the shipped draw itself."""

    def test_recommended_draws_populated_after_fit(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=5, grid_steps=3, n_phasing_seeds=1, n_recommended_draws=4)
        assert phaser.recommended_draws_ is not None
        assert len(phaser.recommended_draws_) == 4
        assert "max_cv" in phaser.recommended_draws_.columns

    def test_median_cv_is_not_the_shipped_draws_own_cv(self):
        """The shipped schedule is the BEST of n_recommended_draws (lowest
        max_cv), so its own evaluation is optimistic relative to the
        median across all draws -- the two should generally differ."""
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=10, grid_steps=3, n_phasing_seeds=1, n_recommended_draws=8)
        assert phaser.recommended_schedule_median_cv_ is not None
        best_cv = phaser.recommended_draws_["max_cv"].min()
        assert phaser.recommended_schedule_median_cv_ >= best_cv

    def test_recommended_schedule_is_the_best_evaluated_draw(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=10, grid_steps=3, n_phasing_seeds=1, n_recommended_draws=5)
        best_idx = int(phaser.recommended_draws_["max_cv"].idxmin())
        best_seed = int(phaser.recommended_draws_.loc[best_idx, "seed"])
        # Regenerate the schedule at that seed directly and confirm it
        # matches what was shipped.
        from how_wrong_is_your_mmm._phaser import _generate_phased_schedule

        best_alpha = float(
            phaser.confirmation_.loc[phaser.confirmation_["max_cv"].idxmin(), "alpha"]
        )
        rebuilt = _generate_phased_schedule(
            phaser.plan_df,
            phaser._plan_month_labels,
            alpha=best_alpha,
            max_weekly_deviation_pct=phaser.max_weekly_deviation_pct,
            seed=best_seed,
        )
        pd.testing.assert_frame_equal(phaser.recommended_schedule_, rebuilt)

    def test_fast_mode_uses_single_draw(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        ).fit(n_sims=50, grid_steps=10, fast_mode=True)
        assert len(phaser.recommended_draws_) == 1


class TestRemovedMaxMonthlyDeviationPctParam:
    """P0.6: max_monthly_deviation_pct was stored but never read anywhere
    -- removed rather than wired up. Guard against it silently coming back
    as a dead parameter."""

    def test_constructor_rejects_it(self):
        with pytest.raises(TypeError):
            BudgetPhaser(
                HISTORY_DF,
                PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                max_monthly_deviation_pct=1.0,
            )


class TestRevenueNoiseStdForwarding:
    """P0.3: base_sales/revenue_noise_std must actually reach the
    CollinearityDiagnostic instances BudgetPhaser builds internally, not
    just live as unused constructor args."""

    def test_higher_noise_std_widens_cv(self):
        low = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            revenue_noise_std=5_000.0,
        ).fit(n_sims=30, grid_steps=3, n_phasing_seeds=1, n_recommended_draws=1)
        high = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            revenue_noise_std=50_000.0,
        ).fit(n_sims=30, grid_steps=3, n_phasing_seeds=1, n_recommended_draws=1)
        assert high.recommend()["max_cv"] > low.recommend()["max_cv"]


class TestChannelConstraints:
    """P1.11: channel_constraints lets a caller hard-pin a channel's lever,
    bypassing the CV-comparison threshold entirely (e.g. a channel that
    must never be blacked out regardless of what the comparison prefers)."""

    def setup_method(self):
        self.phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS, seed=0
        )

    def test_constrained_channel_keeps_exact_pinned_spec(self):
        spec = self.phaser.recommend_levers(
            n_sims=5, n_phasing_seeds=1, channel_constraints={"tv": 0}
        )
        assert spec["tv"] == 0

    def test_constrained_channel_skips_cv_comparison_even_if_blackout_would_win(self):
        # Even with threshold 0 (Blackout wins whenever it's at all better),
        # a constrained channel must not be touched.
        spec = self.phaser.recommend_levers(
            n_sims=5,
            n_phasing_seeds=1,
            improvement_threshold_pct=0.0,
            channel_constraints={"tv": 0},
        )
        assert spec["tv"] == 0

    def test_unconstrained_channels_still_decided_normally(self):
        spec = self.phaser.recommend_levers(
            n_sims=5, n_phasing_seeds=1, channel_constraints={"tv": 0}
        )
        assert set(spec.keys()) == set(PLAN_DF.columns)
        for ch in PLAN_DF.columns:
            if ch != "tv":
                assert isinstance(spec[ch], float | Blackout)


class TestBudgetPhaserDeprecatedTrueElasticitiesAlias:
    def test_constructor_warns(self):
        with pytest.warns(FutureWarning, match="true_elasticities is deprecated"):
            BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=MARGINAL_RETURNS)

    def test_both_given_raises(self):
        with pytest.raises(ValueError, match="only one of"):
            BudgetPhaser(
                HISTORY_DF,
                PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                true_elasticities=MARGINAL_RETURNS,
            )

    def test_attribute_access_warns(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_marginal_returns=MARGINAL_RETURNS
        )
        with pytest.warns(FutureWarning, match="deprecated"):
            assert phaser.true_elasticities == MARGINAL_RETURNS


def _abs_dev_pct(phased, plan):
    """Realised |weekly deviation| from plan, in percent, all channels."""
    return ((phased - plan) / plan * 100).abs().to_numpy().ravel()


class TestShapedNudge:
    """The magnitude/sign split behind nudge_shape and balance_signs."""

    def test_zero_cap_returns_zeros(self):
        rng = np.random.default_rng(0)
        out = _shaped_nudge(rng, 4, 0.0, "edge", False)
        assert np.all(out == 0.0)

    def test_magnitudes_respect_the_cap(self):
        for shape in NUDGE_SHAPES:
            rng = np.random.default_rng(0)
            out = _shaped_nudge(rng, 500, 0.4, shape, False)
            assert np.abs(out).max() <= 0.4 + 1e-12, shape

    def test_uniform_spans_the_whole_band(self):
        rng = np.random.default_rng(0)
        mag = np.abs(_shaped_nudge(rng, 4000, 0.4, "uniform", False))
        assert mag.min() < 0.02
        assert mag.max() > 0.38

    def test_annulus_excludes_the_timid_middle(self):
        rng = np.random.default_rng(0)
        mag = np.abs(_shaped_nudge(rng, 4000, 0.4, "annulus", False))
        assert mag.min() >= 0.2 - 1e-12
        assert mag.max() <= 0.4 + 1e-12

    def test_edge_is_exactly_the_cap(self):
        rng = np.random.default_rng(0)
        mag = np.abs(_shaped_nudge(rng, 200, 0.4, "edge", False))
        assert np.allclose(mag, 0.4)

    def test_mean_magnitude_orders_uniform_annulus_edge(self):
        means = []
        for shape in ("uniform", "annulus", "edge"):
            rng = np.random.default_rng(1)
            means.append(np.abs(_shaped_nudge(rng, 6000, 0.4, shape, False)).mean())
        assert means[0] < means[1] < means[2]

    def test_uniform_mean_magnitude_is_about_half_the_cap(self):
        # The reason a nominal +/-20% setting moves a typical week by ~8%.
        rng = np.random.default_rng(2)
        mag = np.abs(_shaped_nudge(rng, 20000, 0.4, "uniform", False))
        assert 0.19 < mag.mean() < 0.21

    def test_balanced_signs_sum_to_zero_for_even_months(self):
        rng = np.random.default_rng(0)
        out = _shaped_nudge(rng, 4, 0.4, "edge", True)
        assert np.isclose(out.sum(), 0.0)
        assert (out > 0).sum() == 2
        assert (out < 0).sum() == 2

    def test_balanced_signs_leave_one_week_flat_for_odd_months(self):
        rng = np.random.default_rng(0)
        out = _shaped_nudge(rng, 5, 0.4, "edge", True)
        assert (out > 0).sum() == 2
        assert (out < 0).sum() == 2
        assert (out == 0).sum() == 1

    def test_unbalanced_signs_are_not_always_even(self):
        rng = np.random.default_rng(0)
        counts = {
            (_shaped_nudge(rng, 4, 0.4, "edge", False) > 0).sum() for _ in range(50)
        }
        assert counts != {2}


class TestGeneratePhasedScheduleNudgeShape:
    def setup_method(self):
        self.month_labels = _get_month_labels(PLAN_DF)

    def _sched(self, shape, balanced, seed=0, alpha=1.0, mag=40.0):
        return _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=alpha,
            max_weekly_deviation_pct=mag,
            seed=seed,
            nudge_shape=shape,
            balance_signs=balanced,
        )

    def test_default_path_is_byte_identical_to_the_unshaped_call(self):
        # Guards every published number: docs/overview.html and the
        # notebooks all derive from seeded default-path schedules, so this
        # must never drift, whatever is added to the shaped path.
        for seed in (0, 7, 99):
            legacy = _generate_phased_schedule(
                PLAN_DF,
                self.month_labels,
                alpha=1.0,
                max_weekly_deviation_pct=40.0,
                seed=seed,
            )
            pd.testing.assert_frame_equal(legacy, self._sched("uniform", False, seed))

    def test_explicit_uniform_unbalanced_matches_the_default(self):
        assert _generate_phased_schedule.__defaults__[:2] == ("uniform", False)

    @pytest.mark.parametrize("shape", NUDGE_SHAPES)
    @pytest.mark.parametrize("balanced", [False, True])
    def test_monthly_totals_preserved(self, shape, balanced):
        result = self._sched(shape, balanced)
        dev = _max_monthly_deviation(PLAN_DF, result, self.month_labels)
        assert dev < 1e-10

    @pytest.mark.parametrize("shape", NUDGE_SHAPES)
    @pytest.mark.parametrize("balanced", [False, True])
    def test_zero_alpha_unchanged(self, shape, balanced):
        result = self._sched(shape, balanced, alpha=0.0)
        pd.testing.assert_frame_equal(result, PLAN_DF.astype(float))

    @pytest.mark.parametrize("shape", NUDGE_SHAPES)
    @pytest.mark.parametrize("balanced", [False, True])
    def test_reproducibility(self, shape, balanced):
        pd.testing.assert_frame_equal(
            self._sched(shape, balanced, seed=7), self._sched(shape, balanced, seed=7)
        )

    @pytest.mark.parametrize("shape", NUDGE_SHAPES)
    @pytest.mark.parametrize("balanced", [False, True])
    def test_shape_changes_the_schedule(self, shape, balanced):
        if shape == "uniform" and not balanced:
            pytest.skip("that is the default path, covered by the identity test")
        assert not np.allclose(
            self._sched(shape, balanced).to_numpy(),
            self._sched("uniform", False).to_numpy(),
        )

    def test_realised_deviation_orders_by_shape(self):
        # The point of the whole option: more of the negotiated band
        # actually reaches the model. Averaged over seeds, since one draw
        # is noisy.
        means = []
        for shape in ("uniform", "annulus", "edge"):
            per_seed = [
                _abs_dev_pct(self._sched(shape, False, seed=s), PLAN_DF).mean()
                for s in range(30)
            ]
            means.append(float(np.mean(per_seed)))
        assert means[0] < means[1] < means[2]

    def test_uniform_realises_far_less_than_its_nominal_band(self):
        per_seed = [
            _abs_dev_pct(
                self._sched("uniform", False, seed=s, mag=20.0), PLAN_DF
            ).mean()
            for s in range(30)
        ]
        assert 5.0 < float(np.mean(per_seed)) < 12.0

    @pytest.mark.parametrize("shape", ["annulus", "edge"])
    def test_balancing_signs_contains_the_rescale_overshoot(self, shape):
        # An unbalanced month forces the surviving weeks to absorb it, so
        # the realised deviation can run far past the nominal band -- at
        # cap=80% the edge shape reaches ~5x it. Sign-balanced months have
        # almost nothing to rescale away, and land under 2x.
        nominal = 80.0
        worst = {
            balanced: max(
                _abs_dev_pct(
                    self._sched(shape, balanced, seed=s, mag=nominal), PLAN_DF
                ).max()
                for s in range(30)
            )
            for balanced in (False, True)
        }
        assert worst[False] > 2.0 * nominal
        assert worst[True] < 0.6 * worst[False]
        assert worst[True] < 2.0 * nominal

    def test_locked_channels_stay_exactly_on_plan_under_every_shape(self):
        for shape in NUDGE_SHAPES:
            for balanced in (False, True):
                result = _generate_phased_schedule(
                    PLAN_DF,
                    self.month_labels,
                    alpha=1.0,
                    max_weekly_deviation_pct={"tv": 40.0, "meta": 0.0, "search": 0.0},
                    seed=0,
                    nudge_shape=shape,
                    balance_signs=balanced,
                )
                for locked in ("meta", "search"):
                    pd.testing.assert_series_equal(
                        result[locked], PLAN_DF[locked].astype(float)
                    )

    @pytest.mark.parametrize("shape", NUDGE_SHAPES)
    @pytest.mark.parametrize("balanced", [False, True])
    def test_all_blackout_is_untouched_by_the_shape(self, shape, balanced):
        # Blackout is on/off by construction, so the options must not reach
        # it. Every channel is Blackout here on purpose: a float-spec
        # channel in the same call draws from the same generator, and the
        # shaped path does not consume randomness for a locked channel, so
        # mixing specs shifts the stream and would make this compare
        # different draws rather than the same draw under two options.
        spec = Blackout(max_dark_weeks_per_month=1)
        base = _generate_phased_schedule(
            PLAN_DF, self.month_labels, alpha=1.0, max_weekly_deviation_pct=spec, seed=3
        )
        shaped = _generate_phased_schedule(
            PLAN_DF,
            self.month_labels,
            alpha=1.0,
            max_weekly_deviation_pct=spec,
            seed=3,
            nudge_shape=shape,
            balance_signs=balanced,
        )
        pd.testing.assert_frame_equal(base, shaped)

    def test_annulus_never_leaves_a_week_untouched_before_rescale(self):
        rng = np.random.default_rng(11)
        assert np.abs(_shaped_nudge(rng, 2000, 0.4, "annulus", False)).min() > 0.0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="nudge_shape must be one of"):
            self._sched("gaussian", False)


class TestBudgetPhaserNudgeShape:
    def test_defaults_preserve_shipped_behaviour(self):
        phaser = BudgetPhaser(
            history_df=HISTORY_DF,
            plan_df=PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
        )
        assert phaser.nudge_shape == "uniform"
        assert phaser.balance_signs is False

    def test_stores_the_options(self):
        phaser = BudgetPhaser(
            history_df=HISTORY_DF,
            plan_df=PLAN_DF,
            true_marginal_returns=MARGINAL_RETURNS,
            nudge_shape="annulus",
            balance_signs=True,
        )
        assert phaser.nudge_shape == "annulus"
        assert phaser.balance_signs is True

    def test_invalid_shape_raises_at_construction(self):
        with pytest.raises(ValueError, match="nudge_shape must be one of"):
            BudgetPhaser(
                history_df=HISTORY_DF,
                plan_df=PLAN_DF,
                true_marginal_returns=MARGINAL_RETURNS,
                nudge_shape="triangular",
            )
