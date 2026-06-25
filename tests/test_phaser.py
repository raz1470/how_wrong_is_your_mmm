"""Tests for _phaser.py — BudgetPhaser (monthly-constrained spend phasing)."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._phaser import (
    BudgetPhaser,
    _compute_weights,
    _generate_phased_schedule,
    _get_month_labels,
    _max_monthly_deviation,
)

ELASTICITIES = {"tv": 0.3, "meta": 0.5, "search": 0.4}

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


class TestComputeWeights:
    def test_uniform_all_ones(self):
        w = _compute_weights(208, 52, "uniform", 5.0, 52)
        assert np.allclose(w, 1.0)

    def test_uniform_length(self):
        w = _compute_weights(208, 52, "uniform", 5.0, 52)
        assert len(w) == 260

    def test_binary_history_weight_is_one(self):
        w = _compute_weights(208, 52, "binary", 5.0, 52)
        assert np.allclose(w[:208], 1.0)

    def test_binary_plan_weight(self):
        w = _compute_weights(208, 52, "binary", 5.0, 52)
        assert np.allclose(w[208:], 5.0)

    def test_binary_custom_plan_weight(self):
        w = _compute_weights(10, 5, "binary", 3.0, 52)
        assert np.allclose(w[10:], 3.0)

    def test_decay_most_recent_is_one(self):
        w = _compute_weights(208, 52, "decay", 5.0, 52)
        assert abs(w[-1] - 1.0) < 1e-10

    def test_decay_monotone_increasing(self):
        w = _compute_weights(208, 52, "decay", 5.0, 52)
        assert np.all(np.diff(w) > 0)

    def test_decay_half_life(self):
        """Weight at half_life weeks from end should be ~0.5."""
        half_life = 52
        w = _compute_weights(208, 52, "decay", 5.0, half_life)
        # index of observation that is half_life weeks from the end
        idx = len(w) - 1 - half_life
        assert abs(w[idx] - 0.5) < 1e-10

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="weighting"):
            _compute_weights(10, 5, "invalid", 5.0, 52)

    def test_length_matches_total(self):
        w = _compute_weights(100, 20, "binary", 5.0, 52)
        assert len(w) == 120


class TestBudgetPhaser:
    def test_fit_returns_self(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES)
        assert phaser.fit(n_sims=5, grid_steps=3) is phaser

    def test_results_shape(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=5
        )
        assert len(phaser.results_) == 5

    def test_summary_columns(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=3
        )
        cols = set(phaser.summary().columns)
        assert {
            "alpha",
            "actual_correlation",
            "max_cv",
            "max_monthly_deviation_pct",
        }.issubset(cols)
        assert {"cv_tv", "cv_meta", "cv_search"}.issubset(cols)

    def test_fast_mode(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=50, grid_steps=20, fast_mode=True
        )
        assert len(phaser.results_) == 10

    def test_recommend_is_min_cv(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=5
        )
        rec = phaser.recommend()
        assert rec["max_cv"] == phaser.results_["max_cv"].min()

    def test_recommended_schedule_shape(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=3
        )
        assert phaser.recommended_schedule_.shape == PLAN_DF.shape

    def test_recommended_schedule_index(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=3
        )
        pd.testing.assert_index_equal(phaser.recommended_schedule_.index, PLAN_DF.index)

    def test_monthly_totals_preserved_in_recommended_schedule(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=3
        )
        month_labels = _get_month_labels(PLAN_DF)
        dev = _max_monthly_deviation(
            PLAN_DF, phaser.recommended_schedule_, month_labels
        )
        assert dev < 1e-10

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).summary()

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPhaser(
                HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES
            ).recommend()

    def test_no_datetime_index_on_history_raises(self):
        df = simulate_spend(n_obs=208, correlation=0.7, seed=0)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BudgetPhaser(df, PLAN_DF, true_elasticities=ELASTICITIES)

    def test_no_datetime_index_on_plan_raises(self):
        df = simulate_spend(n_obs=52, correlation=0.7, seed=0)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BudgetPhaser(HISTORY_DF, df, true_elasticities=ELASTICITIES)

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
                HISTORY_DF, plan_2ch, true_elasticities={"tv": 0.3, "meta": 0.5}
            )

    def test_invalid_weighting_raises(self):
        with pytest.raises(ValueError, match="weighting"):
            BudgetPhaser(HISTORY_DF, PLAN_DF, weighting="bad")

    def test_alpha_starts_at_zero(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=5
        )
        assert phaser.results_["alpha"].iloc[0] == 0.0

    def test_alpha_ends_at_one(self):
        phaser = BudgetPhaser(HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES).fit(
            n_sims=5, grid_steps=5
        )
        assert phaser.results_["alpha"].iloc[-1] == 1.0

    def test_weighting_uniform(self):
        phaser = BudgetPhaser(
            HISTORY_DF, PLAN_DF, true_elasticities=ELASTICITIES, weighting="uniform"
        ).fit(n_sims=5, grid_steps=3)
        assert phaser.results_ is not None

    def test_weighting_decay(self):
        phaser = BudgetPhaser(
            HISTORY_DF,
            PLAN_DF,
            true_elasticities=ELASTICITIES,
            weighting="decay",
            half_life=52,
        ).fit(n_sims=5, grid_steps=3)
        assert phaser.results_ is not None
