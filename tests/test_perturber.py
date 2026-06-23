"""Tests for _perturber.py — _perturb_spend and BudgetPerturber.

NOTE: BudgetPerturber is pending a full rebuild in Sprint B (3-channel
monthly-budget-constrained perturbation). Tests here cover the updated
_perturb_spend (N-channel) and the minimally-patched BudgetPerturber.
"""

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._perturber import BudgetPerturber, _perturb_spend

CHANNELS = ["tv", "meta", "search"]
ELASTICITIES = {"tv": 0.3, "meta": 0.5, "search": 0.4}


class TestPerturbSpend:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.7, seed=0)

    def test_output_shape(self):
        result = _perturb_spend(self.spend_df, perturbation_std=1000.0)
        assert result.shape == self.spend_df.shape
        assert list(result.columns) == CHANNELS

    def test_zero_perturbation_unchanged(self):
        result = _perturb_spend(self.spend_df, perturbation_std=0.0)
        pd.testing.assert_frame_equal(result, self.spend_df.reset_index(drop=True))

    def test_total_spend_preserved(self):
        import numpy as np

        result = _perturb_spend(self.spend_df, perturbation_std=5000.0, seed=0)
        original_total = self.spend_df.sum(axis=1).to_numpy()
        perturbed_total = result.sum(axis=1).to_numpy()
        assert (abs(original_total - perturbed_total) < 1e-6).all()

    def test_reproducibility(self):
        r1 = _perturb_spend(self.spend_df, 5000.0, seed=42)
        r2 = _perturb_spend(self.spend_df, 5000.0, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_larger_perturbation_reduces_mean_correlation(self):
        import numpy as np

        def mean_corr(df):
            c = df.corr().to_numpy()
            n = len(df.columns)
            return float(np.mean([c[i, j] for i in range(n) for j in range(i + 1, n)]))

        low = _perturb_spend(self.spend_df, 1_000.0, seed=0)
        high = _perturb_spend(self.spend_df, 50_000.0, seed=0)
        assert mean_corr(high) < mean_corr(low)


class TestBudgetPerturber:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.7, seed=0)

    def test_fit_returns_self(self):
        perturber = BudgetPerturber(self.spend_df, true_elasticities=ELASTICITIES)
        assert perturber.fit(n_sims=5, grid_steps=3) is perturber

    def test_results_shape(self):
        perturber = BudgetPerturber(
            self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5, grid_steps=5)
        assert len(perturber.results_) == 5

    def test_fast_mode(self):
        perturber = BudgetPerturber(
            self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=50, grid_steps=20, fast_mode=True)
        assert len(perturber.results_) == 10

    def test_recommend_is_min_cv(self):
        perturber = BudgetPerturber(
            self.spend_df, true_elasticities=ELASTICITIES
        ).fit(n_sims=5, grid_steps=5)
        rec = perturber.recommend()
        assert rec["max_cv"] == perturber.results_["max_cv"].min()

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPerturber(self.spend_df, true_elasticities=ELASTICITIES).summary()

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPerturber(self.spend_df, true_elasticities=ELASTICITIES).recommend()
