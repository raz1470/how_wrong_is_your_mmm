"""Tests for _perturber.py — BudgetPerturber."""

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._perturber import BudgetPerturber, _perturb_spend


RESULT_COLS = {
    "perturbation_std",
    "perturbation_pct",
    "actual_correlation",
    "tv_cv",
    "meta_cv",
    "max_cv",
}


class TestPerturbSpend:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.7, seed=0)

    def test_output_shape(self):
        result = _perturb_spend(self.spend_df, perturbation_std=1000.0)
        assert result.shape == self.spend_df.shape
        assert list(result.columns) == ["tv", "meta"]

    def test_zero_perturbation_unchanged(self):
        result = _perturb_spend(self.spend_df, perturbation_std=0.0)
        pd.testing.assert_frame_equal(result, self.spend_df.reset_index(drop=True))

    def test_total_spend_preserved(self):
        """TV + Meta total should be the same before and after perturbation."""
        result = _perturb_spend(self.spend_df, perturbation_std=5000.0, seed=0)
        original_total = (self.spend_df["tv"] + self.spend_df["meta"]).to_numpy()
        perturbed_total = (result["tv"] + result["meta"]).to_numpy()
        import numpy as np
        assert (abs(original_total - perturbed_total) < 1e-6).all()

    def test_reproducibility(self):
        r1 = _perturb_spend(self.spend_df, 5000.0, seed=42)
        r2 = _perturb_spend(self.spend_df, 5000.0, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_larger_perturbation_changes_correlation(self):
        """Higher perturbation_std should reduce inter-channel correlation."""
        low = _perturb_spend(self.spend_df, 1_000.0, seed=0)
        high = _perturb_spend(self.spend_df, 50_000.0, seed=0)
        corr_low = low["tv"].corr(low["meta"])
        corr_high = high["tv"].corr(high["meta"])
        assert corr_high < corr_low


class TestBudgetPerturber:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.7, seed=0)

    def test_fit_returns_self(self):
        perturber = BudgetPerturber(self.spend_df)
        result = perturber.fit(n_sims=5, grid_steps=3, fast_mode=False)
        assert result is perturber

    def test_results_shape(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=5)
        assert len(perturber.results_) == 5

    def test_summary_columns(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=3)
        assert set(perturber.summary().columns) == RESULT_COLS

    def test_fast_mode(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=50, grid_steps=20, fast_mode=True)
        assert len(perturber.results_) == 10

    def test_recommend_returns_series(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=5)
        rec = perturber.recommend()
        assert isinstance(rec, pd.Series)
        assert "max_cv" in rec.index

    def test_recommend_is_min_cv(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=5)
        rec = perturber.recommend()
        assert rec["max_cv"] == perturber.results_["max_cv"].min()

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPerturber(self.spend_df).summary()

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BudgetPerturber(self.spend_df).recommend()

    def test_mean_weekly_total_set_after_fit(self):
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=3)
        assert perturber.mean_weekly_total_ > 0

    def test_perturbation_pct_range(self):
        """Grid should start at 0% and end at max_perturbation_pct * 100."""
        perturber = BudgetPerturber(self.spend_df, max_perturbation_pct=0.5).fit(
            n_sims=5, grid_steps=5
        )
        assert perturber.results_["perturbation_pct"].iloc[0] == 0.0
        assert perturber.results_["perturbation_pct"].iloc[-1] <= 50.0

    def test_higher_perturbation_reduces_correlation(self):
        """Correlation should generally decrease as perturbation increases."""
        perturber = BudgetPerturber(self.spend_df).fit(n_sims=5, grid_steps=5)
        corrs = perturber.results_["actual_correlation"].to_numpy()
        assert corrs[0] > corrs[-1]
