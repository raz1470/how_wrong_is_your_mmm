"""Tests for _diagnostic.py — CollinearityDiagnostic."""

import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic


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
        result = diag.fit(n_sims=5)
        assert result is diag

    def test_results_shape(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        # 2 channels × 5 sims
        assert diag.results_.shape == (10, 6)

    def test_summary_columns(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLS

    def test_summary_rows(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert len(diag.summary()) == 2

    def test_cv_positive(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=10)
        assert (diag.summary()["coef_of_variation"] > 0).all()

    def test_actual_correlation(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=5)
        assert 0 < diag.actual_correlation < 1

    def test_fast_mode_overrides_n_sims(self):
        diag = CollinearityDiagnostic(correlation=0.7).fit(n_sims=50, fast_mode=True)
        assert len(diag.results_) == 2 * 10  # fast_mode caps at 10

    def test_summary_before_fit_raises(self):
        diag = CollinearityDiagnostic(correlation=0.7)
        with pytest.raises(RuntimeError):
            diag.summary()

    def test_actual_correlation_before_fit_raises(self):
        diag = CollinearityDiagnostic(correlation=0.7)
        with pytest.raises(RuntimeError):
            _ = diag.actual_correlation


class TestRealSpendPath:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.6, seed=99)

    def test_fit_runs(self):
        diag = CollinearityDiagnostic(spend_df=self.spend_df).fit(n_sims=5)
        assert diag.results_ is not None

    def test_summary_columns(self):
        diag = CollinearityDiagnostic(spend_df=self.spend_df).fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLS

    def test_spend_df_not_mutated(self):
        original = self.spend_df.copy()
        CollinearityDiagnostic(spend_df=self.spend_df).fit(n_sims=5)
        pd.testing.assert_frame_equal(self.spend_df, original)

    def test_actual_correlation_matches_input(self):
        diag = CollinearityDiagnostic(spend_df=self.spend_df).fit(n_sims=5)
        expected = self.spend_df["tv"].corr(self.spend_df["meta"])
        assert abs(diag.actual_correlation - expected) < 1e-10

    def test_correlation_param_ignored_when_spend_df_supplied(self):
        """correlation kwarg should have no effect when spend_df is passed."""
        diag_low = CollinearityDiagnostic(spend_df=self.spend_df, correlation=0.1).fit(n_sims=5)
        diag_high = CollinearityDiagnostic(spend_df=self.spend_df, correlation=0.9).fit(n_sims=5)
        assert abs(diag_low.actual_correlation - diag_high.actual_correlation) < 1e-10
