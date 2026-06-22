"""Tests for _dgp.py — simulate_spend and simulate_sales."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend


class TestSimulateSpend:
    def test_output_shape(self):
        df = simulate_spend(n_obs=52, correlation=0.5)
        assert df.shape == (52, 2)
        assert list(df.columns) == ["tv", "meta"]

    def test_default_shape(self):
        df = simulate_spend()
        assert df.shape == (104, 2)

    def test_correlation_direction(self):
        """Higher target correlation should produce higher actual correlation."""
        low = simulate_spend(correlation=0.2, seed=0)["tv"].corr(
            simulate_spend(correlation=0.2, seed=0)["meta"]
        )
        high = simulate_spend(correlation=0.9, seed=0)["tv"].corr(
            simulate_spend(correlation=0.9, seed=0)["meta"]
        )
        assert high > low

    def test_reproducibility(self):
        df1 = simulate_spend(seed=42)
        df2 = simulate_spend(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_spend(seed=0)
        df2 = simulate_spend(seed=1)
        assert not df1.equals(df2)

    def test_invalid_correlation(self):
        with pytest.raises(ValueError):
            simulate_spend(correlation=0.0)
        with pytest.raises(ValueError):
            simulate_spend(correlation=1.0)


class TestSimulateSales:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.5, seed=0)

    def test_output_length(self):
        sales = simulate_sales(self.spend_df)
        assert len(sales) == len(self.spend_df)

    def test_output_name(self):
        sales = simulate_sales(self.spend_df)
        assert sales.name == "sales"

    def test_reproducibility(self):
        s1 = simulate_sales(self.spend_df, seed=7)
        s2 = simulate_sales(self.spend_df, seed=7)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = simulate_sales(self.spend_df, seed=0)
        s2 = simulate_sales(self.spend_df, seed=1)
        assert not s1.equals(s2)

    def test_elasticity_direction(self):
        """Higher elasticity should produce higher mean sales."""
        low = simulate_sales(self.spend_df, true_elast_tv=0.1, revenue_noise_std=0).mean()
        high = simulate_sales(self.spend_df, true_elast_tv=0.9, revenue_noise_std=0).mean()
        assert high > low
