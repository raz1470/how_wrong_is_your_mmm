"""Tests for _dgp.py — simulate_spend and simulate_sales."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend

CHANNELS = ["tv", "meta", "search"]
ELASTICITIES = {"tv": 0.3, "meta": 0.5, "search": 0.4}


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
        assert len(simulate_sales(self.spend_df, ELASTICITIES)) == len(self.spend_df)

    def test_output_name(self):
        assert simulate_sales(self.spend_df, ELASTICITIES).name == "sales"

    def test_reproducibility(self):
        s1 = simulate_sales(self.spend_df, ELASTICITIES, seed=7)
        s2 = simulate_sales(self.spend_df, ELASTICITIES, seed=7)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = simulate_sales(self.spend_df, ELASTICITIES, seed=0)
        s2 = simulate_sales(self.spend_df, ELASTICITIES, seed=1)
        assert not s1.equals(s2)

    def test_elasticity_direction(self):
        """Higher elasticity should produce higher mean sales."""
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

    def test_default_elasticities(self):
        # should not raise when using default elasticities on default channels
        sales = simulate_sales(self.spend_df)
        assert len(sales) == len(self.spend_df)
