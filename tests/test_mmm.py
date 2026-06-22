"""Tests for _mmm.py — fit_ols."""

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend
from how_wrong_is_your_mmm._mmm import fit_ols


class TestFitOls:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.3, seed=0)

    def test_output_keys(self):
        sales = simulate_sales(self.spend_df, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert set(result.keys()) == {"tv", "meta"}

    def test_output_types(self):
        sales = simulate_sales(self.spend_df, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert isinstance(result["tv"], float)
        assert isinstance(result["meta"], float)

    def test_recovers_true_elasticities(self):
        """With low collinearity and no noise, OLS should recover true elasticities."""
        spend_df = simulate_spend(n_obs=500, correlation=0.1, seed=0)
        sales = simulate_sales(
            spend_df,
            true_elast_tv=0.3,
            true_elast_meta=0.5,
            revenue_noise_std=0.0,
            seed=0,
        )
        result = fit_ols(spend_df, sales)
        assert abs(result["tv"] - 0.3) < 0.01
        assert abs(result["meta"] - 0.5) < 0.01
