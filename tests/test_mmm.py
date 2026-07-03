"""Tests for _mmm.py — fit_ols."""

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend
from how_wrong_is_your_mmm._mmm import fit_ols

ELASTICITIES = {"tv": 0.3, "meta": 0.5, "search": 0.4}


class TestFitOls:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.3, seed=0)

    def test_output_keys(self):
        sales = simulate_sales(self.spend_df, ELASTICITIES, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert set(result.keys()) == {"tv", "meta", "search"}

    def test_output_types(self):
        sales = simulate_sales(self.spend_df, ELASTICITIES, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert all(isinstance(v, float) for v in result.values())

    def test_recovers_true_elasticities(self):
        """With low collinearity and no noise, OLS should recover true elasticities."""
        spend_df = simulate_spend(n_obs=500, correlation=0.1, seed=0)
        sales = simulate_sales(
            spend_df,
            ELASTICITIES,
            revenue_noise_std=0.0,
            seed=0,
        )
        result = fit_ols(spend_df, sales)
        assert abs(result["tv"] - 0.3) < 0.01
        assert abs(result["meta"] - 0.5) < 0.01
        assert abs(result["search"] - 0.4) < 0.01

    def test_two_channel_spend(self):
        """fit_ols should work with any number of channels."""
        spend_2ch = simulate_spend(
            n_obs=104, correlation=0.3, channels=["tv", "meta"], seed=0
        )
        sales = simulate_sales(spend_2ch, {"tv": 0.3, "meta": 0.5}, seed=0)
        result = fit_ols(spend_2ch, sales)
        assert set(result.keys()) == {"tv", "meta"}
