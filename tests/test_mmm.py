"""Tests for _mmm.py — fit_ols."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import (
    calibrate_baseline,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._mmm import fit_ols

MARGINAL_RETURNS = {"tv": 0.3, "meta": 0.5, "search": 0.4}


class TestFitOls:
    def setup_method(self):
        self.spend_df = simulate_spend(n_obs=104, correlation=0.3, seed=0)

    def test_output_keys(self):
        sales = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert set(result.keys()) == {"tv", "meta", "search"}

    def test_output_types(self):
        sales = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=0)
        result = fit_ols(self.spend_df, sales)
        assert all(isinstance(v, float) for v in result.values())

    def test_recovers_true_marginal_returns(self):
        """With low collinearity and no noise, OLS should recover the true
        marginal returns."""
        spend_df = simulate_spend(n_obs=500, correlation=0.1, seed=0)
        sales = simulate_sales(
            spend_df,
            MARGINAL_RETURNS,
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


class TestFitOlsControls:
    """Toggling a control on and off is how omitted-variable bias is measured.

    The DGP couples spend to demand, so a fit that omits demand is
    misspecified. Including it is correct but not free: the control is more
    collinear with each channel than the channels are with each other.
    """

    def setup_method(self):
        self.spend_df, self.demand = simulate_spend(
            n_obs=104, correlation=0.7, seed=0, return_demand=True
        )
        self.sales = simulate_sales(self.spend_df, MARGINAL_RETURNS, seed=0)

    def test_controls_none_matches_omitting_the_argument(self):
        assert fit_ols(self.spend_df, self.sales) == fit_ols(
            self.spend_df, self.sales, controls=None
        )

    def test_control_coefficient_is_returned(self):
        result = fit_ols(self.spend_df, self.sales, controls=self.demand)
        assert set(result) == {"tv", "meta", "search", "demand"}

    def test_dataframe_controls_accepted(self):
        controls = self.demand.to_frame(name="demand")
        result = fit_ols(self.spend_df, self.sales, controls=controls)
        assert "demand" in result

    def test_unnamed_series_gets_a_default_name(self):
        result = fit_ols(
            self.spend_df, self.sales, controls=pd.Series(self.demand.to_numpy())
        )
        assert "control" in result

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows but spend_df has"):
            fit_ols(self.spend_df, self.sales, controls=self.demand.iloc[:-1])

    def test_name_collision_raises(self):
        colliding = self.demand.rename("tv")
        with pytest.raises(ValueError, match="collides with a channel name"):
            fit_ols(self.spend_df, self.sales, controls=colliding)

    def test_omitted_confounder_biases_and_the_control_removes_it(self):
        """The headline behaviour: same data, same estimator, one regressor
        apart, and the answer moves by more than 50%."""
        spend_df, demand = simulate_spend(
            n_obs=104, correlation=0.7, seed=0, return_demand=True
        )
        cal = calibrate_baseline(
            spend_df, MARGINAL_RETURNS, baseline_share=0.72, baseline_cv=0.05
        )
        omitted, controlled = [], []
        for sim in range(120):
            sales = simulate_sales(
                spend_df,
                MARGINAL_RETURNS,
                base_sales=cal.baseline_level,
                seed=sim,
                demand=demand,
                demand_coef=cal.demand_coef,
            )
            omitted.append(fit_ols(spend_df, sales)["tv"])
            controlled.append(fit_ols(spend_df, sales, controls=demand)["tv"])

        truth = MARGINAL_RETURNS["tv"]
        omitted_bias = abs(np.mean(omitted) - truth) / truth
        controlled_bias = abs(np.mean(controlled) - truth) / truth
        assert omitted_bias > 0.5
        assert controlled_bias < 0.1
        assert controlled_bias < omitted_bias

    def test_the_control_costs_width(self):
        """Correct specification is not free: corr(spend, demand) exceeds the
        channel-channel correlation, so the control widens every estimate."""
        spend_df, demand = simulate_spend(
            n_obs=104, correlation=0.7, seed=0, return_demand=True
        )
        omitted, controlled = [], []
        for sim in range(200):
            sales = simulate_sales(spend_df, MARGINAL_RETURNS, seed=sim)
            omitted.append(fit_ols(spend_df, sales)["tv"])
            controlled.append(fit_ols(spend_df, sales, controls=demand)["tv"])
        assert np.std(controlled) > np.std(omitted)

    def test_decoupled_spend_makes_the_control_cheap(self):
        """At demand_share=0 the channels are just as correlated with each
        other but no longer track demand, so the control barely costs width.
        This is the mechanism phasing is meant to exploit."""

        def width_penalty(demand_share):
            spend_df, demand = simulate_spend(
                n_obs=104,
                correlation=0.7,
                seed=0,
                demand_share=demand_share,
                return_demand=True,
            )
            omitted, controlled = [], []
            for sim in range(200):
                sales = simulate_sales(spend_df, MARGINAL_RETURNS, seed=sim)
                omitted.append(fit_ols(spend_df, sales)["tv"])
                controlled.append(fit_ols(spend_df, sales, controls=demand)["tv"])
            return np.std(controlled) / np.std(omitted)

        assert width_penalty(0.0) < width_penalty(1.0)
