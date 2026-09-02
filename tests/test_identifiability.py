"""Tests for _identifiability.py — IdentifiabilityDiagnostic."""

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm._dgp import simulate_demand, simulate_spend
from how_wrong_is_your_mmm._identifiability import IdentifiabilityDiagnostic

N_OBS = 60
DEMAND = simulate_demand(N_OBS, process="seasonal", seed=0)
SPEND_DF = simulate_spend(
    N_OBS, correlation=0.5, seed=1, demand=DEMAND, demand_share=1.0
)

# Small grid — real default is 33x31; tests only need structural coverage.
B_GRID = np.round(np.linspace(0.4, 1.0, 4), 4)
LAM_GRID = np.round(np.linspace(0.0, 0.6, 4), 4)

SUMMARY_INDEX = {
    "b_mean",
    "b_sd",
    "b_bias",
    "lam_mean",
    "lam_sd",
    "lam_bias",
    "valley_pct",
}


def make_diag(**kwargs):
    defaults = dict(
        spend_df=SPEND_DF,
        demand=DEMAND,
        b_candidates=B_GRID,
        lam_candidates=LAM_GRID,
    )
    defaults.update(kwargs)
    return IdentifiabilityDiagnostic(**defaults)


class TestConstruction:
    def test_defaults_to_default_marginal_returns(self):
        diag = make_diag()
        assert diag.true_marginal_returns == {"tv": 0.5, "meta": 1.0, "search": 1.5}

    def test_channels_from_spend_df_columns(self):
        diag = make_diag()
        assert diag.channels_ == list(SPEND_DF.columns)

    def test_accepts_series_demand(self):
        diag = make_diag(demand=pd.Series(DEMAND))
        assert diag.demand.shape == (N_OBS,)

    def test_wrong_length_demand_raises(self):
        with pytest.raises(ValueError, match="length len\\(spend_df\\)"):
            make_diag(demand=DEMAND[:-1])

    def test_saturation_out_of_range_raises(self):
        with pytest.raises(ValueError, match="true_saturation"):
            make_diag(true_saturation=1.5)

    def test_saturation_zero_raises(self):
        with pytest.raises(ValueError, match="true_saturation"):
            make_diag(true_saturation=0.0)

    def test_adstock_out_of_range_raises(self):
        with pytest.raises(ValueError, match="true_adstock"):
            make_diag(true_adstock=1.0)

    def test_adstock_negative_raises(self):
        with pytest.raises(ValueError, match="true_adstock"):
            make_diag(true_adstock=-0.1)

    def test_invalid_spend_data_raises(self):
        bad = SPEND_DF.copy()
        bad["tv"] = 0.0
        with pytest.raises(ValueError):
            make_diag(spend_df=bad)

    def test_results_and_surface_none_before_fit(self):
        diag = make_diag()
        assert diag.results_ is None
        assert diag.rss_surface_ is None


class TestFit:
    def test_fit_returns_self(self):
        diag = make_diag()
        assert diag.fit(n_sims=3) is diag

    def test_results_shape(self):
        diag = make_diag().fit(n_sims=4)
        assert diag.results_.shape == (4, 3)
        assert list(diag.results_.columns) == ["sim", "recovered_b", "recovered_lam"]

    def test_rss_surface_shape_matches_grids(self):
        diag = make_diag().fit(n_sims=3)
        assert diag.rss_surface_.shape == (len(B_GRID), len(LAM_GRID))

    def test_fast_mode_overrides_n_sims(self):
        diag = make_diag().fit(n_sims=50, fast_mode=True)
        assert len(diag.results_) == 10

    def test_recovered_values_within_grid_bounds(self):
        diag = make_diag().fit(n_sims=5)
        assert diag.results_["recovered_b"].between(B_GRID.min(), B_GRID.max()).all()
        assert (
            diag.results_["recovered_lam"].between(LAM_GRID.min(), LAM_GRID.max()).all()
        )

    def test_reproducible_given_same_seed_offset(self):
        d1 = make_diag().fit(n_sims=5, noise_seed_offset=7)
        d2 = make_diag().fit(n_sims=5, noise_seed_offset=7)
        pd.testing.assert_frame_equal(d1.results_, d2.results_)

    def test_different_seed_offset_gives_different_draw(self):
        d1 = make_diag().fit(n_sims=5, noise_seed_offset=1)
        d2 = make_diag().fit(n_sims=5, noise_seed_offset=2)
        assert not d1.results_.equals(d2.results_)


class TestValleyWidth:
    def test_raises_before_fit(self):
        diag = make_diag()
        with pytest.raises(RuntimeError, match="Call fit"):
            diag.valley_width()

    def test_returns_fraction_between_zero_and_one(self):
        diag = make_diag().fit(n_sims=5)
        width = diag.valley_width()
        assert 0.0 < width <= 1.0

    def test_wider_tolerance_gives_wider_or_equal_valley(self):
        diag = make_diag().fit(n_sims=5)
        narrow = diag.valley_width(tol=0.001)
        wide = diag.valley_width(tol=0.5)
        assert wide >= narrow


class TestSummary:
    def test_raises_before_fit(self):
        diag = make_diag()
        with pytest.raises(RuntimeError, match="Call fit"):
            diag.summary()

    def test_summary_index(self):
        diag = make_diag().fit(n_sims=5)
        assert set(diag.summary().index) == SUMMARY_INDEX

    def test_bias_is_mean_minus_truth(self):
        diag = make_diag(true_saturation=0.7, true_adstock=0.3).fit(n_sims=5)
        s = diag.summary()
        assert s["b_bias"] == pytest.approx(s["b_mean"] - 0.7)
        assert s["lam_bias"] == pytest.approx(s["lam_mean"] - 0.3)

    def test_valley_pct_matches_valley_width_times_100(self):
        diag = make_diag().fit(n_sims=5)
        assert diag.summary(tol=0.02)["valley_pct"] == pytest.approx(
            100 * diag.valley_width(tol=0.02)
        )


class TestIdentifiabilityShrinksWithMoreCurvature:
    def test_narrower_spend_range_widens_the_valley(self):
        # A near-constant spend pattern gives adstock/saturation almost
        # nothing to bite on -- the RSS surface should be flatter (wider
        # valley) than a design with real variation in it.
        flat_spend = SPEND_DF.copy()
        rng = np.random.default_rng(0)
        for ch in flat_spend.columns:
            mean = flat_spend[ch].mean()
            # Tiny noise, not exactly constant -- zero variance is rejected
            # by _validate_spend_data (no elasticity is estimable at all).
            flat_spend[ch] = mean + rng.normal(scale=mean * 1e-6, size=len(flat_spend))

        varied = make_diag().fit(n_sims=10)
        flat = make_diag(spend_df=flat_spend).fit(n_sims=10)

        assert flat.valley_width() >= varied.valley_width()
