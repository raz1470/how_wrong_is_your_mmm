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
CHANNELS = list(SPEND_DF.columns)

# Small grid — real default is 33x31; tests only need structural coverage.
B_GRID = np.round(np.linspace(0.4, 1.0, 4), 4)
LAM_GRID = np.round(np.linspace(0.0, 0.6, 4), 4)

SUMMARY_COLUMNS = {
    "channel",
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
        assert diag.channels_ == CHANNELS

    def test_accepts_series_demand(self):
        diag = make_diag(demand=pd.Series(DEMAND))
        assert diag.demand.shape == (N_OBS,)

    def test_wrong_length_demand_raises(self):
        with pytest.raises(ValueError, match="length len\\(spend_df\\)"):
            make_diag(demand=DEMAND[:-1])

    def test_float_saturation_broadcasts_to_every_channel(self):
        diag = make_diag(true_saturation=0.8)
        assert diag.true_saturation == {ch: 0.8 for ch in CHANNELS}

    def test_dict_saturation_kept_per_channel(self):
        values = {"tv": 0.6, "meta": 0.8, "search": 1.0}
        diag = make_diag(true_saturation=values)
        assert diag.true_saturation == values

    def test_dict_saturation_missing_channel_raises(self):
        with pytest.raises(KeyError, match="true_saturation"):
            make_diag(true_saturation={"tv": 0.6, "meta": 0.8})

    def test_saturation_out_of_range_raises(self):
        with pytest.raises(ValueError, match="true_saturation"):
            make_diag(true_saturation=1.5)

    def test_saturation_zero_raises(self):
        with pytest.raises(ValueError, match="true_saturation"):
            make_diag(true_saturation=0.0)

    def test_dict_saturation_one_bad_channel_raises(self):
        with pytest.raises(ValueError, match="true_saturation\\['meta'\\]"):
            make_diag(true_saturation={"tv": 0.6, "meta": 1.5, "search": 1.0})

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

    def test_results_keyed_by_channel(self):
        diag = make_diag().fit(n_sims=4)
        assert set(diag.results_) == set(CHANNELS)
        for ch in CHANNELS:
            assert diag.results_[ch].shape == (4, 3)
            assert list(diag.results_[ch].columns) == [
                "sim",
                "recovered_b",
                "recovered_lam",
            ]

    def test_rss_surface_shape_matches_grids_per_channel(self):
        diag = make_diag().fit(n_sims=3)
        assert set(diag.rss_surface_) == set(CHANNELS)
        for ch in CHANNELS:
            assert diag.rss_surface_[ch].shape == (len(B_GRID), len(LAM_GRID))

    def test_fast_mode_overrides_n_sims(self):
        diag = make_diag().fit(n_sims=50, fast_mode=True)
        for ch in CHANNELS:
            assert len(diag.results_[ch]) == 10

    def test_recovered_values_within_grid_bounds(self):
        diag = make_diag().fit(n_sims=5)
        for ch in CHANNELS:
            r = diag.results_[ch]
            assert r["recovered_b"].between(B_GRID.min(), B_GRID.max()).all()
            assert r["recovered_lam"].between(LAM_GRID.min(), LAM_GRID.max()).all()

    def test_reproducible_given_same_seed_offset(self):
        d1 = make_diag().fit(n_sims=5, noise_seed_offset=7)
        d2 = make_diag().fit(n_sims=5, noise_seed_offset=7)
        for ch in CHANNELS:
            pd.testing.assert_frame_equal(d1.results_[ch], d2.results_[ch])

    def test_different_seed_offset_gives_different_draw(self):
        d1 = make_diag().fit(n_sims=5, noise_seed_offset=1)
        d2 = make_diag().fit(n_sims=5, noise_seed_offset=2)
        assert not all(d1.results_[ch].equals(d2.results_[ch]) for ch in CHANNELS)


class TestValleyWidth:
    def test_raises_before_fit(self):
        diag = make_diag()
        with pytest.raises(RuntimeError, match="Call fit"):
            diag.valley_width()

    def test_returns_fraction_between_zero_and_one_per_channel(self):
        diag = make_diag().fit(n_sims=5)
        widths = diag.valley_width()
        assert set(widths) == set(CHANNELS)
        for w in widths.values():
            assert 0.0 < w <= 1.0

    def test_wider_tolerance_gives_wider_or_equal_valley(self):
        diag = make_diag().fit(n_sims=5)
        narrow = diag.valley_width(tol=0.001)
        wide = diag.valley_width(tol=0.5)
        for ch in CHANNELS:
            assert wide[ch] >= narrow[ch]


class TestSummary:
    def test_raises_before_fit(self):
        diag = make_diag()
        with pytest.raises(RuntimeError, match="Call fit"):
            diag.summary()

    def test_summary_columns(self):
        diag = make_diag().fit(n_sims=5)
        assert set(diag.summary().columns) == SUMMARY_COLUMNS

    def test_summary_one_row_per_channel(self):
        diag = make_diag().fit(n_sims=5)
        assert set(diag.summary()["channel"]) == set(CHANNELS)

    def test_bias_is_mean_minus_truth(self):
        diag = make_diag(true_saturation=0.7, true_adstock=0.3).fit(n_sims=5)
        s = diag.summary().set_index("channel")
        for ch in CHANNELS:
            assert s.loc[ch, "b_bias"] == pytest.approx(s.loc[ch, "b_mean"] - 0.7)
            assert s.loc[ch, "lam_bias"] == pytest.approx(s.loc[ch, "lam_mean"] - 0.3)

    def test_bias_uses_each_channels_own_truth(self):
        # Per-channel true_saturation, not a shared value -- each row's
        # b_bias should be measured against ITS OWN channel's truth.
        truths = {"tv": 0.5, "meta": 0.7, "search": 0.9}
        diag = make_diag(true_saturation=truths).fit(n_sims=5)
        s = diag.summary().set_index("channel")
        for ch in CHANNELS:
            assert s.loc[ch, "b_bias"] == pytest.approx(
                s.loc[ch, "b_mean"] - truths[ch]
            )

    def test_valley_pct_matches_valley_width_times_100(self):
        diag = make_diag().fit(n_sims=5)
        summary_pct = diag.summary(tol=0.02).set_index("channel")["valley_pct"]
        widths = diag.valley_width(tol=0.02)
        for ch in CHANNELS:
            assert summary_pct[ch] == pytest.approx(100 * widths[ch])


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

        varied_widths = varied.valley_width()
        flat_widths = flat.valley_width()
        for ch in CHANNELS:
            assert flat_widths[ch] >= varied_widths[ch]

    def test_flattening_one_channel_only_widens_that_channels_valley(self):
        # The point of per-channel profiling: flattening ONLY tv's spend
        # should barely move meta/search's own identifiability, since each
        # channel is now profiled holding every other channel at ITS OWN
        # (unflattened) true curvature -- not a shared grid point.
        flat_tv = SPEND_DF.copy()
        rng = np.random.default_rng(0)
        mean = flat_tv["tv"].mean()
        flat_tv["tv"] = mean + rng.normal(scale=mean * 1e-6, size=len(flat_tv))

        varied = make_diag().fit(n_sims=10)
        flat = make_diag(spend_df=flat_tv).fit(n_sims=10)

        varied_widths = varied.valley_width()
        flat_widths = flat.valley_width()
        assert flat_widths["tv"] > varied_widths["tv"]
        # Other channels shouldn't blow out just because tv went flat.
        for ch in ("meta", "search"):
            assert flat_widths[ch] == pytest.approx(varied_widths[ch], abs=0.35)
