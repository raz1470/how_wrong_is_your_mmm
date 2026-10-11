"""Exact collinearity: spend whose transformed design is rank deficient.

Every channel is a fixed multiple of one series and every channel has the
same saturation and adstock, so after the curvature transform the columns
are exact multiples of each other. OLS has no unique answer on that design.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from how_wrong_is_your_mmm import UnidentifiableWarning
from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._diagnostic import (
    CollinearityDiagnostic,
    _design_is_rank_deficient,
)
from how_wrong_is_your_mmm._discovery_report import DiscoveryReport
from how_wrong_is_your_mmm._phaser import BudgetPhaser

MARGINAL_RETURNS = {"tv": 0.8, "meta": 1.2, "search": 2.0}
SHARES = {"tv": 1.0, "meta": 0.6, "search": 0.3}


def _exact_spend(n_obs: int, start: str, seed: int = 0) -> pd.DataFrame:
    one = simulate_spend(
        n_obs=n_obs, correlation=0.5, channels=["x", "y"], seed=seed, start_date=start
    )["x"]
    return pd.DataFrame({ch: one * share for ch, share in SHARES.items()})


EXACT = _exact_spend(156, "2020-01-06")


def _diag(spend, **kwargs):
    defaults = dict(
        spend_df=spend,
        true_marginal_returns=MARGINAL_RETURNS,
        saturation=0.7,
        adstock=0.3,
    )
    defaults.update(kwargs)
    return CollinearityDiagnostic(**defaults)


class TestDesignIsRankDeficient:
    def test_full_rank(self):
        rng = np.random.default_rng(0)
        x = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
        assert not _design_is_rank_deficient(x)

    def test_exact_multiple_column(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=50)
        x = np.column_stack([np.ones(50), a, 3.7 * a])
        assert _design_is_rank_deficient(x)

    def test_zero_column(self):
        x = np.column_stack([np.ones(10), np.arange(10.0), np.zeros(10)])
        assert _design_is_rank_deficient(x)

    def test_scale_does_not_matter(self):
        # Spend in the millions next to the intercept is still full rank.
        rng = np.random.default_rng(1)
        x = np.column_stack([np.ones(50), 1e7 * rng.random(size=(50, 2))])
        assert not _design_is_rank_deficient(x)


class TestCollinearityDiagnostic:
    def test_warns_and_reports_inf_cv(self):
        diag = _diag(EXACT)
        with pytest.warns(UnidentifiableWarning, match="cannot be separated"):
            diag.fit(n_sims=5)
        assert diag.identifiable_ is False
        summ = diag.summary()
        assert np.isinf(summ["coef_of_variation"]).all()
        assert summ["mean_estimated"].isna().all()
        assert diag.results_["estimated_marginal_return"].isna().all()

    def test_linear_curves_too(self):
        diag = _diag(EXACT, saturation=1.0, adstock=0.0)
        with pytest.warns(UnidentifiableWarning):
            diag.fit(n_sims=3)
        assert np.isinf(diag.summary()["coef_of_variation"]).all()

    def test_with_controls(self):
        demand = np.linspace(0.0, 1.0, len(EXACT))
        diag = _diag(EXACT, demand=demand, demand_coef=1000.0)
        with pytest.warns(UnidentifiableWarning):
            diag.fit(n_sims=3, controls=True)
        assert np.isinf(diag.summary()["coef_of_variation"]).all()

    def test_analytic_cv_is_inf(self):
        diag = _diag(EXACT)
        with pytest.warns(UnidentifiableWarning):
            diag.fit(n_sims=2)
        with pytest.warns(UnidentifiableWarning):
            cv = diag.analytic_cv()
        assert list(cv.index) == list(SHARES)
        assert np.isinf(cv).all()

    def test_different_curves_break_the_dependence(self):
        diag = _diag(EXACT, adstock={"tv": 0.5, "meta": 0.3, "search": 0.1})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnidentifiableWarning)
            diag.fit(n_sims=5)
        assert diag.identifiable_ is True
        assert np.isfinite(diag.summary()["coef_of_variation"]).all()

    def test_near_collinear_is_not_flagged(self):
        spend = simulate_spend(
            n_obs=156, correlation=0.999, channels=list(SHARES), seed=1
        )
        diag = _diag(spend)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnidentifiableWarning)
            diag.fit(n_sims=5)
            cv = diag.analytic_cv()
        assert diag.identifiable_ is True
        assert np.isfinite(cv).all()


class TestBudgetPhaser:
    def test_unphased_plan_cannot_win(self):
        phaser = BudgetPhaser(
            history_df=EXACT.iloc[:104],
            plan_df=EXACT.iloc[104:],
            true_marginal_returns=MARGINAL_RETURNS,
            saturation=0.7,
            adstock=0.3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnidentifiableWarning)
            phaser.fit(fast_mode=True)
        grid = phaser.summary().set_index("alpha")
        assert np.isinf(grid.loc[0.0, "max_cv"])
        assert np.isfinite(grid.drop(index=0.0)["max_cv"]).all()
        rec = phaser.recommend()
        assert rec["alpha"] > 0
        assert np.isfinite(rec["max_cv"])
        assert np.isfinite(phaser.recommended_schedule_median_cv_)


class TestDiscoveryReport:
    def test_raises_on_unidentifiable_plan(self):
        report = DiscoveryReport(
            history_df=EXACT.iloc[:104],
            plan_df=EXACT.iloc[104:],
            true_marginal_returns=MARGINAL_RETURNS,
            saturation=0.7,
            adstock=0.3,
        )
        with pytest.raises(ValueError, match="Use BudgetPhaser instead"):
            report.fit(fast_mode=True)
