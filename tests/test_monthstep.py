"""Tests for the MonthStep strategy marker (_phaser.py).

MonthStep scales each month's whole spend by 1 +/- step, with the sign taken
from a Hadamard column (balanced, mutually orthogonal across channels). It
moves budget BETWEEN months, so what has to hold exactly is per-channel
conservation over each 12-month block; monthly totals change by design.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import how_wrong_is_your_mmm
from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    BudgetPhaser,
    MonthStep,
    Redistribute,
    _generate_phased_schedule,
    _get_month_labels,
    _month_step_hadamard,
    _redistribute_blocks,
    _resolve_channel_specs,
)

# Calendars with different month-length patterns, including 1-2 week partial
# first/last months.
START_DATES = [
    "2023-01-02",
    "2023-01-09",
    "2023-01-16",
    "2023-01-23",
    "2023-01-30",
    "2023-02-06",
    "2023-03-06",
    "2023-05-01",
]
# 52 weeks from this Monday span exactly Jan..Dec 2024: one 12-month block
# of 12 full months, the clean case for balance/orthogonality checks.
ALIGNED_START = "2024-01-01"


def make_plan(n_weeks=52, n_channels=3, start=ALIGNED_START, seed=1):
    channels = [f"ch{i}" for i in range(n_channels)]
    return simulate_spend(
        n_obs=n_weeks,
        correlation=0.7,
        channels=channels,
        seed=seed,
        start_date=start,
    )


def phase(plan, spec, seed=0, alpha=1.0):
    return _generate_phased_schedule(
        plan,
        _get_month_labels(plan),
        alpha=alpha,
        max_weekly_deviation_pct=spec,
        seed=seed,
    )


def monthly(df, labels):
    return df.groupby(labels).sum()


def block_rows(labels):
    return _redistribute_blocks(labels)


def month_signs(plan, out):
    """Recover each (month, channel) step sign: every full month's ratio to
    plan is c*(1 +/- step) for a per-channel constant c, and a balanced
    column has mean ratio exactly c, so the sign is ratio vs that mean."""
    labels = _get_month_labels(plan)
    ratio = monthly(out, labels) / monthly(plan, labels)
    return np.sign(ratio - ratio.mean()).to_numpy()


class TestMonthStepMarker:
    def test_default(self):
        assert MonthStep().step_pct == 40.0

    def test_exported_at_top_level(self):
        assert how_wrong_is_your_mmm.MonthStep is MonthStep
        assert "MonthStep" in how_wrong_is_your_mmm.__all__

    @pytest.mark.parametrize("bad", [-0.1, 100.5, 250])
    def test_step_validated(self, bad):
        with pytest.raises(ValueError, match="step_pct"):
            MonthStep(step_pct=bad)

    @pytest.mark.parametrize("ok", [0, 20, 100])
    def test_step_bounds_inclusive(self, ok):
        assert MonthStep(step_pct=ok).step_pct == float(ok)

    def test_repr_and_eq(self):
        assert MonthStep(40.0) == MonthStep(40.0)
        assert MonthStep(40.0) != MonthStep(60.0)
        assert MonthStep() != Redistribute()
        assert MonthStep() != 40.0
        assert "step_pct=60.0" in repr(MonthStep(60.0))


class TestResolveChannelSpecs:
    def test_single_month_step_broadcasts(self):
        m = MonthStep(20.0)
        assert _resolve_channel_specs(m, ["a", "b"]) == {"a": m, "b": m}

    def test_mixed_dict(self):
        m = MonthStep()
        out = _resolve_channel_specs(
            {"a": m, "b": 40.0, "c": Blackout(), "d": Redistribute()}, list("abcd")
        )
        assert out["a"] is m
        assert out["b"] == (-40.0, 40.0)
        assert isinstance(out["c"], Blackout)
        assert isinstance(out["d"], Redistribute)

    def test_step_may_differ_per_channel(self):
        _resolve_channel_specs(
            {"a": MonthStep(20.0), "b": MonthStep(60.0)},
            ["a", "b"],
        )


class TestHadamardColumns:
    def test_shape_and_entries(self):
        h = _month_step_hadamard()
        assert h.shape == (12, 11)
        assert set(np.unique(h)) == {-1.0, 1.0}

    def test_columns_balanced(self):
        assert (_month_step_hadamard().sum(axis=0) == 0).all()

    def test_columns_orthogonal(self):
        h = _month_step_hadamard()
        np.testing.assert_array_equal(h.T @ h, 12 * np.eye(11))

    def test_cached(self):
        assert _month_step_hadamard() is _month_step_hadamard()


class TestConservation:
    @pytest.mark.parametrize("start", START_DATES)
    @pytest.mark.parametrize("step", [0.0, 20.0, 60.0, 100.0])
    @pytest.mark.parametrize("n_channels", [3, 10])
    def test_block_total_conserved_per_channel(self, start, step, n_channels):
        plan = make_plan(52, n_channels, start)
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(step), seed=2)
        for rows in block_rows(labels):
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )

    @pytest.mark.parametrize("n_channels", [11, 12, 13, 25])
    def test_conserved_beyond_hadamard_columns(self, n_channels):
        plan = make_plan(52, n_channels)
        labels = _get_month_labels(plan)
        with (
            pytest.warns(UserWarning) if n_channels > 11 else warnings.catch_warnings()
        ):
            out = phase(plan, MonthStep(60.0), seed=1)
        for rows in block_rows(labels):
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )

    @pytest.mark.parametrize("n_weeks", [104, 156, 160])
    def test_multi_year_block_totals_conserved(self, n_weeks):
        plan = make_plan(n_weeks, 10)
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(40.0), seed=1)
        blocks = block_rows(labels)
        assert len(blocks) >= 2
        for rows in blocks:
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_monthly_totals_do_change(self):
        # by design: only the block total is preserved
        plan = make_plan()
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(40.0))
        rel = (monthly(out, labels) / monthly(plan, labels) - 1).abs()
        assert (rel.max() > 0.3).all()

    def test_no_negative_spend(self):
        for start in START_DATES:
            plan = make_plan(52, 10, start)
            for step in (20.0, 60.0, 100.0):
                out = phase(plan, MonthStep(step), seed=7)
                assert (out.to_numpy() >= 0.0).all()
                assert np.isfinite(out.to_numpy()).all()

    def test_weekly_shape_within_a_month_unchanged(self):
        plan = make_plan(52, 4)
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(60.0), seed=3)
        ratio = out / plan
        for m in np.unique(labels):
            rows = np.where(labels == m)[0]
            for ch in plan.columns:
                r = ratio[ch].to_numpy()[rows]
                np.testing.assert_allclose(r, r[0], rtol=1e-12)

    def test_at_most_two_levels_per_channel(self):
        # every full month is c*(1+step) or c*(1-step): one change per month
        plan = make_plan(52, 5)
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(40.0), seed=4)
        ratio = monthly(out, labels) / monthly(plan, labels)
        for ch in plan.columns:
            assert len(np.unique(np.round(ratio[ch], 9))) == 2


class TestBalanceAndOrthogonality:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n_channels", [2, 5, 11])
    def test_full_block_columns_balanced_and_orthogonal(self, seed, n_channels):
        plan = make_plan(52, n_channels)
        assert len(np.unique(_get_month_labels(plan))) == 12
        out = phase(plan, MonthStep(40.0), seed=seed)
        s = month_signs(plan, out)
        assert s.shape == (12, n_channels)
        assert (np.abs(s) == 1).all()
        # each channel: equal up and down months
        assert (s.sum(axis=0) == 0).all()
        # sign columns mutually orthogonal within the block
        np.testing.assert_array_equal(s.T @ s, 12 * np.eye(n_channels))

    def test_orthogonal_across_all_three_years(self):
        plan = make_plan(156, 8)
        out = phase(plan, MonthStep(40.0), seed=1)
        s = month_signs(plan, out)
        for a in range(0, 36, 12):
            block = s[a : a + 12]
            np.testing.assert_array_equal(block.T @ block, 12 * np.eye(8))

    def test_each_block_gets_a_fresh_shuffle(self):
        plan = make_plan(156, 6)
        s = month_signs(plan, phase(plan, MonthStep(40.0), seed=1))
        assert not np.array_equal(s[0:12], s[12:24])
        assert not np.array_equal(s[12:24], s[24:36])

    def test_more_than_eleven_channels_warns_and_stays_balanced(self):
        plan = make_plan(52, 14)
        with pytest.warns(UserWarning, match="11th"):
            out = phase(plan, MonthStep(40.0), seed=0)
        s = month_signs(plan, out)
        assert (s.sum(axis=0) == 0).all()  # every column still 6 up / 6 down
        gram = s.T @ s
        # the first 11 stay orthogonal; the extras lose orthogonality
        assert not np.array_equal(gram, 12 * np.eye(14))

    def test_eleven_or_fewer_channels_do_not_warn(self):
        plan = make_plan(52, 11)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            phase(plan, MonthStep(40.0), seed=0)

    def test_orthogonality_is_among_month_step_channels_only(self):
        plan = make_plan(52, 5)
        spec = {
            "ch0": MonthStep(40.0),
            "ch1": 40.0,
            "ch2": MonthStep(40.0),
            "ch3": Blackout(max_dark_weeks_per_month=1),
            "ch4": MonthStep(40.0),
        }
        out = phase(plan, spec, seed=2)
        s = month_signs(plan[["ch0", "ch2", "ch4"]], out[["ch0", "ch2", "ch4"]])
        np.testing.assert_array_equal(s.T @ s, 12 * np.eye(3))


class TestDeterminismAndAlpha:
    def test_same_seed_same_schedule(self):
        plan = make_plan(156, 10)
        spec = MonthStep(60.0)
        pd.testing.assert_frame_equal(
            phase(plan, spec, seed=3), phase(plan, spec, seed=3)
        )

    def test_different_seed_different_schedule(self):
        plan = make_plan(52, 10)
        spec = MonthStep(60.0)
        assert not np.allclose(
            phase(plan, spec, seed=3).to_numpy(), phase(plan, spec, seed=4).to_numpy()
        )

    def test_alpha_zero_is_unchanged(self):
        plan = make_plan(52, 5)
        pd.testing.assert_frame_equal(phase(plan, MonthStep(60.0), alpha=0.0), plan)

    def test_step_zero_is_unchanged(self):
        plan = make_plan(52, 5)
        out = phase(plan, MonthStep(0.0))
        np.testing.assert_allclose(out.to_numpy(), plan.to_numpy(), rtol=1e-12)

    def test_alpha_scales_the_step(self):
        # alpha=0.5 at 60% is the same draw as alpha=1 at 30%
        plan = make_plan(52, 4)
        a = phase(plan, MonthStep(60.0), alpha=0.5, seed=1)
        b = phase(plan, MonthStep(30.0), alpha=1.0, seed=1)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)

    def test_larger_step_moves_more(self):
        plan = make_plan(52, 4)
        d20 = (phase(plan, MonthStep(20.0)) - plan).abs().to_numpy().sum()
        d60 = (phase(plan, MonthStep(60.0)) - plan).abs().to_numpy().sum()
        assert d60 > d20

    def test_input_not_mutated(self):
        plan = make_plan(52, 4)
        before = plan.copy()
        phase(plan, MonthStep(60.0), seed=1)
        pd.testing.assert_frame_equal(plan, before)


class TestPartialMonthsAndBlocks:
    def test_one_week_partial_month_is_left_at_plan(self):
        # 2023-01-30 start: January holds a single week.
        plan = make_plan(52, 4, "2023-01-30")
        labels = _get_month_labels(plan)
        month_len = pd.Series(labels).value_counts()
        stub = month_len.index[month_len == 1][0]
        assert month_len[stub] == 1
        out = phase(plan, MonthStep(40.0), seed=1)
        ratio = monthly(out, labels) / monthly(plan, labels)
        for ch in plan.columns:
            r = ratio[ch]
            full = r.drop(stub)
            # stub sits at c, the full months at c*(1 +/- 0.4)
            assert full.max() / r[stub] == pytest.approx(1.4)
            assert full.min() / r[stub] == pytest.approx(0.6)

    def test_thirteen_month_block(self):
        # 52 weeks from 2023-01-30 touch 13 calendar months -> one block
        plan = make_plan(52, 6, "2023-01-30")
        labels = _get_month_labels(plan)
        assert len(np.unique(labels)) == 13
        assert len(block_rows(labels)) == 1
        out = phase(plan, MonthStep(60.0), seed=2)
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )
        assert (out.to_numpy() >= 0).all()

    @pytest.mark.parametrize("start", START_DATES)
    @pytest.mark.parametrize("n_weeks", [26, 48, 56, 100])
    def test_plans_not_a_multiple_of_52_weeks(self, start, n_weeks):
        plan = make_plan(n_weeks, 10, start)
        labels = _get_month_labels(plan)
        out = phase(plan, MonthStep(60.0), seed=0)
        assert out.shape == plan.shape
        assert (out.to_numpy() >= 0).all()
        for rows in block_rows(labels):
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )

    def test_single_month_plan_is_unchanged(self):
        # one month: the step is undone by the block rescale
        plan = make_plan(3, 3, "2024-01-01")
        assert len(np.unique(_get_month_labels(plan))) == 1
        np.testing.assert_allclose(
            phase(plan, MonthStep(60.0)).to_numpy(), plan.to_numpy(), rtol=1e-12
        )

    def test_one_week_plan_does_not_crash(self):
        plan = make_plan(1, 3, "2024-01-01")
        np.testing.assert_allclose(
            phase(plan, MonthStep(60.0)).to_numpy(), plan.to_numpy(), rtol=1e-12
        )

    def test_short_final_block_is_stepped_on_its_own(self):
        # 42 months -> blocks of 12, 12, 12 and a 6-month tail (kept)
        idx = pd.date_range("2024-01-01", periods=42 * 4, freq="W-MON")
        labels = pd.period_range("2024-01", periods=42, freq="M").repeat(4).to_numpy()
        plan = pd.DataFrame(
            np.random.default_rng(0).uniform(50, 150, size=(len(idx), 3)),
            index=idx,
            columns=list("abc"),
        )
        out = _generate_phased_schedule(plan, labels, 1.0, MonthStep(40.0), 0)
        blocks = block_rows(labels)
        assert [len(np.unique(labels[b])) for b in blocks] == [12, 12, 12, 6]
        for rows in blocks:
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )
        assert not np.allclose(out.iloc[-24:].to_numpy(), plan.iloc[-24:].to_numpy())


class TestEdgeCases:
    def test_all_zero_channel_stays_zero_without_nan(self):
        plan = make_plan(52, 3).copy()
        plan["ch0"] = 0.0
        out = phase(plan, MonthStep(60.0))
        assert (out["ch0"].to_numpy() == 0.0).all()
        assert np.isfinite(out.to_numpy()).all()
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_zero_spend_months_stay_zero(self):
        plan = make_plan(52, 3).copy()
        labels = _get_month_labels(plan)
        plan.loc[labels == np.unique(labels)[3], "ch1"] = 0.0
        out = phase(plan, MonthStep(60.0), seed=2)
        assert (out.loc[labels == np.unique(labels)[3], "ch1"] == 0.0).all()
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_step_hundred_can_zero_a_month_but_never_goes_negative(self):
        plan = make_plan(52, 3)
        out = phase(plan, MonthStep(100.0), seed=0)
        labels = _get_month_labels(plan)
        assert (monthly(out, labels).min() == 0.0).all()
        assert (out.to_numpy() >= 0.0).all()


class TestMixedSpecs:
    def test_locked_channel_untouched(self):
        plan = make_plan(52, 3)
        out = phase(plan, {"ch0": MonthStep(40.0), "ch1": 0.0, "ch2": 0.0})
        pd.testing.assert_series_equal(out["ch1"], plan["ch1"])
        pd.testing.assert_series_equal(out["ch2"], plan["ch2"])
        assert not np.allclose(out["ch0"], plan["ch0"])

    def test_float_and_blackout_channels_keep_monthly_totals(self):
        plan = make_plan(52, 4)
        labels = _get_month_labels(plan)
        spec = {
            "ch0": MonthStep(40.0),
            "ch1": MonthStep(80.0),
            "ch2": 40.0,
            "ch3": Blackout(max_dark_weeks_per_month=1),
        }
        out = phase(plan, spec, seed=0)
        for ch in ("ch2", "ch3"):
            np.testing.assert_allclose(
                monthly(out, labels)[ch].to_numpy(),
                monthly(plan, labels)[ch].to_numpy(),
                rtol=1e-12,
            )
        for ch in ("ch0", "ch1"):
            np.testing.assert_allclose(out[ch].sum(), plan[ch].sum(), rtol=1e-12)

    def test_month_step_and_redistribute_together_conserve(self):
        plan = make_plan(52, 4)
        spec = {
            "ch0": MonthStep(40.0),
            "ch1": MonthStep(60.0),
            "ch2": Redistribute(edge_cap_pct=20.0),
            "ch3": Redistribute(edge_cap_pct=20.0),
        }
        out = phase(plan, spec, seed=1)
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )
        assert (out.to_numpy() >= 0).all()

    def test_existing_specs_unchanged_by_month_step_support(self):
        plan = make_plan(52, 3)
        labels = _get_month_labels(plan)
        for spec in (40.0, Blackout(max_dark_weeks_per_month=1)):
            out = phase(plan, spec, seed=0)
            np.testing.assert_allclose(
                monthly(out, labels).to_numpy(),
                monthly(plan, labels).to_numpy(),
                rtol=1e-12,
            )


class TestBudgetPhaserAcceptsMonthStep:
    def test_constructs_and_reports_nonzero_monthly_deviation(self):
        history = make_plan(104, 3, "2022-01-03", seed=0)
        plan = make_plan(52, 3, "2024-01-01", seed=1)
        phaser = BudgetPhaser(
            history,
            plan,
            true_marginal_returns={"ch0": 0.5, "ch1": 1.0, "ch2": 1.5},
            max_weekly_deviation_pct=MonthStep(20.0),
        )
        row = phaser._evaluate_spec_at_alpha(
            phaser.max_weekly_deviation_pct,
            alpha=1.0,
            n_sims=2,
            n_phasing_seeds=1,
            seed_offset=0,
        )
        # by design: budget moves between months (annual total conserved)
        assert row["max_monthly_deviation_pct"] > 0.0
