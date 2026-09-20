"""Tests for the Redistribute strategy marker (_phaser.py).

Redistribute = round-robin blackout + recipient month + edge layer. Unlike
Blackout and a symmetric range it moves budget BETWEEN months, so what has
to hold exactly is per-channel conservation over each 12-month block, and
conservation of every month the channel's blackout/recipient pair doesn't
touch. The scratch phasing search found a conservation bug once when two
recipient months collided (loop 9), so collisions get explicit tests.
"""

import numpy as np
import pandas as pd
import pytest

import how_wrong_is_your_mmm
from how_wrong_is_your_mmm._dgp import simulate_spend
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    BudgetPhaser,
    Redistribute,
    _generate_phased_schedule,
    _get_month_labels,
    _redistribute_blocks,
    _resolve_channel_specs,
)

# Calendars with different month-length patterns, including 1-2 week
# partial first/last months (start dates from the scratch calendar loop).
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


def make_plan(n_weeks=52, n_channels=3, start="2024-01-01", seed=1):
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


class TestRedistributeMarker:
    def test_defaults(self):
        r = Redistribute()
        assert (r.dark_weeks, r.min_recipient_weeks, r.edge_cap_pct) == (4, 4, 15.0)

    def test_exported_at_top_level(self):
        assert how_wrong_is_your_mmm.Redistribute is Redistribute
        assert "Redistribute" in how_wrong_is_your_mmm.__all__

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_dark_weeks_validated(self, bad):
        with pytest.raises(ValueError, match="dark_weeks"):
            Redistribute(dark_weeks=bad)

    @pytest.mark.parametrize("bad", [0, -3, 1.5])
    def test_min_recipient_weeks_validated(self, bad):
        with pytest.raises(ValueError, match="min_recipient_weeks"):
            Redistribute(min_recipient_weeks=bad)

    @pytest.mark.parametrize("bad", [-1.0, 100.5])
    def test_edge_cap_validated(self, bad):
        with pytest.raises(ValueError, match="edge_cap_pct"):
            Redistribute(edge_cap_pct=bad)

    def test_repr_and_eq(self):
        assert Redistribute(3, 5, 40.0) == Redistribute(3, 5, 40.0)
        assert Redistribute(3, 5, 40.0) != Redistribute(3, 5, 60.0)
        assert Redistribute() != Blackout()
        assert "edge_cap_pct=40.0" in repr(Redistribute(edge_cap_pct=40.0))


class TestResolveChannelSpecs:
    def test_single_redistribute_broadcasts(self):
        r = Redistribute(edge_cap_pct=20.0)
        out = _resolve_channel_specs(r, ["a", "b"])
        assert out == {"a": r, "b": r}

    def test_mixed_dict(self):
        r = Redistribute()
        out = _resolve_channel_specs({"a": r, "b": 40.0, "c": Blackout()}, list("abc"))
        assert out["a"] is r
        assert out["b"] == (-40.0, 40.0)
        assert isinstance(out["c"], Blackout)

    def test_edge_cap_may_differ_per_channel(self):
        _resolve_channel_specs(
            {
                "a": Redistribute(edge_cap_pct=20.0),
                "b": Redistribute(edge_cap_pct=60.0),
            },
            ["a", "b"],
        )

    def test_shared_round_robin_params_must_agree(self):
        with pytest.raises(ValueError, match="share the same dark_weeks"):
            _resolve_channel_specs(
                {"a": Redistribute(dark_weeks=3), "b": Redistribute(dark_weeks=4)},
                ["a", "b"],
            )
        with pytest.raises(ValueError, match="share the same dark_weeks"):
            _resolve_channel_specs(
                {
                    "a": Redistribute(min_recipient_weeks=2),
                    "b": Redistribute(min_recipient_weeks=4),
                },
                ["a", "b"],
            )


class TestRedistributeBlocks:
    def _labels(self, n_months):
        idx = pd.date_range("2024-01-01", periods=n_months * 4, freq="W-MON")
        # one label per month regardless of week count: synthesise directly
        return pd.period_range("2024-01", periods=n_months, freq="M").repeat(
            4
        ).to_numpy(), idx

    @pytest.mark.parametrize(
        "n_months, expected_sizes",
        [
            (12, [12]),
            (8, [8]),  # shorter than a year: one block
            (24, [12, 12]),
            (36, [12, 12, 12]),
            (37, [12, 12, 13]),  # 1-month tail merged
            (41, [12, 12, 17]),  # 5-month tail merged
            (42, [12, 12, 12, 6]),  # 6-month tail kept
            (17, [17]),  # 5-month tail merged into the only block
        ],
    )
    def test_block_sizes(self, n_months, expected_sizes):
        labels, _ = self._labels(n_months)
        blocks = _redistribute_blocks(labels)
        assert [len(np.unique(labels[b])) for b in blocks] == expected_sizes

    def test_blocks_partition_all_rows_in_order(self):
        labels, _ = self._labels(37)
        blocks = _redistribute_blocks(labels)
        assert np.array_equal(np.concatenate(blocks), np.arange(len(labels)))


class TestConservation:
    @pytest.mark.parametrize("start", START_DATES)
    @pytest.mark.parametrize("cap", [0.0, 20.0, 80.0])
    @pytest.mark.parametrize("n_channels", [3, 10])
    def test_block_total_conserved_per_channel(self, start, cap, n_channels):
        plan = make_plan(52, n_channels, start)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=cap), seed=2)
        for rows in block_rows(labels):
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )

    @pytest.mark.parametrize("n_channels", [10, 20, 30])
    def test_conserved_when_channels_outnumber_months(self, n_channels):
        # More channels than eligible months forces channels to share a
        # blackout month AND a recipient month (the loop-9 failure mode).
        plan = make_plan(52, n_channels)
        labels = _get_month_labels(plan)
        for seed in range(6):
            out = phase(plan, Redistribute(edge_cap_pct=60.0), seed=seed)
            for rows in block_rows(labels):
                np.testing.assert_allclose(
                    out.iloc[rows].sum().to_numpy(),
                    plan.iloc[rows].sum().to_numpy(),
                    rtol=1e-12,
                )

    def test_multi_year_block_totals_conserved(self):
        plan = make_plan(156, 10)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=40.0), seed=1)
        blocks = block_rows(labels)
        assert len(blocks) == 3
        for rows in blocks:
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_edge_layer_preserves_post_redistribution_monthly_totals(self):
        plan = make_plan(52, 5)
        labels = _get_month_labels(plan)
        base = phase(plan, Redistribute(edge_cap_pct=0.0), seed=4)
        for cap in (20.0, 60.0, 100.0):
            out = phase(plan, Redistribute(edge_cap_pct=cap), seed=4)
            np.testing.assert_allclose(
                monthly(out, labels).to_numpy(),
                monthly(base, labels).to_numpy(),
                rtol=1e-12,
            )

    @pytest.mark.parametrize("start", START_DATES)
    def test_only_blackout_and_recipient_months_change(self, start):
        plan = make_plan(52, 6, start)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=30.0), seed=3)
        diff = (monthly(out, labels) - monthly(plan, labels)).abs()
        rel = diff / monthly(plan, labels)
        for rows in block_rows(labels):
            block_months = np.unique(labels[rows])
            touched = (rel.loc[block_months] > 1e-9).sum()
            # one blackout month + one recipient month per channel per block
            assert (touched <= 2).all()

    def test_no_negative_spend(self):
        for start in START_DATES:
            plan = make_plan(52, 10, start)
            for cap in (20.0, 80.0, 100.0):
                out = phase(plan, Redistribute(edge_cap_pct=cap), seed=7)
                assert (out.to_numpy() >= 0.0).all()
                assert np.isfinite(out.to_numpy()).all()


class TestRedistributeMechanics:
    """cap=0 isolates the round-robin blackout + recipient step."""

    def _analyse(self, plan, out, labels, dark_weeks=4):
        plan_m = monthly(plan, labels)
        out_m = monthly(out, labels)
        info = {}
        for ch in plan.columns:
            zeros = np.where(out[ch].to_numpy() == 0.0)[0]
            delta = out_m[ch] - plan_m[ch]
            info[ch] = {
                "zeros": zeros,
                "lost": delta[delta < -1e-9],
                "gained": delta[delta > 1e-9],
            }
        return info

    @pytest.mark.parametrize("start", START_DATES)
    def test_one_consecutive_dark_run_per_channel(self, start):
        plan = make_plan(52, 4, start)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(dark_weeks=4, edge_cap_pct=0.0), seed=1)
        for ch, info in self._analyse(plan, out, labels).items():
            zeros = info["zeros"]
            assert len(zeros) == 4
            assert np.array_equal(zeros, np.arange(zeros[0], zeros[0] + 4))
            assert len(np.unique(labels[zeros])) == 1  # inside a single month

    @pytest.mark.parametrize("start", START_DATES)
    def test_freed_budget_lands_in_one_recipient_month(self, start):
        plan = make_plan(52, 4, start)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=1)
        for ch, info in self._analyse(plan, out, labels).items():
            assert len(info["lost"]) == 1 and len(info["gained"]) == 1
            freed = plan[ch].to_numpy()[info["zeros"]].sum()
            assert info["lost"].iloc[0] == pytest.approx(-freed, rel=1e-9)
            assert info["gained"].iloc[0] == pytest.approx(freed, rel=1e-9)
            assert info["lost"].index[0] != info["gained"].index[0]

    @pytest.mark.parametrize("start", START_DATES)
    def test_recipient_month_has_at_least_min_recipient_weeks(self, start):
        plan = make_plan(52, 10, start)
        labels = _get_month_labels(plan)
        month_len = pd.Series(labels).value_counts()
        for mrw in (2, 4):
            out = phase(
                plan, Redistribute(min_recipient_weeks=mrw, edge_cap_pct=0.0), seed=5
            )
            for info in self._analyse(plan, out, labels).values():
                for m in info["gained"].index:
                    assert month_len[m] >= mrw

    def test_redistribution_is_proportional_to_own_weekly_spend(self):
        plan = make_plan(52, 3)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=1)
        for ch, info in self._analyse(plan, out, labels).items():
            m = info["gained"].index[0]
            rows = np.where(labels == m)[0]
            ratio = out[ch].to_numpy()[rows] / plan[ch].to_numpy()[rows]
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
            assert ratio[0] > 1.0

    def test_blackout_months_distinct_when_months_allow(self):
        # 3 channels, plenty of >4-week months: round-robin must not put
        # two channels' blackouts in the same month.
        plan = make_plan(52, 3, "2023-05-01")
        labels = _get_month_labels(plan)
        month_len = pd.Series(labels).value_counts()
        assert (month_len > 4).sum() >= 3
        for seed in range(10):
            out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=seed)
            bo = [
                labels[info["zeros"][0]]
                for info in self._analyse(plan, out, labels).values()
            ]
            assert len(set(bo)) == 3

    def test_dark_weeks_capped_below_month_length(self):
        plan = make_plan(52, 3)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(dark_weeks=3, edge_cap_pct=0.0), seed=1)
        month_len = pd.Series(labels).value_counts()
        for info in self._analyse(plan, out, labels).values():
            assert len(info["zeros"]) == 3
            assert month_len[labels[info["zeros"][0]]] > 3

    def test_dark_weeks_stay_dark_under_edge_layer(self):
        plan = make_plan(52, 4)
        base = phase(plan, Redistribute(edge_cap_pct=0.0), seed=1)
        out = phase(plan, Redistribute(edge_cap_pct=60.0), seed=1)
        assert ((base.to_numpy() == 0.0) <= (out.to_numpy() == 0.0)).all()

    def test_edge_layer_moves_active_weeks(self):
        plan = make_plan(52, 4)
        base = phase(plan, Redistribute(edge_cap_pct=0.0), seed=1)
        out = phase(plan, Redistribute(edge_cap_pct=60.0), seed=1)
        assert not np.allclose(out.to_numpy(), base.to_numpy())

    def test_cap_zero_is_pure_redistribution_and_deterministic(self):
        plan = make_plan(52, 4)
        a = phase(plan, Redistribute(edge_cap_pct=0.0), seed=9)
        b = phase(plan, Redistribute(edge_cap_pct=0.0), seed=9)
        pd.testing.assert_frame_equal(a, b)


class TestDeterminismAndAlpha:
    def test_same_seed_same_schedule(self):
        plan = make_plan(156, 10)
        spec = Redistribute(edge_cap_pct=60.0)
        pd.testing.assert_frame_equal(
            phase(plan, spec, seed=3), phase(plan, spec, seed=3)
        )

    def test_different_seed_different_schedule(self):
        plan = make_plan(52, 10)
        spec = Redistribute(edge_cap_pct=60.0)
        assert not np.allclose(
            phase(plan, spec, seed=3).to_numpy(), phase(plan, spec, seed=4).to_numpy()
        )

    def test_alpha_zero_is_unchanged(self):
        plan = make_plan(52, 5)
        out = phase(plan, Redistribute(edge_cap_pct=60.0), alpha=0.0)
        pd.testing.assert_frame_equal(out, plan)

    def test_alpha_scales_edge_layer_only(self):
        plan = make_plan(52, 4)
        half = phase(plan, Redistribute(edge_cap_pct=60.0), alpha=0.5, seed=1)
        # still redistributes (dark weeks exist) and still conserves
        assert (half.to_numpy() == 0.0).sum() == 4 * 4
        np.testing.assert_allclose(
            half.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_input_not_mutated(self):
        plan = make_plan(52, 4)
        before = plan.copy()
        phase(plan, Redistribute(edge_cap_pct=60.0), seed=1)
        pd.testing.assert_frame_equal(plan, before)


class TestEdgeCases:
    @pytest.mark.parametrize("start", START_DATES)
    @pytest.mark.parametrize("n_weeks", [48, 52, 56])
    def test_partial_month_calendars_do_not_crash(self, start, n_weeks):
        plan = make_plan(n_weeks, 10, start)
        labels = _get_month_labels(plan)
        out = phase(plan, Redistribute(edge_cap_pct=60.0), seed=0)
        assert out.shape == plan.shape
        assert (out.to_numpy() >= 0).all()
        for rows in block_rows(labels):
            np.testing.assert_allclose(
                out.iloc[rows].sum().to_numpy(),
                plan.iloc[rows].sum().to_numpy(),
                rtol=1e-12,
            )

    def test_two_week_partial_month_never_recipient_at_default_floor(self):
        # 2023-01-30 starts on the last Monday of January -> 1-week stub.
        plan = make_plan(52, 10, "2023-01-30")
        labels = _get_month_labels(plan)
        month_len = pd.Series(labels).value_counts()
        assert month_len.min() <= 2
        for seed in range(8):
            base = phase(plan, Redistribute(edge_cap_pct=0.0), seed=seed)
            gained = monthly(base, labels) > monthly(plan, labels) * (1 + 1e-9)
            for ch in plan.columns:
                for m in gained.index[gained[ch]]:
                    assert month_len[m] >= 4

    def test_no_eligible_blackout_month_leaves_plan_unchanged(self):
        # 3 weeks in a single month: nothing is longer than dark_weeks=4.
        plan = make_plan(3, 3, "2024-01-01")
        out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=0)
        pd.testing.assert_frame_equal(out, plan)

    def test_only_one_eligible_month_means_no_recipient_left(self):
        # One 5-week month: eligible to black out, but the only eligible
        # recipient is itself, so the channel must stay at plan.
        plan = make_plan(5, 3, "2024-01-01")
        assert len(np.unique(_get_month_labels(plan))) == 1
        out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=0)
        pd.testing.assert_frame_equal(out, plan)

    def test_few_eligible_months_still_conserves(self):
        # ~8 weeks -> 2 months. Both eligible; must conserve.
        plan = make_plan(9, 6, "2024-01-01")
        out = phase(plan, Redistribute(edge_cap_pct=60.0), seed=0)
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )
        assert (out.to_numpy() >= 0).all()

    def test_short_min_recipient_and_large_dark_weeks_do_not_crash(self):
        plan = make_plan(52, 4)
        for dw in (1, 2, 5, 8):
            out = phase(
                plan, Redistribute(dark_weeks=dw, min_recipient_weeks=1), seed=0
            )
            np.testing.assert_allclose(
                out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
            )

    def test_all_zero_channel_stays_zero(self):
        plan = make_plan(52, 3).copy()
        # zero out a whole channel: nothing to free, nothing to receive.
        plan["ch0"] = 0.0
        out = phase(plan, Redistribute(edge_cap_pct=0.0), seed=0)
        assert (out["ch0"].to_numpy() == 0.0).all()
        np.testing.assert_allclose(
            out.sum().to_numpy(), plan.sum().to_numpy(), rtol=1e-12
        )

    def test_zero_recipient_month_with_freed_budget_conserves(self):
        # Force a recipient month whose own spend is 0 for one channel by
        # zeroing every month but the first block's blackout candidate --
        # the even-split fallback must still conserve.
        plan = make_plan(52, 1, "2024-01-01")
        labels = _get_month_labels(plan)
        for seed in range(20):
            p = plan.copy()
            rng = np.random.default_rng(seed)
            # zero a random subset of months (keep at least the blackout one)
            months = np.unique(labels)
            for m in rng.choice(months, size=len(months) // 2, replace=False):
                p.loc[labels == m, "ch0"] = 0.0
            out = phase(p, Redistribute(edge_cap_pct=0.0), seed=seed)
            np.testing.assert_allclose(
                out.sum().to_numpy(), p.sum().to_numpy(), rtol=1e-12, atol=1e-9
            )
            assert (out.to_numpy() >= 0).all()


class TestMixedSpecs:
    def test_locked_channel_untouched(self):
        plan = make_plan(52, 3)
        spec = {"ch0": Redistribute(edge_cap_pct=40.0), "ch1": 0.0, "ch2": 0.0}
        out = phase(plan, spec, seed=0)
        pd.testing.assert_series_equal(out["ch1"], plan["ch1"])
        pd.testing.assert_series_equal(out["ch2"], plan["ch2"])
        assert not np.allclose(out["ch0"], plan["ch0"])

    def test_mixed_with_float_and_blackout_conserves(self):
        plan = make_plan(52, 4)
        labels = _get_month_labels(plan)
        spec = {
            "ch0": Redistribute(edge_cap_pct=40.0),
            "ch1": Redistribute(edge_cap_pct=80.0),
            "ch2": 40.0,
            "ch3": Blackout(max_dark_weeks_per_month=1),
        }
        out = phase(plan, spec, seed=0)
        # float and Blackout channels: every MONTH conserved
        for ch in ("ch2", "ch3"):
            np.testing.assert_allclose(
                monthly(out, labels)[ch].to_numpy(),
                monthly(plan, labels)[ch].to_numpy(),
                rtol=1e-12,
            )
        # Redistribute channels: block total conserved
        for ch in ("ch0", "ch1"):
            np.testing.assert_allclose(out[ch].sum(), plan[ch].sum(), rtol=1e-12)

    def test_existing_specs_unchanged_by_redistribute_support(self):
        # A float/Blackout-only spec must not touch the Redistribute path.
        plan = make_plan(52, 3)
        labels = _get_month_labels(plan)
        for spec in (40.0, Blackout(max_dark_weeks_per_month=1)):
            out = phase(plan, spec, seed=0)
            np.testing.assert_allclose(
                monthly(out, labels).to_numpy(),
                monthly(plan, labels).to_numpy(),
                rtol=1e-12,
            )


class TestBudgetPhaserAcceptsRedistribute:
    def test_constructs_and_reports_nonzero_monthly_deviation(self):
        history = make_plan(104, 3, "2022-01-03", seed=0)
        plan = make_plan(52, 3, "2024-01-01", seed=1)
        phaser = BudgetPhaser(
            history,
            plan,
            true_marginal_returns={"ch0": 0.5, "ch1": 1.0, "ch2": 1.5},
            max_weekly_deviation_pct=Redistribute(edge_cap_pct=20.0),
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
