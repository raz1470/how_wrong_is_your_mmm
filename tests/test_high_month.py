"""Tests for Redistribute's optional high month (high_month_pct).

The high month is a third month per channel per 12-month block, placed in a
month the blackout/recipient round-robin left untouched, scaled to
(1 + high_month_pct/100) x plan and funded by the same % cut to every other
untouched month of that channel. What must hold exactly: block totals,
the blackout and recipient months are unchanged by it, and
high_month_pct=0 is draw-for-draw identical to plain Redistribute.
"""

import numpy as np
import pytest

from how_wrong_is_your_mmm._phaser import (
    Blackout,
    Redistribute,
    _generate_phased_schedule,
    _get_month_labels,
    _redistribute_blocks,
)
from tests.test_redistribute import START_DATES, make_plan, monthly, phase


def month_ratio(plan, sched):
    labels = _get_month_labels(plan)
    return monthly(sched, labels) / monthly(plan, labels)


def no_edge(h=150.0, **kw):
    return Redistribute(edge_cap_pct=0.0, high_month_pct=h, **kw)


class TestMarker:
    def test_default_is_off(self):
        assert Redistribute().high_month_pct == 0.0

    @pytest.mark.parametrize("bad", [-1.0, 300.5])
    def test_validated(self, bad):
        with pytest.raises(ValueError, match="high_month_pct"):
            Redistribute(high_month_pct=bad)

    @pytest.mark.parametrize("ok", [0.0, 150.0, 300.0])
    def test_bounds_inclusive(self, ok):
        assert Redistribute(high_month_pct=ok).high_month_pct == ok

    def test_repr_and_eq(self):
        assert Redistribute(high_month_pct=150.0) == Redistribute(high_month_pct=150.0)
        assert Redistribute(high_month_pct=150.0) != Redistribute()
        assert "high_month_pct=150.0" in repr(Redistribute(high_month_pct=150.0))
        assert "high_month_pct" not in repr(Redistribute())


class TestZeroIsIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_bit_identical_to_plain_redistribute(self, seed):
        plan = make_plan(n_channels=4)
        a = phase(plan, Redistribute(edge_cap_pct=20.0), seed=seed)
        b = phase(plan, Redistribute(edge_cap_pct=20.0, high_month_pct=0.0), seed=seed)
        assert np.array_equal(a.to_numpy(), b.to_numpy())

    def test_other_channels_unmoved_by_a_high_month_elsewhere(self):
        # The high month has its own RNG stream: adding one on channel 0
        # leaves every other channel's schedule exactly as it was.
        plan = make_plan(n_channels=4)
        base = {c: Redistribute(edge_cap_pct=20.0) for c in plan}
        with_high = dict(base)
        with_high["ch0"] = Redistribute(edge_cap_pct=20.0, high_month_pct=150.0)
        a = phase(plan, base, seed=3)
        b = phase(plan, with_high, seed=3)
        assert np.array_equal(a[["ch1", "ch2", "ch3"]], b[["ch1", "ch2", "ch3"]])
        assert not np.allclose(a["ch0"], b["ch0"])


class TestConservation:
    @pytest.mark.parametrize("start", START_DATES)
    @pytest.mark.parametrize("n_weeks", [52, 104, 156, 70])
    def test_block_totals_preserved(self, start, n_weeks):
        plan = make_plan(n_weeks=n_weeks, n_channels=4, start=start)
        sched = phase(plan, Redistribute(edge_cap_pct=20.0, high_month_pct=150.0))
        labels = _get_month_labels(plan)
        for rows in _redistribute_blocks(labels):
            assert np.allclose(sched.iloc[rows].sum(), plan.iloc[rows].sum(), rtol=1e-9)

    @pytest.mark.parametrize("seed", range(5))
    def test_edge_layer_keeps_month_totals_of_the_high_schedule(self, seed):
        plan = make_plan(n_channels=4)
        without_edge = phase(plan, no_edge(), seed=seed)
        with_edge = phase(
            plan, Redistribute(edge_cap_pct=20.0, high_month_pct=150.0), seed=seed
        )
        labels = _get_month_labels(plan)
        assert np.allclose(
            monthly(with_edge, labels), monthly(without_edge, labels), rtol=1e-9
        )

    def test_no_negative_spend(self):
        for seed in range(10):
            plan = make_plan(n_channels=6, seed=seed)
            sched = phase(plan, no_edge(300.0), seed=seed)
            assert (sched.to_numpy() >= 0).all()


class TestMechanics:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("h", [50.0, 150.0, 300.0])
    def test_exactly_one_high_month_at_the_target_multiple(self, seed, h):
        plan = make_plan(n_channels=4, seed=seed)
        ratio = month_ratio(plan, phase(plan, no_edge(h), seed=seed))
        for c in plan:
            assert np.isclose(ratio[c], 1 + h / 100, rtol=1e-9).sum() == 1

    @pytest.mark.parametrize("seed", range(8))
    def test_blackout_and_recipient_months_untouched(self, seed):
        plan = make_plan(n_channels=4, seed=seed)
        plain = month_ratio(
            plan, phase(plan, Redistribute(edge_cap_pct=0.0), seed=seed)
        )
        high = month_ratio(plan, phase(plan, no_edge(), seed=seed))
        special = ~np.isclose(plain, 1.0)  # blackout + recipient months
        assert (special.sum(axis=0) == 2).all()
        assert np.allclose(
            high.to_numpy()[special], plain.to_numpy()[special], rtol=1e-9
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_funding_is_one_equal_cut(self, seed):
        plan = make_plan(n_channels=4, seed=seed)
        plain = month_ratio(
            plan, phase(plan, Redistribute(edge_cap_pct=0.0), seed=seed)
        )
        high = month_ratio(plan, phase(plan, no_edge(), seed=seed))
        for c in plan:
            untouched = np.isclose(plain[c], 1.0)
            funded = high[c][untouched & ~np.isclose(high[c], 2.5)]
            assert len(funded) >= 1
            assert np.allclose(funded, funded.iloc[0])
            assert funded.iloc[0] < 1.0

    @pytest.mark.parametrize("n_channels", [2, 4, 8, 10])
    @pytest.mark.parametrize("seed", range(4))
    def test_channels_peak_in_different_months(self, n_channels, seed):
        plan = make_plan(n_channels=n_channels, seed=seed)
        ratio = month_ratio(plan, phase(plan, no_edge(), seed=seed))
        high_months = [ratio.index[np.isclose(ratio[c], 2.5)][0] for c in plan]
        assert len(set(high_months)) == n_channels

    def test_deterministic_per_seed_and_varies_across_seeds(self):
        plan = make_plan(n_channels=4)
        spec = Redistribute(edge_cap_pct=20.0, high_month_pct=150.0)
        assert np.array_equal(phase(plan, spec, seed=5), phase(plan, spec, seed=5))
        assert not np.allclose(phase(plan, spec, seed=5), phase(plan, spec, seed=6))

    def test_one_high_month_per_block(self):
        plan = make_plan(n_weeks=156, n_channels=3)
        ratio = month_ratio(plan, phase(plan, no_edge(), seed=2))
        assert (np.isclose(ratio, 2.5).sum(axis=0) == 3).all()

    def test_alpha_zero_changes_nothing(self):
        plan = make_plan(n_channels=3)
        out = phase(plan, no_edge(), alpha=0.0)
        assert np.allclose(out, plan)


class TestTargetingAndMixing:
    def test_high_month_on_one_channel_only(self):
        plan = make_plan(n_channels=4)
        spec = {c: Redistribute(edge_cap_pct=0.0) for c in plan}
        spec["ch0"] = no_edge()
        ratio = month_ratio(plan, phase(plan, spec, seed=1))
        assert np.isclose(ratio["ch0"], 2.5).sum() == 1
        for c in ["ch1", "ch2", "ch3"]:
            assert not np.isclose(ratio[c], 2.5).any()

    def test_mixes_with_other_specs(self):
        plan = make_plan(n_channels=3)
        spec = {"ch0": no_edge(), "ch1": 20.0, "ch2": Blackout()}
        out = phase(plan, spec, seed=0)
        assert np.allclose(out.sum(), plan.sum())

    def test_mismatched_round_robin_still_raises(self):
        plan = make_plan(n_channels=2)
        with pytest.raises(ValueError, match="share the same dark_weeks"):
            phase(
                plan,
                {
                    "ch0": Redistribute(dark_weeks=3, high_month_pct=150.0),
                    "ch1": Redistribute(dark_weeks=4),
                },
            )


class TestEdgeCases:
    @pytest.mark.parametrize("n_weeks", [4, 8, 13, 20])
    def test_short_plans_do_not_crash_and_conserve(self, n_weeks):
        plan = make_plan(n_weeks=n_weeks, n_channels=3)
        out = phase(plan, no_edge())
        assert np.allclose(out.sum(), plan.sum())
        assert (out.to_numpy() >= 0).all()

    def test_skips_when_funding_would_go_negative(self):
        # ~3 months: at most one untouched month to fund a 4x month from,
        # so the step is skipped rather than pushing spend below zero.
        plan = make_plan(n_weeks=13, n_channels=1)
        out = _generate_phased_schedule(
            plan,
            _get_month_labels(plan),
            alpha=1.0,
            max_weekly_deviation_pct=no_edge(300.0),
            seed=0,
        )
        assert (out.to_numpy() >= 0).all()
        assert np.allclose(out.sum(), plan.sum())


class TestPeakMonthAlone:
    """dark_weeks=0: no blackout run and no recipient month, so a high
    month (with or without the edge layer) is the only between-month move."""

    def test_dark_weeks_zero_is_allowed(self):
        assert Redistribute(dark_weeks=0).dark_weeks == 0

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_no_dark_weeks_and_annual_total_kept(self, seed):
        plan = make_plan(n_channels=4)
        sched = phase(plan, no_edge(dark_weeks=0), seed=seed)
        assert (sched.to_numpy() > 0).all()
        assert np.allclose(sched.sum(), plan.sum())

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_one_month_at_high_multiple_rest_at_or_below_plan(self, seed):
        plan = make_plan(n_channels=4)
        ratio = month_ratio(plan, phase(plan, no_edge(dark_weeks=0), seed=seed))
        for ch in ratio.columns:
            col = np.sort(ratio[ch].to_numpy())
            assert col[-1] == pytest.approx(2.5)
            assert (col[:-1] <= 1.0 + 1e-9).all()

    def test_channels_peak_in_different_months(self):
        plan = make_plan(n_channels=4)
        ratio = month_ratio(plan, phase(plan, no_edge(dark_weeks=0), seed=0))
        assert ratio.idxmax().nunique() == 4

    def test_nothing_switched_on_is_the_plan(self):
        plan = make_plan(n_channels=4)
        sched = phase(plan, Redistribute(dark_weeks=0, edge_cap_pct=0.0), seed=0)
        assert np.allclose(sched.to_numpy(), plan.to_numpy())


class TestEdgeQuietMonthsOnly:
    """edge_quiet_months_only: the edge layer skips the blackout, recipient
    and high months and nudges every other month."""

    def quiet(self, **kw):
        return Redistribute(
            edge_cap_pct=20.0, high_month_pct=150.0, edge_quiet_months_only=True, **kw
        )

    def test_default_is_off_and_repr_eq(self):
        assert Redistribute().edge_quiet_months_only is False
        assert self.quiet() == self.quiet()
        assert self.quiet() != Redistribute(edge_cap_pct=20.0, high_month_pct=150.0)
        assert "edge_quiet_months_only=True" in repr(self.quiet())
        assert "edge_quiet_months_only" not in repr(Redistribute())

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_big_move_months_match_the_no_edge_schedule(self, seed):
        plan = make_plan(n_channels=4)
        labels = _get_month_labels(plan)
        bare = phase(plan, no_edge(), seed=seed)
        sched = phase(plan, self.quiet(), seed=seed)
        ratio = month_ratio(plan, bare)
        nudged_somewhere = False
        for ch in plan.columns:
            for month in ratio.index:
                rows = labels == month
                same = np.allclose(sched.loc[rows, ch], bare.loc[rows, ch])
                big = ratio.loc[month, ch] > 1.3 or ratio.loc[month, ch] < 0.6
                if big:
                    assert same
                else:
                    nudged_somewhere |= not same
        assert nudged_somewhere

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_quiet_months_get_the_same_nudge_as_without_the_flag(self, seed):
        plan = make_plan(n_channels=4)
        labels = _get_month_labels(plan)
        ratio = month_ratio(plan, phase(plan, no_edge(), seed=seed))
        every = phase(
            plan, Redistribute(edge_cap_pct=20.0, high_month_pct=150.0), seed=seed
        )
        quiet = phase(plan, self.quiet(), seed=seed)
        for ch in plan.columns:
            for month in ratio.index:
                if 0.6 < ratio.loc[month, ch] < 1.3:
                    rows = labels == month
                    assert np.allclose(quiet.loc[rows, ch], every.loc[rows, ch])

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_totals_kept_and_peak_week_not_raised(self, seed):
        plan = make_plan(n_channels=4)
        bare = phase(plan, no_edge(), seed=seed)
        quiet = phase(plan, self.quiet(), seed=seed)
        every = phase(
            plan, Redistribute(edge_cap_pct=20.0, high_month_pct=150.0), seed=seed
        )
        assert np.allclose(quiet.sum(), plan.sum())
        peak = lambda s: float((s / plan).to_numpy().max())  # noqa: E731
        assert peak(quiet) == pytest.approx(peak(bare))
        assert peak(quiet) < peak(every)

    def test_off_is_bit_identical_to_before(self):
        plan = make_plan(n_channels=4)
        a = phase(plan, Redistribute(edge_cap_pct=20.0, high_month_pct=150.0), seed=3)
        b = phase(
            plan,
            Redistribute(
                edge_cap_pct=20.0, high_month_pct=150.0, edge_quiet_months_only=False
            ),
            seed=3,
        )
        assert np.array_equal(a.to_numpy(), b.to_numpy())

    def test_plain_dark_month_with_quiet_nudge(self):
        plan = make_plan(n_channels=4)
        sched = phase(
            plan, Redistribute(edge_cap_pct=20.0, edge_quiet_months_only=True), seed=0
        )
        assert np.allclose(sched.sum(), plan.sum())
        assert ((sched.to_numpy() < 1e-9).sum(axis=0) == 4).all()
