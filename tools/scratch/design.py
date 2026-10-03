"""Prototype of the orthogonal phasing design generator (the project scope notes, Build
order item 8). Deliberately NOT wired into _phaser.py / BudgetPhaser --
this validates the approach first, per the project's own established
pattern (earlier work did the same thing for the Blackout/seesaw
exploration: prototype outside the class, run through the real scoring
pipeline via a DiscoveryReport._phase() override, decide whether to build
the real API only once the numbers look right).

Design, from the project scope notes' "Orthogonal phasing design generator" section:
  - Month layer: each channel gets a +/-1 sign per calendar month, drawn
    from the rows of a Hadamard matrix so that channels' month-sign
    sequences are pairwise orthogonal (dot product zero) by construction.
  - Within-month layer: a zero-sum weekly hold pattern within each month
    (2-week holds: ++-- / --++ for 4-week months, ++0-- for 5-week
    months), so applying it does not move the monthly total on its own
    (still passed through the same per-month rescale _generate_phased_
    schedule already does, for exact budget closure against the small
    residual the unequal 4-vs-5-week pattern lengths leave).
  - Two-layer version: each channel also picks a *rotation* of the
    within-month hold pattern (which specific weeks are up vs down, not
    just the month's overall sign), refined by coordinate descent to
    minimise the sum of squared pairwise correlations across channels'
    full 52-week deviation vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hadamard import hadamard

# ---------------------------------------------------------------------
# Within-month hold patterns (DesignLayer primitive: within-month layer)
# ---------------------------------------------------------------------


def _base_hold_pattern(n_weeks: int) -> np.ndarray:
    """Canonical zero-sum +/-1(/0) pattern for a month of n_weeks: first
    half up, second half down, an odd middle week left at 0 (matches
    the project scope notes' "++--" / "++0--" for 4- and 5-week months, generalised to
    any month length so the same code handles a 3- or 6-week month too)."""
    half = n_weeks // 2
    p = np.zeros(n_weeks)
    p[:half] = 1.0
    p[n_weeks - half :] = -1.0
    return p


def hold_pool(n_weeks: int) -> list[np.ndarray]:
    """All cyclic rotations of the base hold pattern -- the "shape" pool
    a channel picks a rotation from. Rotating staggers WHICH weeks are up
    vs down within the month, while staying zero-sum. Deduplicated (a
    pattern with repeated structure can produce identical rotations)."""
    base = _base_hold_pattern(n_weeks)
    seen: list[np.ndarray] = []
    for shift in range(n_weeks):
        p = np.roll(base, shift)
        if not any(np.array_equal(p, s) for s in seen):
            seen.append(p)
    return seen


# ---------------------------------------------------------------------
# Month layer (DesignLayer primitive: month layer)
# ---------------------------------------------------------------------


def month_sign_rows(n_channels: int, n_months: int) -> np.ndarray:
    """n_channels x n_months matrix of +/-1, one Hadamard row per channel.

    Uses a Hadamard matrix of order >= n_months (exact order when we have
    a construction for it, e.g. 12), truncated to the first n_months
    columns, and assigns channels the first n_channels rows -- skipping
    row 0 (the trivial all-ones row) when there are enough rows to spare,
    since an all-ones row carries no month-to-month contrast of its own
    (harmless, still orthogonal to the rest, but a worse use of a slot
    when better rows are available).
    """
    order = n_months
    while True:
        try:
            H_full = hadamard(order)
            break
        except ValueError:
            order += 1
    H = H_full[:, :n_months]
    n_avail = H.shape[0]
    if n_channels < n_avail:
        rows = list(range(1, n_avail))[:n_channels]
    else:
        rows = [i % n_avail for i in range(n_channels)]
    return H[rows, :]


# ---------------------------------------------------------------------
# Full weekly pattern for one channel from (month signs, per-month shape
# rotation choice)
# ---------------------------------------------------------------------


def build_weekly_pattern(
    month_lengths: list[int], month_signs: np.ndarray, rotation: int
) -> np.ndarray:
    """One channel's full-year unit-amplitude deviation vector: for each
    month, that month's Hadamard sign times a fixed rotation of the
    length-appropriate hold pattern (the SAME rotation index reused across
    every month of a given length for this channel -- keeps the search
    space small: one rotation choice per channel, not one per channel per
    month)."""
    chunks = []
    pools: dict[int, list[np.ndarray]] = {}
    for n_weeks, sign in zip(month_lengths, month_signs, strict=True):
        if n_weeks not in pools:
            pools[n_weeks] = hold_pool(n_weeks)
        pool = pools[n_weeks]
        pat = pool[rotation % len(pool)]
        chunks.append(sign * pat)
    return np.concatenate(chunks)


def pairwise_corr_stats(patterns: np.ndarray) -> tuple[float, float]:
    """max |r| and mean |r| off-diagonal, for an n_channels x n_weeks
    matrix of realised deviation vectors."""
    n = patterns.shape[0]
    if n < 2:
        return 0.0, 0.0
    C = np.corrcoef(patterns)
    mask = ~np.eye(n, dtype=bool)
    vals = np.abs(C[mask])
    return float(vals.max()), float(vals.mean())


# ---------------------------------------------------------------------
# One-layer generator: month-level Hadamard sign only, rotation=0 for
# every channel (no shape optimisation) -- the "4 channels, 1-layer
# Hadamard" benchmark.
# ---------------------------------------------------------------------


def one_layer_design(
    channels: list[str], month_lengths: list[int]
) -> dict[str, np.ndarray]:
    n_months = len(month_lengths)
    signs = month_sign_rows(len(channels), n_months)
    return {
        ch: build_weekly_pattern(month_lengths, signs[i], rotation=0)
        for i, ch in enumerate(channels)
    }


# ---------------------------------------------------------------------
# Naive random baseline -- independent +/-1 per channel per month, no
# structure at all. The "random +/-20% months" benchmark.
# ---------------------------------------------------------------------


def random_design(
    channels: list[str], month_lengths: list[int], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_months = len(month_lengths)
    out = {}
    for ch in channels:
        signs = rng.choice([-1.0, 1.0], size=n_months)
        out[ch] = build_weekly_pattern(month_lengths, signs, rotation=0)
    return out


# ---------------------------------------------------------------------
# Two-layer, coordinate-descent-optimised generator.
# ---------------------------------------------------------------------


def two_layer_design(
    channels: list[str],
    month_lengths: list[int],
    seed: int = 0,
    n_iter: int = 30,
) -> dict[str, np.ndarray]:
    n_channels = len(channels)
    n_months = len(month_lengths)
    rng = np.random.default_rng(seed)

    H = month_sign_rows(n_channels, n_months)  # seed assignment: row i -> channel i
    row_pool = list(range(H.shape[0]))
    max_rotation = max(len(hold_pool(nw)) for nw in set(month_lengths))

    # current assignment: channel -> (row index into H's rows actually
    # available as *candidates*, but simplest: keep each channel's row
    # fixed to its seeded Hadamard row -- with n_channels <= n_months this
    # already gives exact month-level orthogonality; what coordinate
    # descent optimises is each channel's ROTATION, which the seeded
    # (rotation=0 for all) version leaves unoptimised and is exactly what
    # differs between channels on unequal-length (5-week) months.
    rotations = [0] * n_channels

    def patterns_for(rotations: list[int]) -> np.ndarray:
        return np.stack(
            [
                build_weekly_pattern(month_lengths, H[i], rotations[i])
                for i in range(n_channels)
            ]
        )

    best_max, best_mean = pairwise_corr_stats(patterns_for(rotations))

    order = list(range(n_channels))
    for _ in range(n_iter):
        rng.shuffle(order)
        improved = False
        for c in order:
            best_choice = rotations[c]
            best_score = None
            for r in range(max_rotation):
                trial = list(rotations)
                trial[c] = r
                patterns = patterns_for(trial)
                # objective: sum of squared pairwise correlations
                C = np.corrcoef(patterns)
                n = C.shape[0]
                score = float((C[~np.eye(n, dtype=bool)] ** 2).sum())
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = r
            if best_choice != rotations[c]:
                rotations[c] = best_choice
                improved = True
        if not improved:
            break

    final = patterns_for(rotations)
    best_max, best_mean = pairwise_corr_stats(final)
    return (
        {ch: final[i] for i, ch in enumerate(channels)},
        {"max_abs_r": best_max, "mean_abs_r": best_mean, "rotations": rotations},
    )


# ---------------------------------------------------------------------
# Applying a unit-amplitude pattern to real (non-uniform) weekly spend,
# with the same per-month total-preserving rescale _generate_phased_
# schedule uses -- this is what actually flows into a DiscoveryReport,
# and is where exact Hadamard orthogonality (computed on the idealised
# +/-1 pattern) picks up real-world residual correlation, since spend
# isn't flat within a month.
# ---------------------------------------------------------------------


def apply_design(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    unit_patterns: dict[str, np.ndarray],
    cap_pct: dict[str, float],
) -> pd.DataFrame:
    channels = list(plan_df.columns)
    new_spend = plan_df.to_numpy().copy().astype(float)
    months = np.unique(month_labels)
    for ci, ch in enumerate(channels):
        raw_full = unit_patterns[ch] * (cap_pct[ch] / 100.0)
        for month in months:
            mask = np.where(month_labels == month)[0]
            orig_weeks = plan_df.iloc[mask, ci].to_numpy()
            monthly_total = orig_weeks.sum()
            new_weeks = orig_weeks * (1.0 + raw_full[mask])
            if new_weeks.sum() > 0:
                new_spend[mask, ci] = new_weeks * (monthly_total / new_weeks.sum())
            else:
                new_spend[mask, ci] = orig_weeks
    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


def realised_corr_stats(
    plan_df: pd.DataFrame, phased_df: pd.DataFrame
) -> tuple[float, float]:
    """max/mean |r| of the REALISED fractional weekly deviations
    ((phased - plan) / plan), after the monthly rescale -- what actually
    reaches CollinearityDiagnostic, not the idealised unit pattern."""
    dev = (phased_df.to_numpy() - plan_df.to_numpy()) / plan_df.to_numpy()
    n = dev.shape[1]
    C = np.corrcoef(dev.T)
    mask = ~np.eye(n, dtype=bool)
    vals = np.abs(C[mask])
    return float(vals.max()), float(vals.mean())


# ---------------------------------------------------------------------
# The discrete-levels idea: each week, independently pick from a
# fixed array of %-of-plan deviations (including 0%, i.e. "leave it
# alone" is itself one of the options), rather than drawing a continuous
# magnitude or holding a fixed cap. Same per-month rescale convention as
# every other lever, so monthly totals still land exactly on plan.
# ---------------------------------------------------------------------


def random_levels_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    levels: tuple[float, ...] = (0.0, -10.0, -20.0, 10.0, 20.0),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    new_spend = plan_df.to_numpy().copy().astype(float)
    months = np.unique(month_labels)
    for ci, _ch in enumerate(channels):
        for month in months:
            mask = np.where(month_labels == month)[0]
            n_weeks = len(mask)
            raw = rng.choice(levels, size=n_weeks) / 100.0
            orig_weeks = plan_df.iloc[mask, ci].to_numpy()
            monthly_total = orig_weeks.sum()
            new_weeks = orig_weeks * (1.0 + raw)
            if new_weeks.sum() > 0:
                new_spend[mask, ci] = new_weeks * (monthly_total / new_weeks.sum())
            else:
                new_spend[mask, ci] = orig_weeks
    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# The cross-month blackout idea: a few consecutive weeks in ONE month
# go dark, and the freed budget is NOT redistributed within that same
# month (which is what the package's existing Blackout does) -- it's
# added to a DIFFERENT month instead. Deliberately breaks the package's
# per-month budget-conservation invariant: only the ANNUAL total for the
# channel is unchanged, not each month's own total. This is the
# "loosen the conservation unit from monthly to annual" sketch from
# An earlier note, tried here as its own lever rather
# than folded into item 8's orthogonal design.
# ---------------------------------------------------------------------


def cross_month_blackout_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)

    # BUG (found while sanity-checking the max-week-spike numbers): a
    # month shorter than dark_weeks+1 -- e.g. this plan's partial 1-week
    # first month -- used to fall through the old `n_dark <= 0: continue`
    # check and silently skip the WHOLE channel, blackout and
    # redistribution both, rather than just picking a different month.
    # Fix: only consider months with enough weeks to survive a blackout
    # (>= dark_weeks + 1) as blackout-month candidates in the first
    # place, and only consider months with >= 2 weeks as redistribution
    # recipients (dumping a whole freed budget onto one single week would
    # just recreate the same single-week-spike problem this lever exists
    # to avoid).
    eligible_blackout_months = [m for m in months if month_len[m] > dark_weeks]
    eligible_recipient_months = [m for m in months if month_len[m] >= 2]

    for ci, _ch in enumerate(channels):
        blackout_month = rng.choice(eligible_blackout_months)
        recipient_pool = [m for m in eligible_recipient_months if m != blackout_month]
        recipient_month = rng.choice(recipient_pool)

        b_mask = np.where(month_labels == blackout_month)[0]
        n_b = len(b_mask)
        n_dark = min(dark_weeks, n_b - 1)
        start = rng.integers(0, n_b - n_dark + 1)
        dark_idx = b_mask[start : start + n_dark]

        freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
        new_spend[dark_idx, ci] = 0.0

        r_mask = np.where(month_labels == recipient_month)[0]
        r_weeks = plan_df.iloc[r_mask, ci].to_numpy()
        # spread the freed budget across the recipient month's weeks in
        # proportion to their existing plan spend (keeps the recipient
        # month's own weekly SHAPE, just scales it up).
        r_total = r_weeks.sum()
        if r_total > 0:
            new_spend[r_mask, ci] = r_weeks + freed * (r_weeks / r_total)
        else:
            new_spend[r_mask, ci] = r_weeks + freed / len(r_mask)

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# The refinement: instead of dumping the blackout's freed budget into
# ONE recipient month (cross_month_blackout_schedule above), scatter it
# as +10%/+20% bumps across random weeks spread over the rest of the
# year -- softer concentration, closer to the original
# sketch ("small randomised nudges spread across random weeks in the
# ~9 months not otherwise touched"). dark_weeks (t) is the sweep knob:
# try 2, 3, 4.
# ---------------------------------------------------------------------


def cross_month_blackout_scatter_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int,
    bump_levels: tuple[float, ...] = (10.0, 20.0),
    n_bump_weeks: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)
    n_weeks_total = len(plan_df)

    eligible_blackout_months = [m for m in months if month_len[m] > dark_weeks]

    for ci, _ch in enumerate(channels):
        blackout_month = rng.choice(eligible_blackout_months)
        b_mask = np.where(month_labels == blackout_month)[0]
        n_b = len(b_mask)
        n_dark = min(dark_weeks, n_b - 1)
        start = rng.integers(0, n_b - n_dark + 1)
        dark_idx = b_mask[start : start + n_dark]

        freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
        new_spend[dark_idx, ci] = 0.0

        # bump pool: every week NOT in the blackout month itself (already
        # touched) and not already dark.
        pool = np.array(
            [w for w in range(n_weeks_total) if month_labels[w] != blackout_month]
        )
        n_bump = min(n_bump_weeks, len(pool))
        bump_idx = rng.choice(pool, size=n_bump, replace=False)
        bump_pct = rng.choice(bump_levels, size=n_bump) / 100.0

        orig_bump = plan_df.iloc[bump_idx, ci].to_numpy()
        raw_uplift = orig_bump * bump_pct
        raw_total = raw_uplift.sum()
        if raw_total > 0:
            scale = freed / raw_total
            new_spend[bump_idx, ci] = orig_bump + raw_uplift * scale
        else:
            new_spend[bump_idx, ci] = orig_bump + freed / n_bump

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# Loop 1 (phasing_strategy_loop.md): sustained-hold scatter. Targets
# identifiability-adstock specifically -- the full-comparison run under
# real overview adstock/saturation found single-week scattered bumps get
# attenuated by adstock's low-pass filtering before they can inform decay
# estimation, while a whole recipient MONTH (a sustained, multi-week
# step) survives it much better. This tries to keep the single-recipient
# month's "sustained step" property while still spreading the
# redistribution across several LOCATIONS in the year (for cross-channel
# variance decorrelation, which wants spread, not concentration).
# ---------------------------------------------------------------------


def cross_month_blackout_hold_scatter_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 3,
    hold_weeks: int = 2,
    n_holds: int = 3,
    bump_levels: tuple[float, ...] = (10.0, 20.0),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)

    eligible_blackout_months = [m for m in months if month_len[m] > dark_weeks]
    eligible_hold_months = [m for m in months if month_len[m] >= hold_weeks]

    for ci, _ch in enumerate(channels):
        blackout_month = rng.choice(eligible_blackout_months)
        b_mask = np.where(month_labels == blackout_month)[0]
        n_b = len(b_mask)
        n_dark = min(dark_weeks, n_b - 1)
        start = rng.integers(0, n_b - n_dark + 1)
        dark_idx = b_mask[start : start + n_dark]

        freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
        new_spend[dark_idx, ci] = 0.0

        hold_pool_months = [m for m in eligible_hold_months if m != blackout_month]
        n_use = min(n_holds, len(hold_pool_months))
        chosen_months = rng.choice(hold_pool_months, size=n_use, replace=False)

        all_hold_idx = []
        for hm in chosen_months:
            h_mask = np.where(month_labels == hm)[0]
            n_h = len(h_mask)
            hw = min(hold_weeks, n_h)
            hstart = rng.integers(0, n_h - hw + 1)
            all_hold_idx.append(h_mask[hstart : hstart + hw])
        hold_idx = np.concatenate(all_hold_idx)

        level_per_hold = rng.choice(bump_levels, size=n_use)
        bump_pct = np.repeat(level_per_hold, [len(h) for h in all_hold_idx])

        orig_hold = plan_df.iloc[hold_idx, ci].to_numpy()
        raw_uplift = orig_hold * (bump_pct / 100.0)
        raw_total = raw_uplift.sum()
        if raw_total > 0:
            scale = freed / raw_total
            new_spend[hold_idx, ci] = orig_hold + raw_uplift * scale
        else:
            new_spend[hold_idx, ci] = orig_hold + freed / len(hold_idx)

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# Loop 5 (phasing_strategy_loop.md): single-recipient-month redistribution
# (loop 4's new leader) but with blackout/recipient MONTHS assigned
# round-robin across channels instead of each channel drawing
# independently at random. Targets variance specifically -- "structured
# beats random" has held everywhere else this design has been tested
# (item 8's whole original premise); right now two channels can, by
# chance, both black out the same month, which works against
# cross-channel decorrelation for no reason.
# ---------------------------------------------------------------------


def cross_month_blackout_roundrobin_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 3,
    min_recipient_weeks: int = 4,
    min_blackout_month_weeks: int | None = None,
) -> pd.DataFrame:
    # Loop 18: min_blackout_month_weeks (None = dark_weeks+1, old behaviour)
    # Loop 15: min_recipient_weeks (default 4 since loop 15; 2 = pre-loop-15 behaviour)
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)

    _mb = (
        (dark_weeks + 1)
        if min_blackout_month_weeks is None
        else min_blackout_month_weeks
    )
    eligible_blackout_months = [m for m in months if month_len[m] >= _mb]
    eligible_recipient_months = [
        m for m in months if month_len[m] >= min_recipient_weeks
    ]

    # round-robin, but shuffle the assignment order once per seed so it
    # isn't always the same channel->month mapping across seeds.
    bo_order = rng.permutation(len(eligible_blackout_months))
    re_order = rng.permutation(len(eligible_recipient_months))

    for ci, _ch in enumerate(channels):
        blackout_month = eligible_blackout_months[
            bo_order[ci % len(eligible_blackout_months)]
        ]
        # offset recipient assignment so it doesn't just mirror blackout's
        # round-robin position (which would make channel i's recipient
        # always the same relative distance from its blackout).
        recipient_pool = [m for m in eligible_recipient_months if m != blackout_month]
        recipient_month = recipient_pool[
            re_order[
                (ci + len(eligible_recipient_months) // 2)
                % len(eligible_recipient_months)
            ]
            % len(recipient_pool)
        ]

        b_mask = np.where(month_labels == blackout_month)[0]
        n_b = len(b_mask)
        n_dark = min(dark_weeks, n_b - 1)
        start = rng.integers(0, n_b - n_dark + 1)
        dark_idx = b_mask[start : start + n_dark]

        freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
        new_spend[dark_idx, ci] = 0.0

        r_mask = np.where(month_labels == recipient_month)[0]
        r_weeks = plan_df.iloc[r_mask, ci].to_numpy()
        r_total = r_weeks.sum()
        if r_total > 0:
            new_spend[r_mask, ci] = r_weeks + freed * (r_weeks / r_total)
        else:
            new_spend[r_mask, ci] = r_weeks + freed / len(r_mask)

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# Loop 8: same as cross_month_blackout_roundrobin_schedule but spreads
# the freed budget EVENLY across the recipient month's weeks instead of
# proportional to their existing spend (which stacks the top-up onto
# whichever week was already biggest in that month).
# ---------------------------------------------------------------------


def cross_month_blackout_roundrobin_even_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)

    eligible_blackout_months = [m for m in months if month_len[m] > dark_weeks]
    eligible_recipient_months = [m for m in months if month_len[m] >= 2]

    bo_order = rng.permutation(len(eligible_blackout_months))
    re_order = rng.permutation(len(eligible_recipient_months))

    for ci, _ch in enumerate(channels):
        blackout_month = eligible_blackout_months[
            bo_order[ci % len(eligible_blackout_months)]
        ]
        recipient_pool = [m for m in eligible_recipient_months if m != blackout_month]
        recipient_month = recipient_pool[
            re_order[
                (ci + len(eligible_recipient_months) // 2)
                % len(eligible_recipient_months)
            ]
            % len(recipient_pool)
        ]

        b_mask = np.where(month_labels == blackout_month)[0]
        n_b = len(b_mask)
        n_dark = min(dark_weeks, n_b - 1)
        start = rng.integers(0, n_b - n_dark + 1)
        dark_idx = b_mask[start : start + n_dark]

        freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
        new_spend[dark_idx, ci] = 0.0

        r_mask = np.where(month_labels == recipient_month)[0]
        r_weeks = plan_df.iloc[r_mask, ci].to_numpy()
        new_spend[r_mask, ci] = r_weeks + freed / len(r_mask)

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# Loop 9: TWO smaller blackouts per channel (each t=2), each with its
# own round-robin recipient month, instead of one t=4 event. Targets
# identifiability specifically -- earlier work found repeated dark/decay
# cycles across the year earned Blackout(dark=4,prob=1.0) most of its
# rigor, not one long dark run. Also gentler per-event (t=2 not t=4).
# ---------------------------------------------------------------------


def cross_month_blackout_roundrobin_double_schedule(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = list(plan_df.columns)
    months = np.unique(month_labels)
    month_len = {m: len(np.where(month_labels == m)[0]) for m in months}
    new_spend = plan_df.to_numpy().copy().astype(float)

    eligible_blackout_months = [m for m in months if month_len[m] > dark_weeks]
    eligible_recipient_months = [m for m in months if month_len[m] >= 2]
    n_bo = len(eligible_blackout_months)
    n_re = len(eligible_recipient_months)

    bo_order = rng.permutation(n_bo)
    re_order = rng.permutation(n_re)

    for ci, _ch in enumerate(channels):
        # two blackout months per channel, offset by half the pool so a
        # channel's two events land far apart in the year, not adjacent.
        bo1 = eligible_blackout_months[bo_order[(2 * ci) % n_bo]]
        bo2 = eligible_blackout_months[bo_order[(2 * ci + n_bo // 2) % n_bo]]
        if bo1 == bo2:
            bo2 = eligible_blackout_months[bo_order[(2 * ci + n_bo // 2 + 1) % n_bo]]

        for k, blackout_month in enumerate((bo1, bo2)):
            recipient_pool = [
                m for m in eligible_recipient_months if m not in (bo1, bo2)
            ]
            recipient_month = recipient_pool[
                re_order[(2 * ci + k + n_re // 3) % n_re] % len(recipient_pool)
            ]

            b_mask = np.where(month_labels == blackout_month)[0]
            n_b = len(b_mask)
            n_dark = min(dark_weeks, n_b - 1)
            start = rng.integers(0, n_b - n_dark + 1)
            dark_idx = b_mask[start : start + n_dark]

            freed = plan_df.iloc[dark_idx, ci].to_numpy().sum()
            new_spend[dark_idx, ci] = 0.0

            r_mask = np.where(month_labels == recipient_month)[0]
            r_weeks = plan_df.iloc[r_mask, ci].to_numpy()
            r_total = r_weeks.sum()
            if r_total > 0:
                new_spend[r_mask, ci] = r_weeks + freed * (r_weeks / r_total)
            else:
                new_spend[r_mask, ci] = r_weeks + freed / len(r_mask)

    return pd.DataFrame(new_spend, index=plan_df.index, columns=plan_df.columns)


# ---------------------------------------------------------------------
# Loop 10: t=4 round-robin (current leader) PLUS a light +/-10%
# edge+balanced weekly layer on top, applied to the already-redistributed
# schedule. Targets variance/identifiability specifically -- edge+
# balanced's fine, independent weekly variation is what's been winning
# those two metrics everywhere; testing whether layering a SMALL dose of
# it on top of the structural leader closes the remaining gap without
# giving up bias's current lead.
# ---------------------------------------------------------------------


def roundrobin_plus_edge_layer(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 4,
    edge_cap_pct: float = 15.0,
    min_recipient_weeks: int = 4,
    min_blackout_month_weeks: int | None = None,
) -> pd.DataFrame:
    # Loop 11 (2026-09-18): edge_cap_pct is a smooth rigor/cost/spike
    # dial with no plateau (swept 0-100%, monotonic the whole way) --
    # 15% chosen as a deliberate small step past the loop-10 default of
    # 10%, staying inside the ~1.5-2.4x spike range every other
    # accepted candidate in phasing_strategy_loop.md has stayed in.
    # This is a deployability judgment call, not an optimum.
    from how_wrong_is_your_mmm._phaser import _generate_phased_schedule

    base = cross_month_blackout_roundrobin_schedule(
        plan_df,
        month_labels,
        seed,
        dark_weeks=dark_weeks,
        min_recipient_weeks=min_recipient_weeks,
        min_blackout_month_weeks=min_blackout_month_weeks,
    )
    channels = list(plan_df.columns)
    spec = {c: edge_cap_pct for c in channels}
    return _generate_phased_schedule(
        base,
        month_labels,
        alpha=1.0,
        max_weekly_deviation_pct=spec,
        seed=seed + 10_000,
        nudge_shape="edge",
        balance_signs=True,
    )


def edge_layer_frequency_ablation(
    base_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    cap_pct: float = 15.0,
    frequency: float = 1.0,
) -> pd.DataFrame:
    """Loop 12 ablation: an edge+balanced-style layer where MAGNITUDE
    (cap_pct, the fixed +/-deviation size on a touched week) and
    FREQUENCY (the fraction of weeks in a month that get touched at all,
    the rest staying at `base_df`) are independent knobs -- unlike
    `_generate_phased_schedule`'s "edge" shape, which always touches
    every week (frequency=1.0) and only lets magnitude vary.

    Built to test whether loop 11's rigor gains from a bigger edge_cap
    are really a MAGNITUDE effect, a FREQUENCY effect, or both, by
    holding a "deviation budget" H = frequency * cap_pct**2 (a
    variance-like quantity) constant across the frequency/magnitude
    split and checking whether the rigor scores move at matched H.

    Applied on top of `base_df` (typically a
    `cross_month_blackout_roundrobin_schedule` output), same
    preserve-the-monthly-total rescale as `_generate_phased_schedule`.
    """
    rng = np.random.default_rng(seed)
    channels = list(base_df.columns)
    new_spend = base_df.to_numpy().copy().astype(float)
    cap = cap_pct / 100.0

    for month in np.unique(month_labels):
        mask = np.where(month_labels == month)[0]
        n = len(mask)
        for ci, _ch in enumerate(channels):
            n_touch = int(round(n * frequency))
            n_touch -= n_touch % 2  # keep it even so +/- balance exactly
            if n_touch < 2:
                continue
            touch_idx = rng.choice(n, size=n_touch, replace=False)
            signs = np.array([1.0] * (n_touch // 2) + [-1.0] * (n_touch // 2))
            rng.shuffle(signs)
            dev = np.zeros(n)
            dev[touch_idx] = signs * cap

            month_spend = base_df.iloc[mask, ci].to_numpy()
            raw = month_spend * (1.0 + dev)
            total_before = month_spend.sum()
            total_after = raw.sum()
            if total_after > 0:
                raw = raw * (total_before / total_after)
            new_spend[mask, ci] = raw

    return pd.DataFrame(new_spend, index=base_df.index, columns=channels)


def roundrobin_plus_freq_ablation(
    plan_df: pd.DataFrame,
    month_labels: np.ndarray,
    seed: int,
    dark_weeks: int = 4,
    cap_pct: float = 15.0,
    frequency: float = 1.0,
) -> pd.DataFrame:
    """Same structural base as `roundrobin_plus_edge_layer`, but with the
    frequency/magnitude-isolating layer instead of the always-every-week
    "edge" shape. frequency=1.0, cap_pct=X should closely match
    `roundrobin_plus_edge_layer(edge_cap_pct=X)` (sanity check)."""
    base = cross_month_blackout_roundrobin_schedule(
        plan_df, month_labels, seed, dark_weeks=dark_weeks
    )
    return edge_layer_frequency_ablation(
        base, month_labels, seed=seed + 10_000, cap_pct=cap_pct, frequency=frequency
    )
