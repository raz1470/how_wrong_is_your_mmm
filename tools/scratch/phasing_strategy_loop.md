# Phasing strategy loop — tracking log

Started 2026-09-18. It may be picked up over
several days, so this file (not chat) is the source of truth
for current status. Scratch/throwaway by convention:
nothing here is committed or wired into `_phaser.py` until a design
earns that.

## Scope

Design a phasing strategy that is actually suitable for a client to run
(not a rigor-only sweep winner like `Blackout(dark=4, prob=1.0)`, which
forces a single week to absorb 3-6x its normal budget) while bringing
down variance, bias, AND identifiability together — not optimising one
at the expense of the others.

## Design principles (what each outcome actually needs)

Pulled from the project scope notes' "Orthogonal phasing design generator" section
and confirmed/extended by what we've found so far. Every loop entry
below should name which of these it's targeting, not just which
parameter changed:

- **Variance** (cross-channel collinearity): needs each channel's
  variation orthogonal to other channels' and to controls. Structured,
  coordinated-across-channels designs beat independent random draws —
  confirmed hard by the random-baseline vs Hadamard-structured result.
- **Bias** (demand confounding): shrinks with the SHARE of exogenous
  (demand-independent) variance relative to the plan's existing natural
  variance. Needs deviations that are large in magnitude, not just
  present — diluting average magnitude (e.g. a lever that's often 0%)
  visibly hurts this.
- **Identifiability — saturation**: needs 3+ distinct positive spend
  levels to trace curvature. A blackout anchors the intercept (the
  zero-spend point) but can't pin the curve's bend anywhere else on its
  own.
- **Identifiability — adstock**: needs clean step changes HELD LONGER
  than the decay half-life. Adstock is a low-pass filter — confirmed
  directly this pass: single-week scattered bumps get attenuated
  before they inform decay estimation, while a sustained multi-week
  hold (or a whole recipient month) survives it much better.
- **Cost**: scales with curvature and the SIZE of the deviation from
  reference spend (Jensen's inequality) — a blackout costs roughly
  spend x (average ROI - marginal ROI).
- **Deployability / learning-phase**: bounded by EDIT COUNT and STEP
  SIZE, not by total variance injected — a "client suitable" design
  should minimise the number of large single-week jumps and cap how
  big any one jump is, not just minimise aggregate variance.

## Scenario (fixed across loops unless noted)

10 channels (`ch0`..`ch9`), history 104wk + plan 52wk, `correlation=0.7`,
saturation/adstock cycling the real `docs/overview.html` values (tv
b=0.60/lambda=0.50, meta b=0.75/lambda=0.30, search_generic
b=0.90/lambda=0.10, tiktok b=0.70/lambda=0.20), true marginal returns
0.5-1.4. Fixed as of 2026-09-18 (previously flat saturation=0.7/adstock=0
placeholder, corrected this pass).

> **STATUS update 2026-09-20: `Redistribute` is built into `_phaser.py` and swept by `DiscoveryReport` (branch `feat/redistribute-strategy`, uncommitted); loop 23 (bottom) is the like-for-like acceptance run through the real class and supersedes loop 20's comparison. Open: sweep-runtime decision and default-row choice. (Loops 21-22 were run in a parallel chat and are separate scratch designs, not built into `src/`.)**
>
> **STATUS as of 2026-09-20 (end of day): loops 14-23 are appended at the bottom and supersede the caveats below. Research is DONE; remaining work is two build tasks scoped in the project scope notes "Open items": `Redistribute` (the round-robin winner; branch `feat/redistribute-strategy`) and `MonthStep` (loops 21-23; month-level orthogonal steps, beat the winner on variance/bias/cost/spike and lost only on identifiability at 40%, matched it at 60% for higher cost). Decision: default sweep gets only the 20% row of each new family; 40/60/80 stay pinnable. Rejected/closed: 20 channels (under-powered fit), blackout+month-step hybrids, floor-only blackout, two-blackout variants. Untested and dropped by choice: block-length, protected-month, locked-channel variants and a 50% step. Nothing here is committed; `tools/scratch/` is git-untracked and the project scope notes is gitignored.**

## Current best (client-suitable candidates only)

**As of loop 11 (2026-09-18): `roundrobin_plus_edge_layer(dark_weeks=4, edge_cap_pct=15.0)`**
in `design.py` -- one t=4 blackout per channel (round-robin assigned
month, no cross-channel collisions), its freed budget redistributed
proportionally into one round-robin recipient month, plus a light
edge+balanced weekly layer on top. See "Batch-of-10 summary" below for
the full writeup on how this was found, and "Loop 11" below for why the
edge_cap moved from 10% to 15%.

| candidate | variance_cv | bias % | identifiability | cost % | max week spike |
|---|---|---|---|---|---|
| Unphased | 0.305 | 58.0 | 40.4 | 0.00 | 1.0x |
| `roundrobin_plus_edge_layer` (t=4, edge_cap=10%, loop-10 winner) | 0.176 | 20.2 | 16.0 | 1.66 | ~2.2-2.3x |
| **`roundrobin_plus_edge_layer` (t=4, edge_cap=15%, current pick)** | **0.168** | **19.7** | **15.6** | 1.70 | ~2.4x |
| +/-80% edge+balanced (reference, not client-suitable) | 0.103 | 24.8 | 14.3 | 2.68 | 1.0x (edge, no spike by construction) |
| `Blackout(dark=4, prob=1.0)` (rigor-only, explicitly excluded) | 0.078 | 16.1 | 2.35 | 14.4 | ~6.5x |

**Important caveat found in loop 11: `edge_cap_pct` is a smooth
rigor/cost/deployability dial, not a free parameter with a sweet spot.**
Swept 0% to 100% -- every increment kept improving all three rigor
metrics with NO plateau, right up to 100% (spike 4.8x, cost 6.2%,
approaching `Blackout`'s own excluded territory). So picking this value
is a deployability judgment call, same category of decision as
excluding `Blackout` itself, not something a rigor-only loop should be
left to auto-maximise. 15% was picked as a small, deliberate step up
from the loop-10 default that stays inside the ~1.5-2.4x spike range
every other accepted candidate in this doc has stayed in; 20%+ remains
on the table to trade more deployability risk for more
rigor, but nothing past 15% is currently recommended by default.

Not yet done: full seed-robustness check on the winner across many more
seeds on the canonical plan (deferred to the "run all of them at the
end" pass); a check on a different CALENDAR structure (different
start_date/length -- loop 13 confirmed robustness to different plan
spend-level draws on this same calendar, but not yet a different
calendar); channel-count sensitivity.

**Multi-year impact (loop 16, added 2026-09-19; most MMMs use ~3 years).**
Design applied EVERY year (fresh 52wk round-robin per year), 52wk unphased
warm-up history, 3 seeds, canonical plan spend seed=1. Means:

| years phased | lever | variance_cv | bias % | identifiability | cost % (per yr) | max spike |
|---|---|---|---|---|---|---|
| 1 | Unphased | 0.433 | 95.5 | 43.3 | 0 | 1.0x |
| 1 | winner (t=4, edge 15%) | 0.192 | 25.2 | 13.5 | 1.77 | 2.47x |
| 2 | Unphased | 0.317 | 70.4 | 45.5 | 0 | 1.0x |
| 2 | winner | 0.107 | 16.0 | 9.6 | 1.72 | 2.64x |
| 3 | Unphased | 0.252 | 50.0 | 45.9 | 0 | 1.0x |
| **3** | **winner** | **0.091** | **10.6** | **8.6** | 1.74 | 2.77x |

Gains compound with more phased years (bias 25 -> 11, id 13.5 -> 8.6)
at flat annual cost; spike creeps up slightly (2.5 -> 2.8x). Note the
1-year row is not comparable to the older 104wk-history tables (different
history length).

## Loop log

(most recent last)

### Loop 1 — sustained-hold scatter (targets: identifiability-adstock, deployability)

**Design.** `cross_month_blackout_hold_scatter_schedule(dark_weeks=3, hold_weeks=2, n_holds=3)`:
same t=3 blackout as before, but instead of scattering the freed budget
across `n_bump_weeks` isolated single weeks, spread it across `n_holds=3`
separate 2-week HELD blocks in different months. Rationale: adstock needs
a clean step change held longer than the decay half-life (design
principle above) -- a single week is too short to survive the low-pass
filter, so hold length rather than scatter count should be the lever
for identifiability, while still spreading LOCATIONS across the year for
cross-channel variance.

**Result** (6 seeds, real adstock/saturation):

| metric | single-week scatter (t=3, 10 bumps) | sustained-hold (t=3, 3x2wk) | single recipient month (t=4) |
|---|---|---|---|
| variance_cv | 0.177 | 0.184 | 0.191 |
| bias % | 38.7 | 35.2 | 27.1 |
| identifiability | 23.5 | 22.0 | 16.6 |
| cost % | 0.77 | 0.97 | 1.61 |
| max week spike | ~2x | ~1.7-1.9x | ~3.3x |

**Decision.** Confirms the hypothesis directionally -- holding for 2 weeks
instead of 1 improves bias and identifiability over pure single-week
scatter, at a small cost/deployability cost that's still much better than
the single-recipient-month version's ~3.3x spike. But it doesn't close
the gap to single-recipient's rigor. Next: the recipient month's hold is
effectively 4-5 weeks (a whole month), not 2 -- loop 2 tries lengthening
`hold_weeks` (e.g. 4) while keeping multiple holds for spread, to see if
hold LENGTH is really the main lever, independent of scatter count.

### Loop 2 — longer holds, same spread (targets: identifiability-adstock, bias)

**Design.** Same as loop 1 but `hold_weeks=2 -> 4` (`n_holds=3` unchanged):
does lengthening the held block keep buying identifiability, or does
spreading the SAME freed budget over more weeks per hold dilute it back
down (bias's known sensitivity to average magnitude)?

**Result** (6 seeds):

| metric | loop 1 (3x2wk) | loop 2 (3x4wk) | single recipient month (t=4) |
|---|---|---|---|
| variance_cv | 0.184 | 0.192 | 0.191 |
| bias % | 35.2 | 32.4 (std 9.3, noisier) | 27.1 |
| identifiability | 22.0 | 23.1 | 16.6 |
| cost % | 0.97 | 0.91 | 1.61 |
| max week spike | ~1.7-1.9x | ~1.4-1.5x | ~3.3x |

**Decision.** Mixed, not a clean win: bias improved a bit (noisily),
deployability improved further (best max-spike yet), but identifiability
did NOT improve with longer holds -- plateaued, even ticked up slightly.
So hold length isn't the main further lever for identifiability past
~2 weeks; something else (total budget concentrated per location?) is
likely doing more of single-recipient-month's work. Loop 3: hold length
fixed at 4wk, but fewer holds (`n_holds=3 -> 2`) so each location has to
carry more of the freed budget -- isolates whether per-location
CONCENTRATION (not just duration) is what identifiability actually wants.

### Loop 3 — fewer, still-4wk holds (targets: identifiability via concentration)

**Design.** `n_holds=3 -> 2`, `hold_weeks=4` unchanged: does concentrating
the same freed budget into fewer locations (each carrying more of it)
help identifiability where longer-but-more-spread holds (loop 2) didn't?

**Result** (6 seeds): variance 0.179 (best of the hold-scatter variants
so far), bias 30.3, identifiability 23.0 (essentially unchanged from
loop 2's 23.1), cost 0.90, max spike ~1.5-1.6x.

**Decision.** Concentration alone doesn't move identifiability either --
it's been stuck at 22-23.5 across all three hold-scatter variants
regardless of length or count. That's itself informative: worked out the
actual adstock half-lives for this scenario's decay values (lambda
0.10-0.50) -- all are under 1.5 weeks, so a 2-week hold was NEVER the
bottleneck. Something else must explain single-recipient-month's much
lower 16.6. Loop 4 checks the obvious confound before chasing anything
subtler: single-recipient-month's number was measured at `dark_weeks=4`,
every hold-scatter variant here has been at `dark_weeks=3`. Isolate it.

### Loop 4 — single-recipient-month at matched dark_weeks=3 (confound check)

**Design.** Existing `cross_month_blackout_schedule` (one recipient
month gets ALL the freed budget), but at `dark_weeks=3` to match every
loop-1-3 comparison exactly instead of the earlier dark_weeks=4 number.

**Result** (6 seeds): variance **0.170** (best of everything tested this
pass), bias 28.5, identifiability **20.8**, cost 1.07, max spike
~1.75-2.0x.

**Decision — this is the new leaderboard candidate.** Confirms the
earlier single-recipient-vs-scatter gap was partly a real dark_weeks
confound (t=3->t=4 alone buys real identifiability, matching the
original t-sweep finding), but even AT MATCHED t=3, single-recipient
(20.8) still beats every hold-scatter variant (22.0-23.1) on
identifiability, and beats all of them on variance too (0.170 vs
0.177-0.192). Apparently concentrating the whole redistribution into
ONE coherent month, rather than splitting it into several smaller holds,
is doing real work beyond just "sustained enough for adstock" -- open
question for a future loop, not resolved. Deployability (~1.75-2.0x
spike) is still far short of plain Blackout's 6.5x and better than the
single-recipient-at-t=4's ~3.3x. Loop 5: does explicitly avoiding
cross-channel month collisions (right now each channel's blackout/
recipient months are independent random draws) improve this further --
targets variance specifically, per the "structured beats random"
principle that's been true everywhere else.

### Loop 5 — round-robin month assignment (targets: variance)

**Design.** Same single-recipient-month structure as loop 4, but
blackout/recipient months assigned round-robin (shuffled-but-non-
colliding) across channels instead of each channel drawing independently
at random, which could put two channels' blackouts in the same month by
chance.

**Result** (6 seeds): variance **0.160** (down from loop 4's 0.170),
bias 28.1 (~same), identifiability 20.1 (~same, marginally better), cost
1.02 (~same), max spike ~1.7-1.9x (~same).

**Decision — new leader.** Confirms "structured beats random" once more:
clean variance win, no cost anywhere else. New best client-suitable
candidate so far: **single-recipient-month, round-robin, dark_weeks=3**
-- variance 0.160 / bias 28.1 / identifiability 20.1 / cost 1.02% / max
spike ~1.8x. Loop 6: re-check dark_weeks under round-robin (t=2/4) -- the
earlier t-sweep found t=4 best for identifiability under the OLD random
assignment; worth confirming that still holds now the confound is fixed.

### Loop 6 — re-sweep dark_weeks under round-robin (targets: bias, identifiability)

**Result** (6 seeds each):

| t | variance_cv | bias % | identifiability | cost % | max week spike |
|---|---|---|---|---|---|
| t=2 | 0.187 | 32.1 | 26.7 | 0.54 | -- |
| t=3 | 0.160 | 28.1 | 20.1 | 1.02 | ~1.7-1.9x |
| t=4 | 0.182 | **20.5** | **16.5** | 1.62 | ~2.0-2.1x |

**Decision — new leader.** t=4 round-robin is a big jump on bias and
identifiability over t=3 (both much better) for a modest cost increase
(1.02% -> 1.62%) and still-reasonable deployability (~2x max spike, vs
plain Blackout's 6.5x). Variance is slightly worse than t=3's (0.182 vs
0.160) but not by much. This is now within reach of +/-80% edge+balanced
(0.103/24.8/14.3/2.68%) -- beats it on bias outright (20.5 vs 24.8), at
40% lower cost, though edge+balanced still wins on variance and
identifiability. **New best client-suitable candidate: single-recipient-
month, round-robin, dark_weeks=4** -- variance 0.182 / bias 20.5 /
identifiability 16.5 / cost 1.62% / max spike ~2.0-2.1x. Loop 7: does
the trend keep improving at t=5, or has it peaked?

### Loop 7 — t=5 (checking the ceiling)

**Result.** Crashes: no month in this calendar has more than 5 weeks, so
a t=5 blackout can never leave a week standing anywhere -- `t=4` is the
hard ceiling for a single-month blackout on this calendar, not a
truncated sweep. Useful boundary to know, not a bug.

**Decision.** Stop sweeping `dark_weeks` here; t=4 round-robin remains
the leader. Loop 8: t=4 round-robin currently spreads the recipient
month's uplift PROPORTIONAL to its existing weekly spend (so the
week that was already biggest in that month gets the biggest top-up --
flagged as a minor concern back when this schedule was first built).
Try spreading it EVENLY across the recipient month's weeks instead, to
see if that improves deployability (max spike) or any rigor metric.

### Loop 8 — even split instead of proportional (targets: deployability)

**Result** (6 seeds): variance 0.186 (slightly worse than loop 6/7's
0.182), bias 20.3 (~same), identifiability 16.8 (slightly worse), cost
1.61 (~same), **max spike 2.2-2.8x -- worse than proportional's 2.0-2.1x**.

**Decision — discard, keep proportional.** Counter to the hypothesis:
spreading evenly can dump a large ABSOLUTE addition onto a week that was
naturally small, which is a bigger fractional jump than proportional
scaling (which adds more to already-big weeks, a smaller relative jump).
Proportional was already the better choice; confirmed, not changed.
Leader unchanged: single-recipient-month, round-robin, dark_weeks=4,
proportional redistribution -- variance 0.182 / bias 20.5 /
identifiability 16.5 / cost 1.62% / max spike ~2.0-2.1x.

### Loop 9 — two smaller blackouts instead of one (targets: identifiability, deployability)

**Design.** Each channel gets TWO t=2 blackout months (round-robin,
non-colliding) each funding its own round-robin recipient month, instead
of one t=4 event. Rationale: earlier work found repeated dark/decay cycles
across a year earned Blackout most of its rigor, not one long run --
tests whether that generalises to this redistribution design too.

**Result** (6 seeds): variance 0.185 (~same as t=4 single), bias 28.3
(worse than t=4 single's 20.5), identifiability 22.6 (worse than t=4
single's 16.5), cost 1.19 (cheaper), max spike 1.55-1.67x (gentler than
t=4 single's ~2.0-2.1x). **Caveat: found a real budget-conservation bug
while checking this** -- when a channel's two recipient months happen to
land on the same month, the second uplift overwrites rather than adds to
the first, causing up to 3.4% annual budget drift in one of the 6 seeds
tested. Not fixed (this loop is being discarded anyway), but the
direction of the finding (worse than one bigger event) is consistent
with loops 1-3's pattern and unlikely to reverse if fixed.

**Decision — discard, keep single-event t=4 as leader.** Splitting into
multiple smaller events keeps losing to one larger, better-targeted
event on this design, same pattern as the hold-scatter loops (1-3) vs
single-recipient. Gentler deployability (1.6x vs 2.0x) is real but not
worth the bias/identifiability cost here. Loop 10 (last of this batch):
try the opposite combination -- keep the winning t=4 single-event
structure, and layer a SMALL amount of edge+balanced-style fine weekly
noise on top, to see if that closes the remaining variance gap to
+/-80% edge+balanced (0.182 vs 0.103) without giving up t=4's bias lead.

### Loop 10 — t=4 round-robin + light edge layer on top (targets: variance, identifiability, keeping bias's lead)

**Result** (6 seeds): variance **0.176** (down from t=4-only's 0.182),
bias 20.2 (~same, still strong), identifiability **16.0** (down from
16.5), cost 1.66% (~same), max spike ~2.2-2.3x (slightly worse than
t=4-only's 2.0-2.1x, still far short of plain Blackout's 6.5x).

**Decision — new leader, best of the whole batch.** Clean improvement on
two of three rigor metrics with bias basically unchanged, for
negligible extra cost. The structural redistribution (round-robin,
t=4) is doing the bias/identifiability heavy lifting; the light edge
layer adds the fine-grained, demand-independent weekly variance that's
been winning variance everywhere else it's been tried. This combination
-- not either piece alone -- is the best client-suitable candidate found
in this 10-loop batch.

---

## Batch-of-10 summary (2026-09-18)

**Winner: `roundrobin_plus_edge_layer(dark_weeks=4, edge_cap_pct=10.0)`**
-- variance_cv 0.176, bias 20.2%, identifiability 16.0, cost 1.66%, max
single-week spike ~2.2-2.3x plan. For comparison: Unphased is
0.305/58.0/40.4/0%; +/-80% edge+balanced (the strongest "pure" existing
lever, itself not considered client-suitable either) is
0.103/24.8/14.3/2.68%; plain `Blackout(dark=4,prob=1.0)` (rigor-only
winner, explicitly excluded from consideration) is 0.078/16.1/2.35/14.4%
with a 6.5x single-week spike.

**What moved the needle, most to least:**
1. Fixing the scenario's saturation/adstock to match `docs/overview.html`
   instead of a flat placeholder (before this batch) -- changed which
   candidates even looked promising.
2. Concentrating the redistribution into ONE month rather than several
   smaller holds/events (loops 1-3, 9) -- consistently won on
   identifiability, contrary to the "spread it out" instinct that helped
   in the no-adstock world.
3. `dark_weeks=4` over 3 (loop 6) -- big bias/identifiability jump for
   modest extra cost; 4 is also the hard ceiling this calendar allows.
4. Round-robin, non-colliding month assignment (loop 5) -- clean,
   free variance win, no downside found anywhere it was tried.
5. A light dusting of edge+balanced noise on top of the structural
   design (loop 10) -- small but real further gain on variance and
   identifiability.

**What didn't work:** more/smaller scattered single-week bumps (worse
than sustained holds); longer or fewer holds beyond ~2 weeks (identifi-
ability plateaued regardless -- this scenario's adstock half-lives are
all under 1.5 weeks, so hold length was never really the bottleneck);
splitting into two smaller blackout events instead of one (loop 9, and
found a real budget-conservation bug doing it -- not fixed, this
direction was discarded anyway); even (vs proportional) redistribution
within the recipient month (loop 8 -- makes deployability slightly
worse, not better).

**Not yet resolved:** why edge+balanced-style variation structurally
outperforms everything else on bias specifically is still not
mechanistically understood, just empirically worked around by grafting
a small amount of it onto the structural design (loop 10). Also
untested: seed-robustness of this final winner across MORE seeds
(everything in this batch used 6), and whether it holds up on a
different plan draw / channel count, not just this one canonical
scenario.

**Repo state:** nothing committed. All code in
`tools/scratch/{hadamard.py, design.py, loop_harness.py,
validate_via_discovery_report.py, seed_robustness.py}`, all functions
from this batch appended to `design.py`. Raw per-seed CSVs for every
loop in `tools/scratch/loop*_raw.csv` / `loop*_v2_raw.csv` (git-
untracked, same as the rest of this directory).

### Loop 11 — tune `edge_cap_pct` in the loop-10 winner (targets: variance, bias, identifiability; check cost/deployability tradeoff)

Only 10% had been tested for the edge layer added in loop 10. Swept
`edge_cap_pct` in {0, 5, 10, 15, 20, 25}% first (6 seeds each, dark_weeks=4
fixed), then extended to {30, 40, 50, 70, 100}% when no plateau appeared.

**Result** (means across 6 seeds, `roundrobin_plus_edge_layer`):

| edge_cap | variance_cv | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| 0% | 0.182 | 20.5 | 16.48 | 1.62 | 2.05x |
| 5% | 0.181 | 20.6 | 16.33 | 1.63 | 2.15x |
| 10% (loop-10) | 0.176 | 20.2 | 16.02 | 1.66 | 2.26x |
| 15% | 0.168 | 19.7 | 15.62 | 1.70 | 2.37x |
| 20% | 0.159 | 19.0 | 15.04 | 1.77 | 2.49x |
| 25% | 0.150 | 18.1 | 14.39 | 1.85 | 2.61x |
| 30% | 0.142 | 17.2 | 13.72 | 1.95 | 2.73x |
| 40% | 0.126 | 15.2 | 12.40 | 2.21 | 2.98x |
| 50% | 0.114 | 13.6 | 11.14 | 2.56 | 3.25x |
| 70% | 0.094 | 11.2 | 8.79 | 3.54 | 3.81x |
| 100% | 0.073 | 8.3 | 5.50 | 6.20 | 4.80x |

No crossover anywhere in 0-100%: every extra point of `edge_cap_pct`
keeps improving all three rigor metrics, monotonically, at a smoothly
rising cost and spike. At 100% it's still short of plain `Blackout`'s
6.5x spike / 14.4% cost, but it's clearly headed the same direction --
this is the same "rigor wants to go to the deployability extreme"
pattern that got `Blackout` excluded in the first place, just playing
out on a continuous dial instead of a step function.

**Decision — this is a deployability choice, not an optimisation
target.** Since there's no data-driven "best" cap, the right answer is
to pick a value inside the deployability envelope already established
by every other accepted candidate in this doc (~1.5-2.3x spike), or
just past it. Moved the default from 10% to **15%** (spike ~2.4x) --
small, deliberate extra rigor for negligible extra cost, still close to
the established range. Did NOT chase this further: past 15-20% the
gains keep coming but so does the same deployability problem this whole
exercise exists to avoid. `edge_cap_pct` should probably ship as a
user-configurable knob (with a sane low default) rather than something
the design generator picks on its own via rigor-maximisation.

**Next up (loop 12):** the still-unresolved mechanistic question from
the batch-10 summary -- why edge+balanced-style variation specifically
helps bias -- via an ablation that isolates magnitude vs frequency of
the edge layer, independent of the cap-size sweep just done here (which
varied both at once through the existing edge+balanced machinery).

### Loop 12 — isolate magnitude vs frequency in the edge layer (targets: mechanistic understanding of bias, per batch-10's "not yet resolved" item)

Built `edge_layer_frequency_ablation` / `roundrobin_plus_freq_ablation` in
`design.py` -- lets magnitude (per-touched-week cap %) and frequency
(fraction of a month's weeks touched) vary independently, unlike
`_generate_phased_schedule`'s "edge" shape which always touches every
week. Held a deviation "budget" H = frequency x cap_pct^2 ~= 225
constant while trading frequency against magnitude, 8 seeds per
condition.

**Result:**

| frequency | cap% | H | variance_cv | bias % | identifiability | cost % |
|---|---|---|---|---|---|---|
| 1.00 | 15.0 | 225 | 0.166 | 21.4 | 16.4 | 1.70 |
| 0.75 | 17.3 | 224 | 0.172 | 19.8 | 16.0 | 1.68 |
| 0.50 | 21.2 | 225 | 0.170 | 19.9 | 15.7 | 1.69 |
| 0.25 | 30.0 | 225 | 0.181 | 21.9 | 16.8 | 1.61 |
| 0.125 | 42.4 | 225 | 0.181 | 21.9 | 16.8 | 1.61 |

**Found a real structural constraint, not a bug to fix:** the 0.25 and
0.125 rows are identical, because with this calendar's 4-5 week months
you cannot balance +/- across fewer than 2 touched weeks, and 2 out of
4-5 weeks is already ~40-50% frequency -- so those two conditions
silently touched zero weeks and collapsed to the "no edge layer"
baseline (matches loop 11's edge_cap_0 row). Frequency below ~0.5 is
simply not constructible on a calendar this granular.

**Decision — answered the mechanistic question, with a caveat.** Across
the achievable range (frequency 0.5-1.0), matched-budget conditions land
in roughly the same place on all three rigor metrics (variance
0.166-0.172, bias 19.8-21.4, identifiability 15.7-16.4). So it's the
TOTAL injected deviation budget (frequency x cap^2, a variance-like
quantity) that drives the bias/variance/identifiability gains seen in
loop 11 -- not frequency or magnitude independently. How that budget is
split between "touch more weeks a little" and "touch fewer weeks a lot"
barely matters, at least down to the calendar's practical floor. This
resolves the batch-10 "not yet understood" item: edge+balanced-style
variation helps because it's exogenous variance, full stop -- there's no
separate frequency mechanism to chase. Practical implication: no reason
to add a frequency knob to the design generator API: cap % alone
(against the deployability constraint from loop 11) is a sufficient
control surface.

**Next up (loop 13):** seed-robustness / different-scenario check has
been deferred to "run all of them at the end". In the
meantime, the remaining genuinely open item from the batch-10 summary is
whether the winning design (`roundrobin_plus_edge_layer`, t=4,
edge_cap=15%) holds up on a DIFFERENT plan draw (not just a different
report seed) -- everything so far, including loops 11 and 12, has used
the same single canonical `plan_df` (seed=1). Loop 13 should vary the
plan-generation seed itself, not just the phasing/report seed, to check
the design isn't overfit to this one plan's particular month lengths /
spend levels.

### Loop 13 — robustness to a different plan draw (targets: confirm the winner isn't overfit to this one plan)

Ran the winning design (`roundrobin_plus_edge_layer`, t=4, edge_cap=15%)
against 5 different plan draws (`simulate_spend` plan seeds 1-5, seed=1
being the canonical plan used everywhere else in this doc), 3 report
seeds each (15 runs total), same fixed history (seed=0) and calendar
start date throughout.

**Result:** stays in the same range across all 5 plan draws -- variance_cv
0.160-0.188 (mean 0.172, std 0.009), bias 12.6-29.5% (mean 19.2%, std
5.2 -- about the same spread report-seed alone already produced),
identifiability 13.6-17.5 (mean 15.5, std 1.05), cost 1.63-1.90% (mean
1.76%), max spike 2.27-2.93x (mean 2.60x, still well under `Blackout`'s
excluded 6.5x). No plan draw where the design degrades or falls apart.

**Caveat (important, not swept under the rug):** all 5 plan draws had
IDENTICAL month lengths ([1,4,4,5,4,4,5,4,4,5,4,4,4], including the
stray 1-week partial first month) -- because month structure comes from
the calendar dates (fixed start_date across all seeds), not from the
spend-simulation seed. So this loop confirms robustness to different
demand/spend-level realisations of the plan, NOT to a different
calendar structure (different start date, different total plan length,
no partial first month). That remains genuinely untested.

**Decision — pass, with a scoped caveat.** The winning design is not an
artifact of one specific random spend draw. Update the "Current best"
table's caveats to reflect this loop closing the "different plan draw"
gap partially (spend-level robustness: done; calendar-structure
robustness: still open).

**Next up (loop 14, or defer to the "run all at the end" pass
called for):** vary the plan's start_date/length to get genuinely
different month-length patterns (e.g. no partial first month, or a
different mix of 4- vs 5-week months) and confirm round-robin
assignment and the t=4 blackout still work when there are e.g. two
5-week months in a row, or exactly 12 clean months with none dropped.

### Loop 14 — calendar-structure robustness (targets: confirm winner isn't overfit to this calendar)

Ran `roundrobin_plus_edge_layer` (t=4, edge_cap=15%) over 8 history start
dates x 3 plan lengths (48/52/56 wk) x 3 report seeds = 72 runs, plan
spend seed fixed at 1. Different start dates give different month-length
patterns (partial first/last months of 1-2 weeks, none, etc). Script:
`loop14_calendar_robustness.py`; raw CSV `loop_data/loop14_calendar_robustness_raw.csv`.

**Result:** zero crashes. Mean variance_cv 0.166 (std 0.010, max 0.200),
bias 23.2% (std 8.0, range 9-39), identifiability 16.2 (std 0.9, range
14.1-18.5), cost 1.79%, max spike 2.77x mean (range 2.27-4.14x).
Unphased on the same calendars: 0.322 / 56.4 / 42.0. Winner beats
Unphased on every calendar tested.

**Caveats:** (1) bias is noisier here than loop 13 (mean 23 vs 19) --
some calendars with 2-week partial months at both ends sit at the higher
end, and the ~2.8x mean spike is above the ~2.4x seen on the canonical
calendar; worst case 4.14x is a deployability flag worth a look (likely a
short partial month acting as recipient). (2) Start dates 2023-01-30 and
2023-05-01 returned IDENTICAL numbers -- probably same month-length
pattern, but unverified; check before trusting them as independent.
(3) Channel-count sensitivity still untested.

**Next up (loop 15):** find which calendars produce the >3.5x spikes and
test excluding short (<=2wk) months from recipient eligibility.

### Loop 15 — exclude short months as recipients (targets: deployability)

Added `min_recipient_weeks` (default 2 = old behaviour) to
`cross_month_blackout_roundrobin_schedule` / `roundrobin_plus_edge_layer`
in `design.py`. Ran on the 3 spike-prone start dates (2023-01-02/16/23) x
3 plan lengths x 3 seeds, min_recipient_weeks 2 vs 4. All >3.3x spikes in
loop 14 came from calendars with 2-3 week partial months (recipient
eligibility was >=2 weeks).

| min_recipient_weeks | variance | bias % | id | cost % | mean spike | max spike |
|---|---|---|---|---|---|---|
| 2 (old) | 0.165 | 21.0 | 15.9 | 1.82 | 3.06x | 4.14x |
| 4 | 0.171 | 23.7 | 16.1 | 1.76 | 2.60x | 3.03x |

**Decision:** spike guard works (max 4.14x -> 3.03x, mean -0.46x) for a
small bias increase (+2.7pts, inside the ~8pt noise) and slightly lower
cost. Recommend `min_recipient_weeks=4` as the default. Identical on
canonical calendar. Not yet made the default in code (still 2).

### Loop 16 — multi-year phasing: see table under "Current best".

**Next up (loop 17):** channel-count sensitivity (5 / 20 channels) and
re-run multi-year table with min_recipient_weeks=4 and more seeds; then
the "run all at the end" seed-robustness pass.

### Loop 17 — channel-count sensitivity + multi-year re-run with min_recipient_weeks=4

Script `loop17.py` (modes `channels` / `multiyear`), raw CSVs `loop_data/loop17_*_raw.csv`. mrw=4, 4-5 seeds.

Channel count (104wk history, 52wk plan, canonical calendar), winner vs Unphased:

| channels | variance | bias % (unphased) | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| 5 | 0.151 | 8.9 (45.6) | 15.1 | 1.73 | 2.35x |
| 10 | 0.169 | 18.0 (72.4) | 15.6 | 1.69 | 2.38x |
| 20 | 0.210 | **70.7** (103.8) | 18.1 | 1.69 | 2.47x |

**Finding: the design degrades with channel count.** Bias is fine at 5-10 channels but at 20 it recovers only about a third of unphased's bias (70.7 vs 103.8). Likely cause (unverified): with ~13 months in the plan and only 3-4 eligible blackout months (t=4 needs >4-week months), 20 channels must share blackout months, so round-robin can't avoid cross-channel collisions and cross-channel orthogonality is lost. Spike/cost stay flat. Loop 18 should test this: count blackout collisions per month, and try 2 blackout windows per channel group or t=3 for large channel counts.

Multi-year (52wk warm-up history, phased every year, mrw=4, 5 seeds, 10 channels):

| years | variance | bias % (unphased) | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| 1 | 0.188 | 22.7 (99.6) | 12.9 | 1.77 | 2.51x |
| 3 | 0.092 | 13.1 (48.2) | 8.8 | 1.70 | 2.66x |

Confirms loop 16: 3-year gains hold with the 4-week guard (bias 13.1 vs 10.6 with mrw=2, within noise).

### Loop 18 — is the 20-channel bias collapse caused by blackout-month collisions? (targets: channel-count scaling)

Hypothesis from loop 17: only 3 months (>4 wk) are eligible for a blackout, so 20 channels pile ~6.7 per month. Added `min_blackout_month_weeks` (default None = old behaviour) to `cross_month_blackout_roundrobin_schedule` / `roundrobin_plus_edge_layer`; set to 4 so 12 months are eligible (dark weeks capped at 3 in 4-wk months). Script `loop18.py`, 4 seeds, mrw=4.

| channels | eligible blackout months | variance | bias % | id | cost % |
|---|---|---|---|---|---|
| 10 | 3 (old) | 0.169 | 18.0 | 15.6 | 1.69 |
| 10 | 12 | 0.152 | 26.3 | 18.1 | 1.22 |
| 20 | 3 (old) | 0.210 | 70.7 | 18.1 | 1.69 |
| 20 | 12 | 0.199 | 68.7 | 20.4 | 1.19 |

**Hypothesis rejected.** Spreading blackouts over 12 months doesn't fix 20-channel bias (70.7 -> 68.7) and hurts 10-channel bias/id (shorter effective blackout, 3 wk in 4-wk months); it only lowers cost. Keep the default (None).
**Likelier cause:** the package itself warns "156 observations across 20 channels (7.8 per channel) ... estimates tend to be very wide and unstable even before collinearity". Unphased bias at 20 channels is already 104%, so this is a sample-size limit of the fit, not a design flaw.
**Next up (loop 19):** 20 channels with longer history (e.g. 208-260 obs, >=10/channel) to confirm the design recovers when the fit is adequately powered.

**Scope decision (2026-09-19):** 20 channels is out of scope; loop 19 (20ch with longer history) dropped. Supported range is ~5-10 channels; treat 20ch as a known limitation (fit is under-powered at <10 obs/channel), not something to design around.
**Next up (revised):** make `min_recipient_weeks=4` the default; final seed-robustness pass on the winner (more seeds, 5-10 channels, 1- and 3-year).

### Loop 19 — final robustness pass + new default (2026-09-19)

`min_recipient_weeks` default changed 2 -> 4 in `design.py` (loop 15). Final pass: `roundrobin_plus_edge_layer` (t=4, edge_cap=15%, mrw=4), 8 seeds, 52wk warm-up history, phased every year, canonical calendar. Script `loop19_final.py`, raw `loop_data/loop19_final_n{5,10}_raw.csv`.

| channels | years | lever | variance | bias % (sd, max) | identifiability | cost %/yr | spike mean (max) |
|---|---|---|---|---|---|---|---|
| 5 | 1 | Unphased | 0.396 | 66.4 (11.4, 79.0) | 43.7 | 0 | 1.0x |
| 5 | 1 | winner | 0.155 | 11.8 (4.7, 20.0) | 13.0 | 1.73 | 2.46x (2.68) |
| 5 | 3 | Unphased | 0.271 | 19.6 (7.7, 32.7) | 45.1 | 0 | 1.0x |
| 5 | 3 | winner | 0.086 | 5.3 (1.9, 8.2) | 9.7 | 1.70 | 2.56x (2.68) |
| 10 | 1 | Unphased | 0.432 | 102.9 (13.7, 125.5) | 41.3 | 0 | 1.0x |
| 10 | 1 | winner | 0.181 | 26.9 (9.7, 44.2) | 13.0 | 1.77 | 2.50x (2.79) |
| 10 | 3 | Unphased | 0.252 | 50.9 (6.6, 64.3) | 43.4 | 0 | 1.0x |
| 10 | 3 | winner | 0.090 | 12.1 (1.9, 14.7) | 8.8 | 1.70 | 2.63x (2.79) |

**Verdict: winner is robust.** Beats Unphased on every metric in all 4 settings across 8 seeds; worst-case spike 2.79x (well under Blackout's 6.5x); annual cost flat at ~1.7%. Bias is the noisy metric at 1 year (10ch sd 9.7, max 44%) and tightens sharply at 3 years (sd 1.9). Bias falls with more years and fewer channels (5ch/3yr: 5.3%).
Caveat: bias is unphased-relative, so absolute values depend on the scenario's demand process; 1-year 10-channel bias (~27%) is the weakest cell.
**Open / not done:** promote design into `_phaser.py` (nothing wired in yet); `edge_cap_pct` should ship as a configurable knob (loop 11).

### Loop 20 — 10 channels, 3 years, winner vs ALL default strategies (4 seeds)

Script `loop20_allstrat.py`; raw `loop_data/loop20_allstrat_seeds*_raw.csv`. 52wk warm-up + 156wk plan; built-in levers applied across the full plan, winner phased per year. Sorted by bias.

| strategy | variance | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| Blackout (dark=4, prob=1.0) | 0.031 | 3.268 | 1.24 | 15.031 | 13.45x |
| Blackout (dark=3, prob=0.8) | 0.037 | 4.164 | 1.946 | 9.17 | 13.82x |
| +/-80% (edge, balanced) | 0.06 | 5.998 | 9.435 | 2.572 | 2.81x |
| +/-60% (edge, balanced) | 0.078 | 7.749 | 14.847 | 1.36 | 2.19x |
| +/-80% (seesaw) | 0.073 | 8.456 | 16.266 | 1.964 | 2.28x |
| +/-60% (seesaw) | 0.094 | 11.281 | 22.752 | 1.059 | 1.90x |
| Blackout (dark=1) | 0.075 | 11.441 | 14.494 | 1.623 | 1.78x |
| +/-40% (edge, balanced) | 0.109 | 11.59 | 23.182 | 0.584 | 1.70x |
| +/-80% (uniform) | 0.103 | 11.624 | 20.7 | 0.751 | 2.89x |
| roundrobin_plus_edge_layer | 0.092 | 12.946 | 8.822 | 1.692 | 2.69x |
| +/-60% (uniform) | 0.131 | 14.865 | 27.832 | 0.399 | 2.16x |
| +/-40% (seesaw) | 0.129 | 16.388 | 30.403 | 0.46 | 1.56x |
| +/-40% (uniform) | 0.171 | 20.749 | 35.712 | 0.173 | 1.67x |
| +/-20% (edge, balanced) | 0.171 | 23.558 | 35.975 | 0.145 | 1.32x |
| +/-20% (seesaw) | 0.192 | 26.552 | 38.959 | 0.115 | 1.27x |
| +/-20% (uniform) | 0.22 | 34.37 | 42.983 | 0.045 | 1.30x |
| unphased | 0.252 | 49.006 | 45.418 | 0.0 | 1.00x |

**Read:** the winner is NOT dominant at 3 years. It is the best client-suitable option on identifiability among low-cost designs (8.8 vs 9.4 for +/-80% edge, 14.8 for +/-60% edge), but +/-60% edge-balanced beats it on variance (0.078 vs 0.092), bias (7.7 vs 12.9), cost (1.36 vs 1.69%) and spike (2.19x vs 2.69x), losing only on identifiability (14.8 vs 8.8). +/-80% edge-balanced beats it on variance/bias with similar identifiability but higher cost (2.57%). Blackout wins rigor outright but at 9-15% cost and ~13x single-week spikes. Winner's real case: best identifiability per unit of cost/spike; needs a decision on whether identifiability matters more than bias/variance.
**Caveat:** 4 seeds only; bias sd ~2 at 3yr so gaps under ~4 pts are not significant.

### Loop 21 — month-level orthogonal step design (targets: deployability/low edit count while keeping rigor)

Run in parallel with the `Redistribute` build (separate chat); scratch only, `src/` untouched. New `monthstep.py`: each month, each channel's whole month is scaled by (1 +/- step%), signs from a Hadamard(12) non-constant column (balanced 6 up/6 down per channel, orthogonal across channels), weekly shape within a month unchanged, annual total restored per 52wk block, fresh shuffle per block. So <=12 budget changes per channel per year, all at month boundaries, no weekly noise, no pauses. Checked: annual drift ~1e-13, no negatives, mean |cross-channel corr| of log-ratios 0.015. `loop21_monthstep.py`, 10ch, 3yr, 4 seeds, same harness as loop 20.

| strategy | variance | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| monthstep 80% | 0.038 | 4.7 | 4.6 | 5.77 | 1.91x |
| +/-80% edge-balanced | 0.060 | 6.0 | 9.4 | 2.57 | 2.81x |
| monthstep 60% | 0.051 | 6.2 | 8.6 | 2.96 | 1.67x |
| +/-60% edge-balanced | 0.078 | 7.7 | 14.8 | 1.36 | 2.19x |
| **monthstep 40%** | **0.075** | **9.2** | **16.5** | **1.25** | **1.44x** |
| winner (round-robin + edge 15%) | 0.092 | 12.9 | 8.8 | 1.69 | 2.69x |
| monthstep 20% | 0.134 | 18.1 | 31.0 | 0.31 | 1.22x |
| unphased | 0.252 | 49.0 | 45.4 | 0 | 1.0x |

**Read:** promising. monthstep40 beats the winner on variance (0.075 vs 0.092), bias (9.2 vs 12.9), cost (1.25 vs 1.69%) and spike (1.44x vs 2.69x), losing only on identifiability (16.5 vs 8.8). monthstep60 matches the winner's identifiability (8.6) with better variance/bias and a lower spike (1.67x) but higher cost (2.96%). Also far fewer edits than any weekly design (<=12/channel/yr, step held ~4 wk). Cost is real here because steps are large relative to weekly-edge noise: cost scales with step size.
**Caveats:** 4 seeds; bias sd 1-2 pts at >=40%, 5 at 20%. Not run at 5 channels or 1 year; calendar robustness untested (partial months get multiplier 1 if <2 wk). No learning-phase modelling: a 40% month step is a single >20% edit per month, but only ~12/yr vs 52 for weekly edge.

### Loop 22 — partial blackout (floor instead of zero) (targets: deployability, avoiding pause/restart)

`partialblackout.py`: winner's round-robin structure, but dark weeks are cut to floor% of plan (not 0); freed = (1-floor) x plan goes to the recipient month, same 15% edge layer. Same harness, 10ch, 3yr, 4 seeds (floor 0 = the winner; matches loop 20 exactly).

| floor | variance | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| 0% (winner) | 0.092 | 12.9 | 8.8 | 1.69 | 2.69x |
| 10% | 0.100 | 14.2 | 12.7 | 1.23 | 2.54x |
| 25% | 0.114 | 16.2 | 18.4 | 0.81 | 2.32x |
| 50% | 0.144 | 20.3 | 28.4 | 0.38 | 1.94x |

**Read:** another smooth dial, no free lunch. Every step up in floor cuts cost and spike but degrades all three rigor metrics, identifiability fastest (8.8 -> 12.7 -> 18.4 -> 28.4): the zero-spend anchor is what buys identifiability. A 10% floor keeps the campaign live (avoids a true pause) for +3.9 identifiability and +1.3 bias. Even so, spike stays >2.3x because the recipient month still absorbs the freed budget, so a floor does not fix the spike; monthstep40 (1.44x) dominates floor10 on variance, bias, and spike while costing the same (1.25 vs 1.23%), losing on identifiability (16.5 vs 12.7).
**Next:** if month-level steps are worth pursuing: 5ch/1yr/calendar robustness, more seeds, and a step+blackout hybrid (month steps plus a single floor/blackout per channel) to recover identifiability; then decide whether it joins the `Redistribute` family.

### Loop 23 — acceptance re-run through the real `DiscoveryReport` (2026-09-20)

`Redistribute` is now a first-class strategy in `_phaser.py` (branch `feat/redistribute-strategy`, uncommitted), swept by `_default_levers` at +/-20/40/60/80% (label `"+/-X% (redistribute + edge)"`, 20 candidates total). Every row below, including the four new ones, is phased by `DiscoveryReport._phase` -> `_generate_phased_schedule`; no scratch harness, so this is the like-for-like comparison loop 20 could not give. Redistribute is phased per 12-month block inside the class (36 months -> 3 blocks, fresh round-robin each), not per 52wk chunk as in loop 16/19/20. On a 1-year plan the class output is bit-identical to `design.roundrobin_plus_edge_layer` (checked at caps 20/60, seeds 0/3, 10 channels).

Script `loop23_acceptance.py` (summary: `loop23_summarise.py`), raw `loop_data/loop23_*_raw.csv`. Scenario = loop 20's: 10 channels, 52wk warm-up history + 156wk plan, correlation 0.7, overview (b, lambda) pairs cycled, `fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8)`, **8 report seeds (0-7)**. Sorted by bias; spike = max weekly spend / that week's plan (mean over seeds, max over seeds).

| strategy | variance | bias % (sd) | identifiability | cost % | spike mean (max) |
|---|---|---|---|---|---|
| Blackout (dark=4, prob=1.0) | 0.032 | 3.8 (0.9) | 1.22 | 15.01 | 14.5x (20.7) |
| Blackout (dark=3, prob=0.8) | 0.038 | 4.5 (1.5) | 1.96 | 9.17 | 16.2x (20.7) |
| +/-80% (edge, balanced) | 0.061 | 6.2 (1.5) | 9.58 | 2.57 | 2.6x (3.1) |
| **+/-80% (redistribute + edge)** | 0.051 | 6.2 (2.0) | 3.81 | 4.17 | 4.7x (5.5) |
| **+/-60% (redistribute + edge)** | 0.061 | 7.7 (2.5) | 4.94 | 2.97 | 3.9x (4.5) |
| +/-60% (edge, balanced) | 0.080 | 8.3 (2.0) | 15.08 | 1.36 | 2.1x (2.3) |
| **+/-40% (redistribute + edge)** | 0.073 | 9.8 (2.8) | 6.42 | 2.21 | 3.3x (3.7) |
| +/-80% (seesaw) | 0.074 | 10.0 (2.6) | 15.85 | 1.95 | 2.3x (2.4) |
| Blackout (dark=1) | 0.077 | 10.1 (3.3) | 14.27 | 1.63 | 1.8x (1.9) |
| **+/-20% (redistribute + edge)** | 0.087 | 12.0 (3.0) | 8.19 | 1.78 | 2.7x (2.9) |
| +/-80% (uniform) | 0.103 | 12.4 (2.6) | 21.01 | 0.74 | 2.8x (3.1) |
| +/-40% (edge, balanced) | 0.113 | 12.6 (3.3) | 23.29 | 0.59 | 1.7x (1.8) |
| +/-60% (seesaw) | 0.095 | 13.7 (4.0) | 22.05 | 1.05 | 1.9x (2.0) |
| +/-60% (uniform) | 0.131 | 16.4 (3.1) | 28.19 | 0.39 | 2.1x (2.2) |
| +/-40% (seesaw) | 0.129 | 20.4 (6.5) | 29.83 | 0.45 | 1.6x (1.6) |
| +/-40% (uniform) | 0.170 | 23.3 (4.1) | 35.38 | 0.17 | 1.7x (1.7) |
| +/-20% (edge, balanced) | 0.177 | 25.1 (6.1) | 35.15 | 0.15 | 1.3x (1.3) |
| +/-20% (seesaw) | 0.190 | 33.7 (11.3) | 38.14 | 0.11 | 1.3x (1.3) |
| +/-20% (uniform) | 0.219 | 37.3 (5.7) | 41.60 | 0.04 | 1.3x (1.3) |
| unphased | 0.252 | 50.9 (6.6) | 43.38 | 0.00 | 1.0x |

**Read.**
- The built-in rows reproduce loop 20 (e.g. +/-80% edge 0.061 / 6.2 / 9.6 / 2.57%, loop 20: 0.060 / 6.0 / 9.4 / 2.57%), so the harness and the class agree.
- The Redistribute family buys identifiability, and it is the only deployable-cost family that does: 3.8-8.2 across the four rows vs 9.6-41.6 for every edge/seesaw/uniform row and 14 for Blackout(dark=1). Only the two heavy Blackout rows (1.2, 2.0) are lower, at 9-15% cost and 15-20x spikes.
- Against the matching-cost edge row it is not a dominant swap. +/-20% redistribute (1.78% cost, 2.7x spike) vs +/-60% edge (1.36%, 2.1x): identifiability 8.2 vs 15.1 (better), variance 0.087 vs 0.080 and bias 12.0 vs 8.3 (worse; bias gap ~1.5 sd, not clearly significant at 8 seeds). +/-80% redistribute matches +/-80% edge on bias (6.2 vs 6.2), beats it on variance (0.051 vs 0.061) and identifiability (3.8 vs 9.6), for +1.6pts cost and 4.7x vs 2.6x spike.
- Spikes: 20% row 2.7x mean / 2.9x max (scope's expectation ~2.5x: ok); 60% and 80% rows 3.9x / 4.7x mean, 4.5x / 5.5x max. The scope expected 3.5-4x for these; the 80% row overshoots. The edge layer's spike multiplies the recipient-month top-up (+/-X% on top of a ~2.5x month), which is why intensity costs more spike here than for the plain edge rows.
- `_pick_winner` (rigor only) still picks `Blackout (dark=4, prob=1.0)` on all 8 seeds, unchanged by the new rows. Redistribute never wins the sweep.
- Noise: bias sd across seeds is 1-3pts for the rows that matter, so bias gaps under ~3-4pts between adjacent rows are not significant at 8 seeds; identifiability gaps are large and consistent.

**Runtime (decision (a), see below).** Same fit settings, seed 0, one core: 20 levers 285s vs 227s with the four Redistribute rows removed (+26%, ~14s per lever on this 10ch/3yr scenario; the sweep is linear in candidates, phasing itself is negligible). Redistribute rows cost the same to score as any other row. At the real `fit()` defaults (n_sims=50, n_phasing_seeds=5, id_n_sims=20), 20 levers took 773s (~13 min) on this scenario; the 16-lever figure (~610s) is extrapolated from the +26% ratio, not measured. Phasing itself is ~7ms per call, so the whole cost is diagnostics. **Decision (a), 2026-09-20: keep the family in the default sweep.**

**Two things this run turned up.**
1. **Monthly totals are not preserved by Redistribute.** The project scope notes say "monthly (and annual) budget conserved exactly"; the ported algorithm moves the freed budget into a different month, so only each channel's 12-month-block total (and every month other than its blackout and recipient months) is conserved. The edge layer preserves the post-redistribution monthly totals. Tests assert exactly that. Consequences handled in code: the report's "monthly totals unchanged" copy (Section 3 caption, Cost paragraph, Section 4) switches to annual wording when the winner or pinned strategy is a Redistribute one, and `BudgetPhaser`'s `max_monthly_deviation_pct` is nonzero for it.
2. **The report had no max-spike column.** Loop 20 and this run computed it externally. Added (2026-09-20): the appendix comparison table now has a "Peak week" column (`_peak_week_multiple`: largest weekly spend as a multiple of that week's plan, across channels; same definition as `max_spike` here).

**Decided (2026-09-20): (a) family stays in the default sweep; (b) the client-suitable default row is +/-20% (redistribute + edge)** (1.78% cost, 2.7x mean / 2.9x max peak week; chosen because its smaller week-to-week moves are least likely to trigger ad-platform learning-phase resets). The 40/60/80% rows stay in the sweep for comparison. No code default changed: `_pick_winner` is untouched and `Redistribute()`'s own `edge_cap_pct` default is still 15.

**Original open decisions:** (a) keep the family in the default sweep or put it behind a flag, given +26% runtime; (b) which row is the client-suitable default (candidate: +/-20%, 2.7x / 1.78%; or +/-40%, 3.3x / 2.21%, if 3.7x worst-case spikes are deployable). Not decided in code: nothing is committed and the default winner logic is unchanged.

### Loop 23 — hybrid (blackout/floor + month steps) and month-step robustness

Script `loop23.py` (modes `hybrid`, `robust`); raw `loop_data/loop23_*_raw.csv`. Scratch only. Same harness as loops 20-22 (52wk warm-up history, phased every year, canonical plan seed 1).

**Hybrid** (blackout at floor F% then month steps S%, `hyb_F_S`; 10ch, 3yr, 4 seeds):

| strategy | variance | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| ms 60 | 0.051 | 6.2 | 8.6 | 2.96 | 1.67x |
| hyb 0/40 | 0.064 | 8.2 | 6.4 | 2.79 | 3.07x |
| hyb 25/40 | 0.070 | 8.8 | 11.0 | 1.91 | 2.66x |
| ms 40 | 0.075 | 9.2 | 16.5 | 1.25 | 1.44x |
| hyb 0/20 | 0.085 | 12.5 | 8.6 | 1.89 | 2.66x |
| hyb 25/20 | 0.100 | 13.7 | 17.0 | 1.01 | 2.30x |

**Read: hybrids don't earn their keep.** hyb 0/20 lands on the winner's numbers (0.092/12.9/8.8/1.69/2.69x); hyb 0/40 is dominated by ms 60 on variance, bias and spike (3.07x vs 1.67x) for only slightly lower cost and better identifiability. Adding a blackout brings back the 2.3-3x spikes without a compensating gain. Pure month steps sit on the frontier.

**Robustness** of `ms_40` / `ms_60` vs winner (8 seeds unless noted; means):

| config | strategy | variance | bias % (sd, max) | identifiability | cost % | max spike |
|---|---|---|---|---|---|---|
| 10ch 3y | winner | 0.090 | 12.1 (1.9, 14.7) | 8.8 | 1.70 | 2.63x |
| 10ch 3y | ms 40 | 0.076 | 9.4 (1.5, 11.6) | 17.7 | 1.23 | 1.44x |
| 10ch 3y | ms 60 | 0.051 | 6.5 (0.9, 8.2) | 9.4 | 2.92 | 1.67x |
| 10ch 1y | winner | 0.180 | 25.7 (6.2, 33.9) | 12.5 | 1.79 | 2.49x |
| 10ch 1y | ms 40 | 0.134 | 15.5 (2.6, 18.8) | 21.9 | 1.22 | 1.44x |
| 10ch 1y | ms 60 | 0.091 | 10.6 (1.9, 12.4) | 12.6 | 2.93 | 1.67x |
| 5ch 3y | winner | 0.086 | 5.3 (1.9, 8.2) | 9.7 | 1.70 | 2.56x |
| 5ch 3y | ms 40 | 0.078 | 5.3 (1.1, 7.1) | 18.1 | 1.24 | 1.44x |
| 5ch 3y | ms 60 | 0.053 | 3.8 (0.8, 5.2) | 9.8 | 2.93 | 1.67x |
| 5ch 1y | winner | 0.152 | 10.6 (3.8, 16.7) | 12.3 | 1.77 | 2.45x |
| 5ch 1y | ms 40 | 0.134 | 7.9 (2.1, 9.9) | 21.7 | 1.31 | 1.43x |
| 5ch 1y | ms 60 | 0.092 | 5.4 (1.7, 7.4) | 12.4 | 3.07 | 1.65x |
| 10ch 3y, other calendar A (start 01-16, 4 seeds) | ms 40 / ms 60 / winner | 0.082 / 0.056 / 0.098 | 9.3 / 5.6 / 11.8 | 17.8 / 9.6 / 9.6 | 1.23 / 2.91 / 1.70 | 1.46x / 1.70x / 2.72x |
| 10ch 3y, other calendar B (start 01-23, 4 seeds) | ms 40 / ms 60 / winner | 0.078 / 0.053 / 0.096 | 8.9 / 5.4 / 11.8 | 17.1 / 9.2 / 9.1 | 1.22 / 2.92 / 1.73 | 1.46x / 1.71x / 2.64x |

**Verdict:** month-level steps are robust across 5/10 channels, 1/3 years and calendars with 2-3 week partial months (spike stays 1.4-1.7x vs winner's 2.4-2.7x, no calendar sensitivity). Compared to the winner: `ms_40` has better variance and bias and a cheaper cost (1.2 vs 1.7%) with lower spike, but roughly half the identifiability (17.7 vs 8.8); `ms_60` matches the winner's identifiability (9.4 vs 8.8) with clearly better variance/bias/spike but costs ~1.2 pts more (2.9 vs 1.7%). The gap is largest at 1 year (10ch bias 10.6-15.5 vs 25.7). The trade is cost (revenue given up by shifting spend) vs identifiability; a step between 40-60% (e.g. 50%) is untested and likely lands near the winner's identifiability at ~2% cost.
**Not done:** ms 50%; learning-phase modelling (month steps mean ~12 edits/channel/yr, one large step per month); real-platform tolerance for ±40-60% monthly budget moves is unknown and is the main deployability question; identifiability metric definition should be checked since it dominates the ms40-vs-winner call.
**Suggested next:** decide whether month-step joins `Redistribute` as a second strategy family (`MonthStep`, 20/40/60/80) in `_default_levers`.

## Loop 24 — MonthStep as a built-in strategy; acceptance re-run through the real `DiscoveryReport`

**What changed:** `MonthStep(step_pct)` is now a first-class `DeviationSpec` (`_phaser.py`), with four rows at 20/40/60/80% in `_default_levers` (labels like `"+/-40% (month step)"`). The default sweep is now 24 candidates (unphased + 12 float + 4 Redistribute + 4 MonthStep + 3 Blackout). Every row, MonthStep included, is phased by `DiscoveryReport._phase`. Harness: `loop24_monthstep_acceptance.py` (raw CSVs `loop_data/loop24_*_raw.csv`), summariser `loop24_summarise.py`. Same scenario and `fit(n_sims=20, n_phasing_seeds=2, id_n_sims=8)` as loops 20/23, plus an extra `"+/-50% (month step) [extra]"` row (not in the default sweep) to fill the untested 50% step. Sweeps therefore ran 25 rows.

**Deviations from the scratch reference to know about (loop 21's `monthstep.py`):**
- Blocks are 12 *calendar* months from plan start (same rule as Redistribute; tail block < 6 months merged into previous; partial-month stubs < 2 weeks left at plan), not 52-week chunks. Bit-identical to scratch for a 52-week plan starting on a month boundary (max diff 0.0); multi-year plans differ slightly.
- Sign columns: >11 channels reuses Hadamard columns with a fresh per-channel row permutation and emits a `UserWarning` (orthogonality is then approximate). Decision (a), agreed.
- This file has two "Loop 23" headings (Redistribute acceptance, and the hybrid/robustness batch from the other chat); this is Loop 24.

**Results, 10ch / 3y, 8 seeds (0-7), mean over seeds** (`edits` = deviation-ratio changes per channel per year, measured externally, not shown in the report):

| lever | variance | bias % | identifiability | cost % | max spike (mean) | edits/ch/yr |
|---|---|---|---|---|---|---|
| unphased | 0.252 | 50.9 | 43.4 | 0 | 1.0x | 0 |
| Blackout (dark=4, prob=1.0) | 0.032 | 3.80 | 1.22 | 15.0 | 14.5x | 20.7 |
| +/-80% (month step) | 0.038 | 4.83 | 4.93 | 5.85 | 1.93x | 6.7 |
| +/-80% (edge, balanced) | 0.061 | 6.15 | 9.58 | 2.57 | 2.64x | 40.8 |
| +/-80% (redistribute + edge) | 0.051 | 6.24 | 3.81 | 4.17 | 4.69x | 38.8 |
| +/-60% (month step) | 0.051 | 6.41 | 9.45 | 2.98 | 1.68x | 6.7 |
| +/-50% (month step) [extra] | 0.061 | 7.61 | 12.9 | 2.01 | 1.57x | 6.7 |
| +/-40% (month step) | 0.075 | 9.38 | 17.7 | 1.25 | 1.45x | 6.7 |
| +/-20% (month step) | 0.134 | 18.2 | 31.4 | 0.31 | 1.22x | 6.7 |

Sweep winner (pure rigor): `Blackout (dark=4, prob=1.0)` in 8/8 seeds, as before. MonthStep +/-80% is 3rd on bias overall, ahead of every float and Redistribute row, at 5.85% cost vs Blackout's 15.0%, and with a 1.9x max spike vs 14.5x. MonthStep +/-60% (bias 6.4%, cost 3.0%) is roughly level with edge and redistribute +/-80% (6.2% / 6.2%, cost 2.6% / 4.2%) but with 1.7x spike and 6.7 edits/ch/yr vs 2.6-4.7x and ~40. The 50% step sits exactly between 40 and 60 (bias 7.6 vs 9.4 / 6.4), so the family is smooth and nothing unexpected happens between rows.

**Other configurations (4 seeds each, +/-80% MonthStep vs Blackout dark=4):**

| config | ms80 bias % / cost % | Blackout bias % / cost % | unphased bias % | sweep winner |
|---|---|---|---|---|
| 5ch 3y | 3.08 / 5.86 | 1.56 / 14.6 | 19.7 | Blackout 4/4 |
| 10ch 1y | 7.76 / 5.84 | 9.14 / 15.4 | 82.0 | MonthStep 80 in 2/4, Blackout 2/4 |
| 5ch 1y | 4.13 / 5.94 | 3.87 / 15.1 | 33.1 | Blackout 3/4, MonthStep 80 in 1/4 |

At 1 year the +/-80% month step matches or beats Blackout on bias for 10ch, at well under half the cost.

**Runtime (10ch/3y, 2 CPUs):** 20 rows (no MonthStep, no extra) 283s; 25 rows 355-364s per fit (mean 360s). Default 24 rows is therefore about 340s (about +20% vs the pre-MonthStep sweep; ~+14s per row). 5ch/3y: 98s; 10ch/1y: 199s; 5ch/1y: 56s.

**Checks:** annual conservation exact (tests, all seeds); no negatives; deterministic per seed; balanced 6-up/6-down blocks; sign columns orthogonal; existing levers unchanged (all pre-existing lever rows unchanged in spec and label). `to_html()` renders with the MonthStep rows and annual-not-monthly wording when a MonthStep row wins or is pinned. Full suite 961 passed, 1 skipped; ruff format/check clean on `src` and `tests`.

**Decision (b):** MonthStep stays in the default sweep (24 rows, about +20% fit time). **Not done:** learning-phase modelling; notebook 05 still shows the old 16-row output.
