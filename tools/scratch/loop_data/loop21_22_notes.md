
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
