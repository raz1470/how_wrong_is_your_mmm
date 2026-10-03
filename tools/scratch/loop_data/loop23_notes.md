
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
