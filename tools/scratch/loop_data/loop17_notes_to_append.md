
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
