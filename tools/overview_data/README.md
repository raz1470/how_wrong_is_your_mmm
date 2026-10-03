# Data behind `docs/overview.html`

The scripts that compute the article's numbers, and the results they
produced. The article's charts read constants written into the page, so
these outputs were copied in by hand. Nothing here runs in CI.

Run every script from the repo root with `uv run python <script>`.

| Article section | Script | Result |
|---|---|---|
| 1, 2, 4, 5: the example scenario, the unphased problem, the strategy table, the winner's impact, time to benefit | `example_scenario.py` | `results/example_scenario.json` |
| 3: bias against the demand link, variance against channel correlation | `corr_sweep.py` | `results/corr_sweep.json` |
| 4: one phased schedule per strategy | `phase_panels.py` | `results/phase_panels.json` |
| 5: the winner's phased spend by channel | `winner_spend.py` | `results/winner_spend.json` |
| 4: Monte Carlo standard errors for the strategy table (not quoted in the article) | `mc_error.py` | `results/mc_error.json` |
| 5 and 9: cost in £ and as a share of sales, and its range across saturation exponents | `cost.py` | `results/cost.json` |
| 6: does it scale, 5 to 15 channels | `section6/` | `section6/results/scale_summary_4seed.json` |

## Section 6

Each point is averaged over four simulated datasets, with (demand,
history, plan) seeds (6, 37, 13), (106, 112, 113), (206, 212, 213) and
(306, 312, 313), tagged `base`, `s1`, `s2`, `s3`.

1. `fullfit.py <n> <tag>` for n in 5, 10, 15 and each tag: variance,
   saturation, adstock and cost. About 23 minutes per 15-channel fit.
2. `harness.py <n> 100 <tag>` for the same twelve runs: bias at 100
   draws. A few minutes each.
3. `summarise.py`: improvement on the unphased plan, as
   (mean unphased - mean strategy) / mean unphased over the four datasets.

The per-draw bias files (`results/bias_*.csv`) are not committed. Step 2
regenerates them.

## What has been checked

- `summarise.py` reproduces `scale_summary_4seed.json`, which matches the
  `SCALE` constant in the article.
- Every script for sections 1 to 5 was re-run on history seed 37, and the
  article's chart constants were rebuilt from the results.
- `mc_error.py` reproduces the table's bias column from its own draws.
- `phase_panels.py` and `winner_spend.py` give the same results as on
  seed 12, since they depend on the plan only.
- The section 6 full fits and bias runs were re-run on the new example
  dataset and noise level.

## The example dataset

Sections 1 to 5 use demand seed 6, history seed 37 and plan seed 13.

The history seed was 12 until the article was close to done. It was
changed because of how the unphased bias splits across channels. Across
25 history seeds the average bias is steady (24% to 32% unphased, about
15% under Combined) and each channel's bias under Combined barely moves
(TV about 30%, Meta 13%, Search Generic 8.5%, TikTok 9.5%). What does move
is how the unphased bias is shared out: Search Generic's ran from -2% to
21%. On seed 12 it was 7.6%, already below its phased level, so the
article had one channel that did not improve. All four channels improve
on 10 of the 25 seeds, and seed 37 is the one with the clearest margins.
The headline figures barely changed: bias 28.8% to 15.3% (was 28.3% to
15.1%), variance 0.23 to 0.07.

Section 6 was re-run to match. Its `base` dataset uses history seed 37,
and all four datasets use the four-channel noise level from the new
example, so all twelve fits and bias runs were redone.

## Bias draws

Every bias figure in the article is an average over 100 demand draws, the
`DiscoveryReport.fit()` default.
