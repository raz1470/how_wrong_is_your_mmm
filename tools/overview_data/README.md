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
| 4: Monte Carlo standard errors under the strategy table | `mc_error.py` | `results/mc_error.json` |
| 5 and 9: cost in £ and as a share of sales, and its range across saturation exponents | `cost.py` | `results/cost.json` |
| 6: does it scale, 5 to 15 channels | `section6/` | `section6/results/scale_summary_4seed.json` |

## Section 6

Each point is averaged over four simulated datasets, with (demand,
history, plan) seeds (6, 12, 13), (106, 112, 113), (206, 212, 213) and
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
- `example_scenario.py` and `corr_sweep.py` were re-run on the current
  package at 100 bias draws. Only the bias figures moved; variance,
  saturation, adstock, cost and the winner are identical to the earlier
  run.
- `mc_error.py` reproduces the table's bias column from its own draws.
- `phase_panels.py` and `winner_spend.py` reproduce their committed
  results.
- The section 6 full fits have not been re-run since the article was
  built.

## Bias draws

Every bias figure in the article is an average over 100 demand draws, the
`DiscoveryReport.fit()` default. Sections 1 to 5 were first built at 15
draws; at 100, Search Generic's bias no longer improves under Combined
(7.6% to 8.6%, within simulation error) and TV's unphased bias is 49%, not
42%.
