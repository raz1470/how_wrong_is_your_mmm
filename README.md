# How Wrong Is Your MMM?

**How using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.**

Take the marketing mix model (MMM) you use to allocate your marketing budget across channels. Run it again on a slightly different slice of history, same channels, same market. Would it give you the same answer?

For most brands, no. TV, Meta, and Search budgets move together because the same planning cycle drives them all, and that makes it hard for an MMM to tell their individual effects apart. The result is elasticity estimates that shift every time you refit, not because the market changed, but because the data was never informative enough to pin them down.

This package quantifies that problem and recommends a fix.

---

## The three-part solution

**Part 1 — Diagnose.** Simulate many plausible histories of your market and measure how much your elasticity estimates swing. The coefficient of variation (CV) is your "how wrong" number: at mean channel correlation 0.7, TV's CV is ~36%.

**Part 2 — Phase.** Recommend a weekly spend schedule that breaks the correlation between channels while keeping monthly totals exactly the same, either a continuous nudge to each week's split, or `Blackout`, a harder on/off switch that takes a channel fully dark some weeks and makes it up on the weeks it stays on. At the package default (±40% continuous), one year of phasing cuts CV by ~30%. `Blackout` alone reaches 47–59%, using no more than one dark week a month.

**Part 3 — Retrain.** Refit your MMM on the phased data. The de-correlated spend does the work: elasticity estimates come back measurably tighter, without waiting years for it to accumulate.

---

## Guides

[**Introduction**](https://raz1470.github.io/how_wrong_is_your_mmm/introduction.html)
An introduction to the problem and the fix.

[**Research**](https://raz1470.github.io/how_wrong_is_your_mmm/research.html)
How the diagnostic and the phasing algorithm actually work, and the research behind every number.

[**API Reference**](https://raz1470.github.io/how_wrong_is_your_mmm/api/)
Full class and function docs for `CollinearityDiagnostic`, `BudgetPhaser`, and `Blackout`.

[**Example report**](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html)
A full `ReportBuilder` report on simulated data, end to end — the same output `to_html()` produces for a real client.

---

## Quick start

```bash
pip install how-wrong-is-your-mmm  # coming to PyPI
```

Or from source:

```bash
git clone https://github.com/raz1470/how_wrong_is_your_mmm
cd how_wrong_is_your_mmm
uv venv --python 3.12 && uv sync
```

```python
from how_wrong_is_your_mmm import CollinearityDiagnostic, BudgetPhaser

# Diagnose — synthetic spend
diag = CollinearityDiagnostic(correlation=0.7, spend_seed=0)
diag.fit()
diag.summary()
# channel  true_elasticity  mean_estimated  coef_of_variation
#      tv             0.30           0.329              0.357
#    meta             0.50           0.503              0.288
#  search             0.40           0.357              0.623

# Diagnose — your own spend data
diag = CollinearityDiagnostic(spend_df=my_spend_df)
diag.fit()
diag.summary()   # same output, personalised to your correlation structure

# Phase — recommend a de-correlated spend schedule
# history: your multi-year spend history (DatetimeIndex)
# plan:    the upcoming year's spend plan (DatetimeIndex, same channels)
phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_   # 52-week DataFrame, monthly totals guaranteed to match
```

```python
# Report — diagnose + phase, packaged into one client-ready HTML report
from how_wrong_is_your_mmm import ReportBuilder

rb = ReportBuilder(history_df=history, plan_df=plan, client_name="Acme Co")
rb.fit()
rb.to_html("reports/acme_co.html")     # self-contained HTML, open it in a browser
rb.schedule_csv("reports/acme_co_schedule.csv")   # the recommended weekly schedule as a CSV
```

`reports/` is git-ignored by default (see `.gitignore`) — client data has no business in a public repo. Save your own generated reports there, or wherever suits your workflow. See it end to end at the [example report](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html) above.

---

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| [`01_diagnostic_walkthrough`](notebooks/01_diagnostic_walkthrough.ipynb) | Shows how unreliable your elasticity estimates get as your channels become more correlated, then runs the same check on your own spend data |
| [`02_phaser_walkthrough`](notebooks/02_phaser_walkthrough.ipynb) | Walks through `BudgetPhaser` end to end: how much phasing helps, and the actual recommended weekly schedule it produces |
| [`03_time_to_benefit`](notebooks/03_time_to_benefit.ipynb) | How long you need to phase your budget before you see a real improvement |
| [`04_channel_scaling_walkthrough`](notebooks/04_channel_scaling_walkthrough.ipynb) | Checks that the diagnostic and the fix still work when you have more than three channels |
| [`05_bayesian_comparison`](notebooks/05_bayesian_comparison.ipynb) | Checks whether switching to a Bayesian model fixes the problem on its own (it doesn't, not by much) |

---

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

222 tests. Python 3.12+. MIT licence.

The [API reference](https://raz1470.github.io/how_wrong_is_your_mmm/api/) is built with `mkdocs` + `mkdocstrings` from the docstrings in `src/`, and the built site is committed under `docs/api/` (this repo has no CI build step for GitHub Pages, so the site has to be built and committed locally, same as the notebooks). To preview changes locally:

```bash
uv run mkdocs serve
```

To rebuild the committed site before a release:

```bash
uv run mkdocs build --strict
```
