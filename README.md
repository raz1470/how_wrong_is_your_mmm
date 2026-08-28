# How Wrong Is Your MMM?

[![CI](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml)

**How using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.**

Take the marketing mix model (MMM) you use to allocate your marketing budget across channels. Run it again on a slightly different slice of history, same channels, same market. Would it give you the same answer?

For most brands, no. TV, Meta, and Search budgets move together because the same planning cycle drives them all, and that makes it hard for an MMM to tell their individual effects apart. The result is marginal-return estimates (£ revenue per £ spend, sometimes called mROAS) that shift every time you refit, not because the market changed, but because the data was never informative enough to pin them down.

This package quantifies that problem and recommends a fix. **Important scope note:** it measures whether your spend design can identify each channel's effect precisely — sampling variance under a model that's correctly specified by construction. It does not check whether your model *is* correctly specified: an omitted driver (seasonality, adstock, saturation, a competitor event) can leave this diagnostic looking healthy while the underlying estimate is badly biased. See the [research page](https://raz1470.github.io/how_wrong_is_your_mmm/collinearity_research.html) for the full scope discussion.

![The ranges tighten, and keep tightening toward the true answer — incremental revenue model-estimated range today vs after 1 year vs after 2 years of budget phasing, every channel narrowing roughly 55-60% off the same £12.3m plan, no extra spend. Dashed line marks the true marginal return's implied revenue on this demo scenario.](https://raw.githubusercontent.com/raz1470/how_wrong_is_your_mmm/main/assets/readme-honest-ranges.png)

This chart shows the impact of the phasing algorithm: the estimated revenue range for each channel gets tighter the longer you phase, roughly 60% tighter after 2 years, off the same budget. On this scenario, the ranges for TV, Meta, and Search actually overlap today — you can't confidently say which channel is doing best — and phasing is what pulls them apart into a clear order.

---

## The three-part solution

**Part 1 — Diagnose.** Simulate many plausible histories of your market and measure how much your marginal-return estimates swing — exposing uncertainty the model has been hiding, not a rounding error. (The width of that range is what matters; where it's centred depends entirely on the marginal return you assume, and there's no universal default — see Quick start below.)

**Part 2 — Phase.** Recommend a weekly spend schedule that breaks the correlation between channels while keeping monthly totals exactly the same. Choose a continuous nudge to each week's split, or Blackout: a harder on/off switch that takes a channel fully dark some weeks and makes it up on the weeks it stays on. Get the overall scale of your marginal returns wrong but the channels' proportions right, and the percentage reduction phasing buys you barely moves — only the absolute £ figures above shift. Get the *proportions between channels* wrong, though, and the reduction can move too: it changes which channel looks least identified, which changes which phasing intensity gets recommended for it.

**Part 3 — Retrain.** Refit your MMM on the phased data. The de-correlated spend does the work: marginal-return estimates come back measurably tighter, without waiting years for it to accumulate.

---

## Guides

[**Overview**](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html)
An overview of how using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.

[**Multicollinearity Research**](https://raz1470.github.io/how_wrong_is_your_mmm/collinearity_research.html)
The multicollinearity research behind how using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.

[**API Reference**](https://raz1470.github.io/how_wrong_is_your_mmm/api/)
Full class and function docs for `CollinearityDiagnostic`, `BudgetPhaser`, and `Blackout`.

[**Example report**](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html)
A real example report, viewable end to end — built on simulated data, but the exact HTML `to_html()` produces for a client.

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

### 1. Diagnose your collinearity risk

Simulate synthetic spend at whatever channel correlation you want to stress-test, and see how much your marginal-return estimates swing:

```python
from how_wrong_is_your_mmm import CollinearityDiagnostic

diag = CollinearityDiagnostic(correlation=0.7, spend_seed=0)
diag.fit()
diag.summary()
# channel  true_marginal_return  mean_estimated  coef_of_variation
#      tv                  0.50           0.538              0.284
#    meta                  1.00           1.003              0.188
#  search                  1.50           1.445              0.200
```

The defaults above (`true_marginal_returns={"tv": 0.5, "meta": 1.0, "search": 1.5}`,
`revenue_noise_std=26_000`) are illustrative only, and only apply when your channels
are literally named `tv`/`meta`/`search` — any other channel naming raises a clear
`ValueError` telling you to supply your own. There is no universal default for
either: run the same check on your own spend history and your own assumptions
instead:

```python
diag = CollinearityDiagnostic(
    spend_df=my_spend_df,
    true_marginal_returns={"tv": 1.8, "paid_social": 2.4, "search": 4.1},  # your own numbers
    revenue_noise_std=my_residual_std,  # from your own model's residuals, not the package default
)
diag.fit()
diag.summary()  # same output, personalised to your correlation structure and assumptions
```

### 2. Phase your budget

Recommend a de-correlated spend schedule that keeps your monthly totals exactly the same:

```python
from how_wrong_is_your_mmm import BudgetPhaser

# history: your multi-year spend history (DatetimeIndex)
# plan:    the upcoming year's spend plan (DatetimeIndex, same channels)
phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_  # 52-week DataFrame, monthly totals guaranteed to match
```

### 3. Build a client-ready report

Package the diagnosis and the phased schedule into one self-contained HTML report:

```python
from how_wrong_is_your_mmm import ReportBuilder

rb = ReportBuilder(
    history_df=history,
    plan_df=plan,
    true_marginal_returns={"tv": 1.8, "paid_social": 2.4, "search": 4.1},  # your own numbers
    revenue_noise_std=my_residual_std,  # from your own model's residuals
    client_name="Example Brand",
)
rb.fit()
rb.to_html("reports/example_brand.html")  # self-contained HTML, open it in a browser
rb.schedule_csv(
    "reports/example_brand_schedule.csv"
)  # the recommended weekly schedule as a CSV
```

`reports/` is git-ignored by default (see `.gitignore`). Save your own generated reports there, or wherever suits your workflow. See it end to end at the [example report](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html) above.

`true_marginal_returns` and `revenue_noise_std` are the two most important inputs
to get right — every CV and every £ range in the report is anchored to them, and
CV is exactly inversely proportional to both. The demo defaults above (matching
`tv`/`meta`/`search` channel names) are illustrative only; for your own data,
always supply your own values.

---

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| [`01_diagnostic_walkthrough`](notebooks/01_diagnostic_walkthrough.ipynb) | Shows how unreliable your marginal-return estimates get as your channels become more correlated, then runs the same check on your own spend data |
| [`02_phaser_walkthrough`](notebooks/02_phaser_walkthrough.ipynb) | Walks through `BudgetPhaser` end to end: how much phasing helps, and the actual recommended weekly schedule it produces |
| [`03_time_to_benefit`](notebooks/03_time_to_benefit.ipynb) | How long you need to phase your budget before you see a real improvement |
| [`04_channel_scaling_walkthrough`](notebooks/04_channel_scaling_walkthrough.ipynb) | Checks that the diagnostic and the fix still work when you have more than three channels |
| [`05_bayesian_comparison`](notebooks/05_bayesian_comparison.ipynb) | Checks whether switching to a Bayesian model fixes the problem on its own (it doesn't, not by much) |
| [`06_phasing_cost`](notebooks/06_phasing_cost.ipynb) | What the phasing lever actually costs: how much of your signed-off band reaches the model, and what phasing gives up in revenue once the response curve saturates |
| [`07_omitted_variable_bias`](notebooks/07_omitted_variable_bias.ipynb) | The other half of "how wrong": bias, not variance. Budgets follow demand, so a model fitted on spend alone is confounded — and unlike collinearity, nothing about the width of your estimates warns you. Measures how much phasing helps, what a realistic demand proxy adds on top, and what the whole thing is worth once an inflated marginal return is allowed to set the budget rather than just the mix |

---

## Future advancements

**A notebook on what the CV number does and doesn't tell you.** This diagnostic measures sampling variance under a model that's correctly specified by construction — it's silent on misspecification, and a confounder that inflates your point estimate can leave CV looking *healthier*, not worse. Planned: a walkthrough making that scope unmistakable.

**Does phasing help with omitted-variable bias, not just collinearity?** Directionally yes, and for a principled reason: phasing noise is exogenous by construction, so it raises spend variance without raising its covariance with an unobserved confounder. The caveat: phasing only helps with *unobserved* confounders — a *known* one (e.g. a seasonal driver you can see) should be controlled for directly (a Fourier term, free) rather than randomised away (expensive).

**Does phasing help identify adstock and saturation too?** This package targets cross-channel collinearity in marginal returns. There's planned research into whether phasing also helps identify adstock decay and saturation curvature. See the ["Does this help with adstock and saturation too?"](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html#faq) FAQ on the overview page for the reasoning so far.

**Bring-your-own-estimator.** `ReportBuilder` currently fits with OLS internally. A hook to swap in your own estimator instead (Bayesian, regularised, whatever your team already trusts) while still returning the same diagnostics and phased schedule is on the list.

---

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

271 tests. Python 3.12+. MIT licence.

The [API reference](https://raz1470.github.io/how_wrong_is_your_mmm/api/) is built with `mkdocs` + `mkdocstrings` from the docstrings in `src/`. A GitHub Actions workflow (`docs-deploy.yml`) rebuilds it and deploys the whole `docs/` site on every push to `main`, so there's nothing to build or commit locally for a release. To preview changes locally:

```bash
uv run mkdocs serve
```

To catch doc errors before pushing, rather than waiting on CI:

```bash
uv run mkdocs build --strict
```
