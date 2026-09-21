# How Wrong Is Your MMM?

[![CI](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml)

**How using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.**

Take the marketing mix model (MMM) you use to allocate your marketing budget across channels. Run it again on a slightly different slice of history, same channels, same market. Would it give you the same answer?

For most brands, no. TV, Meta, and Search budgets move together because the same planning cycle drives them all, and that makes it hard for an MMM to tell their individual effects apart. The result is marginal-return estimates (£ revenue per £ spend, sometimes called mROAS) that shift every time you refit, not because the market changed, but because the data was never informative enough to pin them down.

This package quantifies that problem and recommends a fix, on three fronts: variance (can your spend design tell channels apart at all), bias (how far an unobserved demand driver could be pushing the estimate), and identifiability (whether adstock and saturation come back as anything more than a guess). **Important scope note:** all three are measured by simulation against assumptions you supply (your own plausible marginal returns, and optionally a demand proxy), not verified against ground truth in your actual historical data. A confounder you didn't think to simulate can still leave every one of these diagnostics looking healthy while the underlying estimate is wrong. See [the overview](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html) for the full scope discussion.

![A year of phasing tightens the estimated range for the same plan — incremental revenue model-estimated range today vs after one year of budget phasing, every channel narrowing 63-72% off the same £17.28m plan, no extra spend. Dashed line marks the true marginal return's implied revenue on this demo scenario.](https://raw.githubusercontent.com/raz1470/how_wrong_is_your_mmm/main/assets/readme-honest-ranges.png)

This chart shows the impact of the phasing algorithm: the estimated revenue range for each channel gets 63-72% tighter after one year of phasing, off the same budget. On this scenario, Meta and TikTok's ranges actually overlap today — you can't confidently say which channel is doing best — and phasing is what pulls them apart into a clear order.

---

## The three-part solution

**Part 1 — Diagnose.** Simulate many plausible histories of your market and measure how much your marginal-return estimates swing on three fronts: variance (how much the estimate itself moves), bias (how far a plausible unobserved demand driver could push it), and identifiability (whether adstock and saturation come back as anything more than a guess). (The width of the variance range is what matters; where it's centred depends entirely on the marginal return you assume, and there's no universal default — see Quick start below.)

**Part 2 — Phase.** Recommend a weekly spend schedule that breaks the correlation between channels while keeping monthly totals exactly the same. Choose a continuous nudge to each week's split, or Blackout: a harder on/off switch that takes a channel fully dark some weeks and makes it up on the weeks it stays on. A third option, Redistribute, blacks each channel out for a few consecutive weeks in a round-robin month, moves that budget into one recipient month, and layers a light edge nudge on top: it keeps each channel's annual total exactly, but (unlike the other two) moves budget between months. A fourth, MonthStep, scales each whole month's spend up or down by a fixed step (orthogonal Hadamard sign patterns across channels, weekly shape within a month untouched, so at most 12 budget changes per channel per year); it keeps annual totals exactly but not monthly ones. Get the overall scale of your marginal returns wrong but the channels' proportions right, and the percentage reduction phasing buys you barely moves — only the absolute £ figures above shift. Get the *proportions between channels* wrong, though, and the reduction can move too: it changes which channel looks least identified, which changes which phasing intensity gets recommended for it.

**Part 3 — Retrain.** Refit your MMM on the phased data. The de-correlated spend does the work: marginal-return estimates come back measurably tighter on all three fronts, without waiting years for it to accumulate.

---

## Guides

[**Overview**](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html)
An overview of how using a budget phasing algorithm can dramatically tighten the confidence in your MMM results.

[**API Reference**](https://raz1470.github.io/how_wrong_is_your_mmm/api/)
Full class and function docs for `CollinearityDiagnostic`, `IdentifiabilityDiagnostic`, `BudgetPhaser`, `Blackout`, `Redistribute`, `MonthStep`, and `DiscoveryReport`.

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

Sweep candidate phasing strategies (or pin one directly, with per-channel
overrides) and package the diagnosis into one self-contained HTML report,
plus the resulting weekly schedule as a CSV:

```python
from how_wrong_is_your_mmm import DiscoveryReport

report = DiscoveryReport(
    history_df=history,
    plan_df=plan,
    true_marginal_returns={"tv": 1.8, "paid_social": 2.4, "search": 4.1},  # your own numbers
    revenue_noise_std=my_residual_std,  # from your own model's residuals
    client_name="Example Brand",
    # optional: pin a strategy instead of sweeping for one, with per-channel overrides
    # strategy_pct=60.0,
    # channel_constraints={"paid_social": 20.0},
)
report.fit()
report.to_html("reports/example_brand.html")  # self-contained HTML, open it in a browser
report.schedule_csv(
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
| [`01_scenario_walkthrough`](notebooks/01_scenario_walkthrough.ipynb) | Runs the full pipeline once on the same four-channel scenario as the live example report: inputs, the problem across variance/bias/identifiability, correlation cost, the algorithm, impact, phased spend |
| [`02_channel_scaling`](notebooks/02_channel_scaling.ipynb) | Sweeps channel count (3 to 20) and spend correlation to check the fix holds at realistic scale, not just the 4-channel demo |
| [`03_time_to_benefit`](notebooks/03_time_to_benefit.ipynb) | How many weeks of phasing it takes to meaningfully reduce marginal-return uncertainty, and how that depends on starting correlation and which lever you use |
| [`04_adstock_threat`](notebooks/04_adstock_threat.ipynb) | Checks whether phasing's bias reduction survives real carryover (adstock), across four demand-process shapes |
| [`05_strategy_comparison`](notebooks/05_strategy_comparison.ipynb) | Compares `DiscoveryReport`'s default phasing shapes/intensities (7 candidates, all at +/-20%: uniform/edge/seesaw, Redistribute, MonthStep and the original Blackout) on variance, bias, identifiability and cost — the same criteria the live report scores every candidate on |

---

## Future advancements

**Does phasing help with omitted-variable bias, not just collinearity?** Directionally yes, and for a principled reason: phasing noise is exogenous by construction, so it raises spend variance without raising its covariance with an unobserved confounder. The caveat: phasing only helps with *unobserved* confounders — a *known* one (e.g. a seasonal driver you can see) should be controlled for directly (a Fourier term, free) rather than randomised away (expensive).

**Bring-your-own-estimator.** `DiscoveryReport` currently fits with OLS internally. A hook to swap in your own estimator instead (Bayesian, regularised, whatever your team already trusts) while still returning the same diagnostics and phased schedule is on the list.

---

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

963 tests. Python 3.12+. MIT licence.

The [API reference](https://raz1470.github.io/how_wrong_is_your_mmm/api/) is built with `mkdocs` + `mkdocstrings` from the docstrings in `src/`. A GitHub Actions workflow (`docs-deploy.yml`) rebuilds it and deploys the whole `docs/` site on every push to `main`, so there's nothing to build or commit locally for a release. To preview changes locally:

```bash
uv run mkdocs serve
```

To catch doc errors before pushing, rather than waiting on CI:

```bash
uv run mkdocs build --strict
```
