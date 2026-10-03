# How Wrong Is Your MMM?

[![CI](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/how_wrong_is_your_mmm/actions/workflows/ci.yml)

**Measure how reliable your MMM is, and phase your budget to make it more reliable.**

Marketing budgets are planned together and follow demand, so a marketing mix model (MMM) struggles to tell channels apart. This package simulates your own spend to measure the damage on three fronts (variance, bias, and whether saturation and adstock can be recovered at all), then recommends a weekly spend schedule that fixes it without changing any channel's annual budget.

Everything is measured by simulation against marginal returns you supply, not against ground truth in your data.

**[Read the overview](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html)** for the full walkthrough on a worked example.

![One pipeline, three steps. Diagnose: simulate plausible histories and measure all three problems. Phase: vary each channel on its own while the annual budget stays the same. Retrain: refit on the phased data and get a tighter answer back.](https://raw.githubusercontent.com/raz1470/how_wrong_is_your_mmm/main/assets/readme-pipeline.png)

- [**Overview**](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html): the problem, the fix and what it costs.
- [**Example report**](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html): the HTML report the package produces, on simulated data.
- [**API reference**](https://raz1470.github.io/how_wrong_is_your_mmm/api/): every class and function.

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

### 1. Diagnose

How much do your marginal-return estimates swing, given your own spend history?

```python
from how_wrong_is_your_mmm import CollinearityDiagnostic

diag = CollinearityDiagnostic(
    spend_df=my_spend_df,
    true_marginal_returns={"tv": 1.8, "paid_social": 2.4, "search": 4.1},  # your own numbers
    revenue_noise_std=my_noise_std,  # weekly noise sd in GBP, your own assumption
)
diag.fit()
diag.summary()
```

### 2. Phase

Get a weekly schedule that breaks the correlation between channels:

```python
from how_wrong_is_your_mmm import BudgetPhaser

# history: your multi-year spend history (DatetimeIndex)
# plan:    the upcoming year's spend plan (DatetimeIndex, same channels)
phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_  # 52-week DataFrame, monthly totals guaranteed to match
```

### 3. Report

Compare the phasing strategies and package the result as one HTML report, with the winning schedule as a CSV:

```python
from how_wrong_is_your_mmm import DiscoveryReport

report = DiscoveryReport(
    history_df=history,
    plan_df=plan,
    true_marginal_returns={"tv": 1.8, "paid_social": 2.4, "search": 4.1},  # your own numbers
    revenue_noise_pct=0.02,  # weekly noise sd as a share of average weekly sales (default)
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
report.growth_readout()  # each channel's return at +50% and +100% spend, unphased vs phased
```

`true_marginal_returns` and the noise level are the two inputs that matter most. Supply your own for both.

---

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| [`01_scenario_walkthrough`](notebooks/01_scenario_walkthrough.ipynb) | The full pipeline on the example report's four-channel scenario |
| [`02_channel_scaling`](notebooks/02_channel_scaling.ipynb) | Whether the fix holds from 3 to 20 channels and across spend correlations |
| [`03_time_to_benefit`](notebooks/03_time_to_benefit.ipynb) | How many weeks of phasing it takes to cut the uncertainty |
| [`04_adstock_threat`](notebooks/04_adstock_threat.ipynb) | Whether the bias reduction survives carryover |
| [`05_strategy_comparison`](notebooks/05_strategy_comparison.ipynb) | The six default strategies compared on variance, bias, identifiability and cost |

---

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

Python 3.12+. MIT licence.

The API reference is rebuilt and deployed with the rest of `docs/` on every push to `main`. To preview it locally:

```bash
uv run mkdocs serve
```
