# how_wrong_is_your_mmm

Collinearity diagnostics and budget phasing for Marketing Mix Models.

This is the API reference, generated from the package's own docstrings. For
what the package does and why, start with one of these instead:

- [**Overview**](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html):
  the problem, the fix and what it costs, on a worked example.
- [**Example report**](https://raz1470.github.io/how_wrong_is_your_mmm/example-report.html):
  the HTML report `DiscoveryReport` produces, on simulated data.
- [**Notebooks**](https://github.com/raz1470/how_wrong_is_your_mmm/tree/main/notebooks):
  hands-on walkthroughs with real output.

## Install

```bash
pip install how-wrong-is-your-mmm  # coming to PyPI
```

## The classes

| Class | What it does |
|---|---|
| [`CollinearityDiagnostic`](api/diagnostic.md) | Quantifies how identifiable OLS marginal returns are, given your spend data. |
| [`IdentifiabilityDiagnostic`](api/identifiability.md) | Quantifies whether your spend data pins down each channel's saturation and adstock. |
| [`BudgetPhaser`](api/phaser.md) | Recommends a de-correlated weekly spend schedule. |
| [`Blackout`](api/phaser.md#how_wrong_is_your_mmm.Blackout) | A harder on/off phasing lever for `BudgetPhaser`, in place of a continuous weekly range. |
| [`Redistribute`](api/phaser.md#how_wrong_is_your_mmm.Redistribute) | A dark month, an optional peak month and a light weekly nudge, for `BudgetPhaser`/`DiscoveryReport`; preserves annual (not monthly) totals. |
| [`MonthStep`](api/phaser.md#how_wrong_is_your_mmm.MonthStep) | A whole-month step phasing lever for `BudgetPhaser`/`DiscoveryReport`: each month scaled up or down in orthogonal Hadamard patterns across channels; preserves annual (not monthly) totals. |
| [`DiscoveryReport`](api/discovery_report.md) | Sweeps candidate phasing strategies (or pins one directly, with per-channel overrides) into a single client-ready HTML report, plus the resulting schedule as a CSV. |

The lower-level building blocks are also exported, mainly useful if you're
extending the package rather than just using it: the
[simulation functions](api/dgp.md) (`simulate_spend`, `simulate_sales`,
`simulate_demand` and friends) and [`fit_ols`](api/mmm.md).

## Minimal example

```python
from how_wrong_is_your_mmm import CollinearityDiagnostic, BudgetPhaser, DiscoveryReport

diag = CollinearityDiagnostic(spend_df=my_spend_df)
diag.fit()
diag.summary()

phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_

report = DiscoveryReport(history_df=history, plan_df=plan, client_name="Example Brand")
report.fit()
report.to_html("reports/example_brand.html")
report.schedule_csv("reports/example_brand_schedule.csv")
```

Source: [github.com/raz1470/how_wrong_is_your_mmm](https://github.com/raz1470/how_wrong_is_your_mmm)
