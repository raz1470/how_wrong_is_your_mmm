# how_wrong_is_your_mmm

Collinearity diagnostics and budget phasing for Marketing Mix Models.

This is the API reference, generated from the package's own docstrings. If
you're looking for the narrative version of what this package does and why,
start with one of these instead:

- [**Overview**](https://raz1470.github.io/how_wrong_is_your_mmm/overview.html) —
  a two-minute read: the headline numbers and the fix, no maths.
- [**Multicollinearity Research**](https://raz1470.github.io/how_wrong_is_your_mmm/collinearity_research.html) —
  the full method and the research behind it.
- [**Notebooks**](https://github.com/raz1470/how_wrong_is_your_mmm/tree/main/notebooks) —
  hands-on walkthroughs with real output.

## Install

```bash
pip install how-wrong-is-your-mmm  # coming to PyPI
```

## The classes

| Class | What it does |
|---|---|
| [`CollinearityDiagnostic`](api/diagnostic.md) | Quantifies how identifiable OLS marginal returns are, given your spend data. |
| [`BudgetPhaser`](api/phaser.md) | Recommends a de-correlated weekly spend schedule, monthly totals preserved exactly. |
| [`Blackout`](api/phaser.md#how_wrong_is_your_mmm.Blackout) | A harder on/off phasing lever for `BudgetPhaser`, in place of a continuous weekly range. |
| [`ReportBuilder`](api/report.md) | Packages `CollinearityDiagnostic` + `BudgetPhaser` into a single client-ready HTML report. |

Two lower-level building blocks — [`simulate_spend`/`simulate_sales`](api/dgp.md)
and [`fit_ols`](api/mmm.md) — are also exported, mainly useful if you're
extending the package rather than just using it.

## Minimal example

```python
from how_wrong_is_your_mmm import CollinearityDiagnostic, BudgetPhaser, ReportBuilder

diag = CollinearityDiagnostic(spend_df=my_spend_df)
diag.fit()
diag.summary()

phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_

report = ReportBuilder(history_df=history, plan_df=plan, client_name="Example Brand")
report.fit()
report.to_html("reports/example_brand.html")
```

Source: [github.com/raz1470/how_wrong_is_your_mmm](https://github.com/raz1470/how_wrong_is_your_mmm)
