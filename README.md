# how_wrong_is_your_mmm

**Collinearity diagnostics and budget phasing for Marketing Mix Models.**

TV, Meta, and Search budgets move together because the same planning cycle drives them all. When channels are correlated, an MMM can't reliably distinguish their individual effects — not because the model is wrong, but because the data was never designed to answer the question. The result is elasticity estimates that shift every time you refit, not because the market changed, but because the data was never informative enough to pin them down.

This package quantifies that problem and recommends a fix.

---

## The three-part solution

**Part 1 — Diagnose.** Simulate many plausible histories of your market and measure how much the estimated elasticities vary. The coefficient of variation (CV) is your "how wrong" number. At mean channel correlation 0.7, TV elasticity CV is ~36% — a third of the estimate's magnitude is unexplained variance.

**Part 2 — Phase.** Recommend a 52-week spend schedule that varies the weekly channel split independently, preserving monthly totals exactly. This introduces the independent variation the model needs to distinguish channel effects. At ±40% weekly deviation, one year of phasing reduces CV by ~30%. At ±80%, it's ~53%.

**Part 3 — Weight.** Upweight recent de-correlated observations in the model fit so the improvement takes effect without waiting for the phased history to dominate.

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
diag = CollinearityDiagnostic(correlation=0.7, seed=0)
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
phaser = BudgetPhaser(history_df=history, plan_df=plan)
phaser.fit()
phaser.recommended_schedule_   # 52-week DataFrame, monthly totals guaranteed to match
```

---

## Key result

At mean pairwise channel correlation **0.70**, with 50 noise seeds:

| Channel | True elasticity | 80% range | Width |
|---------|----------------|-----------|-------|
| TV      | 0.30           | [0.20, 0.46] | 0.26 |
| Meta    | 0.50           | [0.34, 0.66] | 0.32 |
| Search  | 0.40           | [0.11, 0.62] | 0.51 |

The model isn't broken. The data design is.

---

## Notebooks

Outputs committed — view without running.

| Notebook | What it shows |
|----------|--------------|
| [`01_dgp_diagnostic_walkthrough`](notebooks/01_dgp_diagnostic_walkthrough.ipynb) | Correlation sweep 0.1→0.9; elasticity estimates across 50 seeds; personalised diagnostic on real spend |
| [`02_phaser_walkthrough`](notebooks/02_phaser_walkthrough.ipynb) | BudgetPhaser end-to-end; CV curve vs phasing amplitude; elasticity fan chart before/after |
| [`03_time_to_benefit`](notebooks/03_time_to_benefit.ipynb) | Research study: how long phasing takes; correlation sensitivity; deviation amplitude lever |

---

## Practitioner guide

[`docs/guide.html`](docs/guide.html) — theory-led, no code, SVG charts. Four sections: the problem, the diagnostic, the fix, time to benefit.

---

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

87 tests. Python 3.12+. MIT licence.

---

*Part of a Bayesian marketing science stack alongside [`bayesian_vecm`](https://github.com/raz1470/bayesian_vecm).*
