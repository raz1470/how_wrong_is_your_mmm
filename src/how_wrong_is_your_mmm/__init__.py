"""how_wrong_is_your_mmm: collinearity diagnostics and budget perturbation for MMMs."""

from how_wrong_is_your_mmm._dgp import simulate_sales, simulate_spend
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._mmm import fit_ols
from how_wrong_is_your_mmm._perturber import BudgetPerturber

__version__ = "0.1.0"

__all__ = [
    "BudgetPerturber",
    "CollinearityDiagnostic",
    "fit_ols",
    "simulate_sales",
    "simulate_spend",
]
