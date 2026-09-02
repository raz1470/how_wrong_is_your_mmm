"""how_wrong_is_your_mmm: collinearity diagnostics and budget phasing for MMMs."""

from how_wrong_is_your_mmm._dgp import (
    DEMAND_PROCESSES,
    BaselineCalibration,
    apply_adstock,
    calibrate_baseline,
    simulate_demand,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._mmm import fit_ols
from how_wrong_is_your_mmm._phaser import Blackout, BudgetPhaser
from how_wrong_is_your_mmm._report import ReportBuilder

__version__ = "0.1.0"

__all__ = [
    "DEMAND_PROCESSES",
    "BaselineCalibration",
    "Blackout",
    "BudgetPhaser",
    "CollinearityDiagnostic",
    "ReportBuilder",
    "apply_adstock",
    "calibrate_baseline",
    "fit_ols",
    "simulate_demand",
    "simulate_sales",
    "simulate_spend",
]
