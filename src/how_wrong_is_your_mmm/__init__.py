"""how_wrong_is_your_mmm: collinearity diagnostics and budget phasing for MMMs."""

from how_wrong_is_your_mmm._dgp import (
    DEFAULT_DEMAND_SPEND_CORR,
    DEMAND_PROCESSES,
    BaselineCalibration,
    LinkedDemand,
    apply_adstock,
    calibrate_baseline,
    link_demand_to_spend,
    simulate_demand,
    simulate_demand_proxy,
    simulate_sales,
    simulate_spend,
)
from how_wrong_is_your_mmm._diagnostic import CollinearityDiagnostic
from how_wrong_is_your_mmm._discovery_report import DiscoveryReport
from how_wrong_is_your_mmm._identifiability import IdentifiabilityDiagnostic
from how_wrong_is_your_mmm._mmm import fit_ols
from how_wrong_is_your_mmm._phaser import (
    Blackout,
    BudgetPhaser,
    MonthStep,
    Redistribute,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DEMAND_SPEND_CORR",
    "DEMAND_PROCESSES",
    "BaselineCalibration",
    "Blackout",
    "BudgetPhaser",
    "CollinearityDiagnostic",
    "DiscoveryReport",
    "IdentifiabilityDiagnostic",
    "LinkedDemand",
    "MonthStep",
    "Redistribute",
    "apply_adstock",
    "calibrate_baseline",
    "fit_ols",
    "link_demand_to_spend",
    "simulate_demand",
    "simulate_demand_proxy",
    "simulate_sales",
    "simulate_spend",
]
