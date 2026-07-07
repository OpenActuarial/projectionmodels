"""Deterministic actuarial projections at a caller-selected grain."""

from .adjustments import Adjustment, Scenario, Sensitivity
from .assumptions import (
    Assumption,
    AssumptionSet,
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)
from .calculations import (
    Calculation,
    CalculationContext,
    CashFlow,
    Metric,
    RollForward,
)
from .claims import ClaimExperience, ClaimProjection
from .data import (
    DateCohort,
    ProjectionData,
    ProjectionDataset,
    ProjectionDates,
    ProjectionTable,
)
from .exceptions import (
    AdjustmentError,
    AssumptionResolutionError,
    DependencyError,
    ProjectionModelsError,
    ValidationError,
)
from .expenses import ExpenseProjection
from .horizon import ProjectionHorizon
from .model import ProjectionModel
from .results import ProjectionResults

__all__ = [
    "Adjustment",
    "AdjustmentError",
    "Assumption",
    "AssumptionResolutionError",
    "AssumptionSet",
    "Calculation",
    "CalculationContext",
    "CashFlow",
    "ClaimExperience",
    "ClaimProjection",
    "CompletionAssumption",
    "CredibilityAssumption",
    "DateCohort",
    "DependencyError",
    "ExpenseProjection",
    "Metric",
    "ProjectionData",
    "ProjectionDataset",
    "ProjectionDates",
    "ProjectionHorizon",
    "ProjectionModel",
    "ProjectionModelsError",
    "ProjectionResults",
    "ProjectionTable",
    "RollForward",
    "Scenario",
    "SeasonalityAssumption",
    "Sensitivity",
    "TrendAssumption",
    "ValidationError",
]

__version__ = "0.2.1"
