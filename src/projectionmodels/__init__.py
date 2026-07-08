"""Focused actuarial claim, premium, membership, and expense projections."""

from __future__ import annotations

import warnings

from .adjustments import Adjustment, Scenario
from .assumptions import (
    Assumption,
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)
from .claims import ClaimExperience, ClaimProjection
from .data import DateCohort, ProjectionDates
from .exceptions import ProjectionModelsError, ValidationError
from .expenses import ExpenseProjection
from .horizon import ProjectionHorizon
from .premiums import PremiumProjection, RenewalRateActions
from .results import ProjectionResults

__all__ = [
    "Adjustment",
    "Assumption",
    "ClaimExperience",
    "ClaimProjection",
    "CompletionAssumption",
    "CredibilityAssumption",
    "DateCohort",
    "ExpenseProjection",
    "PremiumProjection",
    "ProjectionDates",
    "ProjectionHorizon",
    "ProjectionModelsError",
    "ProjectionResults",
    "RenewalRateActions",
    "Scenario",
    "SeasonalityAssumption",
    "TrendAssumption",
    "ValidationError",
]

__version__ = "0.4.0"

# Backward-compatible access for the 0.3 advanced API.  These names are no
# longer advertised at the package root and will move permanently to
# projectionmodels.advanced in 1.0.
_ADVANCED_NAMES = {
    "AssumptionSet",
    "Calculation",
    "CalculationContext",
    "CashFlow",
    "Metric",
    "ProjectionData",
    "ProjectionDataset",
    "ProjectionModel",
    "ProjectionTable",
    "RollForward",
    "Sensitivity",
}


def __getattr__(name: str):
    if name == "actuarialpy_adapter":
        from . import actuarialpy_adapter

        return actuarialpy_adapter
    if name in _ADVANCED_NAMES:
        warnings.warn(
            f"projectionmodels.{name} is an advanced API; import it from "
            f"projectionmodels.advanced instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import advanced

        return getattr(advanced, name)
    raise AttributeError(f"module 'projectionmodels' has no attribute {name!r}")
