"""Focused actuarial claim, premium, and expense projection workflows on supplied exposure."""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .adjustments import Adjustment, Scenario
from .assumptions import (
    Assumption,
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)
from .book import BookProjection
from .claims import ClaimExperience, ClaimProjection
from .data import DateCohort, ProjectionDates
from .exceptions import ProjectionModelsError, ValidationError
from .expenses import ExpenseProjection
from .group import GroupProjection, new_business
from .horizon import ProjectionHorizon
from .pmpm import PMPMProjection
from .premium import PremiumRollforward
from .premiums import PremiumProjection, RenewalRateActions
from .results import ProjectionResults

__all__ = [
    "Adjustment",
    "Assumption",
    "BookProjection",
    "ClaimExperience",
    "ClaimProjection",
    "CompletionAssumption",
    "CredibilityAssumption",
    "DateCohort",
    "ExpenseProjection",
    "GroupProjection",
    "PMPMProjection",
    "PremiumProjection",
    "PremiumRollforward",
    "ProjectionDates",
    "ProjectionHorizon",
    "ProjectionModelsError",
    "ProjectionResults",
    "RenewalRateActions",
    "Scenario",
    "SeasonalityAssumption",
    "TrendAssumption",
    "ValidationError",
    "new_business",
]

try:
    __version__ = _package_version("projectionmodels")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

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