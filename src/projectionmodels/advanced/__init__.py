"""Advanced building blocks for custom deterministic projection models.

Most users should start with the concrete workflows exported from
:mod:`projectionmodels`: ``ClaimProjection``, ``PremiumProjection``, and
``ExpenseProjection``.  This namespace contains the lower-level calculation
engine for users who need to define a custom roll-forward.
"""

from ..adjustments import Sensitivity
from ..assumptions import AssumptionSet
from ..calculations import Calculation, CalculationContext, CashFlow, Metric, RollForward
from ..data import ProjectionData, ProjectionDataset, ProjectionTable
from ..model import ProjectionModel

__all__ = [
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
]
