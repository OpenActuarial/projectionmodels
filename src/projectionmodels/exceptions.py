"""Package-specific exceptions."""


class ProjectionModelsError(Exception):
    """Base exception for projectionmodels."""


class ValidationError(ProjectionModelsError, ValueError):
    """Raised when projection inputs are structurally invalid."""


class AssumptionResolutionError(ProjectionModelsError, ValueError):
    """Raised when an assumption cannot be matched to projection rows."""


class AdjustmentError(ProjectionModelsError, ValueError):
    """Raised when a scenario adjustment is invalid or cannot be applied."""


class DependencyError(ProjectionModelsError, ValueError):
    """Raised when calculated-variable dependencies are invalid."""
