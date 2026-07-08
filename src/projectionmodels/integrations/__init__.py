"""Optional integrations with other OpenActuarial libraries."""

from .actuarialpy import (
    estimate_completion,
    estimate_credibility,
    estimate_seasonality,
    estimate_trend,
    remove_seasonality,
)

__all__ = [
    "estimate_completion",
    "estimate_credibility",
    "estimate_seasonality",
    "estimate_trend",
    "remove_seasonality",
]
