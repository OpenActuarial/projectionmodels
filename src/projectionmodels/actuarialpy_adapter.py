"""Lazy access to actuarialpy.

The projection package orchestrates actuarial primitives but does not copy their
implementations. Keeping the import lazy lets supplied assumptions, model
validation, and result handling remain usable in environments where optional
integration tests intentionally mock actuarialpy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .exceptions import ProjectionModelsError


def require_actuarialpy() -> Any:
    """Import and return :mod:`actuarialpy` with a useful error message."""

    try:
        return import_module("actuarialpy")
    except ModuleNotFoundError as exc:
        raise ProjectionModelsError(
            "actuarialpy is required to estimate or apply actuarial assumptions. "
            "Install projectionmodels with its declared dependencies."
        ) from exc


def actuarialpy_function(name: str) -> Any:
    module = require_actuarialpy()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ProjectionModelsError(
            f"the installed actuarialpy version does not expose {name!r}"
        ) from exc
