"""Calculated variables and roll-forwards."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DependencyError, ValidationError

Formula = Callable[["CalculationContext"], Any]


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass
class CalculationContext:
    """Values available to a roll-forward or calculated-variable formula."""

    frame: pd.DataFrame
    prior_values: Mapping[str, pd.Series]
    record_weight_col: str | None = None

    def __getitem__(self, name: str) -> pd.Series:
        try:
            return self.frame[name]
        except KeyError as exc:
            raise KeyError(f"calculation input {name!r} is not available") from exc

    def prior(self, name: str) -> pd.Series:
        try:
            return self.prior_values[name]
        except KeyError as exc:
            raise KeyError(f"prior value {name!r} is not available") from exc

    @property
    def year_fraction(self) -> pd.Series:
        return self["year_fraction"]

    @property
    def weight(self) -> pd.Series:
        if self.record_weight_col is None:
            return pd.Series(1.0, index=self.frame.index)
        return self[self.record_weight_col]


def _coerce_result(value: Any, frame: pd.DataFrame, name: str) -> pd.Series:
    if np.isscalar(value):
        return pd.Series(value, index=frame.index, name=name)
    if isinstance(value, pd.Series):
        if not value.index.equals(frame.index):
            value = pd.Series(value.to_numpy(), index=frame.index)
        return value.rename(name)
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != len(frame):
        raise ValidationError(
            f"formula {name!r} must return a scalar or one value per projection row"
        )
    return pd.Series(array, index=frame.index, name=name)


@dataclass(frozen=True)
class VariableDefinition:
    name: str
    aggregation: str = "sum"
    grain: tuple[str, ...] | list[str] | None = None
    reporting_role: str | None = None
    numerator: str | tuple[str, ...] | list[str] | None = None
    denominator: str | None = None
    depends_on: tuple[str, ...] | list[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.aggregation not in {
            "sum",
            "mean",
            "first",
            "max",
            "min",
            "recalculate",
        }:
            raise ValidationError(f"unsupported aggregation {self.aggregation!r}")
        object.__setattr__(self, "grain", None if self.grain is None else _as_tuple(self.grain))
        object.__setattr__(self, "depends_on", _as_tuple(self.depends_on))
        if self.aggregation == "recalculate":
            if self.numerator is None or self.denominator is None:
                raise ValidationError(
                    f"recalculated variable {self.name!r} requires numerator and denominator"
                )
            object.__setattr__(self, "numerator", _as_tuple(self.numerator))


@dataclass(frozen=True)
class RollForward(VariableDefinition):
    """A value carried from one projection period to the next."""

    initial: str | float | int | Callable[[pd.DataFrame], Any] = 0.0
    formula: Formula | None = None
    adjustable: bool = True

    def initial_values(self, frame: pd.DataFrame) -> pd.Series:
        if isinstance(self.initial, str):
            if self.initial not in frame.columns:
                raise ValidationError(
                    f"initial column {self.initial!r} for {self.name!r} is missing"
                )
            return frame[self.initial].rename(self.name)
        if callable(self.initial):
            return _coerce_result(self.initial(frame), frame, self.name)
        return pd.Series(self.initial, index=frame.index, name=self.name)

    def calculate(self, context: CalculationContext) -> pd.Series:
        if self.formula is None:
            return context.prior(self.name).rename(self.name)
        return _coerce_result(self.formula(context), context.frame, self.name)


@dataclass(frozen=True)
class Calculation(VariableDefinition):
    """A projected value calculated for each period."""

    formula: Formula | None = None
    adjustable: bool = False

    def calculate(self, context: CalculationContext) -> pd.Series:
        if self.formula is None:
            if self.name not in context.frame.columns:
                raise ValidationError(
                    f"calculation {self.name!r} has no formula and no source column"
                )
            return context.frame[self.name].rename(self.name)
        return _coerce_result(self.formula(context), context.frame, self.name)


class CashFlow(Calculation):
    """Semantic alias for a monetary calculation."""


class Metric(Calculation):
    """Semantic alias for a calculated actuarial metric."""


def order_variables(variables: Iterable[VariableDefinition]) -> list[VariableDefinition]:
    """Topologically order variables using explicit ``depends_on`` metadata."""

    items = list(variables)
    names = [item.name for item in items]
    if len(names) != len(set(names)):
        raise DependencyError("variable names must be unique")
    by_name = {item.name: item for item in items}
    result: list[VariableDefinition] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise DependencyError(f"circular dependency involving {name!r}")
        visiting.add(name)
        item = by_name[name]
        for dependency in item.depends_on:
            if dependency in by_name:
                visit(dependency)
        visiting.remove(name)
        visited.add(name)
        result.append(item)

    for item in items:
        visit(item.name)
    return result
