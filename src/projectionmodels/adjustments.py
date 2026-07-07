"""Scenario and sensitivity adjustments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .assumptions import Assumption
from .exceptions import AdjustmentError

_ALLOWED_METHODS = {"set", "add", "multiply", "floor", "cap"}


def _filter_mask(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, condition in filters.items():
        if column not in frame.columns:
            raise AdjustmentError(f"adjustment filter column {column!r} is missing")
        if callable(condition):
            selected = condition(frame[column])
            selected = pd.Series(selected, index=frame.index)
        elif isinstance(condition, (list, tuple, set, frozenset, pd.Index, np.ndarray)):
            selected = frame[column].isin(condition)
        else:
            selected = frame[column].eq(condition)
        mask &= selected.fillna(False)
    return mask


@dataclass(frozen=True)
class Adjustment:
    """Modify a named input, assumption, or roll-forward under a scenario."""

    target: str
    method: str
    value: Any
    name: str | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)
    effective_from: Any | None = None
    effective_to: Any | None = None
    priority: int = 100

    def __post_init__(self) -> None:
        if self.method not in _ALLOWED_METHODS:
            raise AdjustmentError(
                f"method must be one of {sorted(_ALLOWED_METHODS)}"
            )

    @property
    def label(self) -> str:
        return self.name or f"{self.target}:{self.method}"

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        mask = _filter_mask(frame, self.filters)
        if self.effective_from is not None or self.effective_to is not None:
            if "period_start" not in frame.columns:
                raise AdjustmentError(
                    "date-effective adjustments require a period_start column"
                )
            dates = pd.to_datetime(frame["period_start"])
            if self.effective_from is not None:
                mask &= dates >= pd.Timestamp(self.effective_from)
            if self.effective_to is not None:
                mask &= dates <= pd.Timestamp(self.effective_to)
        return mask

    def _resolved_value(self, frame: pd.DataFrame) -> pd.Series:
        if isinstance(self.value, Assumption):
            return self.value.resolve(frame)
        if np.isscalar(self.value):
            return pd.Series(self.value, index=frame.index)
        if isinstance(self.value, pd.Series):
            if len(self.value) == len(frame):
                return pd.Series(self.value.to_numpy(), index=frame.index)
            raise AdjustmentError(
                f"adjustment {self.label!r} Series length does not match projection rows"
            )
        if callable(self.value):
            resolved = self.value(frame)
            if np.isscalar(resolved):
                return pd.Series(resolved, index=frame.index)
            return pd.Series(resolved, index=frame.index)
        raise AdjustmentError(
            "adjustment values must be scalar, Series, Assumption, or callable"
        )

    def apply(
        self,
        frame: pd.DataFrame,
        values: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        mask = self.mask(frame)
        adjustment = self._resolved_value(frame)
        result = values.copy()
        before = result.loc[mask].copy()
        selected = adjustment.loc[mask]
        if self.method == "set":
            result.loc[mask] = selected
        elif self.method == "add":
            result.loc[mask] = before + selected
        elif self.method == "multiply":
            result.loc[mask] = before * selected
        elif self.method == "floor":
            result.loc[mask] = np.maximum(before, selected)
        elif self.method == "cap":
            result.loc[mask] = np.minimum(before, selected)

        audit = pd.DataFrame(
            {
                "row_index": frame.index[mask],
                "adjustment": self.label,
                "target": self.target,
                "method": self.method,
                "before": before.to_numpy(),
                "adjustment_value": selected.to_numpy(),
                "after": result.loc[mask].to_numpy(),
            }
        )
        for column in (
            "scenario",
            "projection_period",
            "period_start",
            "period_end",
        ):
            if column in frame.columns:
                audit[column] = frame.loc[mask, column].to_numpy()
        return result, audit


@dataclass(frozen=True)
class Scenario:
    """A named collection of ordered adjustments."""

    name: str = "baseline"
    adjustments: tuple[Adjustment, ...] | list[Adjustment] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adjustments",
            tuple(sorted(self.adjustments, key=lambda item: item.priority)),
        )

    def for_target(self, target: str) -> tuple[Adjustment, ...]:
        return tuple(item for item in self.adjustments if item.target == target)

    def apply(
        self,
        target: str,
        frame: pd.DataFrame,
        values: pd.Series,
    ) -> tuple[pd.Series, list[pd.DataFrame]]:
        result = values
        audits: list[pd.DataFrame] = []
        for adjustment in self.for_target(target):
            result, audit = adjustment.apply(frame, result)
            if not audit.empty:
                audits.append(audit)
        return result, audits


@dataclass(frozen=True)
class Sensitivity:
    """Generate one scenario for each selected adjustment value."""

    target: str
    values: tuple[Any, ...] | list[Any]
    method: str = "set"
    filters: Mapping[str, Any] = field(default_factory=dict)
    effective_from: Any | None = None
    effective_to: Any | None = None
    name_template: str = "{target}={value}"

    def scenarios(self) -> list[Scenario]:
        output: list[Scenario] = []
        for value in self.values:
            name = self.name_template.format(target=self.target, value=value)
            output.append(
                Scenario(
                    name=name,
                    adjustments=[
                        Adjustment(
                            name=name,
                            target=self.target,
                            method=self.method,
                            value=value,
                            filters=self.filters,
                            effective_from=self.effective_from,
                            effective_to=self.effective_to,
                        )
                    ],
                )
            )
        return output
