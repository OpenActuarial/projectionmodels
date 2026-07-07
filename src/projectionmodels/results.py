"""Projection results, grain-aware summarization, and scenario comparison."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calculations import VariableDefinition
from .exceptions import ValidationError


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass
class ProjectionResults:
    """Detailed projection output with mathematically explicit aggregation rules."""

    frame: pd.DataFrame
    measures: Mapping[str, VariableDefinition]
    projection_keys: tuple[str, ...]
    component_keys: tuple[str, ...] = field(default_factory=tuple)
    assumption_audit_data: pd.DataFrame | None = None
    adjustment_audit_data: pd.DataFrame | None = None

    def detail(self, *, copy: bool = True) -> pd.DataFrame:
        return self.frame.copy() if copy else self.frame

    def to_frame(self, *, copy: bool = True) -> pd.DataFrame:
        return self.detail(copy=copy)

    def measure_names(self, *, role: str | None = None) -> list[str]:
        if role is None:
            return list(self.measures)
        return [
            name
            for name, definition in self.measures.items()
            if definition.reporting_role == role
        ]

    def summarize(
        self,
        by: str | Iterable[str],
        *,
        measures: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Summarize measures without double-counting coarser-grain values.

        A measure's declared ``grain`` identifies where it is unique. Requested
        grouping fields outside that grain are retained, which permits an entity-
        level exposure to repeat once for each displayed claim type while still
        counting only once when claim type is omitted from the summary.
        """

        group_columns = _as_list(by)
        missing = [column for column in group_columns if column not in self.frame.columns]
        if missing:
            raise ValidationError(f"summary columns are missing: {missing}")
        requested = self.measure_names() if measures is None else _as_list(measures)
        unknown = [name for name in requested if name not in self.measures]
        if unknown:
            raise ValidationError(f"unknown measures: {unknown}")

        # Recalculated metrics depend on summarized numerator and denominator
        # measures. Resolve those dependencies internally so callers can request
        # only the metric they want to display.
        selected = list(requested)
        index = 0
        while index < len(selected):
            definition = self.measures[selected[index]]
            if definition.aggregation == "recalculate":
                dependencies = [
                    *list(definition.numerator or ()),
                    *([definition.denominator] if definition.denominator else []),
                ]
                missing_definitions = [
                    name for name in dependencies if name not in self.measures
                ]
                if missing_definitions:
                    raise ValidationError(
                        f"cannot recalculate {definition.name!r}; measure definitions "
                        f"are missing: {missing_definitions}"
                    )
                for dependency in dependencies:
                    if dependency not in selected:
                        selected.append(dependency)
            index += 1

        base = self.frame.loc[:, group_columns].drop_duplicates().reset_index(drop=True)
        output = base
        deferred: list[str] = []

        for name in selected:
            definition = self.measures[name]
            if definition.aggregation == "recalculate":
                deferred.append(name)
                continue
            if name not in self.frame.columns:
                continue
            grain = list(definition.grain or (self.projection_keys + self.component_keys))
            natural_keys = [
                column
                for column in ["scenario", "projection_period", *grain]
                if column in self.frame.columns
            ]
            display_dimensions = [
                column for column in group_columns if column not in natural_keys
            ]
            dedupe = list(dict.fromkeys(natural_keys + display_dimensions))
            work = self.frame.loc[:, list(dict.fromkeys(dedupe + [name]))].drop_duplicates(
                dedupe
            )
            grouped = work.groupby(group_columns, dropna=False, sort=False)[name]
            if definition.aggregation == "sum":
                summary = grouped.sum(min_count=1)
            elif definition.aggregation == "mean":
                summary = grouped.mean()
            elif definition.aggregation == "first":
                summary = grouped.first()
            elif definition.aggregation == "max":
                summary = grouped.max()
            elif definition.aggregation == "min":
                summary = grouped.min()
            else:  # pragma: no cover - validated on construction
                raise ValidationError(
                    f"unsupported aggregation {definition.aggregation!r}"
                )
            output = output.merge(
                summary.rename(name).reset_index(),
                on=group_columns,
                how="left",
                validate="one_to_one",
            )

        pending = list(deferred)
        while pending:
            progressed = False
            for name in list(pending):
                definition = self.measures[name]
                numerators = list(definition.numerator or ())
                denominator = definition.denominator
                required = numerators + ([denominator] if denominator else [])
                if any(column not in output.columns for column in required):
                    continue
                numerator = output[numerators].sum(axis=1, min_count=1)
                denom = output[denominator]  # type: ignore[index]
                output[name] = np.divide(
                    numerator,
                    denom,
                    out=np.full(len(output), np.nan, dtype=float),
                    where=denom.to_numpy(dtype=float) != 0,
                )
                pending.remove(name)
                progressed = True
            if not progressed:
                missing = {
                    name: [
                        column
                        for column in [
                            *list(self.measures[name].numerator or ()),
                            *(
                                [self.measures[name].denominator]
                                if self.measures[name].denominator
                                else []
                            ),
                        ]
                        if column not in output.columns
                    ]
                    for name in pending
                }
                raise ValidationError(
                    f"cannot recalculate metrics; summarized inputs are missing: {missing}"
                )

        visible = [*group_columns, *[name for name in requested if name in output.columns]]
        return output.loc[:, list(dict.fromkeys(visible))]

    def compare_scenarios(
        self,
        *,
        baseline: str,
        comparison: str,
        by: str | Iterable[str],
        measures: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        grouping = [column for column in _as_list(by) if column != "scenario"]
        selected = self.measure_names() if measures is None else _as_list(measures)
        summary = self.summarize(
            by=["scenario", *grouping],
            measures=selected,
        )
        base = summary.loc[summary["scenario"] == baseline].drop(columns="scenario")
        comp = summary.loc[summary["scenario"] == comparison].drop(columns="scenario")
        merged = base.merge(
            comp,
            on=grouping,
            suffixes=("_baseline", "_comparison"),
            validate="one_to_one",
        )
        for measure in selected:
            left = f"{measure}_baseline"
            right = f"{measure}_comparison"
            if left not in merged.columns or right not in merged.columns:
                continue
            merged[f"{measure}_change"] = merged[right] - merged[left]
            merged[f"{measure}_pct_change"] = np.divide(
                merged[f"{measure}_change"],
                merged[left],
                out=np.full(len(merged), np.nan, dtype=float),
                where=merged[left].to_numpy(dtype=float) != 0,
            )
        return merged

    def assumption_audit(self) -> pd.DataFrame:
        if self.assumption_audit_data is None:
            return pd.DataFrame()
        return self.assumption_audit_data.copy()

    def adjustment_audit(self) -> pd.DataFrame:
        if self.adjustment_audit_data is None:
            return pd.DataFrame()
        return self.adjustment_audit_data.copy()

    @classmethod
    def combine(cls, *results: "ProjectionResults") -> "ProjectionResults":
        """Combine compatible result sets column-wise on their common identifiers."""

        if not results:
            raise ValidationError("at least one ProjectionResults is required")
        base = results[0]
        identifiers = [
            column
            for column in (
                "scenario",
                "projection_period",
                "period_start",
                "period_end",
                *base.projection_keys,
                *base.component_keys,
            )
            if column in base.frame.columns
        ]
        frame = base.frame.copy()
        measures = dict(base.measures)
        for other in results[1:]:
            common = [column for column in identifiers if column in other.frame.columns]
            value_columns = [
                column
                for column in other.measures
                if column in other.frame.columns and column not in frame.columns
            ]
            frame = frame.merge(
                other.frame.loc[:, common + value_columns],
                on=common,
                how="outer",
                validate="one_to_one",
            )
            measures.update(other.measures)
        assumption_audits = [
            item.assumption_audit_data
            for item in results
            if item.assumption_audit_data is not None
        ]
        adjustment_audits = [
            item.adjustment_audit_data
            for item in results
            if item.adjustment_audit_data is not None
        ]
        return cls(
            frame=frame,
            measures=measures,
            projection_keys=base.projection_keys,
            component_keys=base.component_keys,
            assumption_audit_data=(
                pd.concat(assumption_audits, ignore_index=True)
                if assumption_audits
                else None
            ),
            adjustment_audit_data=(
                pd.concat(adjustment_audits, ignore_index=True)
                if adjustment_audits
                else None
            ),
        )
