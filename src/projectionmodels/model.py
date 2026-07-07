"""General deterministic projection engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .adjustments import Scenario, Sensitivity
from .assumptions import Assumption, AssumptionSet
from .calculations import (
    Calculation,
    CalculationContext,
    RollForward,
    VariableDefinition,
    order_variables,
)
from .data import ProjectionData, ProjectionDataset
from .exceptions import AdjustmentError, ValidationError
from .horizon import ProjectionHorizon
from .results import ProjectionResults


@dataclass
class ProjectionModel:
    """Advance actuarial projection records through a deterministic horizon.

    Formulas are vectorized across records and evaluated sequentially by period.
    Record grain, assumption lookup grain, adjustment filters, and reporting grain
    are independently declared.
    """

    assumptions: AssumptionSet | Iterable[Assumption] = field(
        default_factory=AssumptionSet
    )
    roll_forwards: Iterable[RollForward] = field(default_factory=tuple)
    calculations: Iterable[Calculation] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.assumptions, AssumptionSet):
            self.assumptions = AssumptionSet(*list(self.assumptions))
        self.roll_forwards = tuple(
            item for item in order_variables(self.roll_forwards) if isinstance(item, RollForward)
        )
        self.calculations = tuple(
            item for item in order_variables(self.calculations) if isinstance(item, Calculation)
        )
        names = [
            *self.assumptions.assumptions.keys(),
            *[item.name for item in self.roll_forwards],
            *[item.name for item in self.calculations],
        ]
        if len(names) != len(set(names)):
            raise ValidationError(
                "assumptions, roll-forwards, and calculations must have unique names"
            )

    def project(
        self,
        data: ProjectionData | ProjectionDataset,
        horizon: ProjectionHorizon,
        *,
        scenarios: Scenario | Iterable[Scenario] | None = None,
    ) -> ProjectionResults:
        dataset = data if isinstance(data, ProjectionDataset) else ProjectionDataset(data)
        scenario_list = self._normalize_scenarios(scenarios)
        self._validate_adjustment_targets(dataset, scenario_list)

        expanded = dataset.records.expand(horizon)
        periods = horizon.to_frame()
        scenario_frames: list[pd.DataFrame] = []
        adjustment_audits: list[pd.DataFrame] = []

        for scenario in scenario_list:
            prior_values = {
                item.name: item.initial_values(dataset.records.frame).reset_index(drop=True)
                for item in self.roll_forwards
            }
            period_frames: list[pd.DataFrame] = []

            for period_index in periods["projection_index"]:
                current = expanded.loc[
                    expanded["projection_index"] == period_index
                ].reset_index(drop=True)
                current = dataset.merge_tables(current)
                if len(current) != len(dataset.records.frame):
                    raise ValidationError(
                        "supporting tables changed the number of projection records; "
                        "use unique many-to-one lookup tables"
                    )
                current["scenario"] = scenario.name

                reserved = {
                    *self.assumptions.assumptions.keys(),
                    *[item.name for item in self.roll_forwards],
                    *[item.name for item in self.calculations],
                }
                for target in {
                    item.target
                    for item in scenario.adjustments
                    if item.target in current.columns and item.target not in reserved
                }:
                    adjusted, audits = scenario.apply(
                        target, current, current[target].copy()
                    )
                    current[target] = adjusted
                    adjustment_audits.extend(audits)

                for assumption in self.assumptions:
                    values = assumption.resolve(current)
                    values, audits = scenario.apply(assumption.name, current, values)
                    current[assumption.name] = values
                    adjustment_audits.extend(audits)

                context = CalculationContext(
                    frame=current,
                    prior_values=prior_values,
                    record_weight_col=dataset.records.record_weight,
                )
                for roll_forward in self.roll_forwards:
                    values = roll_forward.calculate(context)
                    if roll_forward.adjustable:
                        values, audits = scenario.apply(
                            roll_forward.name, current, values
                        )
                        adjustment_audits.extend(audits)
                    current[roll_forward.name] = values
                    context.frame = current

                context = CalculationContext(
                    frame=current,
                    prior_values=prior_values,
                    record_weight_col=dataset.records.record_weight,
                )
                for calculation in self.calculations:
                    values = calculation.calculate(context)
                    if calculation.adjustable:
                        values, audits = scenario.apply(
                            calculation.name, current, values
                        )
                        adjustment_audits.extend(audits)
                    current[calculation.name] = values
                    context.frame = current

                prior_values = {
                    item.name: current[item.name].reset_index(drop=True)
                    for item in self.roll_forwards
                }
                period_frames.append(current)

            scenario_frames.append(pd.concat(period_frames, ignore_index=True))

        result_frame = pd.concat(scenario_frames, ignore_index=True)
        measures: dict[str, VariableDefinition] = {
            item.name: item for item in [*self.roll_forwards, *self.calculations]
        }
        adjustment_audit = (
            pd.concat(adjustment_audits, ignore_index=True)
            if adjustment_audits
            else pd.DataFrame()
        )
        return ProjectionResults(
            frame=result_frame,
            measures=measures,
            projection_keys=dataset.records.projection_keys,
            component_keys=dataset.records.component_keys,
            assumption_audit_data=self.assumptions.audit_frame(),
            adjustment_audit_data=adjustment_audit,
        )

    def run_sensitivity(
        self,
        data: ProjectionData | ProjectionDataset,
        horizon: ProjectionHorizon,
        sensitivity: Sensitivity,
        *,
        include_baseline: bool = True,
    ) -> ProjectionResults:
        scenarios = sensitivity.scenarios()
        if include_baseline:
            scenarios.insert(0, Scenario("baseline"))
        return self.project(data, horizon, scenarios=scenarios)

    @staticmethod
    def _normalize_scenarios(
        scenarios: Scenario | Iterable[Scenario] | None,
    ) -> list[Scenario]:
        if scenarios is None:
            return [Scenario("baseline")]
        if isinstance(scenarios, Scenario):
            return [scenarios]
        output = list(scenarios)
        if not output:
            raise ValidationError("scenarios must not be empty")
        names = [scenario.name for scenario in output]
        if len(names) != len(set(names)):
            raise ValidationError("scenario names must be unique")
        return output

    def _validate_adjustment_targets(
        self,
        dataset: ProjectionDataset,
        scenarios: Iterable[Scenario],
    ) -> None:
        available = set(dataset.records.frame.columns)
        for table in dataset.tables.values():
            available.update(table.frame.columns)
        available.update(self.assumptions.assumptions)
        available.update(item.name for item in self.roll_forwards)
        available.update(
            item.name for item in self.calculations if item.adjustable
        )
        unknown = sorted(
            {
                adjustment.target
                for scenario in scenarios
                for adjustment in scenario.adjustments
                if adjustment.target not in available
            }
        )
        if unknown:
            raise AdjustmentError(
                "adjustments target unavailable or non-adjustable values: "
                f"{unknown}"
            )
