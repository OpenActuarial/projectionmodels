from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import Adjustment, Assumption, ProjectionHorizon, Scenario, ValidationError
from projectionmodels.advanced import (
    Calculation,
    ProjectionData,
    ProjectionDataset,
    ProjectionModel,
    Sensitivity,
)
from projectionmodels.exceptions import AdjustmentError


def simple_records() -> ProjectionData:
    return ProjectionData(
        pd.DataFrame({"id": ["A"], "input_value": [10.0]}), projection_keys=["id"]
    )


def test_model_default_and_single_scenario_normalization():
    model = ProjectionModel(calculations=[Calculation("output", formula=lambda x: x["input_value"])])
    baseline = model.project(simple_records(), ProjectionHorizon("2027-01-01", periods=1))
    assert baseline.frame["scenario"].unique().tolist() == ["baseline"]
    named = model.project(
        simple_records(), ProjectionHorizon("2027-01-01", periods=1), scenarios=Scenario("named")
    )
    assert named.frame["scenario"].unique().tolist() == ["named"]


def test_model_adjusts_source_columns_and_captures_audit():
    model = ProjectionModel(
        calculations=[Calculation("output", formula=lambda x: x["input_value"] * 2)]
    )
    scenario = Scenario(
        "adjusted", [Adjustment("input_value", "add", 5.0, name="source change")]
    )
    result = model.project(
        simple_records(), ProjectionHorizon("2027-01-01", periods=1), scenarios=scenario
    )
    assert result.frame["output"].item() == 30.0
    audit = result.adjustment_audit()
    assert audit["target"].item() == "input_value"
    assert audit["adjustment"].item() == "source change"


def test_adjustable_calculation_can_be_modified():
    model = ProjectionModel(
        calculations=[
            Calculation("output", formula=lambda x: x["input_value"], adjustable=True)
        ]
    )
    result = model.project(
        simple_records(),
        ProjectionHorizon("2027-01-01", periods=1),
        scenarios=Scenario("adjusted", [Adjustment("output", "multiply", 3.0)]),
    )
    assert result.frame["output"].item() == 30.0


def test_nonadjustable_and_unknown_targets_are_rejected():
    model = ProjectionModel(
        calculations=[Calculation("output", formula=lambda x: x["input_value"])]
    )
    horizon = ProjectionHorizon("2027-01-01", periods=1)
    with pytest.raises(AdjustmentError, match="non-adjustable"):
        model.project(
            simple_records(),
            horizon,
            scenarios=Scenario("bad", [Adjustment("output", "set", 1.0)]),
        )
    with pytest.raises(AdjustmentError, match="unavailable"):
        model.project(
            simple_records(),
            horizon,
            scenarios=Scenario("bad", [Adjustment("unknown", "set", 1.0)]),
        )


def test_model_rejects_empty_or_duplicate_scenario_names():
    model = ProjectionModel(calculations=[Calculation("x", formula=lambda _: 1.0)])
    horizon = ProjectionHorizon("2027-01-01", periods=1)
    with pytest.raises(ValidationError, match="must not be empty"):
        model.project(simple_records(), horizon, scenarios=[])
    with pytest.raises(ValidationError, match="must be unique"):
        model.project(
            simple_records(), horizon, scenarios=[Scenario("same"), Scenario("same")]
        )


def test_model_rejects_duplicate_names_across_definition_types():
    with pytest.raises(ValidationError, match="unique names"):
        ProjectionModel(
            assumptions=[Assumption("value", 1.0)],
            calculations=[Calculation("value", formula=lambda _: 2.0)],
        )


def test_run_sensitivity_without_baseline():
    model = ProjectionModel(
        assumptions=[Assumption("trend", 0.0)],
        calculations=[Calculation("output", formula=lambda x: x["trend"])],
    )
    result = model.run_sensitivity(
        simple_records(),
        ProjectionHorizon("2027-01-01", periods=1),
        Sensitivity("trend", [0.04, 0.08]),
        include_baseline=False,
    )
    assert set(result.frame["scenario"]) == {"trend=0.04", "trend=0.08"}


def test_supporting_table_many_to_many_growth_is_rejected():
    records = simple_records()
    dataset = ProjectionDataset(records).add_table(
        "duplicate",
        pd.DataFrame({"id": ["A", "A"], "lookup": [1, 2]}),
        keys=["id"],
        allow_duplicates=True,
    )
    model = ProjectionModel(calculations=[Calculation("x", formula=lambda _: 1.0)])
    with pytest.raises(ValidationError, match="changed the number"):
        model.project(dataset, ProjectionHorizon("2027-01-01", periods=1))


def test_assumption_audit_is_attached_to_results():
    model = ProjectionModel(
        assumptions=[Assumption("trend", 0.05, metadata={"owner": "pricing"})],
        calculations=[Calculation("output", formula=lambda x: x["trend"])],
    )
    results = model.project(simple_records(), ProjectionHorizon("2027-01-01", periods=1))
    audit = results.assumption_audit()
    assert audit.loc[0, "name"] == "trend"
    assert audit.loc[0, "owner"] == "pricing"
