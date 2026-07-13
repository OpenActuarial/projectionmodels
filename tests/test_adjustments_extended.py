from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import Adjustment, Assumption, Scenario
from projectionmodels.advanced import Sensitivity
from projectionmodels.exceptions import AdjustmentError


@pytest.mark.parametrize(
    ("method", "value", "expected"),
    [
        ("set", 7.0, [7.0, 7.0]),
        ("add", 2.0, [7.0, 12.0]),
        ("multiply", 2.0, [10.0, 20.0]),
        ("floor", 8.0, [8.0, 10.0]),
        ("cap", 8.0, [5.0, 8.0]),
    ],
)
def test_adjustment_methods(method, value, expected):
    frame = pd.DataFrame({"id": [1, 2]})
    result, audit = Adjustment("x", method, value).apply(
        frame, pd.Series([5.0, 10.0])
    )
    assert result.tolist() == expected
    assert len(audit) == 2


def test_adjustment_filters_support_lists_and_callables():
    frame = pd.DataFrame({"product": ["PPO", "HMO", "EPO"], "value": [1, 2, 3]})
    list_adjustment = Adjustment(
        "x", "add", 1, filters={"product": ["PPO", "EPO"]}
    )
    callable_adjustment = Adjustment(
        "x", "multiply", 10, filters={"value": lambda values: values > 1}
    )
    first, _ = list_adjustment.apply(frame, pd.Series([0.0, 0.0, 0.0]))
    second, _ = callable_adjustment.apply(frame, first)
    assert second.tolist() == [1.0, 0.0, 10.0]


def test_adjustment_effective_to_and_audit_identifiers():
    frame = pd.DataFrame(
        {
            "scenario": ["s"] * 3,
            "projection_period": ["2027-01", "2027-02", "2027-03"],
            "period_start": pd.date_range("2027-01-01", periods=3, freq="MS"),
            # explicit month-ends: freq="ME" needs pandas>=2.2, freq="M" is
            # deprecated there -- literal dates work on every supported pandas
            "period_end": pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]),
        }
    )
    adjustment = Adjustment(
        "x", "add", 1.0, effective_from="2027-02-01", effective_to="2027-02-28"
    )
    result, audit = adjustment.apply(frame, pd.Series([0.0, 0.0, 0.0]))
    assert result.tolist() == [0.0, 1.0, 0.0]
    assert audit["projection_period"].item() == "2027-02"
    assert audit["scenario"].item() == "s"


def test_adjustment_values_support_assumption_series_and_callable():
    frame = pd.DataFrame({"product": ["PPO", "HMO"]})
    assumption = Assumption(
        "factor",
        pd.DataFrame({"product": ["PPO", "HMO"], "factor": [2.0, 3.0]}),
        lookup=["product"],
    )
    result, _ = Adjustment("x", "multiply", assumption).apply(
        frame, pd.Series([10.0, 10.0])
    )
    assert result.tolist() == [20.0, 30.0]
    result, _ = Adjustment("x", "add", pd.Series([1.0, 2.0])).apply(
        frame, pd.Series([10.0, 10.0])
    )
    assert result.tolist() == [11.0, 12.0]
    result, _ = Adjustment("x", "set", lambda data: data.index + 5).apply(
        frame, pd.Series([0.0, 0.0])
    )
    assert result.tolist() == [5, 6]


def test_adjustment_validation_errors():
    with pytest.raises(AdjustmentError, match="method must be"):
        Adjustment("x", "bad", 1)
    frame = pd.DataFrame({"id": [1]})
    with pytest.raises(AdjustmentError, match="filter column"):
        Adjustment("x", "add", 1, filters={"missing": 1}).apply(frame, pd.Series([0]))
    with pytest.raises(AdjustmentError, match="period_start"):
        Adjustment("x", "add", 1, effective_from="2027-01-01").apply(
            frame, pd.Series([0])
        )
    with pytest.raises(AdjustmentError, match="Series length"):
        Adjustment("x", "add", pd.Series([1, 2])).apply(frame, pd.Series([0]))
    with pytest.raises(AdjustmentError, match="must be scalar"):
        Adjustment("x", "add", {"bad": 1}).apply(frame, pd.Series([0]))


def test_scenario_priority_and_target_filtering():
    scenario = Scenario(
        "ordered",
        [
            Adjustment("x", "add", 2.0, priority=200),
            Adjustment("x", "multiply", 3.0, priority=100),
            Adjustment("y", "set", 99.0),
        ],
    )
    assert [item.method for item in scenario.for_target("x")] == ["multiply", "add"]
    result, audits = scenario.apply("x", pd.DataFrame({"id": [1]}), pd.Series([4.0]))
    assert result.item() == 14.0
    assert len(audits) == 2


def test_sensitivity_generates_named_scenarios_and_dates():
    sensitivity = Sensitivity(
        "trend",
        [0.04, 0.08],
        method="set",
        filters={"product": "PPO"},
        effective_from="2027-01-01",
        name_template="trend_{value:.0%}",
    )
    scenarios = sensitivity.scenarios()
    assert [scenario.name for scenario in scenarios] == ["trend_4%", "trend_8%"]
    assert scenarios[0].adjustments[0].filters == {"product": "PPO"}
    assert scenarios[0].adjustments[0].effective_from == "2027-01-01"
