from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import (
    Calculation,
    CalculationContext,
    DependencyError,
    Metric,
    RollForward,
    ValidationError,
)
from projectionmodels.calculations import order_variables


def test_calculation_context_access_prior_and_weight():
    frame = pd.DataFrame({"value": [2.0, 3.0], "weight": [10.0, 20.0], "year_fraction": [0.5, 0.5]})
    context = CalculationContext(
        frame=frame, prior_values={"state": pd.Series([1.0, 2.0])}, record_weight_col="weight"
    )
    assert context["value"].tolist() == [2.0, 3.0]
    assert context.prior("state").tolist() == [1.0, 2.0]
    assert context.weight.tolist() == [10.0, 20.0]
    assert context.year_fraction.tolist() == [0.5, 0.5]
    with pytest.raises(KeyError, match="calculation input"):
        context["missing"]
    with pytest.raises(KeyError, match="prior value"):
        context.prior("missing")


def test_calculation_context_default_weight_is_one():
    context = CalculationContext(pd.DataFrame({"x": [1, 2]}), prior_values={})
    assert context.weight.tolist() == [1.0, 1.0]


def test_variable_definition_validation():
    with pytest.raises(ValidationError, match="unsupported aggregation"):
        Calculation("x", aggregation="median")
    with pytest.raises(ValidationError, match="requires numerator"):
        Metric("ratio", aggregation="recalculate")
    metric = Metric(
        "ratio", aggregation="recalculate", numerator=["a", "b"], denominator="c"
    )
    assert metric.numerator == ("a", "b")


def test_roll_forward_initial_values_from_column_callable_and_scalar():
    frame = pd.DataFrame({"base": [1.0, 2.0]})
    assert RollForward("x", initial="base").initial_values(frame).tolist() == [1.0, 2.0]
    assert RollForward("x", initial=lambda data: data["base"] * 2).initial_values(frame).tolist() == [2.0, 4.0]
    assert RollForward("x", initial=3.0).initial_values(frame).tolist() == [3.0, 3.0]
    with pytest.raises(ValidationError, match="initial column"):
        RollForward("x", initial="missing").initial_values(frame)


def test_roll_forward_without_formula_carries_prior_value():
    context = CalculationContext(
        pd.DataFrame({"id": [1, 2]}), prior_values={"x": pd.Series([4.0, 5.0])}
    )
    assert RollForward("x").calculate(context).tolist() == [4.0, 5.0]


def test_calculation_source_column_and_formula_validation():
    context = CalculationContext(pd.DataFrame({"x": [1.0, 2.0]}), prior_values={})
    assert Calculation("x").calculate(context).tolist() == [1.0, 2.0]
    with pytest.raises(ValidationError, match="no formula"):
        Calculation("missing").calculate(context)
    with pytest.raises(ValidationError, match="one value per projection row"):
        Calculation("bad", formula=lambda _: [[1, 2], [3, 4]]).calculate(context)


def test_formula_series_index_is_realigned_to_projection_rows():
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=[10, 20])
    context = CalculationContext(frame, prior_values={})
    result = Calculation(
        "y", formula=lambda _: pd.Series([3.0, 4.0], index=[0, 1])
    ).calculate(context)
    assert result.index.tolist() == [10, 20]
    assert result.tolist() == [3.0, 4.0]


def test_dependency_ordering_cycles_and_duplicates():
    variables = [
        Calculation("c", depends_on=["b"]),
        Calculation("a"),
        Calculation("b", depends_on=["a"]),
    ]
    assert [item.name for item in order_variables(variables)] == ["a", "b", "c"]
    with pytest.raises(DependencyError, match="circular dependency"):
        order_variables(
            [Calculation("a", depends_on=["b"]), Calculation("b", depends_on=["a"])]
        )
    with pytest.raises(DependencyError, match="unique"):
        order_variables([Calculation("a"), Calculation("a")])
