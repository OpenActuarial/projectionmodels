from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projectionmodels import (
    Assumption,
    AssumptionResolutionError,
    AssumptionSet,
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
    ValidationError,
)


def test_assumption_resolves_positional_series_and_array():
    frame = pd.DataFrame({"id": [1, 2]})
    series = Assumption("x", pd.Series([3.0, 4.0]))
    array = Assumption("y", np.array([5.0, 6.0]))
    assert series.resolve(frame).tolist() == [3.0, 4.0]
    assert array.resolve(frame).tolist() == [5.0, 6.0]


def test_assumption_resolves_named_series_index():
    frame = pd.DataFrame({"product": ["PPO", "HMO"]})
    values = pd.Series(
        [0.08, 0.05], index=pd.Index(["PPO", "HMO"], name="product")
    )
    assumption = Assumption("trend", values, lookup=["product"])
    assert assumption.resolve(frame).tolist() == [0.08, 0.05]


def test_assumption_strict_false_allows_unmatched_keys():
    assumption = Assumption(
        "trend",
        pd.DataFrame({"product": ["PPO"], "trend": [0.08]}),
        lookup=["product"],
    )
    result = assumption.resolve(pd.DataFrame({"product": ["HMO"]}), strict=False)
    assert result.isna().all()


def test_assumption_invalid_shapes_and_unkeyed_tables_raise():
    frame = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(AssumptionResolutionError):
        Assumption("x", np.ones((2, 2))).resolve(frame)
    with pytest.raises(AssumptionResolutionError):
        Assumption("x", pd.DataFrame({"x": [1, 2]})).resolve(frame)


def test_assumption_missing_projection_lookup_column_raises():
    assumption = Assumption(
        "trend",
        pd.DataFrame({"product": ["PPO"], "trend": [0.08]}),
        lookup=["product"],
    )
    with pytest.raises(AssumptionResolutionError, match="lack lookup columns"):
        assumption.resolve(pd.DataFrame({"group": ["A"]}))


def test_assumption_constructor_validates_columns_and_duplicate_keys():
    with pytest.raises(ValidationError, match="missing columns"):
        Assumption("trend", pd.DataFrame({"product": ["PPO"]}), lookup=["product"])
    with pytest.raises(ValidationError, match="duplicate lookup keys"):
        Assumption(
            "trend",
            pd.DataFrame(
                {"product": ["PPO", "PPO"], "trend": [0.08, 0.09]}
            ),
            lookup=["product"],
        )


def test_estimated_assumption_selection_preserves_indication_and_note():
    history = pd.DataFrame(
        {
            "month": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "claims": np.arange(12) + 1,
        }
    )
    indicated = TrendAssumption.from_experience(
        "trend", history, date_col="month", value_col="claims"
    )
    selected = indicated.select(0.10, note="pricing committee selection")
    assert selected.source == "actuarialpy_estimate_with_selection"
    assert selected.indicated_values is not None
    assert selected.resolve(pd.DataFrame({"row": [1, 2]})).tolist() == [0.10, 0.10]
    audit = selected.audit_frame()
    assert audit["selection_note"].item() == "pricing committee selection"


def test_keyed_series_selection_is_resolved_by_index():
    original = Assumption(
        "trend",
        pd.DataFrame({"product": ["PPO", "HMO"], "trend": [0.08, 0.05]}),
        lookup=["product"],
    )
    selection = pd.Series(
        [0.07, 0.04], index=pd.Index(["PPO", "HMO"], name="product")
    )
    selected = original.select(selection)
    result = selected.resolve(pd.DataFrame({"product": ["HMO", "PPO"]}))
    assert result.tolist() == [0.04, 0.07]
    assert selected.source == "supplied_selection"


def test_invalid_selection_types_and_missing_value_columns_raise():
    assumption = Assumption("x", 1.0)
    with pytest.raises(TypeError):
        assumption.select({"x": 2.0})
    with pytest.raises(ValidationError, match="does not contain"):
        assumption.select(pd.DataFrame({"wrong": [2.0]}))


def test_trend_factor_uses_resolved_keyed_rates():
    trend = TrendAssumption.from_values(
        "trend",
        pd.DataFrame({"product": ["PPO", "HMO"], "rate": [0.12, 0.0]}),
        lookup=["product"],
        rate_col="rate",
    )
    frame = pd.DataFrame({"product": ["PPO", "HMO"]})
    assert trend.factor(frame, [12, 12]).tolist() == pytest.approx([1.12, 1.0])


def test_ungrouped_seasonality_and_completion_estimators():
    history = pd.DataFrame(
        {
            "month": pd.date_range("2025-01-01", periods=24, freq="MS"),
            "claims": np.arange(24) + 1,
        }
    )
    seasonality = SeasonalityAssumption.from_experience(
        "seasonality", history, date_col="month", value_col="claims"
    )
    assert seasonality.lookup == ("season",)
    assert len(seasonality.values) == 12

    transactions = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "paid": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "amount": [50.0, 75.0],
        }
    )
    completion = CompletionAssumption.from_experience(
        "completion",
        transactions,
        origin_col="origin",
        valuation_col="paid",
        amount_col="amount",
    )
    assert completion.lookup == ("development_month",)
    applied = completion.apply(
        pd.DataFrame({"development_month": [0, 1], "claims": [75.0, 100.0]}),
        value_col="claims",
        development_col="development_month",
    )
    assert applied["claims_completed"].tolist() == pytest.approx([100.0, 100.0])


def test_calculated_seasonality_requires_normalized_factors(fake_actuarialpy):
    fake_actuarialpy.seasonality_factors = lambda *args, **kwargs: pd.Series(
        [2.0] * 12, index=pd.Index(range(1, 13), name="season")
    )
    history = pd.DataFrame(
        {
            "month": pd.date_range("2025-01-01", periods=24, freq="MS"),
            "claims": np.arange(24) + 1,
        }
    )
    with pytest.raises(ValidationError, match="not 1.0"):
        SeasonalityAssumption.from_experience(
            "seasonality", history, date_col="month", value_col="claims"
        )


def test_credibility_blend_and_validation():
    credibility = CredibilityAssumption.from_weights("z", 0.25)
    blended = credibility.blend(
        observed=pd.Series([100.0, 80.0]),
        complement=pd.Series([60.0, 60.0]),
        frame=pd.DataFrame({"id": [1, 2]}),
    )
    assert blended.tolist() == pytest.approx([70.0, 65.0])

    history = pd.DataFrame({"group": ["A"], "value": [1.0]})
    with pytest.raises(ValidationError, match="at least one risk key"):
        CredibilityAssumption.from_experience(
            "z", history, method="limited_fluctuation", by=[]
        )
    with pytest.raises(ValidationError, match="requires exposure_col"):
        CredibilityAssumption.from_experience(
            "z", history, method="limited_fluctuation", by=["group"]
        )
    with pytest.raises(ValidationError, match="method must be"):
        CredibilityAssumption.from_experience(
            "z", history, method="unknown", by=["group"]
        )


def test_buhlmann_and_buhlmann_straub_estimators():
    balanced = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "year": [1, 2, 1, 2],
            "rate": [10.0, 12.0, 20.0, 18.0],
            "exposure": [100.0, 110.0, 80.0, 90.0],
        }
    )
    buhlmann = CredibilityAssumption.from_experience(
        "z_buhlmann",
        balanced,
        method="buhlmann",
        by=["group"],
        value_col="rate",
        period_col="year",
    )
    assert buhlmann.values["z_buhlmann"].eq(0.6).all()
    assert buhlmann.metadata["method"] == "buhlmann"

    straub = CredibilityAssumption.from_experience(
        "z_straub",
        balanced,
        method="buhlmann_straub",
        by=["group"],
        value_col="rate",
        weight_col="exposure",
        period_col="year",
    )
    assert straub.values["z_straub"].between(0, 1).all()
    assert straub.metadata["method"] == "buhlmann_straub"


def test_buhlmann_rejects_unbalanced_periods():
    unbalanced = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "year": [1, 2, 1],
            "rate": [10.0, 12.0, 20.0],
        }
    )
    with pytest.raises(ValidationError, match="same complete period set"):
        CredibilityAssumption.from_experience(
            "z",
            unbalanced,
            method="buhlmann",
            by=["group"],
            value_col="rate",
            period_col="year",
        )


def test_assumption_set_resolution_audit_and_duplicates():
    assumptions = AssumptionSet(
        Assumption("a", 1.0),
        b=Assumption("different_name", 2.0),
    )
    resolved = assumptions.resolve(pd.DataFrame({"id": [1, 2]}))
    assert resolved[["a", "b"]].to_dict("list") == {"a": [1.0, 1.0], "b": [2.0, 2.0]}
    assert set(assumptions.audit_frame()["name"]) == {"a", "b"}
    assert AssumptionSet().audit_frame().empty
    with pytest.raises(ValidationError, match="already exists"):
        assumptions.add(Assumption("a", 3.0))
