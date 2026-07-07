import pandas as pd
import pytest

from projectionmodels import ExpenseProjection, ProjectionHorizon, TrendAssumption


def test_expense_projection_multiple_bases():
    expenses = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "expense_type": ["admin", "fixed", "commission"],
            "base_value": [2.0, 100.0, 0.05],
            "basis": ["pmpm", "fixed_monthly", "percent_premium"],
            "base_date": pd.to_datetime(["2027-01-01"] * 3),
        }
    )
    membership = pd.DataFrame(
        {"group": ["A"], "projection_period": ["2027-01"], "member_months": [10.0]}
    )
    premium = pd.DataFrame(
        {"group": ["A"], "projection_period": ["2027-01"], "premium": [1000.0]}
    )
    projection = ExpenseProjection(
        expenses,
        projection_keys=["group"],
        expense_type_col="expense_type",
        base_value_col="base_value",
        basis_col="basis",
        base_date_col="base_date",
        horizon=ProjectionHorizon("2027-01-01", periods=1),
        trend=TrendAssumption.from_values("expense_trend", 0.0),
        membership=membership,
        premium=premium,
    )
    results = projection.project()
    values = results.frame.set_index("expense_type")["projected_expense"]
    assert values["admin"] == pytest.approx(20.0)
    assert values["fixed"] == pytest.approx(100.0)
    assert values["commission"] == pytest.approx(50.0)
