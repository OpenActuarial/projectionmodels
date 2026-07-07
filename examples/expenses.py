"""Expenses with PMPM, fixed, percent-of-premium, and percent-of-claims bases."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    expenses = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "expense_type": ["administration", "overhead", "commission", "claim_admin"],
            "base_value": [32.0, 5_000.0, 0.025, 0.010],
            "basis": ["pmpm", "fixed_monthly", "percent_premium", "percent_claims"],
            "base_date": pd.to_datetime(["2027-01-01"] * 4),
        }
    )

    periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
    membership = pd.DataFrame(
        {
            "group_id": ["A"] * 12,
            "projection_period": periods,
            "member_months": [1_000.0] * 12,
        }
    )
    premium = pd.DataFrame(
        {
            "group_id": ["A"] * 12,
            "projection_period": periods,
            "premium": [550_000.0] * 12,
        }
    )
    claims = pd.DataFrame(
        {
            "group_id": ["A"] * 12,
            "projection_period": periods,
            "projected_claims": [450_000.0] * 12,
        }
    )

    projection = pm.ExpenseProjection(
        expenses,
        projection_keys=["group_id"],
        expense_type_col="expense_type",
        base_value_col="base_value",
        basis_col="basis",
        base_date_col="base_date",
        horizon=pm.ProjectionHorizon("2027-01-01", periods=12),
        trend=pm.TrendAssumption.from_values("expense_trend", 0.04),
        membership=membership,
        premium=premium,
        claims=claims,
    )

    results = projection.project()
    annual = results.summarize(
        by=["calendar_year", "expense_type"],
        measures=["projected_expense"],
    )
    total = results.summarize(
        by=["calendar_year"],
        measures=["projected_expense"],
    )
    return {"results": results, "annual": annual, "total": total}


if __name__ == "__main__":
    print(run_example()["annual"].to_string(index=False))
