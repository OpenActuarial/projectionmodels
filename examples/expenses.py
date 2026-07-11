"""Expenses on all four bases with a per-type trend — including a contractually flat fee."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    expenses = pd.DataFrame(
        {
            "group_id": ["A"] * 5,
            "expense_type": [
                "administration", "network_fee", "overhead", "commission", "claim_admin",
            ],
            "base_value": [32.0, 6.50, 5_000.0, 0.025, 0.010],
            "basis": [
                "per_exposure", "per_exposure", "fixed_monthly",
                "percent_premium", "percent_claims",
            ],
            "base_date": pd.to_datetime(["2027-01-01"] * 5),
        }
    )

    periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
    exposure = pd.DataFrame(
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
        # Keyed by expense type: a zero-trend type is how a contractually
        # flat fee stays flat while its neighbours trend, and the percent
        # bases are rates, so they hold level too.
        trend=pm.TrendAssumption.from_values(
            "expense_trend",
            pd.DataFrame(
                {
                    "expense_type": [
                        "administration", "network_fee", "overhead",
                        "commission", "claim_admin",
                    ],
                    "expense_trend": [0.04, 0.0, 0.04, 0.0, 0.0],
                }
            ),
            lookup="expense_type",
        ),
        exposure=exposure,
        exposure_col="member_months",
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
