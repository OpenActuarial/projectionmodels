"""Produce underwriting-ready projected premium, claims, expenses, and ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    records = pm.ProjectionData(
        pd.DataFrame(
            {
                "group_id": ["A", "B"],
                "product_id": ["PPO", "PPO"],
                "effective_date": pd.to_datetime(["2025-01-01", "2027-04-15"]),
                "initial_members": [1_000.0, 300.0],
                "initial_premium_pmpm": [540.0, 560.0],
                "initial_claim_pmpm": [455.0, 470.0],
                "initial_expense_pmpm": [35.0, 38.0],
            }
        ),
        projection_keys=["group_id", "product_id"],
        dates=pm.ProjectionDates(
            entry_date="effective_date",
            exposure_timing="daily_prorated",
        ),
    ).add_date_cohort(
        pm.DateCohort(
            "business_origin",
            "effective_date",
            split_date="2027-01-01",
            before_label="existing",
            on_or_after_label="new_business",
        )
    )

    model = pm.ProjectionModel(
        assumptions=pm.AssumptionSet(
            pm.Assumption("retention_rate", 0.995),
            pm.Assumption("premium_trend", 0.05),
            pm.Assumption("claim_trend", 0.07),
            pm.Assumption("expense_trend", 0.03),
        ),
        roll_forwards=[
            pm.RollForward(
                "members",
                initial="initial_members",
                formula=lambda x: np.where(
                    ~x["is_active"],
                    0.0,
                    np.where(
                        x.prior("members") == 0,
                        x["initial_members"],
                        x.prior("members") * x["retention_rate"],
                    ),
                ),
                grain=["group_id", "product_id"],
            ),
            pm.RollForward(
                "premium_pmpm",
                initial="initial_premium_pmpm",
                formula=lambda x: x.prior("premium_pmpm")
                * (1 + x["premium_trend"]) ** x.year_fraction,
                aggregation="mean",
                grain=["group_id", "product_id"],
            ),
            pm.RollForward(
                "claim_pmpm",
                initial="initial_claim_pmpm",
                formula=lambda x: x.prior("claim_pmpm")
                * (1 + x["claim_trend"]) ** x.year_fraction,
                aggregation="mean",
                grain=["group_id", "product_id"],
            ),
            pm.RollForward(
                "expense_pmpm",
                initial="initial_expense_pmpm",
                formula=lambda x: x.prior("expense_pmpm")
                * (1 + x["expense_trend"]) ** x.year_fraction,
                aggregation="mean",
                grain=["group_id", "product_id"],
            ),
        ],
        calculations=[
            pm.Calculation(
                "member_months",
                formula=lambda x: x["members"] * x["active_fraction"],
                grain=["group_id", "product_id"],
                reporting_role="exposure",
            ),
            pm.CashFlow(
                "premium",
                formula=lambda x: x["premium_pmpm"] * x["member_months"],
                grain=["group_id", "product_id"],
                reporting_role="revenue",
                depends_on=["member_months"],
            ),
            pm.CashFlow(
                "claims",
                formula=lambda x: x["claim_pmpm"] * x["member_months"],
                grain=["group_id", "product_id"],
                reporting_role="loss",
                depends_on=["member_months"],
            ),
            pm.CashFlow(
                "expenses",
                formula=lambda x: x["expense_pmpm"] * x["member_months"],
                grain=["group_id", "product_id"],
                reporting_role="expense",
                depends_on=["member_months"],
            ),
            pm.CashFlow(
                "underwriting_margin",
                formula=lambda x: x["premium"] - x["claims"] - x["expenses"],
                grain=["group_id", "product_id"],
                reporting_role="margin",
                depends_on=["premium", "claims", "expenses"],
            ),
            pm.Metric(
                "loss_ratio",
                formula=lambda x: x["claims"] / x["premium"],
                aggregation="recalculate",
                numerator="claims",
                denominator="premium",
                grain=["group_id", "product_id"],
                depends_on=["premium", "claims"],
            ),
        ],
    )

    adverse = pm.Scenario(
        "adverse",
        [pm.Adjustment(target="claim_trend", method="add", value=0.02)],
    )
    results = model.project(
        records,
        pm.ProjectionHorizon("2027-01-01", periods=24),
        scenarios=[pm.Scenario("baseline"), adverse],
    )
    exhibit_input = results.summarize(
        by=["scenario", "business_origin", "calendar_year"],
        measures=[
            "member_months",
            "premium",
            "claims",
            "expenses",
            "underwriting_margin",
            "loss_ratio",
        ],
    )
    comparison = results.compare_scenarios(
        baseline="baseline",
        comparison="adverse",
        by=["business_origin", "calendar_year"],
        measures=["claims", "underwriting_margin", "loss_ratio"],
    )
    return {
        "results": results,
        "exhibit_input": exhibit_input,
        "comparison": comparison,
    }


if __name__ == "__main__":
    print(run_example()["exhibit_input"].to_string(index=False))
