"""General deterministic projection with scenarios, audits, and aggregation."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    records = pm.ProjectionData(
        pd.DataFrame(
            {
                "group_id": ["A", "B"],
                "product_id": ["PPO", "HMO"],
                "current_members": [1_000.0, 600.0],
                "current_premium_pmpm": [525.0, 475.0],
            }
        ),
        projection_keys=["group_id", "product_id"],
    )

    model = pm.ProjectionModel(
        assumptions=pm.AssumptionSet(
            pm.Assumption("retention_rate", 0.995),
            pm.TrendAssumption.from_values("premium_trend", 0.05),
        ),
        roll_forwards=[
            pm.RollForward(
                "members",
                initial="current_members",
                formula=lambda x: x.prior("members") * x["retention_rate"],
                grain=["group_id", "product_id"],
            ),
            pm.RollForward(
                "premium_pmpm",
                initial="current_premium_pmpm",
                formula=lambda x: x.prior("premium_pmpm")
                * (1 + x["premium_trend"]) ** x.year_fraction,
                grain=["group_id", "product_id"],
            ),
        ],
        calculations=[
            pm.CashFlow(
                "premium",
                formula=lambda x: x["members"] * x["premium_pmpm"],
                grain=["group_id", "product_id"],
                reporting_role="revenue",
            )
        ],
    )

    # Apply the increase once. The adjusted value then persists through the
    # premium roll-forward in later periods.
    premium_increase = pm.Scenario(
        "premium_increase",
        adjustments=[
            pm.Adjustment(
                name="Group A renewal premium adjustment",
                target="premium_pmpm",
                method="multiply",
                value=1.08,
                filters={"group_id": "A"},
                effective_from="2027-07-01",
                effective_to="2027-07-31",
            )
        ],
    )

    results = model.project(
        records,
        pm.ProjectionHorizon("2027-01-01", periods=24),
        scenarios=[pm.Scenario("baseline"), premium_increase],
    )
    annual = results.summarize(
        by=["scenario", "calendar_year", "product_id"],
        measures=["members", "premium"],
    )
    comparison = results.compare_scenarios(
        baseline="baseline",
        comparison="premium_increase",
        by=["calendar_year", "product_id"],
        measures=["premium"],
    )
    return {
        "results": results,
        "annual": annual,
        "comparison": comparison,
        "assumption_audit": results.assumption_audit(),
        "adjustment_audit": results.adjustment_audit(),
    }


if __name__ == "__main__":
    output = run_example()
    print(output["annual"].to_string(index=False))
    print("\nScenario comparison")
    print(output["comparison"].to_string(index=False))
