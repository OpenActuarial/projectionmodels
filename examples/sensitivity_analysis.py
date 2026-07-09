"""Systematic trend sensitivity with baseline comparison."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm
import projectionmodels.advanced as pma


def run_example() -> dict[str, object]:
    records = pma.ProjectionData(
        pd.DataFrame(
            {
                "product_id": ["PPO", "HMO"],
                "base_claim_pmpm": [500.0, 430.0],
                "member_months": [10_000.0, 8_000.0],
            }
        ),
        projection_keys=["product_id"],
    )
    model = pma.ProjectionModel(
        assumptions=pma.AssumptionSet(pm.Assumption("claim_trend", 0.06)),
        roll_forwards=[
            pma.RollForward(
                "claim_pmpm",
                initial="base_claim_pmpm",
                formula=lambda x: x.prior("claim_pmpm")
                * (1 + x["claim_trend"]) ** x.year_fraction,
                aggregation="mean",
                grain=["product_id"],
            )
        ],
        calculations=[
            pma.CashFlow(
                "projected_claims",
                formula=lambda x: x["claim_pmpm"] * x["member_months"],
                grain=["product_id"],
                reporting_role="loss",
            )
        ],
    )
    sensitivity = pma.Sensitivity(
        target="claim_trend",
        values=[0.04, 0.06, 0.08],
        method="set",
        name_template="trend_{value:.0%}",
    )
    results = model.run_sensitivity(
        records,
        pm.ProjectionHorizon("2027-01-01", periods=24),
        sensitivity,
    )
    annual = results.summarize(
        by=["scenario", "calendar_year"],
        measures=["projected_claims"],
    )
    comparison = results.compare_scenarios(
        baseline="trend_4%",
        comparison="trend_8%",
        by=["calendar_year"],
        measures=["projected_claims"],
    )
    return {"results": results, "annual": annual, "comparison": comparison}


if __name__ == "__main__":
    output = run_example()
    print(output["annual"].to_string(index=False))
    print("\n4% versus 8%")
    print(output["comparison"].to_string(index=False))
