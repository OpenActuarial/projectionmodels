"""Member-level life-style expected-value projection with model-point weights."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm
import projectionmodels.advanced as pma


def run_example() -> dict[str, object]:
    records = pma.ProjectionData(
        pd.DataFrame(
            {
                "member_id": ["M1", "M2", "M3"],
                "group_id": ["G1", "G1", "G2"],
                "product_id": ["term", "term", "whole_life"],
                "issue_date": pd.to_datetime(["2024-01-01", "2026-06-01", "2025-03-01"]),
                "termination_date": pd.to_datetime([None, "2028-06-30", None]),
                "initial_inforce_probability": [1.0, 1.0, 1.0],
                "monthly_premium": [42.0, 55.0, 80.0],
                "face_amount": [100_000.0, 150_000.0, 75_000.0],
                "model_point_weight": [10.0, 5.0, 20.0],
            }
        ),
        projection_keys=["member_id"],
        record_weight="model_point_weight",
        dates=pm.ProjectionDates(
            entry_date="issue_date",
            exit_date="termination_date",
            exposure_timing="daily_prorated",
        ),
    )

    lapse = pm.Assumption(
        "monthly_lapse_rate",
        pd.DataFrame(
            {
                "product_id": ["term", "whole_life"],
                "monthly_lapse_rate": [0.004, 0.002],
            }
        ),
        lookup=["product_id"],
    )
    mortality = pm.Assumption(
        "monthly_mortality_rate",
        pd.DataFrame(
            {
                "product_id": ["term", "whole_life"],
                "monthly_mortality_rate": [0.00020, 0.00035],
            }
        ),
        lookup=["product_id"],
    )
    model = pma.ProjectionModel(
        assumptions=pma.AssumptionSet(lapse, mortality),
        roll_forwards=[
            pma.RollForward(
                "inforce_probability",
                initial="initial_inforce_probability",
                formula=lambda x: x.prior("inforce_probability")
                * (1 - x["monthly_lapse_rate"] - x["monthly_mortality_rate"]),
                aggregation="mean",
                grain=["member_id"],
            )
        ],
        calculations=[
            pma.CashFlow(
                "premium",
                formula=lambda x: x["inforce_probability"]
                * x["monthly_premium"]
                * x.weight
                * x["active_fraction"],
                grain=["member_id"],
                reporting_role="revenue",
            ),
            pma.CashFlow(
                "expected_death_benefit",
                formula=lambda x: x.prior("inforce_probability")
                * x["monthly_mortality_rate"]
                * x["face_amount"]
                * x.weight
                * x["active_fraction"],
                grain=["member_id"],
                reporting_role="loss",
            ),
        ],
    )

    high_lapse = pm.Scenario(
        "high_lapse",
        [
            pm.Adjustment(
                target="monthly_lapse_rate",
                method="add",
                value=0.002,
                filters={"product_id": "term"},
            )
        ],
    )
    results = model.project(
        records,
        pm.ProjectionHorizon("2027-01-01", periods=24),
        scenarios=[pm.Scenario("baseline"), high_lapse],
    )
    annual = results.summarize(
        by=["scenario", "calendar_year", "product_id"],
        measures=["premium", "expected_death_benefit"],
    )
    comparison = results.compare_scenarios(
        baseline="baseline",
        comparison="high_lapse",
        by=["calendar_year", "product_id"],
        measures=["premium"],
    )
    return {"results": results, "annual": annual, "comparison": comparison}


if __name__ == "__main__":
    print(run_example()["annual"].to_string(index=False))
