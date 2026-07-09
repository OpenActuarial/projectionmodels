"""Claims by claim type with supplied trend, seasonality, credibility, and exposure."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    history = pd.DataFrame(
        {
            "group_id": ["A"] * 8,
            "product_id": ["PPO"] * 8,
            "claim_type": ["inpatient"] * 4 + ["outpatient"] * 4,
            "incurred_month": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"] * 2
            ),
            "reported_claims": [
                95_000,
                110_000,
                100_000,
                105_000,
                55_000,
                58_000,
                62_000,
                60_000,
            ],
            "member_months": [1_000.0] * 8,
        }
    )

    experience = pm.ClaimExperience(
        history,
        projection_keys=["group_id", "product_id"],
        claim_type_col="claim_type",
        date_col="incurred_month",
        claims_col="reported_claims",
        exposure_col="member_months",
    )

    trend = pm.TrendAssumption.from_values(
        "claim_trend",
        pd.DataFrame(
            {
                "claim_type": ["inpatient", "outpatient"],
                "annual_trend": [0.075, 0.060],
            }
        ),
        lookup=["claim_type"],
        rate_col="annual_trend",
    )

    seasonality_table = pd.DataFrame(
        {
            "claim_type": [
                claim_type
                for claim_type in ["inpatient", "outpatient"]
                for _ in range(12)
            ],
            "season": list(range(1, 13)) * 2,
            "seasonality_factor": [
                0.96,
                0.95,
                0.98,
                1.00,
                1.01,
                1.02,
                1.03,
                1.01,
                0.99,
                1.00,
                1.02,
                1.03,
            ]
            + [
                0.94,
                0.95,
                0.98,
                1.00,
                1.02,
                1.03,
                1.04,
                1.02,
                1.00,
                0.99,
                1.01,
                1.02,
            ],
        }
    )
    seasonality = pm.SeasonalityAssumption.from_values(
        "claim_seasonality",
        seasonality_table,
        lookup=["claim_type"],
        factor_col="seasonality_factor",
    )

    credibility = pm.CredibilityAssumption.from_weights(
        "claim_credibility",
        pd.DataFrame(
            {
                "claim_type": ["inpatient", "outpatient"],
                "credibility": [0.65, 0.80],
            }
        ),
        lookup=["claim_type"],
        weight_col="credibility",
    )

    manual = pm.Assumption(
        "manual_claim_rate",
        pd.DataFrame(
            {
                "product_id": ["PPO", "PPO"],
                "claim_type": ["inpatient", "outpatient"],
                "manual_claims_per_exposure": [108.0, 62.0],
            }
        ),
        lookup=["product_id", "claim_type"],
        value_col="manual_claims_per_exposure",
    )

    exposure = pd.DataFrame(
        {
            "group_id": ["A"] * 12,
            "product_id": ["PPO"] * 12,
            "projection_period": pd.period_range("2027-01", periods=12, freq="M").astype(
                str
            ),
            "member_months": [1_000.0] * 12,
        }
    )

    projection = pm.ClaimProjection.from_experience(
        experience,
        exposure=exposure,
        exposure_col="member_months",
        horizon=pm.ProjectionHorizon("2027-01-01", periods=12),
        trend=trend,
        seasonality=seasonality,
        credibility=credibility,
        complement=manual,
    )

    results = projection.project()
    by_type = results.summarize(
        by=["projection_period", "claim_type"],
        measures=["member_months", "projected_claims", "claims_per_exposure"],
    )
    total = results.summarize(
        by=["projection_period"],
        measures=["member_months", "projected_claims", "claims_per_exposure"],
    )
    return {"results": results, "by_type": by_type, "total": total}


if __name__ == "__main__":
    print(run_example()["by_type"].to_string(index=False))
