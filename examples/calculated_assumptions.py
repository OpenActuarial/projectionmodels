"""Estimate completion, trend, seasonality, and credibility with actuarialpy."""

from __future__ import annotations

import numpy as np
import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    months = pd.date_range("2025-01-01", periods=24, freq="MS")
    rows: list[dict[str, object]] = []
    for claim_type, base in (("inpatient", 100_000.0), ("outpatient", 60_000.0)):
        for index, month in enumerate(months):
            rows.append(
                {
                    "group_id": "A",
                    "product_id": "PPO",
                    "claim_type": claim_type,
                    "incurred_month": month,
                    "reported_claims": base * (1.005**index),
                    "member_months": 1_000.0,
                    "claim_count": 30.0 if claim_type == "inpatient" else 90.0,
                }
            )
    history = pd.DataFrame(rows)

    transactions = history.loc[
        :, ["claim_type", "incurred_month", "reported_claims"]
    ].copy()
    transactions["paid_month"] = transactions["incurred_month"]
    transactions = transactions.rename(columns={"reported_claims": "paid_claims"})

    completion = pm.CompletionAssumption.from_experience(
        "claim_completion",
        transactions,
        by=["claim_type"],
        origin_col="incurred_month",
        valuation_col="paid_month",
        amount_col="paid_claims",
    )
    seasonality = pm.SeasonalityAssumption.from_experience(
        "claim_seasonality",
        history,
        by=["claim_type"],
        date_col="incurred_month",
        value_col="reported_claims",
        exposure_col="member_months",
    )
    trend = pm.TrendAssumption.from_experience(
        "claim_trend",
        history,
        by=["claim_type"],
        date_col="incurred_month",
        value_col="reported_claims",
        exposure_col="member_months",
    )
    credibility = pm.CredibilityAssumption.from_experience(
        "claim_credibility",
        history,
        method="limited_fluctuation",
        by=["claim_type"],
        exposure_col="claim_count",
        full_credibility_standard=2_000.0,
    )

    experience = pm.ClaimExperience(
        history,
        projection_keys=["group_id", "product_id"],
        claim_type_col="claim_type",
        date_col="incurred_month",
        claims_col="reported_claims",
        exposure_col="member_months",
        valuation_date="2026-12-31",
    )
    complement = pm.Assumption(
        "manual_claim_rate",
        pd.DataFrame(
            {
                "claim_type": ["inpatient", "outpatient"],
                "manual_claim_rate": [115.0, 70.0],
            }
        ),
        lookup=["claim_type"],
        value_col="manual_claim_rate",
    )
    membership = pd.DataFrame(
        {
            "group_id": ["A"] * 6,
            "product_id": ["PPO"] * 6,
            "projection_period": pd.period_range("2027-01", periods=6, freq="M").astype(
                str
            ),
            "member_months": np.repeat(1_000.0, 6),
        }
    )

    projection = pm.ClaimProjection.from_experience(
        experience,
        membership=membership,
        horizon=pm.ProjectionHorizon("2027-01-01", periods=6),
        completion=completion,
        seasonality=seasonality,
        trend=trend,
        credibility=credibility,
        complement=complement,
    )
    results = projection.project()
    summary = results.summarize(
        by=["claim_type"],
        measures=["member_months", "projected_claims", "claim_pmpm"],
    )
    assumption_audit = pd.concat(
        [
            completion.audit_frame(),
            seasonality.audit_frame(),
            trend.audit_frame(),
            credibility.audit_frame(),
        ],
        ignore_index=True,
    )
    return {
        "results": results,
        "summary": summary,
        "completion": completion,
        "seasonality": seasonality,
        "trend": trend,
        "credibility": credibility,
        "assumption_audit": assumption_audit,
    }


if __name__ == "__main__":
    output = run_example()
    print(output["summary"].to_string(index=False))
    print("\nEstimated assumptions")
    print(output["assumption_audit"].to_string(index=False))
