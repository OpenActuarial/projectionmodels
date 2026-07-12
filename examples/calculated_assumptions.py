"""Estimate completion, seasonality, trend, and credibility with real actuarialpy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from actuarialpy import Experience

import projectionmodels as pm
from projectionmodels.integrations import actuarialpy as apx


def _completion_transactions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for claim_type, scale in (("inpatient", 1.0), ("outpatient", 0.6)):
        origins = pd.date_range("2023-01-01", periods=8, freq="MS")
        for origin_index, origin in enumerate(origins):
            # At least two origins overlap at every modeled development age.
            maximum_development = 2 if origin_index <= 5 else 1 if origin_index == 6 else 0
            origin_scale = 1.0 + 0.03 * origin_index
            for development, payment_share in enumerate((0.50, 0.30, 0.20)):
                if development > maximum_development:
                    continue
                rows.append(
                    {
                        "claim_type": claim_type,
                        "incurred_month": origin,
                        "paid_month": origin + pd.DateOffset(months=development),
                        "paid_claims": 100_000.0
                        * scale
                        * origin_scale
                        * payment_share,
                    }
                )
    return pd.DataFrame(rows)


def _claim_history(valuation_date: pd.Timestamp) -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=36, freq="MS")
    seasonal = np.array([0.94, 0.92, 0.97, 0.99, 1.00, 1.01, 1.02, 1.01, 0.99, 1.02, 1.05, 1.08])
    rows: list[dict[str, object]] = []
    for claim_type, base, monthly_trend, claim_count in (
        ("inpatient", 100_000.0, 0.006, 30.0),
        ("outpatient", 60_000.0, 0.004, 90.0),
    ):
        for index, month in enumerate(months):
            ultimate = base * ((1.0 + monthly_trend) ** index) * seasonal[month.month - 1]
            maturity = (valuation_date.year - month.year) * 12 + valuation_date.month - month.month
            completion = 0.50 if maturity == 0 else 0.80 if maturity == 1 else 1.0
            rows.append(
                {
                    "group_id": "A",
                    "product_id": "PPO",
                    "claim_type": claim_type,
                    "incurred_month": month,
                    "reported_claims": ultimate * completion,
                    "member_months": 1_000.0,
                    "claim_count": claim_count,
                }
            )
    return pd.DataFrame(rows)


def run_example() -> dict[str, object]:
    valuation_date = pd.Timestamp("2026-12-31")
    history = _claim_history(valuation_date)

    completion = apx.estimate_completion(
        "claim_completion",
        _completion_transactions(),
        by=["claim_type"],
        origin_col="incurred_month",
        valuation_col="paid_month",
        amount_col="paid_claims",
    )
    completed_history = completion.apply(
        history,
        value_col="reported_claims",
        date_col="incurred_month",
        valuation_date=valuation_date,
        by=["claim_type"],
        out_col="completed_claims",
    )

    seasonality = apx.estimate_seasonality(
        "claim_seasonality",
        completed_history,
        by=["claim_type"],
        date_col="incurred_month",
        value_col="completed_claims",
        exposure_col="member_months",
    )
    deseasonalized_history = apx.remove_seasonality(
        completed_history,
        seasonality,
        date_col="incurred_month",
        value_col="completed_claims",
        by=["claim_type"],
        out_col="deseasonalized_claims",
    )
    trend = apx.estimate_trend(
        "claim_trend",
        deseasonalized_history,
        by=["claim_type"],
        date_col="incurred_month",
        value_col="deseasonalized_claims",
        exposure_col="member_months",
    )
    credibility = apx.estimate_credibility(
        "claim_credibility",
        history,
        method="limited_fluctuation",
        by=["claim_type"],
        exposure_col="claim_count",
        full_credibility_standard=2_000.0,
    )

    experience = Experience(
        history,
        expense="reported_claims",
        exposure="member_months",
        date="incurred_month",
        dimensions=["group_id", "product_id", "claim_type"],
        valuation_date=valuation_date,
    )
    complement = pm.Assumption(
        "manual_claim_rate",
        pd.DataFrame(
            {
                "claim_type": ["inpatient", "outpatient"],
                "manual_claim_rate": [125.0, 75.0],
            }
        ),
        lookup=["claim_type"],
        value_col="manual_claim_rate",
    )
    exposure = pd.DataFrame(
        {
            "group_id": ["A"] * 6,
            "product_id": ["PPO"] * 6,
            "projection_period": pd.period_range("2027-01", periods=6, freq="M").astype(str),
            "member_months": np.repeat(1_000.0, 6),
        }
    )

    projection = pm.project(
        experience,
        exposure=exposure,
        exposure_col="member_months",
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
        measures=["member_months", "projected_claims", "claims_per_exposure"],
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
