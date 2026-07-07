import pandas as pd
import pytest

from projectionmodels import (
    Assumption,
    ClaimExperience,
    ClaimProjection,
    CredibilityAssumption,
    ProjectionHorizon,
    SeasonalityAssumption,
    TrendAssumption,
)


def test_claim_experience_and_projection_by_claim_type():
    history = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "product_id": ["PPO"] * 4,
            "claim_type": ["ip", "ip", "op", "op"],
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"] * 2),
            "claims": [1000.0, 1000.0, 500.0, 500.0],
            "member_months": [100.0, 100.0, 100.0, 100.0],
        }
    )
    experience = ClaimExperience(
        history,
        projection_keys=["group_id", "product_id"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="member_months",
    )
    seasonality_values = pd.DataFrame(
        {
            "claim_type": [item for item in ["ip", "op"] for _ in range(12)],
            "season": list(range(1, 13)) * 2,
            "factor": [1.0] * 24,
        }
    )
    seasonality = SeasonalityAssumption.from_values(
        "claim_seasonality",
        seasonality_values,
        lookup=["claim_type"],
        factor_col="factor",
    )
    trend = TrendAssumption.from_values(
        "claim_trend",
        pd.DataFrame({"claim_type": ["ip", "op"], "trend": [0.12, 0.0]}),
        lookup=["claim_type"],
        rate_col="trend",
    )
    credibility = CredibilityAssumption.from_weights(
        "claim_credibility",
        pd.DataFrame({"claim_type": ["ip", "op"], "z": [0.5, 1.0]}),
        lookup=["claim_type"],
        weight_col="z",
    )
    complement = Assumption(
        "manual",
        pd.DataFrame({"claim_type": ["ip", "op"], "manual": [8.0, 4.0]}),
        lookup=["claim_type"],
        value_col="manual",
    )
    membership = pd.DataFrame(
        {
            "group_id": ["A", "A"],
            "product_id": ["PPO", "PPO"],
            "projection_period": ["2027-01", "2027-02"],
            "member_months": [100.0, 100.0],
        }
    )
    projection = ClaimProjection.from_experience(
        experience,
        membership=membership,
        horizon=ProjectionHorizon("2027-01-01", periods=2),
        trend=trend,
        seasonality=seasonality,
        credibility=credibility,
        complement=complement,
    )
    results = projection.project()
    assert set(results.frame["claim_type"]) == {"ip", "op"}
    ip = results.frame.query("claim_type == 'ip'").sort_values("projection_period")
    op = results.frame.query("claim_type == 'op'").sort_values("projection_period")
    assert ip["credible_claim_rate"].iloc[0] == pytest.approx(9.0)
    assert op["credible_claim_rate"].iloc[0] == pytest.approx(5.0)
    assert ip["projected_claim_rate"].iloc[1] > ip["projected_claim_rate"].iloc[0]
    assert op["projected_claim_rate"].iloc[1] == pytest.approx(
        op["projected_claim_rate"].iloc[0]
    )
    total = results.summarize(by=["projection_period", "group_id"])
    assert total["member_months"].tolist() == [100.0, 100.0]
