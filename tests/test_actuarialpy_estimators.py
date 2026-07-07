import pandas as pd
import pytest

from projectionmodels import (
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)


def test_actuarialpy_estimated_assumptions_share_resolved_interface():
    history = pd.DataFrame(
        {
            "claim_type": ["ip"] * 24 + ["op"] * 24,
            "month": list(pd.date_range("2025-01-01", periods=24, freq="MS")) * 2,
            "claims": [100.0 + i for i in range(24)] + [50.0 + i for i in range(24)],
            "exposure": [10.0] * 48,
        }
    )

    trend = TrendAssumption.from_experience(
        "claim_trend",
        history,
        by=["claim_type"],
        date_col="month",
        value_col="claims",
        exposure_col="exposure",
    )
    seasonality = SeasonalityAssumption.from_experience(
        "claim_seasonality",
        history,
        by=["claim_type"],
        date_col="month",
        value_col="claims",
        exposure_col="exposure",
    )
    credibility = CredibilityAssumption.from_experience(
        "claim_credibility",
        history,
        method="limited_fluctuation",
        by=["claim_type"],
        exposure_col="exposure",
        full_credibility_standard=200.0,
    )

    projection_rows = pd.DataFrame(
        {"claim_type": ["ip", "op"], "season": [1, 2]}
    )
    assert trend.resolve(projection_rows).tolist() == pytest.approx([0.12, 0.12])
    assert seasonality.resolve(projection_rows).tolist() == pytest.approx([1.0, 1.0])
    assert credibility.resolve(projection_rows).between(0, 1).all()
    assert trend.source == "actuarialpy_estimate"
    assert seasonality.source == "actuarialpy_estimate"
    assert credibility.source == "actuarialpy_estimate"


def test_actuarialpy_estimated_completion_can_be_applied():
    transactions = pd.DataFrame(
        {
            "claim_type": ["ip", "ip", "op", "op"],
            "incurred_month": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"]
            ),
            "paid_month": pd.to_datetime(
                ["2026-01-01", "2026-03-01", "2026-01-01", "2026-03-01"]
            ),
            "paid_claims": [50.0, 75.0, 30.0, 45.0],
        }
    )
    completion = CompletionAssumption.from_experience(
        "claim_completion",
        transactions,
        by=["claim_type"],
        origin_col="incurred_month",
        valuation_col="paid_month",
        amount_col="paid_claims",
    )
    observed = pd.DataFrame(
        {
            "claim_type": ["ip", "op"],
            "development_month": [0, 12],
            "reported_claims": [50.0, 75.0],
        }
    )
    completed = completion.apply(
        observed,
        value_col="reported_claims",
        development_col="development_month",
        by=["claim_type"],
        out_col="ultimate_claims",
    )
    assert completed["ultimate_claims"].tolist() == pytest.approx([100.0, 75.0])
    assert completion.source == "actuarialpy_estimate"
