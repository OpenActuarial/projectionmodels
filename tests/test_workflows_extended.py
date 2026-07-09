from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import (
    ClaimExperience,
    ClaimProjection,
    CompletionAssumption,
    CredibilityAssumption,
    ExpenseProjection,
    ProjectionDates,
    ProjectionHorizon,
    TrendAssumption,
    ValidationError,
)


def claim_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["A", "A"],
            "claim_type": ["ip", "ip"],
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "claims": [100.0, 120.0],
            "exposure": [10.0, 10.0],
            "region": ["west", "west"],
        }
    )


def test_claim_experience_prepare_without_optional_adjustments():
    experience = ClaimExperience(
        claim_history(),
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    prepared = experience.prepare()
    assert prepared["claims_completed"].tolist() == [100.0, 120.0]
    assert prepared["claims_completed_deseasonalized"].tolist() == [100.0, 120.0]



def test_claim_experience_midpoint_preserves_datetime_units():
    experience = ClaimExperience(
        claim_history(),
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    rates = experience.to_base_rates()
    midpoint = rates["experience_midpoint"].item()
    assert pd.Timestamp("2026-01-01") <= midpoint <= pd.Timestamp("2026-02-01")

def test_claim_completion_without_development_requires_valuation_date():
    experience = ClaimExperience(
        claim_history(),
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    completion = CompletionAssumption.from_values(
        "completion", pd.Series([0.5], index=pd.Index([0], name="development_month"))
    )
    with pytest.raises(ValidationError, match="valuation_date"):
        experience.prepare(completion=completion)


def test_claim_base_rates_extra_columns_and_scalar_complement():
    experience = ClaimExperience(
        claim_history(),
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    rates = experience.to_base_rates(extra_record_cols=["region"], complement=8.0)
    assert rates["region"].item() == "west"
    assert rates["experience_claim_rate"].item() == pytest.approx(11.0)
    assert rates["complement_claim_rate"].item() == 8.0


def test_claim_base_rates_reject_variable_extra_columns_and_invalid_complement():
    history = claim_history()
    history["region"] = ["west", "east"]
    experience = ClaimExperience(
        history,
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    with pytest.raises(ValidationError, match="not constant"):
        experience.to_base_rates(extra_record_cols=["region"])
    with pytest.raises(ValidationError, match="complement must be"):
        experience.to_base_rates(complement=pd.DataFrame({"rate": [1.0]}))


def test_claim_projection_without_credibility_uses_experience_rate():
    experience = ClaimExperience(
        claim_history(),
        projection_keys=["group"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="exposure",
    )
    exposure = pd.DataFrame(
        {"group": ["A"], "projection_period": ["2027-01"], "exposure": [10.0]}
    )
    projection = ClaimProjection.from_experience(
        experience,
        exposure=exposure,
        horizon=ProjectionHorizon("2027-01-01", periods=1),
        trend=TrendAssumption.from_values("trend", 0.0),
    )
    result = projection.project().frame
    assert result["credible_claim_rate"].item() == pytest.approx(11.0)
    assert result["projected_claims"].item() == pytest.approx(110.0)


def test_claim_projection_validation_errors():
    base = pd.DataFrame(
        {
            "group": ["A"],
            "claim_type": ["ip"],
            "experience_claim_rate": [10.0],
            "experience_midpoint": pd.to_datetime(["2026-01-01"]),
        }
    )
    with pytest.raises(ValidationError, match="exposure is missing"):
        ClaimProjection(
            base,
            projection_keys=["group"],
            claim_type_col="claim_type",
            exposure=pd.DataFrame({"group": ["A"]}),
            horizon=ProjectionHorizon("2027-01-01", periods=1),
            trend=TrendAssumption.from_values("trend", 0.0),
        )
    with pytest.raises(ValidationError, match="credibility requires"):
        ClaimProjection(
            base,
            projection_keys=["group"],
            claim_type_col="claim_type",
            exposure=pd.DataFrame(
                {"group": ["A"], "projection_period": ["2027-01"], "exposure": [1.0]}
            ),
            horizon=ProjectionHorizon("2027-01-01", periods=1),
            trend=TrendAssumption.from_values("trend", 0.0),
            credibility=CredibilityAssumption.from_weights("z", 0.5),
        )


def base_expenses(basis: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["A"],
            "expense_type": ["expense"],
            "base_value": [0.1 if basis.startswith("percent") else 10.0],
            "basis": [basis],
            "base_date": pd.to_datetime(["2027-01-01"]),
        }
    )


def make_expense_projection(basis: str, **kwargs) -> ExpenseProjection:
    return ExpenseProjection(
        base_expenses(basis),
        projection_keys=["group"],
        expense_type_col="expense_type",
        base_value_col="base_value",
        basis_col="basis",
        base_date_col="base_date",
        horizon=ProjectionHorizon("2027-01-01", periods=1),
        trend=TrendAssumption.from_values("trend", 0.0),
        **kwargs,
    )


def test_percent_claims_expense_projection():
    claims = pd.DataFrame(
        {"group": ["A"], "projection_period": ["2027-01"], "projected_claims": [500.0]}
    )
    result = make_expense_projection("percent_claims", claims=claims).project().frame
    assert result["projected_expense"].item() == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("basis", "message"),
    [
        ("per_exposure", "exposure"),
        ("percent_premium", "premium"),
        ("percent_claims", "claims"),
    ],
)
def test_expense_projection_requires_basis_tables(basis, message):
    with pytest.raises(ValidationError, match=message):
        make_expense_projection(basis).project()


def test_expense_validation_and_supporting_table_columns():
    with pytest.raises(ValidationError, match="unknown expense bases"):
        make_expense_projection("unsupported")
    with pytest.raises(ValidationError, match="exposure table is missing"):
        make_expense_projection("per_exposure", exposure=pd.DataFrame({"group": ["A"]})).project()


def test_expense_projection_respects_partial_period_dates():
    expenses = base_expenses("fixed_monthly").assign(entry=pd.to_datetime(["2027-01-16"]))
    projection = ExpenseProjection(
        expenses,
        projection_keys=["group"],
        expense_type_col="expense_type",
        base_value_col="base_value",
        basis_col="basis",
        base_date_col="base_date",
        horizon=ProjectionHorizon("2027-01-01", periods=1),
        trend=TrendAssumption.from_values("trend", 0.0),
        dates=ProjectionDates(entry_date="entry", exposure_timing="daily_prorated"),
    )
    result = projection.project().frame
    assert result["projected_expense"].item() == pytest.approx(10.0 * 16 / 31)
