"""Regression: Example 10's projection-side numbers stay true.

Pins the projectionmodels segments of docs page ``worked-example-contract.md``
(Example 10): the contract-year loss costs, the premium-independent dollar
expenses (a contractually flat fee, a trended fee, and a percent-of-claims
surcharge), and — given the page's solved rates, which are produced and
pinned at full precision in ratingmodels' suite — the contractual ratio and
margin of both pins.
"""

import numpy as np
import pandas as pd
import pytest
from actuarialpy import Experience

import projectionmodels as pm

KEYS = ["group_id"]
RENEWAL = pd.Timestamp("2027-07-01")
HORIZON = pm.ProjectionHorizon("2026-07-01", periods=24)
EXPOSURE = pd.DataFrame({
    "group_id": "A",
    "projection_period": pd.period_range("2026-07", periods=24, freq="M").astype(str),
    "member_months": np.linspace(1_000.0, 1_046.0, 24).round(0),
})


def _contract_year(frame):
    return np.where(frame["projection_period"] < "2027-07", "CY1", "CY2")


def _claims():
    months = pd.date_range("2025-07-01", periods=12, freq="MS")
    rows = [{"group_id": "A", "claim_type": ct, "incurred_month": m,
             "reported_claims": base * (1 + tr) ** (i / 12) * 1_000.0,
             "member_months": 1_000.0}
            for ct, base, tr in (("inpatient", 260.0, 0.075),
                                 ("outpatient", 190.0, 0.060))
            for i, m in enumerate(months)]
    experience = Experience(
        pd.DataFrame(rows), expense="reported_claims",
        exposure="member_months", date="incurred_month",
        dimensions=[*KEYS, "claim_type"])
    trend = pm.TrendAssumption.from_values(
        "claim_trend",
        pd.DataFrame({"claim_type": ["inpatient", "outpatient"],
                      "annual_trend": [0.075, 0.060]}),
        lookup=["claim_type"], rate_col="annual_trend")
    claims = pm.project(
        experience, exposure=EXPOSURE, exposure_col="member_months",
        horizon=HORIZON, trend=trend,
    ).project().summarize(by=["group_id", "projection_period"],
                          measures=["member_months", "projected_claims"])
    claims["cy"] = _contract_year(claims)
    return claims


def _expense_trend(types_and_rates):
    types, rates = zip(*types_and_rates, strict=True)
    return pm.TrendAssumption.from_values(
        "expense_trend",
        pd.DataFrame({"expense_type": list(types), "expense_trend": list(rates)}),
        lookup="expense_type")


def _dollar_expenses(claims):
    return pm.ExpenseProjection(
        pd.DataFrame({"group_id": ["A"] * 3,
                      "expense_type": ["admin_fee", "care_mgmt", "hcra_surcharge"],
                      "base_value": [25.0, 8.0, 0.08],
                      "basis": ["per_exposure", "per_exposure", "percent_claims"],
                      "base_date": pd.Timestamp("2026-07-01")}),
        projection_keys=KEYS, expense_type_col="expense_type",
        base_value_col="base_value", basis_col="basis", base_date_col="base_date",
        horizon=HORIZON,
        trend=_expense_trend([("admin_fee", 0.0), ("care_mgmt", 0.03),
                              ("hcra_surcharge", 0.0)]),
        exposure=EXPOSURE, exposure_col="member_months",
        claims=claims[["group_id", "projection_period", "projected_claims"]],
    ).project().summarize(by=["group_id", "projection_period"],
                          measures=["projected_expense"])


def test_contract_year_loss_costs():
    claims = _claims()
    annual = claims.groupby("cy")[["member_months", "projected_claims"]].sum()
    loss_cost = annual["projected_claims"] / annual["member_months"]
    assert annual["member_months"].tolist() == [12_132.0, 12_420.0]
    assert round(float(loss_cost["CY1"]), 4) == 497.3488
    assert round(float(loss_cost["CY2"]), 4) == 531.5452
    assert round(float(loss_cost["CY2"] / loss_cost["CY1"] - 1), 6) == 0.068757


def test_dollar_expense_pmpm_is_premium_independent():
    claims = _claims()
    dollar = _dollar_expenses(claims)          # no premium table supplied
    dollar["cy"] = _contract_year(dollar)
    mm = claims.groupby("cy")["member_months"].sum()
    f = dollar.groupby("cy")["projected_expense"].sum() / mm
    assert round(float(f["CY1"]), 4) == 72.9073
    assert round(float(f["CY2"]), 4) == 75.8866


@pytest.mark.parametrize(
    "pin, rate1, action, margins",
    [
        ("gross", 585.116189, 0.06875741, (-32_675.87, -10_494.58)),
        ("net", 678.374736, 0.06566698, (1_064_794.44, 1_165_021.95)),
    ],
)
def test_pins_book_to_contract(pin, rate1, action, margins):
    """Given the page's solved rates, the projection books the contract."""
    claims = _claims()
    premium = pm.PremiumProjection(
        premium_data=pd.DataFrame({"group_id": ["A"], "current_premium_rate": [rate1],
                                   "renewal_date": [RENEWAL]}),
        projection_keys=KEYS, exposure=EXPOSURE, exposure_col="member_months",
        horizon=HORIZON,
        rate_actions=pm.RenewalRateActions(
            frame=pd.DataFrame({"group_id": ["A"], "effective_date": [RENEWAL],
                                "rate_action": [action]}),
            projection_keys=KEYS),
    ).project().summarize(by=["group_id", "projection_period"], measures=["premium"])
    premium["cy"] = _contract_year(premium)

    booking = pm.ExpenseProjection(
        pd.DataFrame({"group_id": ["A"] * 4,
                      "expense_type": ["admin_fee", "care_mgmt",
                                       "hcra_surcharge", "commission"],
                      "base_value": [25.0, 8.0, 0.08, 0.03],
                      "basis": ["per_exposure", "per_exposure",
                                "percent_claims", "percent_premium"],
                      "base_date": pd.Timestamp("2026-07-01")}),
        projection_keys=KEYS, expense_type_col="expense_type",
        base_value_col="base_value", basis_col="basis", base_date_col="base_date",
        horizon=HORIZON,
        trend=_expense_trend([("admin_fee", 0.0), ("care_mgmt", 0.03),
                              ("hcra_surcharge", 0.0), ("commission", 0.0)]),
        exposure=EXPOSURE, exposure_col="member_months",
        claims=claims[["group_id", "projection_period", "projected_claims"]],
        premium=premium[["group_id", "projection_period", "premium"]],
    ).project().summarize(by=["group_id", "projection_period"],
                          measures=["projected_expense"])
    booking["cy"] = _contract_year(booking)

    C = claims.groupby("cy")["projected_claims"].sum()
    P = premium.groupby("cy")["premium"].sum()
    E = booking.groupby("cy")["projected_expense"].sum()
    ratio = C / P if pin == "gross" else C / (P - E)
    margin = P * (1 - 0.85) - E if pin == "gross" else P - E - C
    assert np.allclose(ratio, 0.85, atol=1e-6)
    assert float(margin["CY1"]) == pytest.approx(margins[0], abs=25)
    assert float(margin["CY2"]) == pytest.approx(margins[1], abs=25)
