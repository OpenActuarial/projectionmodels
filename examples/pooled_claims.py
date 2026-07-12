"""Large-claim pooling composed ahead of the claim workflow.

Recipe: pool claim-level detail at the pooling point with
``actuarialpy.pool_losses`` *before* building the ``Experience``, so a
handful of catastrophic claims cannot distort the group's experience rate.
Aggregate the retained claims into incurred-month experience, run the
standard ``ClaimProjection`` on the retained basis, and add the pooling
charge back as a flat PMPM ``rate_load``.

The pooling charge — the expected PMPM cost of the ceded excess layer — is a
book-level quantity: one group's few large claims cannot price its own excess
layer. Estimate it upstream (for example from limited expected values on the
book's severity model) and supply the selected value here.

Claims in this example are treated as at ultimate. When developing retained
claims, build the completion factors on the same pooled basis: large claims
develop differently, so unpooled factors do not apply to retained amounts.

Run with: PYTHONPATH=src python examples/pooled_claims.py
"""

from __future__ import annotations

import actuarialpy as ap
import numpy as np
import pandas as pd
from actuarialpy import Experience

import projectionmodels as pm

POOLING_POINT = 100_000.0
POOLING_CHARGE_PMPM = 14.50  # selected; supplied by book-level excess analysis
MEMBER_MONTHS_PER_PERIOD = 1_000.0


def _claim_detail(rng: np.random.Generator) -> pd.DataFrame:
    """Two years of claim-level detail for one group with a heavy right tail."""

    months = pd.date_range("2025-01-01", periods=24, freq="MS")
    rows: list[dict[str, object]] = []
    for month in months:
        count = int(rng.poisson(115))
        amounts = rng.lognormal(mean=6.6, sigma=1.6, size=count)
        rows.extend(
            {
                "group_id": "1102052",
                "claim_type": "med",
                "incurred_month": month,
                "paid_amount": float(amount),
            }
            for amount in amounts
        )
    return pd.DataFrame(rows)


def _monthly_experience(detail: pd.DataFrame, claims_col: str) -> pd.DataFrame:
    keys = ["group_id", "claim_type", "incurred_month"]
    frame = detail.groupby(keys, as_index=False).agg(
        reported_claims=(claims_col, "sum")
    )
    frame["member_months"] = MEMBER_MONTHS_PER_PERIOD
    return frame


def _project(base_rates: pd.DataFrame, exposure: pd.DataFrame,
             horizon: pm.ProjectionHorizon, *, loads) -> pm.ProjectionResults:
    projection = pm.ClaimProjection(
        base_rates=base_rates,
        projection_keys=["group_id"],
        claim_type_col="claim_type",
        exposure=exposure,
        exposure_col="member_months",
        horizon=horizon,
        trend=pm.TrendAssumption.from_values("claim_trend", 0.07),
        credibility=pm.CredibilityAssumption.from_weights("claim_credibility", 0.65),
        rate_loads=loads,
    )
    return projection.project()


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(20260708)
    detail = _claim_detail(rng)

    # 1. Pool claim-level detail at the pooling point.
    pooled = ap.pool_losses(detail, "paid_amount", POOLING_POINT)
    ceded_total = float(pooled["excess_loss"].sum())

    # 2. Aggregate the retained basis into incurred-month experience.
    retained_experience = _monthly_experience(
        pooled.assign(paid_amount=pooled["pooled_loss"]), "paid_amount"
    )
    unpooled_experience = _monthly_experience(detail, "paid_amount")
    experience_member_months = float(retained_experience["member_months"].sum())
    retained_pmpm = float(
        retained_experience["reported_claims"].sum() / experience_member_months
    )
    unpooled_pmpm = float(
        unpooled_experience["reported_claims"].sum() / experience_member_months
    )

    # 3. Base rates from the retained experience; complement quoted at
    #    prospective level and used as stated.
    experience = Experience(
        retained_experience,
        expense="reported_claims",
        exposure="member_months",
        date="incurred_month",
        dimensions=["group_id", "claim_type"],
        valuation_date="2026-12-31",
    )
    base_rates = pm.base_rates(
        experience,
        complement=pm.Assumption("manual_claim_rate", 320.0),
    )

    # 4. Project CY2027 with the pooling charge as a flat rate load, and once
    #    without the charge so the report can show the two pieces.
    horizon = pm.ProjectionHorizon("2027-01-01", periods=12)
    exposure = pd.DataFrame(
        {
            "group_id": "1102052",
            "projection_period": pd.period_range(
                "2027-01", periods=12, freq="M"
            ).astype(str),
            "member_months": MEMBER_MONTHS_PER_PERIOD,
        }
    )
    charge = pm.Assumption("pooling_charge", POOLING_CHARGE_PMPM)
    with_charge = _project(base_rates, exposure, horizon, loads=(charge,))
    without_charge = _project(base_rates, exposure, horizon, loads=())

    summary = with_charge.summarize(by=["projection_period", "group_id"])
    annual = with_charge.summarize(by=["group_id"])
    annual_excluding = without_charge.summarize(by=["group_id"])
    net_pmpm = float(annual["claims_per_exposure"].iloc[0])
    net_pmpm_excluding_charge = float(annual_excluding["claims_per_exposure"].iloc[0])

    return {
        "pooling_point": POOLING_POINT,
        "ceded_total": ceded_total,
        "unpooled_pmpm": unpooled_pmpm,
        "retained_pmpm": retained_pmpm,
        "experience_member_months": experience_member_months,
        "pooling_charge": POOLING_CHARGE_PMPM,
        "net_pmpm_excluding_charge": net_pmpm_excluding_charge,
        "net_pmpm": net_pmpm,
        "summary": summary,
    }


if __name__ == "__main__" or __name__.startswith("projectionmodels_example"):
    output = run_example()
    if __name__ == "__main__":
        print("Large-claim pooling ahead of the claim workflow")
        print(f"  pooling point                 ${output['pooling_point']:,.0f}")
        print(f"  ceded excess                  ${output['ceded_total']:,.0f}")
        print(f"  unpooled experience PMPM       {output['unpooled_pmpm']:,.2f}")
        print(f"  retained experience PMPM       {output['retained_pmpm']:,.2f}")
        print(f"  pooling charge PMPM           +{output['pooling_charge']:,.2f}")
        print()
        print("Prospective CY2027, annual")
        print(
            f"  net claim PMPM                 {output['net_pmpm']:,.2f}"
            f"  (= {output['net_pmpm_excluding_charge']:,.2f} blended retained"
            f" + {output['pooling_charge']:,.2f} charge)"
        )
