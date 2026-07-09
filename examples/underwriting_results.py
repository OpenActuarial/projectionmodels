"""Compose claim, premium, and expense workflows into an underwriting exhibit."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def _exposure() -> pd.DataFrame:
    periods = pd.period_range("2027-01", periods=24, freq="M").astype(str)
    return pd.DataFrame(
        [
            {
                "group_id": group_id,
                "product_id": "PPO",
                "projection_period": period,
                "member_months": members,
            }
            for group_id, members in (("A", 1_000.0), ("B", 300.0))
            for period in periods
        ]
    )


def _claim_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, effective_date, scale in (
        ("A", "2025-01-01", 1.0),
        ("B", "2027-04-15", 0.95),
    ):
        for claim_type, rate in (("inpatient", 100.0), ("outpatient", 60.0)):
            for month in pd.date_range("2026-01-01", periods=12, freq="MS"):
                rows.append(
                    {
                        "group_id": group_id,
                        "product_id": "PPO",
                        "claim_type": claim_type,
                        "incurred_month": month,
                        "reported_claims": rate * scale * 1_000.0,
                        "member_months": 1_000.0,
                        "effective_date": pd.Timestamp(effective_date),
                        "business_origin": (
                            "existing" if pd.Timestamp(effective_date) < pd.Timestamp("2027-01-01") else "new_business"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_example() -> dict[str, object]:
    horizon = pm.ProjectionHorizon("2027-01-01", periods=24)
    exposure = _exposure()
    dates = pm.ProjectionDates(
        entry_date="effective_date",
        exposure_timing="daily_prorated",
    )

    premium_data = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "product_id": ["PPO", "PPO"],
            "effective_date": pd.to_datetime(["2025-01-01", "2027-04-15"]),
            "business_origin": ["existing", "new_business"],
            "current_premium_rate": [540.0, 560.0],
        }
    )
    premium_results = pm.PremiumProjection(
        premium_data=premium_data,
        projection_keys=["group_id", "product_id"],
        exposure=exposure,
        exposure_col="member_months",
        horizon=horizon,
        dates=dates,
    ).project()
    premium = premium_results.summarize(
        by=[
            "projection_period",
            "period_start",
            "calendar_year",
            "group_id",
            "product_id",
            "business_origin",
        ],
        measures=["member_months", "premium"],
    )

    experience = pm.ClaimExperience(
        _claim_history(),
        projection_keys=["group_id", "product_id"],
        claim_type_col="claim_type",
        date_col="incurred_month",
        claims_col="reported_claims",
        exposure_col="member_months",
    )
    claim_projection = pm.ClaimProjection.from_experience(
        experience,
        exposure=exposure,
        exposure_col="member_months",
        horizon=horizon,
        trend=pm.TrendAssumption.from_values(
            "claim_trend",
            pd.DataFrame(
                {
                    "claim_type": ["inpatient", "outpatient"],
                    "claim_trend": [0.07, 0.06],
                }
            ),
            lookup=["claim_type"],
        ),
        extra_record_cols=["effective_date", "business_origin"],
        dates=dates,
    )
    adverse = pm.Scenario(
        "adverse",
        [pm.Adjustment(target="claim_trend", method="add", value=0.02)],
    )
    claim_results = claim_projection.project(
        scenarios=[pm.Scenario("baseline"), adverse]
    )
    claims = claim_results.summarize(
        by=[
            "scenario",
            "projection_period",
            "calendar_year",
            "group_id",
            "product_id",
            "business_origin",
        ],
        measures=["projected_claims"],
    )

    expenses = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "product_id": "PPO",
                "expense_type": expense_type,
                "base_value": value,
                "basis": basis,
                "base_date": pd.Timestamp("2027-01-01"),
            }
            for group_id in ("A", "B")
            for expense_type, value, basis in (
                ("administration", 35.0, "per_exposure"),
                ("commission", 0.025, "percent_premium"),
                ("claim_admin", 0.01, "percent_claims"),
            )
        ]
    )

    frames: list[pd.DataFrame] = []
    for scenario_name in ("baseline", "adverse"):
        scenario_claims = claims.loc[
            claims["scenario"] == scenario_name,
            ["group_id", "product_id", "projection_period", "projected_claims"],
        ]
        expense_results = pm.ExpenseProjection(
            expenses=expenses,
            projection_keys=["group_id", "product_id"],
            expense_type_col="expense_type",
            base_value_col="base_value",
            basis_col="basis",
            base_date_col="base_date",
            horizon=horizon,
            trend=pm.TrendAssumption.from_values("expense_trend", 0.03),
            exposure=exposure,
            exposure_col="member_months",
            premium=premium[
                ["group_id", "product_id", "projection_period", "premium"]
            ],
            claims=scenario_claims,
        ).project(scenarios=pm.Scenario(scenario_name))
        expense_summary = expense_results.summarize(
            by=["scenario", "projection_period", "group_id", "product_id"],
            measures=["projected_expense"],
        )

        scenario_frame = claims.loc[claims["scenario"] == scenario_name].merge(
            premium,
            on=[
                "projection_period",
                "calendar_year",
                "group_id",
                "product_id",
                "business_origin",
            ],
            how="left",
            validate="one_to_one",
        ).merge(
            expense_summary,
            on=["scenario", "projection_period", "group_id", "product_id"],
            how="left",
            validate="one_to_one",
        )
        scenario_frame["underwriting_margin"] = (
            scenario_frame["premium"]
            - scenario_frame["projected_claims"]
            - scenario_frame["projected_expense"]
        )
        scenario_frame["loss_ratio"] = (
            scenario_frame["projected_claims"] / scenario_frame["premium"]
        )
        frames.append(scenario_frame)

    detail = pd.concat(frames, ignore_index=True)
    exhibit_input = (
        detail.groupby(
            ["scenario", "business_origin", "calendar_year"],
            dropna=False,
            sort=False,
        )
        .agg(
            member_months=("member_months", "sum"),
            premium=("premium", "sum"),
            claims=("projected_claims", "sum"),
            expenses=("projected_expense", "sum"),
            underwriting_margin=("underwriting_margin", "sum"),
        )
        .reset_index()
    )
    exhibit_input["loss_ratio"] = exhibit_input["claims"] / exhibit_input["premium"]

    baseline = exhibit_input.loc[exhibit_input["scenario"] == "baseline"].drop(
        columns="scenario"
    )
    comparison = exhibit_input.loc[exhibit_input["scenario"] == "adverse"].drop(
        columns="scenario"
    )
    comparison = baseline.merge(
        comparison,
        on=["business_origin", "calendar_year"],
        suffixes=("_baseline", "_comparison"),
        validate="one_to_one",
    )
    for measure in ("claims", "underwriting_margin", "loss_ratio"):
        comparison[f"{measure}_change"] = (
            comparison[f"{measure}_comparison"]
            - comparison[f"{measure}_baseline"]
        )

    return {
        "premium_results": premium_results,
        "claim_results": claim_results,
        "detail": detail,
        "exhibit_input": exhibit_input,
        "comparison": comparison,
    }


if __name__ == "__main__":
    print(run_example()["exhibit_input"].to_string(index=False))
