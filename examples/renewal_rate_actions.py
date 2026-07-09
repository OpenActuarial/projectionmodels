"""Apply group-specific rate actions at each group's renewal period."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    premium_data = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "renewal_date": pd.to_datetime(["2027-03-01", "2027-07-01"]),
            "current_premium_rate": [100.0, 100.0],
            "rate_action": [0.10, 0.20],
        }
    )
    periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
    exposure = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "projection_period": period,
                "member_months": 1_000.0,
            }
            for group_id in ("A", "B")
            for period in periods
        ]
    )
    projection = pm.PremiumProjection(
        premium_data=premium_data,
        projection_keys=["group_id"],
        exposure=exposure,
        exposure_col="member_months",
        horizon=pm.ProjectionHorizon("2027-01-01", periods=12),
        recurring_rate_action_col="rate_action",
    )
    results = projection.project()
    detail = results.detail().loc[
        :, ["group_id", "projection_period", "is_renewal_period", "projected_premium_rate", "premium"]
    ]
    annual = results.summarize(
        by=["group_id", "calendar_year"],
        measures=["member_months", "premium"],
    )
    return {"results": results, "detail": detail, "annual": annual}


if __name__ == "__main__":
    output = run_example()
    print(output["detail"].to_string(index=False))
