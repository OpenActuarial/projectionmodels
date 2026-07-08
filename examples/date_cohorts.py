"""Date cohorts, new business, renewals, and partial-period membership."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    premium_data = pd.DataFrame(
        {
            "group_id": ["A", "B", "C"],
            "effective_date": pd.to_datetime(
                ["2025-01-01", "2027-02-15", "2027-08-01"]
            ),
            "termination_date": pd.to_datetime([None, None, "2028-03-15"]),
            "renewal_date": pd.to_datetime(
                ["2027-03-01", "2027-02-15", "2027-08-01"]
            ),
            "current_premium_pmpm": [500.0, 525.0, 550.0],
        }
    )
    premium_data = pm.DateCohort(
        name="business_origin",
        date_col="effective_date",
        split_date="2027-01-01",
        before_label="existing",
        on_or_after_label="new_business",
    ).apply(premium_data)

    periods = pd.period_range("2027-01", periods=18, freq="M").astype(str)
    monthly_members = {"A": 1_000.0, "B": 300.0, "C": 150.0}
    membership = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "projection_period": period,
                "member_months": members,
            }
            for group_id, members in monthly_members.items()
            for period in periods
        ]
    )

    projection = pm.PremiumProjection(
        premium_data=premium_data,
        projection_keys=["group_id"],
        membership=membership,
        horizon=pm.ProjectionHorizon("2027-01-01", periods=18),
        dates=pm.ProjectionDates(
            entry_date="effective_date",
            exit_date="termination_date",
            renewal_date="renewal_date",
            exposure_timing="daily_prorated",
        ),
    )
    results = projection.project()
    summary = results.summarize(
        by=["business_origin", "calendar_year"],
        measures=["member_months", "premium"],
    )
    renewal_summary = results.summarize(
        by=["is_renewal_period", "calendar_year"],
        measures=["member_months", "premium"],
    )
    return {
        "results": results,
        "summary": summary,
        "renewal_summary": renewal_summary,
    }


if __name__ == "__main__":
    output = run_example()
    detail = output["results"].detail()[
        [
            "group_id",
            "projection_period",
            "business_origin",
            "active_fraction",
            "is_renewal_period",
            "member_months",
            "premium",
        ]
    ]
    print(detail.head(12).to_string(index=False))
    print("\nSummary")
    print(output["summary"].to_string(index=False))
