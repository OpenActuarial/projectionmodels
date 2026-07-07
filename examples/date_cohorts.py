"""Lifecycle dates, new-business cohorts, renewals, and partial-period exposure."""

from __future__ import annotations

import pandas as pd

import projectionmodels as pm


def run_example() -> dict[str, object]:
    records = pm.ProjectionData(
        pd.DataFrame(
            {
                "group_id": ["A", "B", "C"],
                "effective_date": pd.to_datetime(
                    ["2025-01-01", "2027-02-15", "2027-08-01"]
                ),
                "termination_date": pd.to_datetime([None, None, "2028-03-15"]),
                "renewal_date": pd.to_datetime(
                    ["2027-03-01", "2027-02-15", "2027-08-01"]
                ),
                "monthly_members": [1_000.0, 300.0, 150.0],
            }
        ),
        projection_keys=["group_id"],
        dates=pm.ProjectionDates(
            entry_date="effective_date",
            exit_date="termination_date",
            renewal_date="renewal_date",
            exposure_timing="daily_prorated",
        ),
    ).add_date_cohort(
        pm.DateCohort(
            name="business_origin",
            date_col="effective_date",
            split_date="2027-01-01",
            before_label="existing",
            on_or_after_label="new_business",
        )
    )

    model = pm.ProjectionModel(
        calculations=[
            pm.Calculation(
                "member_months",
                formula=lambda x: x["monthly_members"] * x["active_fraction"],
                grain=["group_id"],
                reporting_role="exposure",
            )
        ]
    )

    results = model.project(records, pm.ProjectionHorizon("2027-01-01", periods=18))
    summary = results.summarize(
        by=["business_origin", "calendar_year"],
        measures=["member_months"],
    )
    renewal_summary = results.summarize(
        by=["is_renewal_period", "calendar_year"],
        measures=["member_months"],
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
        ]
    ]
    print(detail.head(12).to_string(index=False))
    print("\nSummary")
    print(output["summary"].to_string(index=False))
