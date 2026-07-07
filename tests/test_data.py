import pandas as pd
import pytest

from projectionmodels import (
    DateCohort,
    ProjectionData,
    ProjectionDates,
    ProjectionHorizon,
    ValidationError,
)


def test_projection_keys_must_be_unique():
    frame = pd.DataFrame({"group_id": ["A", "A"], "value": [1, 2]})
    with pytest.raises(ValidationError):
        ProjectionData(frame, projection_keys=["group_id"])


def test_component_keys_allow_repeated_entities():
    frame = pd.DataFrame(
        {
            "group_id": ["A", "A"],
            "claim_type": ["inpatient", "outpatient"],
        }
    )
    data = ProjectionData(
        frame,
        projection_keys=["group_id"],
        component_keys=["claim_type"],
    )
    assert data.record_keys == ("group_id", "claim_type")


def test_date_cohort_and_lifecycle_fields():
    frame = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "effective_date": ["2026-01-01", "2027-02-15"],
            "termination_date": [None, None],
            "renewal_date": ["2027-03-01", "2027-02-15"],
        }
    )
    data = ProjectionData(
        frame,
        projection_keys=["group_id"],
        dates=ProjectionDates(
            entry_date="effective_date",
            exit_date="termination_date",
            renewal_date="renewal_date",
            exposure_timing="daily_prorated",
        ),
    ).add_date_cohort(
        DateCohort(
            "business_origin",
            "effective_date",
            split_date="2027-01-01",
            before_label="existing",
            on_or_after_label="new",
        )
    )
    expanded = data.expand(ProjectionHorizon("2027-01-01", periods=3))
    b_jan = expanded.query("group_id == 'B' and projection_period == '2027-01'").iloc[0]
    b_feb = expanded.query("group_id == 'B' and projection_period == '2027-02'").iloc[0]
    assert b_jan["active_fraction"] == 0
    assert 0 < b_feb["active_fraction"] < 1
    assert set(expanded["business_origin"]) == {"existing", "new"}
    assert expanded.query("group_id == 'A' and projection_period == '2027-03'")[
        "is_renewal_period"
    ].item()
