from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import (
    DateCohort,
    ProjectionData,
    ProjectionDataset,
    ProjectionDates,
    ProjectionHorizon,
    ProjectionTable,
    ValidationError,
)


def test_date_cohort_frequency_and_missing_dates():
    frame = pd.DataFrame({"date": ["2027-01-15", "2027-04-01", None]})
    result = DateCohort("quarter", "date", frequency="quarterly").apply(frame)
    assert result["quarter"].astype("string").tolist() == ["2027Q1", "2027Q2", pd.NA]


def test_date_cohort_custom_breaks_and_labels():
    frame = pd.DataFrame({"date": pd.to_datetime(["2027-02-01", "2027-08-01"])})
    result = DateCohort(
        "half",
        "date",
        breaks=("2027-01-01", "2027-07-01", "2028-01-01"),
        labels=("H1", "H2"),
    ).apply(frame)
    assert result["half"].astype(str).tolist() == ["H1", "H2"]


def test_date_cohort_validates_mode_breaks_labels_and_columns():
    with pytest.raises(ValidationError, match="exactly one"):
        DateCohort("x", "date")
    with pytest.raises(ValidationError, match="exactly one"):
        DateCohort("x", "date", split_date="2027-01-01", frequency="year")
    with pytest.raises(ValidationError, match="at least two"):
        DateCohort("x", "date", breaks=("2027-01-01",))
    with pytest.raises(ValidationError, match=r"len\(breaks\) - 1"):
        DateCohort(
            "x",
            "date",
            breaks=("2027-01-01", "2027-07-01", "2028-01-01"),
            labels=("only_one",),
        )
    with pytest.raises(ValidationError, match="missing columns"):
        DateCohort("x", "date", split_date="2027-01-01").apply(
            pd.DataFrame({"other": [1]})
        )


def test_projection_dates_validate_timing_and_columns():
    dates = ProjectionDates(entry_date="entry", exposure_timing="invalid")
    with pytest.raises(ValidationError, match="exposure_timing"):
        dates.validate(pd.DataFrame({"entry": ["2027-01-01"]}))
    with pytest.raises(ValidationError, match="missing columns"):
        ProjectionDates(entry_date="entry").validate(pd.DataFrame({"other": [1]}))


def test_projection_data_validates_keys_components_and_record_weight():
    with pytest.raises(ValidationError, match="must not be empty"):
        ProjectionData(pd.DataFrame({"id": [1]}), projection_keys=[])
    with pytest.raises(ValidationError, match="must be distinct"):
        ProjectionData(
            pd.DataFrame({"id": [1]}), projection_keys=["id"], component_keys=["id"]
        )
    with pytest.raises(ValidationError, match="nonnegative"):
        ProjectionData(
            pd.DataFrame({"id": [1], "weight": [-1.0]}),
            projection_keys=["id"],
            record_weight="weight",
        )


def test_projection_data_attributes_exclude_record_keys():
    data = ProjectionData(
        pd.DataFrame({"group": ["A"], "claim_type": ["ip"], "region": ["west"]}),
        projection_keys=["group"],
        component_keys=["claim_type"],
    )
    assert data.record_keys == ("group", "claim_type")
    assert data.attributes == ("region",)


def test_whole_period_entry_and_exit_flags():
    data = ProjectionData(
        pd.DataFrame(
            {
                "id": ["A"],
                "entry": pd.to_datetime(["2027-02-15"]),
                "exit": pd.to_datetime(["2027-03-10"]),
            }
        ),
        projection_keys=["id"],
        dates=ProjectionDates(entry_date="entry", exit_date="exit"),
    )
    expanded = data.expand(ProjectionHorizon("2027-01-01", periods=4))
    assert expanded["active_fraction"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert expanded["duration_month"].tolist() == [-1, 0, 1, 2]


def test_daily_prorated_exit_fraction():
    data = ProjectionData(
        pd.DataFrame(
            {"id": ["A"], "entry": ["2027-01-01"], "exit": ["2027-01-15"]}
        ),
        projection_keys=["id"],
        dates=ProjectionDates(
            entry_date="entry", exit_date="exit", exposure_timing="daily_prorated"
        ),
    )
    expanded = data.expand(ProjectionHorizon("2027-01-01", periods=2))
    assert expanded.loc[0, "active_fraction"] == pytest.approx(15 / 31)
    assert expanded.loc[1, "active_fraction"] == 0


def test_renewal_anniversary_handles_leap_day():
    data = ProjectionData(
        pd.DataFrame({"id": ["A"], "renewal": pd.to_datetime(["2024-02-29"])}),
        projection_keys=["id"],
        dates=ProjectionDates(renewal_date="renewal"),
    )
    expanded = data.expand(ProjectionHorizon("2027-01-01", periods=3))
    assert expanded.loc[expanded["projection_period"] == "2027-02", "is_renewal_period"].item()


def test_projection_table_validates_keys_and_duplicates():
    with pytest.raises(ValidationError, match="must not be empty"):
        ProjectionTable("x", pd.DataFrame({"id": [1]}), keys=[])
    with pytest.raises(ValidationError, match="duplicate keys"):
        ProjectionTable("x", pd.DataFrame({"id": [1, 1]}), keys=["id"])
    table = ProjectionTable(
        "x", pd.DataFrame({"id": [1, 1], "value": [2, 3]}), keys=["id"], allow_duplicates=True
    )
    assert table.allow_duplicates


def test_projection_dataset_table_management_and_merge_validation():
    records = ProjectionData(pd.DataFrame({"id": [1], "base": [10]}), projection_keys=["id"])
    dataset = ProjectionDataset(records).add_table(
        "rates", pd.DataFrame({"id": [1], "rate": [0.1]}), keys=["id"]
    )
    merged = dataset.merge_tables(records.expand(ProjectionHorizon("2027-01-01", periods=1)))
    assert merged["rate"].item() == 0.1
    assert dataset.get_table("rates").name == "rates"
    with pytest.raises(ValidationError, match="already exists"):
        dataset.add_table("rates", pd.DataFrame({"id": [1]}), keys=["id"])
    with pytest.raises(ValidationError, match="not present"):
        dataset.get_table("missing")


def test_projection_dataset_rejects_missing_join_keys_and_overwrites():
    records = ProjectionData(pd.DataFrame({"id": [1], "value": [10]}), projection_keys=["id"])
    missing = ProjectionDataset(records).add_table(
        "missing", pd.DataFrame({"other": [1], "rate": [0.1]}), keys=["other"]
    )
    with pytest.raises(ValidationError, match="lack keys"):
        missing.merge_tables(records.expand(ProjectionHorizon("2027-01-01", periods=1)))

    overlap = ProjectionDataset(records).add_table(
        "overlap", pd.DataFrame({"id": [1], "value": [20]}), keys=["id"]
    )
    with pytest.raises(ValidationError, match="overwrite columns"):
        overlap.merge_tables(records.expand(ProjectionHorizon("2027-01-01", periods=1)))
