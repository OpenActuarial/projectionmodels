import pandas as pd
import pytest

from projectionmodels import Adjustment, Scenario


def test_filtered_date_effective_adjustment():
    frame = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "period_start": pd.to_datetime(["2027-01-01", "2027-07-01", "2027-07-01"]),
        }
    )
    values = pd.Series([100.0, 100.0, 100.0])
    scenario = Scenario(
        "test",
        [
            Adjustment(
                target="premium",
                method="multiply",
                value=1.1,
                filters={"group": "A"},
                effective_from="2027-07-01",
            )
        ],
    )
    adjusted, audits = scenario.apply("premium", frame, values)
    assert adjusted.tolist() == pytest.approx([100.0, 110.0, 100.0])
    assert len(audits) == 1
    assert audits[0]["after"].item() == pytest.approx(110.0)
