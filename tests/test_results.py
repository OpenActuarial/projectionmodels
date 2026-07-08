import pandas as pd
import pytest

from projectionmodels import ProjectionResults
from projectionmodels.advanced import Calculation, Metric


def test_grain_aware_aggregation_and_ratio_recalculation():
    frame = pd.DataFrame(
        {
            "scenario": ["base", "base"],
            "projection_period": ["2027-01", "2027-01"],
            "group": ["A", "A"],
            "claim_type": ["ip", "op"],
            "claims": [100.0, 50.0],
            "member_months": [10.0, 10.0],
            "claim_pmpm": [10.0, 5.0],
        }
    )
    results = ProjectionResults(
        frame,
        measures={
            "claims": Calculation("claims", grain=["group", "claim_type"]),
            "member_months": Calculation("member_months", grain=["group"]),
            "claim_pmpm": Metric(
                "claim_pmpm",
                aggregation="recalculate",
                numerator="claims",
                denominator="member_months",
                grain=["group", "claim_type"],
            ),
        },
        projection_keys=("group",),
        component_keys=("claim_type",),
    )
    total = results.summarize(by=["scenario", "projection_period", "group"])
    assert total["claims"].item() == 150.0
    assert total["member_months"].item() == 10.0
    assert total["claim_pmpm"].item() == pytest.approx(15.0)

    by_type = results.summarize(
        by=["scenario", "projection_period", "group", "claim_type"]
    )
    assert by_type["member_months"].tolist() == [10.0, 10.0]
