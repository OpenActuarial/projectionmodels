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
            "exposure": [10.0, 10.0],
            "claims_per_exposure": [10.0, 5.0],
        }
    )
    results = ProjectionResults(
        frame,
        measures={
            "claims": Calculation("claims", grain=["group", "claim_type"]),
            "exposure": Calculation("exposure", grain=["group"]),
            "claims_per_exposure": Metric(
                "claims_per_exposure",
                aggregation="recalculate",
                numerator="claims",
                denominator="exposure",
                grain=["group", "claim_type"],
            ),
        },
        projection_keys=("group",),
        component_keys=("claim_type",),
    )
    total = results.summarize(by=["scenario", "projection_period", "group"])
    assert total["claims"].item() == 150.0
    assert total["exposure"].item() == 10.0
    assert total["claims_per_exposure"].item() == pytest.approx(15.0)

    by_type = results.summarize(
        by=["scenario", "projection_period", "group", "claim_type"]
    )
    assert by_type["exposure"].tolist() == [10.0, 10.0]
