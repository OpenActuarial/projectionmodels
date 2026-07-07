import pandas as pd
import pytest

from projectionmodels import (
    Adjustment,
    Assumption,
    AssumptionSet,
    Calculation,
    CashFlow,
    ProjectionData,
    ProjectionHorizon,
    ProjectionModel,
    RollForward,
    Scenario,
)


def test_general_roll_forward_and_scenario():
    records = ProjectionData(
        pd.DataFrame(
            {
                "group_id": ["A", "B"],
                "current_members": [100.0, 50.0],
                "current_rate": [10.0, 20.0],
            }
        ),
        projection_keys=["group_id"],
    )
    model = ProjectionModel(
        assumptions=AssumptionSet(
            Assumption("retention", 0.9),
            Assumption("trend", 0.0),
        ),
        roll_forwards=[
            RollForward(
                "members",
                initial="current_members",
                formula=lambda x: x.prior("members") * x["retention"],
                grain=["group_id"],
            ),
            RollForward(
                "rate",
                initial="current_rate",
                formula=lambda x: x.prior("rate") * (1 + x["trend"]),
                grain=["group_id"],
            ),
        ],
        calculations=[
            CashFlow(
                "amount",
                formula=lambda x: x["members"] * x["rate"],
                grain=["group_id"],
            )
        ],
    )
    adverse = Scenario(
        "adverse",
        [
            Adjustment(
                target="rate",
                method="multiply",
                value=1.1,
                filters={"group_id": "A"},
            )
        ],
    )
    results = model.project(
        records,
        ProjectionHorizon("2027-01-01", periods=2),
        scenarios=[Scenario("baseline"), adverse],
    )
    baseline_a = results.frame.query(
        "scenario == 'baseline' and group_id == 'A'"
    )
    adverse_a = results.frame.query("scenario == 'adverse' and group_id == 'A'")
    assert baseline_a["members"].tolist() == pytest.approx([90, 81])
    assert adverse_a["rate"].tolist() == pytest.approx([11, 12.1])


def test_supporting_table_broadcasts_to_components_without_changing_rows():
    records = ProjectionData(
        pd.DataFrame(
            {
                "group": ["A", "A"],
                "claim_type": ["ip", "op"],
                "rate": [10.0, 5.0],
            }
        ),
        projection_keys=["group"],
        component_keys=["claim_type"],
    )
    from projectionmodels import ProjectionDataset

    dataset = ProjectionDataset(records).add_table(
        "membership",
        pd.DataFrame(
            {"group": ["A"], "projection_period": ["2027-01"], "members": [100.0]}
        ),
        keys=["group", "projection_period"],
    )
    model = ProjectionModel(
        calculations=[
            Calculation(
                "members",
                formula=lambda x: x["members"],
                grain=["group"],
            ),
            Calculation(
                "claims",
                formula=lambda x: x["rate"] * x["members"],
                grain=["group", "claim_type"],
            ),
        ]
    )
    results = model.project(dataset, ProjectionHorizon("2027-01-01", periods=1))
    assert len(results.frame) == 2
    summary = results.summarize(by=["projection_period", "group"])
    assert summary["members"].item() == 100.0
    assert summary["claims"].item() == 1500.0
