from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projectionmodels import ProjectionResults, ValidationError
from projectionmodels.advanced import Calculation, Metric


def make_results() -> ProjectionResults:
    frame = pd.DataFrame(
        {
            "scenario": ["base", "base", "adverse", "adverse"],
            "projection_period": ["2027-01"] * 4,
            "group": ["A", "B", "A", "B"],
            "amount": [100.0, 200.0, 120.0, 220.0],
            "exposure": [10.0, 20.0, 10.0, 20.0],
            "score_mean": [1.0, 3.0, 2.0, 4.0],
            "score_min": [1.0, 3.0, 2.0, 4.0],
            "score_max": [1.0, 3.0, 2.0, 4.0],
            "first": [5.0, 6.0, 7.0, 8.0],
        }
    )
    return ProjectionResults(
        frame,
        measures={
            "amount": Calculation("amount", aggregation="sum", reporting_role="loss"),
            "exposure": Calculation("exposure", aggregation="sum", reporting_role="exposure"),
            "score_mean": Calculation("score_mean", aggregation="mean"),
            "score_min": Calculation("score_min", aggregation="min"),
            "score_max": Calculation("score_max", aggregation="max"),
            "first": Calculation("first", aggregation="first"),
            "rate": Metric(
                "rate",
                aggregation="recalculate",
                numerator="amount",
                denominator="exposure",
            ),
        },
        projection_keys=("group",),
        assumption_audit_data=pd.DataFrame({"name": ["trend"]}),
        adjustment_audit_data=pd.DataFrame({"target": ["amount"]}),
    )


def test_result_detail_copy_and_measure_roles():
    results = make_results()
    copied = results.detail()
    copied.loc[0, "amount"] = -1
    assert results.frame.loc[0, "amount"] == 100.0
    assert results.detail(copy=False) is results.frame
    assert results.to_frame(copy=False) is results.frame
    assert results.measure_names(role="loss") == ["amount"]
    assert results.measure_names(role="missing") == []


def test_result_aggregation_methods():
    results = make_results()
    summary = results.summarize(
        by=["scenario"],
        measures=["amount", "score_mean", "score_min", "score_max", "first"],
    )
    base = summary.loc[summary["scenario"] == "base"].iloc[0]
    assert base["amount"] == 300.0
    assert base["score_mean"] == 2.0
    assert base["score_min"] == 1.0
    assert base["score_max"] == 3.0
    assert base["first"] == 5.0


def test_recalculated_metric_automatically_summarizes_dependencies():
    results = make_results()
    summary = results.summarize(by=["scenario"], measures=["rate"])
    assert summary.columns.tolist() == ["scenario", "rate"]
    assert summary.set_index("scenario").loc["base", "rate"] == pytest.approx(10.0)


def test_nested_recalculated_metrics_are_resolved():
    frame = pd.DataFrame({"group": ["A"], "a": [10.0], "b": [2.0], "c": [5.0]})
    results = ProjectionResults(
        frame,
        measures={
            "a": Calculation("a"),
            "b": Calculation("b"),
            "c": Calculation("c"),
            "first_ratio": Metric(
                "first_ratio", aggregation="recalculate", numerator="a", denominator="b"
            ),
            "second_ratio": Metric(
                "second_ratio",
                aggregation="recalculate",
                numerator="first_ratio",
                denominator="c",
            ),
        },
        projection_keys=("group",),
    )
    summary = results.summarize(by=["group"], measures=["second_ratio"])
    assert summary["second_ratio"].item() == pytest.approx(1.0)


def test_zero_denominator_produces_nan():
    frame = pd.DataFrame({"group": ["A"], "amount": [10.0], "exposure": [0.0]})
    results = ProjectionResults(
        frame,
        measures={
            "amount": Calculation("amount"),
            "exposure": Calculation("exposure"),
            "rate": Metric(
                "rate", aggregation="recalculate", numerator="amount", denominator="exposure"
            ),
        },
        projection_keys=("group",),
    )
    assert np.isnan(results.summarize(by=["group"], measures=["rate"])["rate"].item())


def test_result_summary_validation_errors():
    results = make_results()
    with pytest.raises(ValidationError, match="summary columns"):
        results.summarize(by=["missing"])
    with pytest.raises(ValidationError, match="unknown measures"):
        results.summarize(by=["scenario"], measures=["missing"])

    invalid = ProjectionResults(
        pd.DataFrame({"group": ["A"], "ratio": [1.0]}),
        measures={
            "ratio": Metric(
                "ratio", aggregation="recalculate", numerator="missing", denominator="other"
            )
        },
        projection_keys=("group",),
    )
    with pytest.raises(ValidationError, match="measure definitions are missing"):
        invalid.summarize(by=["group"], measures=["ratio"])


def test_compare_scenarios_calculates_absolute_and_percent_changes():
    results = make_results()
    comparison = results.compare_scenarios(
        baseline="base", comparison="adverse", by=["group"], measures=["amount"]
    )
    assert comparison["amount_change"].tolist() == [20.0, 20.0]
    assert comparison["amount_pct_change"].tolist() == pytest.approx([0.2, 0.1])


def test_compare_scenarios_zero_baseline_percent_is_nan():
    frame = pd.DataFrame(
        {"scenario": ["base", "other"], "group": ["A", "A"], "amount": [0.0, 1.0]}
    )
    results = ProjectionResults(
        frame, measures={"amount": Calculation("amount")}, projection_keys=("group",)
    )
    comparison = results.compare_scenarios(
        baseline="base", comparison="other", by=["group"], measures=["amount"]
    )
    assert np.isnan(comparison["amount_pct_change"].item())


def test_result_audits_return_copies_and_empty_frames():
    results = make_results()
    assumption = results.assumption_audit()
    assumption.loc[0, "name"] = "changed"
    assert results.assumption_audit()["name"].item() == "trend"
    adjustment = results.adjustment_audit()
    adjustment.loc[0, "target"] = "changed"
    assert results.adjustment_audit()["target"].item() == "amount"

    empty = ProjectionResults(pd.DataFrame(), measures={}, projection_keys=())
    assert empty.assumption_audit().empty
    assert empty.adjustment_audit().empty


def test_combine_compatible_results_and_validation():
    frame = pd.DataFrame(
        {
            "scenario": ["base"],
            "projection_period": ["2027-01"],
            "group": ["A"],
            "a": [1.0],
        }
    )
    first = ProjectionResults(
        frame,
        measures={"a": Calculation("a")},
        projection_keys=("group",),
        assumption_audit_data=pd.DataFrame({"name": ["first"]}),
    )
    second = ProjectionResults(
        frame.drop(columns="a").assign(b=2.0),
        measures={"b": Calculation("b")},
        projection_keys=("group",),
        adjustment_audit_data=pd.DataFrame({"target": ["b"]}),
    )
    combined = ProjectionResults.combine(first, second)
    assert combined.frame[["a", "b"]].iloc[0].tolist() == [1.0, 2.0]
    assert set(combined.measures) == {"a", "b"}
    assert combined.assumption_audit()["name"].item() == "first"
    assert combined.adjustment_audit()["target"].item() == "b"
    with pytest.raises(ValidationError, match="at least one"):
        ProjectionResults.combine()
