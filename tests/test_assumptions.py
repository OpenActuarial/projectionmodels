import numpy as np
import pandas as pd
import pytest

from projectionmodels import (
    Assumption,
    AssumptionResolutionError,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)


def test_keyed_assumption_resolution():
    frame = pd.DataFrame({"product": ["PPO", "HMO"]})
    values = pd.DataFrame({"product": ["PPO", "HMO"], "trend": [0.08, 0.05]})
    assumption = Assumption("claim_trend", values, lookup=["product"], value_col="trend")
    assert assumption.resolve(frame).tolist() == [0.08, 0.05]


def test_missing_assumption_is_strict():
    assumption = Assumption(
        "trend",
        pd.DataFrame({"product": ["PPO"], "trend": [0.08]}),
        lookup=["product"],
        value_col="trend",
    )
    with pytest.raises(AssumptionResolutionError):
        assumption.resolve(pd.DataFrame({"product": ["HMO"]}))


def test_trend_from_experience_uses_actuarialpy():
    history = pd.DataFrame(
        {
            "product": ["PPO"] * 4,
            "month": pd.date_range("2026-01-01", periods=4, freq="MS"),
            "claims": [1, 2, 3, 4],
        }
    )
    trend = TrendAssumption.from_experience(
        "claim_trend",
        history,
        by="product",
        date_col="month",
        value_col="claims",
    )
    assert trend.source == "actuarialpy_estimate"
    assert trend.values["claim_trend"].item() == pytest.approx(0.12)


def test_seasonality_from_experience():
    history = pd.DataFrame(
        {
            "product": ["PPO"] * 24,
            "month": pd.date_range("2025-01-01", periods=24, freq="MS"),
            "claims": np.arange(24) + 1,
        }
    )
    seasonality = SeasonalityAssumption.from_experience(
        "claim_seasonality",
        history,
        by="product",
        date_col="month",
        value_col="claims",
    )
    assert len(seasonality.values) == 12
    assert seasonality.values["claim_seasonality"].mean() == pytest.approx(1.0)


def test_limited_fluctuation_credibility():
    history = pd.DataFrame(
        {"group": ["A", "B"], "claims": [25.0, 100.0]}
    )
    credibility = CredibilityAssumption.from_experience(
        "credibility",
        history,
        method="limited_fluctuation",
        by="group",
        exposure_col="claims",
        full_credibility_standard=100,
    )
    assert credibility.values["credibility"].tolist() == [0.5, 1.0]
