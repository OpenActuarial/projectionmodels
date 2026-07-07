from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import (
    ProjectionHorizon,
    ProjectionModelsError,
    ValidationError,
    actuarialpy_adapter,
)


def test_quarterly_horizon_with_end_date():
    frame = ProjectionHorizon(
        "2027-02-15", end="2027-10-01", frequency="quarterly"
    ).to_frame()
    assert frame["projection_period"].tolist() == ["2027Q1", "2027Q2", "2027Q3", "2027Q4"]
    assert frame["season"].tolist() == [1, 2, 3, 4]
    assert frame["year_fraction"].eq(0.25).all()


def test_annual_horizon_fields_and_length():
    horizon = ProjectionHorizon("2027-07-01", periods=2, frequency="annual")
    frame = horizon.to_frame()
    assert frame["projection_period"].tolist() == ["2027", "2028"]
    assert frame["season"].tolist() == [1, 1]
    assert len(horizon) == 2
    assert horizon.normalized_start == pd.Timestamp("2027-01-01")


def test_horizon_validation_errors():
    with pytest.raises(ValidationError, match="frequency"):
        ProjectionHorizon("2027-01-01", periods=1, frequency="weekly")
    with pytest.raises(ValidationError, match="positive"):
        ProjectionHorizon("2027-01-01", periods=0)
    with pytest.raises(ValidationError, match="must not precede"):
        ProjectionHorizon("2027-02-01", end="2027-01-01").to_frame()


def test_actuarialpy_adapter_missing_module(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(actuarialpy_adapter, "import_module", missing)
    with pytest.raises(ProjectionModelsError, match="actuarialpy is required"):
        actuarialpy_adapter.require_actuarialpy()


def test_actuarialpy_adapter_missing_function(monkeypatch):
    monkeypatch.setattr(actuarialpy_adapter, "require_actuarialpy", lambda: object())
    with pytest.raises(ProjectionModelsError, match="does not expose"):
        actuarialpy_adapter.actuarialpy_function("missing")
