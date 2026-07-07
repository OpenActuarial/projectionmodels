import pandas as pd
import pytest

from projectionmodels import ProjectionHorizon, ValidationError


def test_monthly_horizon_fields():
    horizon = ProjectionHorizon("2027-01-15", periods=3, frequency="monthly")
    frame = horizon.to_frame()
    assert frame["projection_period"].tolist() == ["2027-01", "2027-02", "2027-03"]
    assert frame["calendar_month"].tolist() == [1, 2, 3]
    assert frame["year_fraction"].eq(1 / 12).all()
    assert frame.loc[0, "period_start"] == pd.Timestamp("2027-01-01")


def test_horizon_requires_periods_or_end():
    with pytest.raises(ValidationError):
        ProjectionHorizon("2027-01-01")
    with pytest.raises(ValidationError):
        ProjectionHorizon("2027-01-01", periods=12, end="2027-12-01")
