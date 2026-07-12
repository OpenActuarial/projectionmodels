"""Assumption coercion and exposure-column defaults on the project() seam."""
import pandas as pd
import pytest
from actuarialpy import Experience

import projectionmodels as pm


def _exp():
    df = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=6, freq="MS"),
                       "grp": ["A"] * 6, "ct": ["med"] * 6,
                       "claims": [100.0, 110, 105, 115, 120, 118],
                       "member_months": [1.0] * 6})
    return Experience(df, expense="claims", exposure="member_months", date="month",
                      dimensions=["grp", "ct"], valuation_date="2025-06-30")


def _future(col="member_months"):
    return pd.DataFrame({"grp": ["A"] * 3,
                         "projection_period": ["2025-07", "2025-08", "2025-09"],
                         col: [1.0] * 3})


def test_raw_series_completion_and_scalar_credibility_coerce():
    factors = pd.Series({0: 0.6, 1: 0.85})
    proj = pm.project(_exp(), exposure=_future(), horizon=pm.ProjectionHorizon("2025-07-01", periods=3),
                      trend=0.05, completion=factors, credibility=0.9, complement=110.0)
    assert proj.project().frame["projected_claims"].gt(0).all()


def test_exposure_column_defaults_to_the_bound_role_name():
    proj = pm.project(_exp(), exposure=_future("member_months"),
                      horizon=pm.ProjectionHorizon("2025-07-01", periods=3), trend=0.05)
    assert proj.exposure_col == "member_months"


def test_missing_exposure_column_error_lists_the_frame():
    with pytest.raises(pm.ValidationError, match="pass exposure_col"):
        pm.project(_exp(), exposure=_future("lives"),
                   horizon=pm.ProjectionHorizon("2025-07-01", periods=3), trend=0.05)
