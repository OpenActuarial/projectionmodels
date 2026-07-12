"""project() consumes a wide from_tables Experience via recorded-pivot melt."""
import numpy as np
import pandas as pd
import pytest
from actuarialpy import Experience, Measures

import projectionmodels as pm


def _wide(extra_expense=False, months=24):
    idx = pd.date_range("2025-01-01", periods=months, freq="MS")
    membership = pd.DataFrame([{"group_id": "A", "month": t, "member_months": 100.0}
                               for t in idx])
    lines = pd.DataFrame([
        {"group_id": "A", "incurred_date": t, "claim_type": ct,
         "paid_amount": base * (1.0 + 0.1 * np.cos(2 * np.pi * (t.month - 1) / 12))}
        for t in idx for ct, base in (("inpatient", 30_000.0), ("outpatient", 12_000.0))
    ])
    tables = [Measures(lines, expense="paid_amount", wide_by="claim_type",
                       date="incurred_date")]
    if extra_expense:
        fees = pd.DataFrame([{"group_id": "A", "month": t, "admin_fee": 900.0} for t in idx])
        tables.append(Measures(fees, expense="admin_fee"))
    return Experience.from_tables(
        membership, grain=["group_id", "month"], exposure="member_months",
        tables=tables, date="month", period="M", dimensions="group_id",
        valuation_date=idx[-1] + pd.offsets.MonthEnd(0))


def _future():
    return pd.DataFrame({"group_id": ["A"] * 6,
                         "projection_period": pd.period_range("2027-01", periods=6, freq="M").astype(str),
                         "member_months": [100.0] * 6})


def test_wide_experience_projects_identically_to_manual_long():
    exp = _wide()
    horizon = pm.ProjectionHorizon("2027-01-01", periods=6)
    wide_results = pm.project(exp, exposure=_future(), horizon=horizon, trend=0.05).project()

    long = exp.melt()
    manual = pm.project(long, exposure=_future(), horizon=horizon, trend=0.05,
                        grain=["group_id", "claim_type"]).project()
    a = wide_results.frame.sort_values(["claim_type", "projection_period"])["projected_claims"].to_numpy()
    b = manual.frame.sort_values(["claim_type", "projection_period"])["projected_claims"].to_numpy()
    assert a == pytest.approx(b)
    assert set(wide_results.frame["claim_type"]) == {"inpatient", "outpatient"}


def test_non_pivot_expense_columns_are_announced_as_excluded():
    exp = _wide(extra_expense=True)
    with pytest.warns(UserWarning, match="admin_fee.*excluded"):
        pm.project(exp, exposure=_future(),
                   horizon=pm.ProjectionHorizon("2027-01-01", periods=6), trend=0.05)


def test_seasonality_estimate_sentinel_fits_from_bound_history():
    exp = _wide()
    proj = pm.project(exp, exposure=_future(),
                      horizon=pm.ProjectionHorizon("2027-01-01", periods=6),
                      trend=0.0, seasonality="estimate")
    rates = proj.project().frame.query("claim_type == 'inpatient'").sort_values("projection_period")
    assert rates["projected_claim_rate"].nunique() > 1   # seasonal shape reapplied


def test_completion_estimate_sentinel_explains_what_it_needs():
    exp = _wide()
    with pytest.raises(pm.ValidationError, match="estimate_completion"):
        pm.project(exp, exposure=_future(),
                   horizon=pm.ProjectionHorizon("2027-01-01", periods=6),
                   trend=0.05, completion="estimate")
