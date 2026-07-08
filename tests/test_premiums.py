from __future__ import annotations

import pandas as pd
import pytest

from projectionmodels import (
    PremiumProjection,
    ProjectionDates,
    ProjectionHorizon,
    RenewalRateActions,
    ValidationError,
)


def _membership(groups, periods):
    return pd.DataFrame(
        [
            {
                "group_id": group,
                "projection_period": period,
                "member_months": 1.0,
            }
            for group in groups
            for period in pd.period_range("2027-01", periods=periods, freq="M").astype(str)
        ]
    )


def test_recurring_rate_action_applies_at_each_groups_renewal_and_persists():
    data = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "renewal_date": pd.to_datetime(["2027-03-01", "2027-07-01"]),
            "current_premium_pmpm": [100.0, 100.0],
            "rate_action": [0.10, 0.20],
        }
    )
    results = PremiumProjection(
        premium_data=data,
        projection_keys=["group_id"],
        membership=_membership(["A", "B"], 15),
        horizon=ProjectionHorizon("2027-01-01", periods=15),
        recurring_rate_action_col="rate_action",
    ).project().detail()

    rates = results.pivot(index="projection_period", columns="group_id", values="premium_pmpm")
    assert rates.loc["2027-02", "A"] == pytest.approx(100.0)
    assert rates.loc["2027-03", "A"] == pytest.approx(110.0)
    assert rates.loc["2027-06", "B"] == pytest.approx(100.0)
    assert rates.loc["2027-07", "B"] == pytest.approx(120.0)
    assert rates.loc["2028-02", "A"] == pytest.approx(110.0)
    assert rates.loc["2028-03", "A"] == pytest.approx(121.0)


def test_dated_rate_action_applies_once_and_does_not_repeat():
    data = pd.DataFrame(
        {
            "group_id": ["A"],
            "renewal_date": pd.to_datetime(["2027-03-15"]),
            "current_premium_pmpm": [100.0],
        }
    )
    actions = RenewalRateActions(
        pd.DataFrame(
            {
                "group_id": ["A"],
                "effective_date": pd.to_datetime(["2027-03-15"]),
                "rate_action": [0.10],
            }
        ),
        projection_keys=["group_id"],
    )
    results = PremiumProjection(
        premium_data=data,
        projection_keys=["group_id"],
        membership=_membership(["A"], 15),
        horizon=ProjectionHorizon("2027-01-01", periods=15),
        rate_actions=actions,
    ).project().detail().set_index("projection_period")

    assert results.loc["2027-02", "premium_pmpm"] == pytest.approx(100.0)
    assert results.loc["2027-03", "premium_pmpm"] == pytest.approx(110.0)
    assert results.loc["2028-03", "premium_pmpm"] == pytest.approx(110.0)


def test_multiple_dated_actions_compound_in_their_effective_periods():
    data = pd.DataFrame(
        {"group_id": ["A"], "current_premium_pmpm": [100.0]}
    )
    actions = RenewalRateActions(
        pd.DataFrame(
            {
                "group_id": ["A", "A"],
                "effective_date": pd.to_datetime(["2027-03-01", "2028-03-01"]),
                "rate_action": [0.10, 0.05],
            }
        ),
        projection_keys=["group_id"],
    )
    detail = PremiumProjection(
        premium_data=data,
        projection_keys=["group_id"],
        membership=_membership(["A"], 15),
        horizon=ProjectionHorizon("2027-01-01", periods=15),
        rate_actions=actions,
    ).project().detail().set_index("projection_period")
    assert detail.loc["2027-03", "premium_pmpm"] == pytest.approx(110.0)
    assert detail.loc["2028-03", "premium_pmpm"] == pytest.approx(115.5)


def test_premium_equals_rate_times_active_membership():
    data = pd.DataFrame(
        {
            "group_id": ["A"],
            "current_premium_pmpm": [125.0],
            "effective_date": pd.to_datetime(["2027-01-16"]),
        }
    )
    detail = PremiumProjection(
        premium_data=data,
        projection_keys=["group_id"],
        membership=pd.DataFrame(
            {
                "group_id": ["A"],
                "projection_period": ["2027-01"],
                "member_months": [100.0],
            }
        ),
        horizon=ProjectionHorizon("2027-01-01", periods=1),
        dates=ProjectionDates(
            entry_date="effective_date", exposure_timing="daily_prorated"
        ),
    ).project().detail()
    expected = 125.0 * 100.0 * (16 / 31)
    assert detail["premium"].item() == pytest.approx(expected)


def test_rate_action_validation():
    with pytest.raises(ValidationError, match="missing columns"):
        RenewalRateActions(pd.DataFrame({"group_id": ["A"]}), ["group_id"])
    with pytest.raises(ValidationError, match="unique"):
        RenewalRateActions(
            pd.DataFrame(
                {
                    "group_id": ["A", "A"],
                    "effective_date": ["2027-03-01", "2027-03-15"],
                    "rate_action": [0.10, 0.05],
                }
            ),
            ["group_id"],
        ).to_projection_table(ProjectionHorizon("2027-01-01", periods=12))


def test_premium_projection_requires_rate_action_and_membership_columns():
    with pytest.raises(ValidationError, match="premium_data is missing"):
        PremiumProjection(
            premium_data=pd.DataFrame(
                {"group_id": ["A"], "current_premium_pmpm": [100.0]}
            ),
            projection_keys=["group_id"],
            membership=_membership(["A"], 1),
            horizon=ProjectionHorizon("2027-01-01", periods=1),
            recurring_rate_action_col="rate_action",
        )
