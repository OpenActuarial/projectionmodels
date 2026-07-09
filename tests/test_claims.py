import pandas as pd
import pytest

from projectionmodels import (
    Assumption,
    ClaimExperience,
    ClaimProjection,
    CredibilityAssumption,
    ProjectionHorizon,
    SeasonalityAssumption,
    TrendAssumption,
)


def test_claim_experience_and_projection_by_claim_type():
    history = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "product_id": ["PPO"] * 4,
            "claim_type": ["ip", "ip", "op", "op"],
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"] * 2),
            "claims": [1000.0, 1000.0, 500.0, 500.0],
            "member_months": [100.0, 100.0, 100.0, 100.0],
        }
    )
    experience = ClaimExperience(
        history,
        projection_keys=["group_id", "product_id"],
        claim_type_col="claim_type",
        date_col="month",
        claims_col="claims",
        exposure_col="member_months",
    )
    seasonality_values = pd.DataFrame(
        {
            "claim_type": [item for item in ["ip", "op"] for _ in range(12)],
            "season": list(range(1, 13)) * 2,
            "factor": [1.0] * 24,
        }
    )
    seasonality = SeasonalityAssumption.from_values(
        "claim_seasonality",
        seasonality_values,
        lookup=["claim_type"],
        factor_col="factor",
    )
    trend = TrendAssumption.from_values(
        "claim_trend",
        pd.DataFrame({"claim_type": ["ip", "op"], "trend": [0.12, 0.0]}),
        lookup=["claim_type"],
        rate_col="trend",
    )
    credibility = CredibilityAssumption.from_weights(
        "claim_credibility",
        pd.DataFrame({"claim_type": ["ip", "op"], "z": [0.5, 1.0]}),
        lookup=["claim_type"],
        weight_col="z",
    )
    complement = Assumption(
        "manual",
        pd.DataFrame({"claim_type": ["ip", "op"], "manual": [8.0, 4.0]}),
        lookup=["claim_type"],
        value_col="manual",
    )
    membership = pd.DataFrame(
        {
            "group_id": ["A", "A"],
            "product_id": ["PPO", "PPO"],
            "projection_period": ["2027-01", "2027-02"],
            "member_months": [100.0, 100.0],
        }
    )
    projection = ClaimProjection.from_experience(
        experience,
        membership=membership,
        horizon=ProjectionHorizon("2027-01-01", periods=2),
        trend=trend,
        seasonality=seasonality,
        credibility=credibility,
        complement=complement,
    )
    results = projection.project()
    assert set(results.frame["claim_type"]) == {"ip", "op"}
    ip = results.frame.query("claim_type == 'ip'").sort_values("projection_period")
    op = results.frame.query("claim_type == 'op'").sort_values("projection_period")
    # op has z = 1 and zero trend: the blend returns the experience rate and
    # nothing moves between the experience midpoint and the blend basis.
    assert op["trended_experience_rate"].iloc[0] == pytest.approx(
        op["experience_claim_rate"].iloc[0]
    )
    assert op["credible_claim_rate"].iloc[0] == pytest.approx(5.0)
    # ip has z = 0.5: the blend mixes *trended* experience with the complement
    # as stated (8.0) at the prospective midpoint of the horizon.
    assert ip["credible_claim_rate"].iloc[0] == pytest.approx(
        0.5 * ip["trended_experience_rate"].iloc[0] + 0.5 * 8.0
    )
    assert ip["projected_claim_rate"].iloc[1] > ip["projected_claim_rate"].iloc[0]
    assert op["projected_claim_rate"].iloc[1] == pytest.approx(
        op["projected_claim_rate"].iloc[0]
    )
    total = results.summarize(by=["projection_period", "group_id"])
    assert total["member_months"].tolist() == [100.0, 100.0]


# ---------------------------------------------------------------------------
# Regression tests for the 0.5.0 blend/trend ordering fix. Dates are chosen so
# calendar month gaps are exact integers (period midpoints land on day 15),
# making every expected value computable by hand.
# ---------------------------------------------------------------------------


def _base_rates(rate, manual, experience_midpoint):
    return pd.DataFrame(
        {
            "group_id": ["G"],
            "claim_type": ["med"],
            "experience_claim_rate": [float(rate)],
            "experience_midpoint": [pd.Timestamp(experience_midpoint)],
            "complement_claim_rate": [float(manual)],
        }
    )


def _membership(start, periods):
    labels = pd.period_range(start, periods=periods, freq="M").astype(str)
    return pd.DataFrame(
        {
            "group_id": "G",
            "projection_period": labels,
            "member_months": 1_000.0,
        }
    )


def _project(base, *, z, trend, start, periods, basis="prospective",
             loads=(), seasonality=None):
    credibility = (
        None
        if z is None
        else CredibilityAssumption.from_weights("claim_credibility", z)
    )
    projection = ClaimProjection(
        base_rates=base,
        projection_keys=["group_id"],
        claim_type_col="claim_type",
        membership=_membership(start, periods),
        horizon=ProjectionHorizon(start, periods=periods),
        trend=TrendAssumption.from_values("claim_trend", trend),
        seasonality=seasonality,
        credibility=credibility,
        complement_basis=basis,
        rate_loads=loads,
    )
    return projection.project()


def _annual_pmpm(results):
    summary = results.summarize(by=["group_id"])
    return float(summary["claim_pmpm"].iloc[0])


def test_zero_credibility_reproduces_the_complement_as_stated():
    base = _base_rates(380.0, 400.0, "2025-06-15")
    single = _project(base, z=0.0, trend=0.08, start="2027-06-01", periods=1)
    assert _annual_pmpm(single) == pytest.approx(400.0, rel=1e-12)

    year = _project(base, z=0.0, trend=0.08, start="2027-01-01", periods=12)
    # Within-horizon trend around the prospective midpoint nets to ~zero. The
    # pre-0.5.0 ordering produced ~468 here (the manual times two years of
    # trend), so the tolerance below is a regression guard, not slack.
    assert _annual_pmpm(year) == pytest.approx(400.0, rel=5e-4)


def test_full_credibility_is_invariant_to_the_blend_basis():
    base = _base_rates(380.0, 999.0, "2025-06-15")
    kwargs = dict(z=1.0, trend=0.08, start="2027-01-01", periods=12)
    prospective = _project(base, basis="prospective", **kwargs)
    experience = _project(base, basis="experience", **kwargs)
    explicit = _project(base, basis=pd.Timestamp("2026-03-31"), **kwargs)
    reference = prospective.frame["projected_claim_rate"].to_numpy()
    for other in (experience, explicit):
        assert other.frame["projected_claim_rate"].to_numpy() == pytest.approx(
            reference, rel=1e-12
        )


def test_experience_basis_reproduces_the_legacy_blend_then_trend_order():
    base = _base_rates(380.0, 400.0, "2025-06-15")
    legacy = _project(
        base, z=0.0, trend=0.08, start="2027-06-01", periods=1, basis="experience"
    )
    # Exactly 24 months of trend applied to the complement: 400 x 1.08^2.
    assert _annual_pmpm(legacy) == pytest.approx(400.0 * 1.08**2, rel=1e-12)


def test_partial_credibility_blends_trended_experience_with_the_complement():
    base = _base_rates(380.0, 400.0, "2025-06-15")
    results = _project(base, z=0.5, trend=0.08, start="2027-06-01", periods=1)
    expected = 0.5 * 380.0 * 1.08**2 + 0.5 * 400.0
    assert _annual_pmpm(results) == pytest.approx(expected, rel=1e-12)
    frame = results.frame
    assert frame["trended_experience_rate"].iloc[0] == pytest.approx(
        380.0 * 1.08**2
    )
    assert frame["credible_claim_rate"].iloc[0] == pytest.approx(expected)


def test_rate_loads_are_added_flat_after_seasonality_and_outside_the_blend():
    base = _base_rates(400.0, 999.0, "2025-06-15")
    seasonality = SeasonalityAssumption.from_values(
        "claim_seasonality",
        pd.DataFrame({"season": [6, 7], "factor": [1.1, 0.9]}),
        factor_col="factor",
    )
    results = _project(
        base, z=1.0, trend=0.0, start="2027-06-01", periods=2,
        seasonality=seasonality, loads=(14.5,),
    )
    rates = results.frame.sort_values("projection_period")["projected_claim_rate"]
    assert rates.tolist() == pytest.approx(
        [400.0 * 1.1 + 14.5, 400.0 * 0.9 + 14.5]
    )
    assert _annual_pmpm(results) == pytest.approx(400.0 + 14.5, rel=1e-12)
