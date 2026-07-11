from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd
import pytest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


def run_example(name: str) -> dict[str, object]:
    namespace = runpy.run_path(
        str(EXAMPLE_DIR / name), run_name=f"projectionmodels_example_{name}"
    )
    return namespace["run_example"]()


def test_general_projection_example_outputs_and_audits():
    output = run_example("advanced/general_projection.py")
    annual = output["annual"]
    comparison = output["comparison"]
    assert len(annual) == 8
    ppo = comparison.loc[comparison["product_id"] == "PPO"]
    hmo = comparison.loc[comparison["product_id"] == "HMO"]
    assert (ppo["premium_change"] > 0).all()
    assert hmo["premium_change"].eq(0).all()
    assert len(output["assumption_audit"]) == 2
    assert len(output["adjustment_audit"]) == 1


def test_health_claims_example_preserves_entity_exposure():
    output = run_example("health_claims.py")
    by_type = output["by_type"]
    total = output["total"]
    assert len(by_type) == 24
    assert total["member_months"].eq(1_000.0).all()
    assert total["projected_claims"].gt(0).all()
    assert total["claims_per_exposure"].gt(0).all()


def test_expense_example_includes_all_bases():
    output = run_example("expenses.py")
    annual = output["annual"]
    total = output["total"]
    assert set(annual["expense_type"]) == {
        "administration",
        "network_fee",
        "overhead",
        "commission",
        "claim_admin",
    }
    # the contractually flat fee projects at exactly its base value
    fee = annual.loc[annual["expense_type"] == "network_fee", "projected_expense"]
    assert fee.item() == pytest.approx(6.50 * 1_000.0 * 12)
    assert annual["projected_expense"].gt(0).all()
    assert total["projected_expense"].item() == pytest.approx(
        annual["projected_expense"].sum()
    )


def test_date_cohort_example_tracks_partial_exposure_and_renewals():
    output = run_example("date_cohorts.py")
    detail = output["results"].detail()
    group_b_jan = detail.query("group_id == 'B' and projection_period == '2027-01'")
    group_b_feb = detail.query("group_id == 'B' and projection_period == '2027-02'")
    assert group_b_jan["active_fraction"].item() == 0
    assert 0 < group_b_feb["active_fraction"].item() < 1
    assert detail.query("group_id == 'A' and projection_period == '2027-03'")[
        "is_renewal_period"
    ].item()
    assert set(output["summary"]["business_origin"]) == {"existing", "new_business"}


def test_calculated_assumptions_example_uses_actuarialpy_sources():
    output = run_example("calculated_assumptions.py")
    assert len(output["summary"]) == 2
    assert len(output["results"].detail()) == 12
    assert set(output["assumption_audit"]["source"]) == {"actuarialpy_estimate"}
    for name in ("completion", "seasonality", "trend", "credibility"):
        assert output[name].source == "actuarialpy_estimate"


def test_member_level_example_responds_to_lapse_by_product():
    output = run_example("advanced/member_level_projection.py")
    comparison = output["comparison"]
    term = comparison.loc[comparison["product_id"] == "term"]
    whole_life = comparison.loc[comparison["product_id"] == "whole_life"]
    assert (term["premium_change"] < 0).all()
    assert whole_life["premium_change"].eq(0).all()
    assert output["results"].detail()["premium"].ge(0).all()


def test_sensitivity_example_orders_claim_results():
    output = run_example("advanced/sensitivity_analysis.py")
    annual = output["annual"]
    low = annual.loc[annual["scenario"] == "trend_4%"].set_index("calendar_year")
    high = annual.loc[annual["scenario"] == "trend_8%"].set_index("calendar_year")
    assert (high["projected_claims"] > low["projected_claims"]).all()
    assert (output["comparison"]["projected_claims_change"] > 0).all()
    assert set(annual["scenario"]) == {"baseline", "trend_4%", "trend_6%", "trend_8%"}



def test_renewal_rate_action_example_applies_by_group_date():
    output = run_example("renewal_rate_actions.py")
    detail = output["detail"].pivot(
        index="projection_period", columns="group_id", values="projected_premium_rate"
    )
    assert detail.loc["2027-02", "A"] == pytest.approx(100.0)
    assert detail.loc["2027-03", "A"] == pytest.approx(110.0)
    assert detail.loc["2027-06", "B"] == pytest.approx(100.0)
    assert detail.loc["2027-07", "B"] == pytest.approx(120.0)

def test_underwriting_example_produces_exhibit_ready_results():
    output = run_example("underwriting_results.py")
    exhibit = output["exhibit_input"]
    required = {
        "scenario",
        "business_origin",
        "calendar_year",
        "member_months",
        "premium",
        "claims",
        "expenses",
        "underwriting_margin",
        "loss_ratio",
    }
    assert required.issubset(exhibit.columns)
    comparison = output["comparison"]
    assert (comparison["claims_change"] >= 0).all()
    assert (comparison["underwriting_margin_change"] <= 0).all()
    assert pd.api.types.is_numeric_dtype(exhibit["loss_ratio"])


def test_pooled_claims_example_outputs():
    output = run_example("pooled_claims.py")
    assert output["ceded_total"] > 0
    assert output["retained_pmpm"] < output["unpooled_pmpm"]
    summary = output["summary"]
    assert (summary["claims_per_exposure"] > 0).all()
    # The charge is added flat and outside the blend, so the with-charge and
    # without-charge projections differ by exactly the stated PMPM.
    assert output["net_pmpm"] == pytest.approx(
        output["net_pmpm_excluding_charge"] + output["pooling_charge"], rel=1e-9
    )
    # Pooling reconciliation: what left the experience equals ceded per
    # experience member month.
    assert output["unpooled_pmpm"] - output["retained_pmpm"] == pytest.approx(
        output["ceded_total"] / output["experience_member_months"], rel=1e-9
    )
    # Level anchor (data is seeded, so this is deterministic): invariants
    # alone cannot catch basis or scale errors — a projection off by 80x
    # satisfies every ratio check above.
    assert 300.0 < output["net_pmpm"] < 380.0
