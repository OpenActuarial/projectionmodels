"""Regression: the renewal-cycle worked-example page numbers stay true.

Pins docs page ``worked-example-projection.md`` (Example 7) in the
OpenActuarial docs repo — the projectionmodels segment. The pricing
segment (indication and renewal constraints) is pinned in ratingmodels'
suite; the issued actions cross over here as constants.
"""
import numpy as np
import pandas as pd
import pytest

import projectionmodels as pm
from projectionmodels.integrations import actuarialpy as apx


def _inputs():
    rng = np.random.default_rng(20270101)
    months = pd.date_range("2024-01-01", "2026-12-01", freq="MS")
    season = np.array([0.97, 0.95, 0.99, 0.98, 1.00, 1.00,
                       0.99, 1.00, 1.01, 1.03, 1.04, 1.06])
    valuation = pd.Timestamp("2026-12-31")
    completion = {0: 0.55, 1: 0.85, 2: 0.96}
    rows = []
    for group, mm, level in [("A", 5000.0, 1.00), ("B", 700.0, 1.08)]:
        for ct, base, tr in [("inpatient", 175.0, 0.065),
                             ("outpatient", 260.0, 0.080)]:
            for i, m in enumerate(months):
                rate = base * level * (1 + tr) ** (i / 12) * season[m.month - 1]
                rate *= 1 + rng.normal(0, 0.02)
                maturity = ((valuation.year - m.year) * 12
                            + valuation.month - m.month)
                rows.append((group, ct, m,
                             rate * mm * completion.get(maturity, 1.0), mm))
    hist = pd.DataFrame(rows, columns=["group_id", "claim_type",
                                       "incurred_month", "reported_claims",
                                       "member_months"])
    tx = pd.DataFrame(
        [(ct, o, o + pd.DateOffset(months=d), 1_000_000.0 * (1 + 0.02 * i) * s)
         for ct in ("inpatient", "outpatient")
         for i, o in enumerate(pd.date_range("2026-01-01", periods=8, freq="MS"))
         for d, s in enumerate((0.55, 0.30, 0.11, 0.04))],
        columns=["claim_type", "incurred_month", "paid_month", "paid"])
    return hist, tx[tx["paid_month"] <= valuation], valuation


def test_renewal_cycle_page_numbers():
    hist, tx, valuation = _inputs()

    completion = apx.estimate_completion(
        "claim_completion", tx, by=["claim_type"],
        origin_col="incurred_month", valuation_col="paid_month",
        amount_col="paid")
    panel = hist.groupby(["claim_type", "incurred_month"], as_index=False).agg(
        reported_claims=("reported_claims", "sum"),
        member_months=("member_months", "sum"))
    completed = completion.apply(panel, value_col="reported_claims",
                                 date_col="incurred_month",
                                 valuation_date=valuation,
                                 by=["claim_type"], out_col="completed_claims")
    seasonality = apx.estimate_seasonality(
        "claim_seasonality", completed, by=["claim_type"],
        date_col="incurred_month", value_col="completed_claims",
        exposure_col="member_months")
    deseason = apx.remove_seasonality(completed, seasonality,
                                      date_col="incurred_month",
                                      value_col="completed_claims",
                                      by=["claim_type"],
                                      out_col="deseasonalized_claims")
    trend = apx.estimate_trend(
        "claim_trend", deseason, by=["claim_type"],
        date_col="incurred_month", value_col="deseasonalized_claims",
        exposure_col="member_months")
    credibility = apx.estimate_credibility(
        "claim_credibility",
        hist.drop_duplicates(["group_id", "incurred_month"]),
        method="limited_fluctuation", by=["group_id"],
        exposure_col="member_months", full_credibility_standard=120_000.0)

    tsel = trend.selected_values.set_index("claim_type")["claim_trend"]
    assert round(float(tsel["inpatient"]), 4) == 0.0651
    assert round(float(tsel["outpatient"]), 4) == 0.0798
    csel = completion.selected_values
    assert np.allclose(
        csel.loc[csel.claim_type == "inpatient", "completion_factor"],
        [0.55, 0.85, 0.96, 1.00])
    zsel = credibility.selected_values.set_index("group_id")["claim_credibility"]
    assert float(zsel["A"]) == pytest.approx(1.0)
    assert round(float(zsel["B"]), 4) == 0.4583

    horizon = pm.ProjectionHorizon("2027-01-01", periods=12)
    periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
    exposure = pd.DataFrame(
        [{"group_id": g, "projection_period": p, "member_months": mm}
         for g, mm in (("A", 5000.0), ("B", 700.0)) for p in periods])
    experience = pm.ClaimExperience(
        hist, projection_keys=["group_id"], claim_type_col="claim_type",
        date_col="incurred_month", claims_col="reported_claims",
        exposure_col="member_months", valuation_date=valuation)
    manual = pm.Assumption(
        "manual_claim_rate",
        pd.DataFrame({"claim_type": ["inpatient", "outpatient"],
                      "manual_claim_rate": [215.0, 335.0]}),
        lookup=["claim_type"], value_col="manual_claim_rate")
    projection = pm.ClaimProjection.from_experience(
        experience, exposure=exposure, exposure_col="member_months",
        horizon=horizon, completion=completion, seasonality=seasonality,
        trend=trend, credibility=credibility, complement=manual,
        rate_loads=(14.50,))
    results = projection.project(
        scenarios=[pm.Scenario("baseline"),
                   pm.Scenario("adverse",
                               [pm.Adjustment(target="claim_trend",
                                              method="add", value=0.02)])])

    cy = results.summarize(by=["scenario", "group_id"],
                           measures=["projected_claims",
                                     "claims_per_exposure"])
    lc = (cy[cy.scenario == "baseline"]
          .set_index("group_id")["claims_per_exposure"])
    assert round(float(lc["A"]), 4) == 590.3863
    assert round(float(lc["B"]), 4) == 604.9014

    det = results.to_frame()
    row = det[(det.scenario == "baseline") & (det.group_id == "B")
              & (det.projection_period == "2027-07")
              & (det.claim_type == "inpatient")].iloc[0]
    assert round(row["experience_claim_rate"], 2) == 208.01
    assert round(row["trended_experience_rate"], 2) == 236.63
    assert round(row["credible_claim_rate"], 2) == 224.91
    assert round(row["projected_claim_rate"], 2) == 238.75

    # Issued renewal actions from the pricing step. Example 7's indication
    # and cap/floor numbers (A capped at +10%, B at formula +9.42%) are
    # pinned in ratingmodels' suite (test_worked_example_projection_
    # indication.py) — ratingmodels' actuarialpy floor sits above this
    # repo's tested range, so the pricing segment cannot import here.
    current = pd.Series({"A": 585.0, "B": 612.0})
    issued = np.array([0.10, 0.09418300653594769])

    actions = pm.RenewalRateActions(
        pd.DataFrame({"group_id": ["A", "B"],
                      "effective_date": pd.to_datetime(["2027-04-01",
                                                        "2027-09-01"]),
                      "rate_action": issued}),
        projection_keys=["group_id"])
    premium_results = pm.PremiumProjection(
        premium_data=pd.DataFrame({
            "group_id": ["A", "B"],
            "renewal_date": pd.to_datetime(["2027-04-01", "2027-09-01"]),
            "current_premium_rate": current.to_numpy()}),
        projection_keys=["group_id"], exposure=exposure,
        exposure_col="member_months", horizon=horizon,
        rate_actions=actions).project()
    ren = premium_results.detail().query("is_renewal_period")
    assert round(float(ren.loc[ren.group_id == "A", "premium"].iloc[0])) == 3_217_500
    assert round(float(ren.loc[ren.group_id == "B", "premium"].iloc[0])) == 468_748

    claims_pp = results.summarize(
        by=["scenario", "group_id", "projection_period"],
        measures=["projected_claims"])
    premium_pp = premium_results.summarize(
        by=["group_id", "projection_period"], measures=["premium"])
    merged = (claims_pp[claims_pp.scenario == "baseline"]
              .merge(premium_pp, on=["group_id", "projection_period"])
              .groupby("group_id")[["projected_claims", "premium"]].sum())
    lr = merged["projected_claims"] / merged["premium"]
    assert round(float(lr["A"]), 4) == 0.9388
    assert round(float(lr["B"]), 4) == 0.9583

    comp = results.compare_scenarios(baseline="baseline", comparison="adverse",
                                     by="group_id",
                                     measures="projected_claims")
    pct = comp.set_index("group_id")["projected_claims_pct_change"]
    assert round(float(pct["A"]), 4) == 0.0367
    assert round(float(pct["B"]), 4) == 0.0179
