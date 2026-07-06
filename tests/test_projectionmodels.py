"""Test suite for projectionmodels."""
import numpy as np
import pandas as pd
import pytest

import actuarialpy as ap
import projectionmodels as pm
from actuarialpy import credibility_weighted_estimate as blend

EXP_MID = pd.Timestamp("2025-07-01")
PROSP_MID = pd.Timestamp("2027-07-01")
FLAT12 = np.full(12, 1_000.0)
SEASON = np.array([0.90, 0.95, 1.00, 1.05, 1.10, 1.05, 0.95, 0.90, 0.98, 1.04, 1.06, 1.02])


# --- primitives -----------------------------------------------------------------
def test_blend_scalar_and_array():
    assert blend(100.0, 200.0, 0.5) == pytest.approx(150.0)
    assert blend(100.0, 200.0, 1.0) == pytest.approx(100.0)   # full credibility -> experience
    assert blend(100.0, 200.0, 0.0) == pytest.approx(200.0)   # zero credibility -> manual
    out = blend(np.array([100.0, 50.0]), np.array([200.0, 150.0]), np.array([0.5, 0.2]))
    assert out == pytest.approx([150.0, 130.0])


def test_season_normalisation():
    # the demo/test seasonal vector should average ~1 so it only redistributes
    assert SEASON.mean() == pytest.approx(1.0, abs=0.02)


# --- persistency ----------------------------------------------------------------
def test_persistency_probability_and_clip():
    p = pm.Persistency(base_retention=0.90, rate_elasticity=1.0)
    assert float(p.probability(0.10)) == pytest.approx(0.80)
    assert float(p.probability(0.00)) == pytest.approx(0.90)
    assert float(p.probability(2.00)) == pytest.approx(0.0)    # clipped at floor
    assert float(p.probability(-1.0)) == pytest.approx(1.0)    # clipped at cap


def test_persistency_from_history_recovers_line():
    base, elas = 0.95, 1.2
    rc = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    renewed = base - elas * rc                                  # exact line, no noise
    fit = pm.fit_persistency(rc, renewed)
    assert fit.base_retention == pytest.approx(base, abs=1e-6)
    assert fit.rate_elasticity == pytest.approx(elas, abs=1e-6)


# --- PMPM projection ------------------------------------------------------------
def test_pmpm_composition():
    kw = dict(book_pmpm=180.0, claim_trend=0.06, exp_midpoint=EXP_MID, prosp_midpoint=PROSP_MID)
    r = pm.PMPMProjection(group_pmpm=200.0, credibility=0.75, plan_factor=0.98,
                          pooling_pmpm=8.0, **kw).result
    tf = float(ap.midpoint_trend_factor(EXP_MID, PROSP_MID, 0.06))
    expected = blend(200.0, 180.0, 0.75) * tf * 0.98 + 8.0 * tf
    assert r.blended_pmpm == pytest.approx(0.75 * 200 + 0.25 * 180)
    assert r.trend_factor == pytest.approx(tf)
    assert r.projected_pmpm == pytest.approx(expected)


def test_pmpm_derives_group_pmpm_and_credibility():
    r = pm.PMPMProjection(book_pmpm=180.0, claim_trend=0.0, exp_midpoint=EXP_MID,
                          prosp_midpoint=PROSP_MID, group_claims=3_600_000,
                          group_member_months=20_000, group_claim_count=5_000,
                          full_credibility_claims=10_000.0).result
    assert r.group_pmpm == pytest.approx(180.0)                 # 3.6M / 20k
    assert r.credibility == pytest.approx(
        min(float(ap.limited_fluctuation_z(5_000, 10_000.0)), 1.0))


def test_pmpm_credibility_capped_at_one():
    r = pm.PMPMProjection(book_pmpm=180.0, claim_trend=0.0, exp_midpoint=EXP_MID,
                          prosp_midpoint=PROSP_MID, group_pmpm=200.0,
                          group_claim_count=10_000_000, full_credibility_claims=1_082.0).result
    assert r.credibility == pytest.approx(1.0)


def test_pmpm_requires_inputs():
    with pytest.raises(ValueError):
        pm.PMPMProjection(book_pmpm=180.0, claim_trend=0.0, exp_midpoint=EXP_MID,
                          prosp_midpoint=PROSP_MID, credibility=0.5)   # no group_pmpm/claims


# --- premium roll-forward -------------------------------------------------------
def test_premium_rollforward():
    r = pm.PremiumRollforward(current_premium=2_400_000, current_member_months=12_000,
                              rate_action=0.10, plan_change=-0.05).result
    assert r.current_pmpm == pytest.approx(200.0)
    assert r.projected_pmpm == pytest.approx(200.0 * 1.10 * 0.95)


def test_premium_scales_with_membership():
    roll = pm.PremiumRollforward(current_premium=2_400_000, current_member_months=12_000,
                                 rate_action=0.0, plan_change=0.0)
    prem = roll.premium(FLAT12)
    assert prem.sum() == pytest.approx(200.0 * FLAT12.sum())


# --- group projection -----------------------------------------------------------
def _group(**over):
    kw = dict(prospective_membership=FLAT12, seasonal_factors=SEASON,
              current_premium=2_400_000, current_member_months=12_000,
              rate_action=0.06, plan_change=0.0, book_pmpm=180.0, claim_trend=0.06,
              exp_midpoint=EXP_MID, prosp_midpoint=PROSP_MID, group_pmpm=190.0,
              credibility=0.7, pooling_pmpm=5.0)
    kw.update(over)
    return pm.GroupProjection(**kw)


def test_group_loss_ratio_invariant_to_renewal():
    g_full = _group(renewal_prob=1.0).result
    g_half = _group(renewal_prob=0.5).result
    assert g_full.loss_ratio == pytest.approx(g_half.loss_ratio)         # LR unaffected
    assert g_half.expected_premium == pytest.approx(0.5 * g_full.premium)
    assert g_half.expected_claims == pytest.approx(0.5 * g_full.claims)


def test_group_seasonality_preserves_annual_claims():
    # flat membership + mean-1 seasonal -> annual claims == projected_pmpm * total member-months
    g = _group(seasonal_factors=SEASON).result
    total_mm = FLAT12.sum()
    assert g.claims == pytest.approx(g.pmpm.projected_pmpm * total_mm * SEASON.mean())
    # and the seasonal version has the same total as the flat version
    g_flat = _group(seasonal_factors=None).result
    assert g.claims == pytest.approx(g_flat.claims * SEASON.mean(), rel=1e-9)


def test_group_persistency_sets_renewal_prob():
    pers = pm.Persistency(base_retention=0.90, rate_elasticity=1.0)
    g = _group(rate_action=0.10, persistency=pers).result
    assert g.renewal_prob == pytest.approx(0.80)


# --- book projection ------------------------------------------------------------
def test_book_aggregates_expected():
    g1 = _group(renewal_prob=0.9).result
    g2 = _group(current_premium=1_200_000, current_member_months=6_000,
                prospective_membership=np.full(12, 500.0), renewal_prob=0.8).result
    book = pm.BookProjection([g1, g2], labels=["A", "B"]).result
    assert book.premium == pytest.approx(g1.expected_premium + g2.expected_premium)
    assert book.claims == pytest.approx(g1.expected_claims + g2.expected_claims)
    assert book.loss_ratio == pytest.approx(book.claims / book.premium)
    assert list(book.by_group["group"]) == ["A", "B"]
    assert len(book.monthly) == 12


def test_new_business_is_fully_manual():
    nb = pm.new_business(book_pmpm=180.0, claim_trend=0.0, exp_midpoint=EXP_MID,
                         prosp_midpoint=PROSP_MID, prospective_membership=FLAT12,
                         manual_premium_pmpm=205.0, close_ratio=0.3)
    assert nb.pmpm.credibility == pytest.approx(0.0)            # no experience
    assert nb.pmpm.blended_pmpm == pytest.approx(180.0)        # all book
    assert nb.renewal_prob == pytest.approx(0.3)               # close ratio
    assert nb.premium == pytest.approx(205.0 * FLAT12.sum())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
