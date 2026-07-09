"""
PMPM projection -- the claims engine.
=====================================
Credibility-blends the group's own PMPM with the book PMPM, trends and
plan-adjusts it, and adds a large-claim pooling load:

    Z          from the group's claim count (limited fluctuation) unless supplied
    blended    = Z * group_pmpm + (1 - Z) * book_pmpm
    projected  = blended * trend * plan_factor + pooling_pmpm * trend

`projected_pmpm` is a rate per member-month. Call `.claims(membership, seasonal)`
to turn it into projected claim dollars by month; seasonality (factors averaging 1)
redistributes across months without changing the annual total.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import actuarialpy as ap
from actuarialpy import credibility_weighted_estimate


@dataclass(frozen=True)
class PMPMResult:
    group_pmpm: float
    book_pmpm: float
    credibility: float
    blended_pmpm: float
    trend_factor: float
    plan_factor: float
    pooling_pmpm: float
    projected_pmpm: float


class PMPMProjection:
    def __init__(self, *, book_pmpm, claim_trend, exp_midpoint, prosp_midpoint,
                 group_pmpm=None, group_claims=None, group_member_months=None,
                 group_claim_count=None, credibility=None, full_credibility_claims=1082.0,
                 plan_factor=1.0, pooling_pmpm=0.0):
        if group_pmpm is None:
            if group_claims is None or group_member_months is None:
                raise ValueError("supply group_pmpm, or group_claims and group_member_months")
            group_pmpm = ap.pure_premium(group_claims, group_member_months)
        if credibility is None:
            if group_claim_count is None:
                raise ValueError("supply credibility, or group_claim_count to derive it")
            credibility = min(float(ap.limited_fluctuation_z(group_claim_count, full_credibility_claims)), 1.0)

        tf = float(ap.midpoint_trend_factor(exp_midpoint, prosp_midpoint, claim_trend))
        blended = float(credibility_weighted_estimate(group_pmpm, book_pmpm, credibility))
        projected = blended * tf * plan_factor + pooling_pmpm * tf
        self.result = PMPMResult(
            group_pmpm=float(group_pmpm), book_pmpm=float(book_pmpm),
            credibility=float(credibility), blended_pmpm=blended, trend_factor=tf,
            plan_factor=float(plan_factor), pooling_pmpm=float(pooling_pmpm),
            projected_pmpm=float(projected))

    @property
    def projected_pmpm(self) -> float:
        return self.result.projected_pmpm

    def claims(self, membership, seasonal_factors=None):
        """Projected claim dollars by prospective month."""
        E = np.asarray(membership, float)
        s = np.ones_like(E) if seasonal_factors is None else np.asarray(seasonal_factors, float)
        return self.result.projected_pmpm * E * s


def project_pmpm(**kwargs) -> PMPMResult:
    """Functional form: returns the :class:`PMPMResult`."""
    return PMPMProjection(**kwargs).result
