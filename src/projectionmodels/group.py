"""
Group projection -- one group's forward roll, premium and claims together.
==========================================================================
Composes the premium roll-forward and the credibility-blended claims projection on
the given monthly membership, then weights both by the renewal probability. This is
the unit you loop over the in-force book (see :mod:`projectionmodels.book`).

Renewal probability weights premium and claims equally (a lapsed group books
neither), so the projected loss ratio is unaffected by it.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .pmpm import PMPMProjection, PMPMResult
from .premium import PremiumRollforward, PremiumResult


@dataclass(frozen=True)
class GroupProjectionResult:
    monthly: pd.DataFrame               # month, member_months, premium, claims (conditional on renewal)
    premium: float                      # conditional on renewal
    claims: float
    loss_ratio: float
    renewal_prob: float
    expected_premium: float             # renewal-weighted
    expected_claims: float
    pmpm: PMPMResult
    premium_detail: PremiumResult


class GroupProjection:
    def __init__(self, *, prospective_membership, seasonal_factors=None,
                 # premium (from DB)
                 current_premium, current_member_months, rate_action=0.0, plan_change=0.0,
                 # claims (group from DB + book)
                 book_pmpm, claim_trend, exp_midpoint, prosp_midpoint,
                 group_pmpm=None, group_claims=None, group_member_months=None,
                 group_claim_count=None, credibility=None, full_credibility_claims=1082.0,
                 pooling_pmpm=0.0, plan_affects_claims=True,
                 # renewal likelihood -- supplied (e.g. from underwriting), not modelled here
                 renewal_prob=1.0):
        E = np.asarray(prospective_membership, float)
        s = np.ones_like(E) if seasonal_factors is None else np.asarray(seasonal_factors, float)
        plan_factor = (1.0 + plan_change) if plan_affects_claims else 1.0

        prem = PremiumRollforward(current_premium=current_premium,
                                  current_member_months=current_member_months,
                                  rate_action=rate_action, plan_change=plan_change)
        pmpm = PMPMProjection(book_pmpm=book_pmpm, claim_trend=claim_trend,
                              exp_midpoint=exp_midpoint, prosp_midpoint=prosp_midpoint,
                              group_pmpm=group_pmpm, group_claims=group_claims,
                              group_member_months=group_member_months,
                              group_claim_count=group_claim_count, credibility=credibility,
                              full_credibility_claims=full_credibility_claims,
                              plan_factor=plan_factor, pooling_pmpm=pooling_pmpm)

        premium_m = prem.premium(E)
        claims_m = pmpm.claims(E, s)
        monthly = pd.DataFrame({"month": np.arange(1, len(E) + 1),
                                "member_months": E.round().astype(int),
                                "premium": premium_m, "claims": claims_m})
        P, C = float(premium_m.sum()), float(claims_m.sum())
        self.result = GroupProjectionResult(
            monthly=monthly, premium=P, claims=C, loss_ratio=C / P, renewal_prob=renewal_prob,
            expected_premium=renewal_prob * P, expected_claims=renewal_prob * C,
            pmpm=pmpm.result, premium_detail=prem.result)

    # convenient pass-through to the common result fields
    @property
    def monthly(self): return self.result.monthly
    @property
    def premium(self): return self.result.premium
    @property
    def claims(self): return self.result.claims
    @property
    def loss_ratio(self): return self.result.loss_ratio
    @property
    def expected_premium(self): return self.result.expected_premium
    @property
    def expected_claims(self): return self.result.expected_claims
    @property
    def renewal_prob(self): return self.result.renewal_prob


def project_group(**kwargs) -> GroupProjectionResult:
    """Functional form: returns the :class:`GroupProjectionResult`."""
    return GroupProjection(**kwargs).result


def new_business(*, book_pmpm, claim_trend, exp_midpoint, prosp_midpoint,
                 prospective_membership, manual_premium_pmpm, seasonal_factors=None,
                 close_ratio=1.0, plan_change=0.0, pooling_pmpm=0.0) -> GroupProjectionResult:
    """A sold-but-new case: no experience, so claims are fully manual (credibility 0)
    and premium is the manual/target rate on projected membership; `close_ratio`
    plays the role of the renewal probability."""
    E = np.asarray(prospective_membership, float)
    return GroupProjection(
        prospective_membership=E, seasonal_factors=seasonal_factors,
        current_premium=manual_premium_pmpm * E.sum(), current_member_months=E.sum(),
        rate_action=0.0, plan_change=plan_change,
        book_pmpm=book_pmpm, claim_trend=claim_trend, exp_midpoint=exp_midpoint,
        prosp_midpoint=prosp_midpoint, group_pmpm=book_pmpm, credibility=0.0,
        pooling_pmpm=pooling_pmpm, renewal_prob=close_ratio).result
