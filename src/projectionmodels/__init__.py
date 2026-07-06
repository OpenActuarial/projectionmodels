"""
projectionmodels -- forward projection (budgeting) for a book of business
=========================================================================
Part of the OpenActuarial ecosystem. Sits beside `ratingmodels` and `lossmodels`
on top of the `actuarialpy` primitives layer, and depends only downward on
`actuarialpy` -- never sideways on another workflow package.

Rating builds a price from experience; projection takes the book AS IT IS -- stored
premiums, historical claims, membership assumptions -- and rolls it forward for
planning. The dividing line: the moment something computes what you *should* charge
rather than what the projection *is*, it belongs in `ratingmodels`, not here.

Layers
------
    PMPMProjection        credibility-blended, trended, pooled claims PMPM
    PremiumRollforward    stored premium rolled forward by rate action / plan change
    GroupProjection       one group: premium + claims + renewal weighting  (loop unit)
    BookProjection        aggregate in-force renewals + new business -> book budget

Each class computes on construction and exposes a frozen ``*Result`` dataclass via
``.result``; a lowercase functional form (``project_group`` etc.) returns the result
directly.

Quick start
-----------
    from projectionmodels import GroupProjection, BookProjection

    g = GroupProjection(prospective_membership=E, seasonal_factors=s,
                        current_premium=..., current_member_months=...,
                        rate_action=0.06, plan_change=-0.02,
                        book_pmpm=..., claim_trend=0.06,
                        exp_midpoint=..., prosp_midpoint=...,
                        group_claims=..., group_member_months=..., group_claim_count=...,
                        pooling_pmpm=..., renewal_prob=0.90)   # renewal likelihood from underwriting
    book = BookProjection([g_1, g_2, new_biz], labels=[...])
    book.loss_ratio
"""
from .pmpm import PMPMProjection, PMPMResult, project_pmpm
from .premium import PremiumRollforward, PremiumResult, roll_forward_premium
from .group import (GroupProjection, GroupProjectionResult, project_group, new_business)
from .book import BookProjection, BookResult, project_book

__version__ = "0.1.0"

__all__ = [
    "PMPMProjection", "PMPMResult", "project_pmpm",
    "PremiumRollforward", "PremiumResult", "roll_forward_premium",
    "GroupProjection", "GroupProjectionResult", "project_group", "new_business",
    "BookProjection", "BookResult", "project_book",
]
