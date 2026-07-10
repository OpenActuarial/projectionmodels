"""
Premium roll-forward -- the stored premium projected by known factors.
======================================================================
Premiums come from the database; they are NOT rebuilt from loss experience (that is
`ratingmodels`). This just rolls the stored figure forward:

    projected_pmpm = (current_premium / current_member_months)
                     * (1 + rate_action) * (1 + plan_change)

Premium is level per member-month (it earns evenly), so `.premium(membership)`
scales by membership with no seasonal shape -- unlike claims.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PremiumResult:
    current_pmpm: float
    rate_action: float
    plan_change: float
    projected_pmpm: float


class PremiumRollforward:
    def __init__(self, *, current_premium, current_member_months,
                 rate_action=0.0, plan_change=0.0):
        current_pmpm = current_premium / current_member_months
        projected = current_pmpm * (1.0 + rate_action) * (1.0 + plan_change)
        self.result = PremiumResult(float(current_pmpm), float(rate_action),
                                    float(plan_change), float(projected))

    @property
    def projected_pmpm(self) -> float:
        return self.result.projected_pmpm

    def premium(self, membership):
        """Projected premium dollars by prospective month (level per member-month)."""
        return self.result.projected_pmpm * np.asarray(membership, float)


def roll_forward_premium(**kwargs) -> PremiumResult:
    """Functional form: returns the :class:`PremiumResult`."""
    return PremiumRollforward(**kwargs).result
