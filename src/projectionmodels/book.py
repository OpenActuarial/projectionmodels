"""
Book projection -- aggregate group projections into the book budget.
====================================================================
Rolls up in-force renewals and new business into total premium, claims, and loss
ratio (by group and by month). Totals use the EXPECTED (renewal-weighted) figures,
so a group contributes in proportion to how likely it is to renew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .group import GroupProjection


@dataclass(frozen=True)
class BookResult:
    premium: float                      # expected (renewal-weighted) book premium
    claims: float
    loss_ratio: float
    by_group: pd.DataFrame              # per-group expected premium/claims/LR/renewal
    monthly: pd.DataFrame               # expected premium/claims by month, summed over the book


class BookProjection:
    def __init__(self, projections, labels=None):
        results = [p.result if isinstance(p, GroupProjection) else p for p in projections]
        if not results:
            raise ValueError("no projections supplied")
        labels = list(labels) if labels is not None else [f"grp_{i}" for i in range(len(results))]

        rows = []
        prem_m = np.zeros(len(results[0].monthly))
        clm_m = np.zeros_like(prem_m)
        months = results[0].monthly["month"].to_numpy()
        for lab, r in zip(labels, results, strict=True):
            rows.append({"group": lab, "premium": r.expected_premium, "claims": r.expected_claims,
                         "loss_ratio": (r.expected_claims / r.expected_premium
                                        if r.expected_premium else np.nan),
                         "renewal_prob": r.renewal_prob})
            prem_m += r.monthly["premium"].to_numpy() * r.renewal_prob
            clm_m += r.monthly["claims"].to_numpy() * r.renewal_prob

        by_group = pd.DataFrame(rows)
        monthly = pd.DataFrame({"month": months, "premium": prem_m, "claims": clm_m})
        monthly["loss_ratio"] = monthly["claims"] / monthly["premium"]
        P, C = float(by_group["premium"].sum()), float(by_group["claims"].sum())
        self.result = BookResult(premium=P, claims=C, loss_ratio=C / P,
                                 by_group=by_group, monthly=monthly)

    @property
    def premium(self): return self.result.premium
    @property
    def claims(self): return self.result.claims
    @property
    def loss_ratio(self): return self.result.loss_ratio
    @property
    def by_group(self): return self.result.by_group
    @property
    def monthly(self): return self.result.monthly


def project_book(projections, labels=None) -> BookResult:
    """Functional form: returns the :class:`BookResult`."""
    return BookProjection(projections, labels=labels).result
