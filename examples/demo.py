"""projectionmodels: one group, then the book roll-up.

Fit a persistency curve, project one group's premium and credibility-blended
claims onto the given monthly membership, then aggregate in-force renewals and a
new-business case into a book budget.

    pip install -e .
    python examples/demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from projectionmodels import (GroupProjection, BookProjection, Persistency, fit_persistency,
                              new_business, __version__)

pd.set_option("display.width", 120)
SEASON = np.array([0.92, 0.95, 1.02, 1.05, 1.03, 0.95, 0.93, 0.97, 1.02, 1.06, 1.08, 1.02])
EXP_MID, PROSP_MID = pd.Timestamp("2025-07-01"), pd.Timestamp("2027-07-01")

print(f"projectionmodels {__version__}\n" + "=" * 74)

# --- persistency fit from a little renewal history, then one group ---------------
pers = fit_persistency(rate_changes=[0.00, 0.05, 0.08, 0.12, 0.15, 0.20],
                                renewed=[0.95, 0.92, 0.88, 0.83, 0.79, 0.70])
print(f"persistency: base retention {pers.base_retention:.3f}, "
      f"elasticity {pers.rate_elasticity:.3f}  ->  P(renew | +6%) = {pers.probability(0.06):.3f}")

grpA = GroupProjection(
    prospective_membership=np.linspace(1_850, 1_950, 12).round(), seasonal_factors=SEASON,
    current_premium=4_500_000, current_member_months=21_600, rate_action=0.06, plan_change=-0.02,
    book_pmpm=180.0, claim_trend=0.06, exp_midpoint=EXP_MID, prosp_midpoint=PROSP_MID,
    group_claims=3_800_000, group_member_months=21_600, group_claim_count=6_000,
    full_credibility_claims=10_000.0, pooling_pmpm=8.0, persistency=pers)

r, c = grpA.result, grpA.result.pmpm
print("\nGROUP A")
print(f"  claims PMPM: group ${c.group_pmpm:.2f}  book ${c.book_pmpm:.2f}  Z={c.credibility:.2f}"
      f"  -> blended ${c.blended_pmpm:.2f} x trend {c.trend_factor:.3f} (+pool) = ${c.projected_pmpm:.2f}")
print(f"  premium PMPM (rolled fwd): ${r.premium_detail.projected_pmpm:.2f}")
print(f"  conditional: premium ${r.premium:,.0f}  claims ${r.claims:,.0f}  LR {r.loss_ratio:.1%}")
print(f"  renewal P={r.renewal_prob:.2f} -> expected premium ${r.expected_premium:,.0f}  "
      f"claims ${r.expected_claims:,.0f}")

# --- a second in-force group + one new-business case, then the book --------------
grpB = GroupProjection(
    prospective_membership=np.full(12, 620.0), seasonal_factors=SEASON,
    current_premium=1_150_000, current_member_months=7_200, rate_action=0.11, plan_change=0.0,
    book_pmpm=175.0, claim_trend=0.06, exp_midpoint=EXP_MID, prosp_midpoint=PROSP_MID,
    group_claims=980_000, group_member_months=7_200, group_claim_count=1_400,
    full_credibility_claims=10_000.0, pooling_pmpm=6.0, persistency=pers)

nb = new_business(
    book_pmpm=178.0, claim_trend=0.06, exp_midpoint=EXP_MID, prosp_midpoint=PROSP_MID,
    prospective_membership=np.full(12, 400.0), manual_premium_pmpm=205.0,
    seasonal_factors=SEASON, close_ratio=0.30, pooling_pmpm=7.0)

book = BookProjection([grpA, grpB, nb], labels=["GroupA (renew)", "GroupB (renew)", "NewCo (sold)"])
print("\n" + "=" * 74, "\nBOOK ROLL-UP  (expected, renewal/close weighted)\n", "=" * 74, sep="")
bg = book.by_group.copy()
for col in ("premium", "claims"):
    bg[col] = bg[col].map(lambda v: f"${v:,.0f}")
bg["loss_ratio"] = bg["loss_ratio"].map(lambda v: f"{v:.1%}")
bg["renewal_prob"] = bg["renewal_prob"].map(lambda v: f"{v:.2f}")
print(bg.to_string(index=False))
print(f"\nBOOK TOTAL:  premium ${book.premium:,.0f}   claims ${book.claims:,.0f}   "
      f"loss ratio {book.loss_ratio:.1%}")
print(f"quarterly claims: "
      + "  ".join(f"Q{q+1} ${book.monthly['claims'].to_numpy()[q*3:q*3+3].sum():,.0f}" for q in range(4)))
