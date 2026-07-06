# projectionmodels

[![CI](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/projectionmodels)](https://pypi.org/project/projectionmodels/)

Forward projection (budgeting) for a book of business — part of the
[OpenActuarial](https://openactuarial.org) ecosystem.

`ratingmodels` builds a price from experience. `projectionmodels` does the opposite
direction: it takes the book **as it is** — stored premiums, historical claims, and
membership assumptions — and rolls it forward for planning and budgeting. It sits
beside `ratingmodels` and `lossmodels` on top of the `actuarialpy` primitives layer,
and depends only **downward** on `actuarialpy` — never sideways on another workflow
package.

## Install

```bash
pip install projectionmodels
```

## Layers

| object | role |
| --- | --- |
| `PMPMProjection` | credibility-blended, trended, plan-adjusted claims PMPM with a pooling load |
| `PremiumRollforward` | stored premium rolled forward by rate action and plan change |
| `GroupProjection` | one group: premium + claims + renewal weighting — the unit you loop over the book |
| `BookProjection` | aggregate in-force renewals + new business into the book budget |

Each class computes on construction and exposes a frozen `*Result` dataclass via
`.result`; a lowercase functional form (`project_group`, `project_book`, …) returns
the result directly.

## What it computes

**Premium** is a stored value rolled forward (not rebuilt from experience), level
per member-month:

```
prem_pmpm* = (current_premium / current_member_months) · (1 + rate_action) · (1 + plan_change)
```

**Claims** are the actuarial piece — a credibility blend of the group's own PMPM and
the book PMPM, trended, plan-adjusted, and seasonalised onto the given membership:

```
Z         = limited-fluctuation credibility from the group's claim count
blended   = Z · group_pmpm + (1 − Z) · book_pmpm
projected = blended · trend · plan_factor + pooling_pmpm · trend
claims_m  = projected · membership_m · seasonal_m
```

**Renewal probability** is supplied per group (e.g. from underwriting) via
`renewal_prob` — it is an input, not something this package models. It weights
premium and claims equally (a lapsed group books neither), so the projected loss
ratio is unaffected. New business is the same `GroupProjection` with
`group_pmpm = book_pmpm`, `credibility = 0`, and `renewal_prob = close_ratio`.

## Quick start

```python
import numpy as np, pandas as pd
from projectionmodels import GroupProjection, BookProjection

g = GroupProjection(
    prospective_membership=np.full(12, 1900.0), seasonal_factors=season_factors,
    current_premium=4_500_000, current_member_months=21_600,
    rate_action=0.06, plan_change=-0.02,
    book_pmpm=180.0, claim_trend=0.06,
    exp_midpoint=pd.Timestamp("2025-07-01"), prosp_midpoint=pd.Timestamp("2027-07-01"),
    group_claims=3_800_000, group_member_months=21_600, group_claim_count=6_000,
    pooling_pmpm=8.0, renewal_prob=0.90)   # renewal likelihood from underwriting

book = BookProjection([g, ...], labels=["GroupA", ...])
book.loss_ratio          # expected book loss ratio
book.by_group            # per-group expected premium / claims / LR
book.monthly             # book premium & claims by month
```

## Built on `actuarialpy`

This package adds no primitives of its own — it composes existing `actuarialpy`
primitives and depends only downward:

- `credibility_weighted_estimate` — the credibility blend, `Z·group + (1−Z)·book`
- `midpoint_trend_factor` — trend to the prospective midpoint
- `seasonality_factors` / `apply_seasonality` — monthly seasonality
- `pure_premium` — PMPM

`pool_losses` / `excess_over_threshold` in `actuarialpy` cap large claims when you
derive the pooling PMPM upstream.
