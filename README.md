# projectionmodels

The renewal cycle end to end: premium, claims, and expenses projected over a horizon.

[![CI](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/projectionmodels)](https://pypi.org/project/projectionmodels/)
[![Python](https://img.shields.io/pypi/pyversions/projectionmodels)](https://pypi.org/project/projectionmodels/)

## Overview

`projectionmodels` projects premium, claims, and expenses over a monthly
horizon on exposure you supply — by group, by claim type, and at cost levels
that make the pipeline order (completion, then trend, then adjustments)
explicit rather than implicit.

Assumptions are first-class objects: estimate them from history with
`actuarialpy` through the built-in adapter, or state them directly and keep
the projection fully reproducible either way.

## Installation

```bash
pip install projectionmodels
```

Requires Python 3.10 or newer.

## Quick start

```python
import pandas as pd
import projectionmodels as pm

premium_data = pd.DataFrame({
    "group_id": ["A", "B"],
    "renewal_date": pd.to_datetime(["2027-03-01", "2027-07-01"]),
    "current_premium_rate": [100.0, 100.0],
    "rate_action": [0.10, 0.20],
})

periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
exposure = pd.DataFrame(
    {"group_id": g, "projection_period": p, "member_months": 1_000.0}
    for g in ("A", "B") for p in periods
)

results = pm.PremiumProjection(
    premium_data=premium_data,
    projection_keys=["group_id"],
    exposure=exposure,
    exposure_col="member_months",
    horizon=pm.ProjectionHorizon("2027-01-01", periods=12),
    recurring_rate_action_col="rate_action",
).project()

print(results.to_frame().head())
```

## What's inside

- **Premium** — renewal-date-aware rate projection with recurring rate
  actions.
- **Claims** — projection by claim type with explicit cost levels and
  pipeline order (complete, trend, adjust).
- **Expenses** — fixed and variable expense projection alongside the claim
  stream.
- **Assumptions** — assumption objects estimated from history via the
  `actuarialpy` adapter or supplied directly.
- **Book and group** — the same machinery at single-group and whole-book
  level, with results tables by period, group, and component.
- **Advanced** — extension points for custom models and integrations.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/projectionmodels.html](https://openactuarial.org/projectionmodels.html)**.

## The OpenActuarial ecosystem

`projectionmodels` is one of eight packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| [actuarialpy](https://github.com/OpenActuarial/actuarialpy) | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| **[projectionmodels](https://github.com/OpenActuarial/projectionmodels)** | Claim, premium, and expense projection over a renewal horizon |
| [ratingmodels](https://github.com/OpenActuarial/ratingmodels) | Manual and experience rating, credibility, indication, GLM relativities |
| [reservingmodels](https://github.com/OpenActuarial/reservingmodels) | Claims development and stochastic reserving: chain ladder, BF, Mack, ODP bootstrap |
| [lossmodels](https://github.com/OpenActuarial/lossmodels) | Severity and frequency fitting, aggregate loss distributions |
| [extremeloss](https://github.com/OpenActuarial/extremeloss) | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| [risksim](https://github.com/OpenActuarial/risksim) | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/projectionmodels
cd projectionmodels
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).
