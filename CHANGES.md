# OpenActuarial — `projectionmodels` (and why `actuarialpy` is unchanged)

## `actuarialpy`: no changes needed

Reading the real source settled this: `actuarialpy` already has every primitive
`projectionmodels` needs, so nothing is added.

- credibility blend → `credibility_weighted_estimate(observed, complement, z)`
- large-loss pooling → `pool_losses` / `excess_over_threshold` / `retention_for_target_cv`
- trend, seasonality, PMPM → `midpoint_trend_factor`, `seasonality_factors` /
  `apply_seasonality`, `pure_premium`

The `persistency` module that was briefly added has been **removed**. Renewal
likelihood at your shop is supplied by underwriting — it is an input, not a
calculation — so a model/fit for it is dead weight. (The fit I had written was also
a naive OLS of renewal on rate change; the honest version, if you ever did model
it, is a retention study or a logistic fit, not that.) So the earlier
`actuarialpy_0.42_persistency.patch` is withdrawn; your `actuarialpy` repo needs no
edit.

## `projectionmodels` v0.1.0 — the deliverable

A workflow package beside `ratingmodels`/`lossmodels`, `src/` layout matching
`actuarialpy` (hatchling, `packages = ["src/projectionmodels"]`,
`testpaths = ["tests"]`). It adds no primitives — it composes `actuarialpy`'s.

- `PMPMProjection` — credibility-blended, trended, pooled claims PMPM
- `PremiumRollforward` — stored premium rolled forward (not rebuilt from experience)
- `GroupProjection` — one group: premium + claims + renewal weighting (loop unit)
- `BookProjection` — aggregate in-force renewals + new business

**Renewal likelihood is a supplied input**: pass `renewal_prob=<number>` (e.g. from
underwriting) to `GroupProjection`. It weights expected premium and claims (a lapsed
group books neither); the loss ratio is unaffected. New business uses the same call
with `credibility=0` and `renewal_prob=close_ratio`.

12 tests pass; `examples/demo.py` runs after install. Depends on `actuarialpy>=0.40`.

```bash
cd projectionmodels
pip install -e ".[dev]"
pytest -q                       # 12 passed
git init && git add . && git commit -m "projectionmodels 0.1.0"
```
