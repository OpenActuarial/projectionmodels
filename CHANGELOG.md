# Changelog

## [0.6.4] - 2026-07-10

### Changed
- The `actuarialpy` dependency is now an open floor (`>=0.41`) instead of a
  capped range (`>=0.41.0,<0.45.0`). The cap made every `actuarialpy` minor
  release a resolver conflict until a coordinated re-release here; the
  ecosystem policy is now open floors, with the nightly ecosystem smoke
  workflow catching runtime drift. CI keeps a pinned lane at the 0.41.0
  floor edge.

## [0.6.3] - 2026-07-10

Documentation release — no functional changes.

### Added

- **Class docstrings for `GroupProjection`, `BookProjection`,
  `PMPMProjection`, and `PremiumRollforward`.** Sphinx
  `automodule :members:` silently omits objects without a docstring, so the
  0.6.2 group/book family was absent from the API reference on
  openactuarial.org despite being fully exported.
- A guard test (`tests/test_public_api_docstrings.py`) asserting every
  function and class in `__all__` carries a docstring, turning a silent
  autodoc omission into a CI failure.

## [0.6.2] - 2026-07-10

Entry written retroactively in 0.6.3 — 0.6.2 shipped without one.

### Added

- **The group/book projection family**: `GroupProjection` (one group's
  premium and claims projected together on a shared monthly membership,
  renewal-weighted), `BookProjection` (group projections rolled up to book
  totals, per-group and monthly), `PMPMProjection` (the
  credibility-blend → trend → plan → pooling claims engine),
  `PremiumRollforward` (the stored premium rolled forward by rate action
  and plan change), and `new_business` (the sold-but-new case: fully manual
  claims, close ratio in the role of renewal probability).
- `examples/demo.py`.

### Changed

- `__version__` now resolves via `importlib.metadata`, making
  `pyproject.toml` the single source of truth for the version.
- CI: Windows added to the OS matrix, and the actuarialpy lanes
  restructured to a `latest-allowed` default plus pinned range edges
  (0.41.0 and 0.44.0) that cannot go stale when the pin moves.

## [0.6.1] - 2026-07-09

Compatibility release for actuarialpy 0.43/0.44 — no functional changes.

### Changed

- **actuarialpy cap raised to `<0.45.0`** (was `<0.43.0`). actuarialpy
  0.43–0.44 moved presentation and I/O helpers (`UnderwritingSummary`,
  the `profiles` module, `to_excel_report`) into `experiencestudies`;
  every primitive `projectionmodels` calls (`completion_factors`,
  `fit_trend`, `seasonality_factors`, `limited_fluctuation_z`,
  `per_exposure`, and the apply/deseasonalize helpers) is unchanged.
  The full test suite passes against 0.44.
- The installed-package guard test now accepts actuarialpy 0.43/0.44.

## [0.6.0] - 2026-07-08

### Changed

- **Domain-agnostic vocabulary across the workflow API**, matching
  `ClaimExperience` and the rest of the OpenActuarial ecosystem. Renames
  (old → new): `membership` → `exposure`, `membership_col` (default
  `"member_months"`) → `exposure_col` (default `"exposure"`), and
  `membership_period_col` → `exposure_period_col` on `ClaimProjection`,
  `PremiumProjection`, and `ExpenseProjection`; output measure `claim_pmpm`
  → `claims_per_exposure`; calculation `premium_pmpm` →
  `projected_premium_rate`; `current_rate_col` default
  `"current_premium_pmpm"` → `"current_premium_rate"`; expense basis
  `"pmpm"` → `"per_exposure"`. Domain units are named through
  `exposure_col` — the health examples pass
  `exposure_col="member_months"` and keep PMPM in presentation labels only.
  Purely a rename: results are numerically identical.

## [0.5.0] - 2026-07-08

### Fixed

- **Credibility complements are no longer trended from the experience
  midpoint.** `ClaimProjection` previously blended the experience rate with
  the complement and then trended the blended rate, silently restating the
  complement at experience-period cost level (a zero-credibility projection
  returned the manual times the full trend factor). Blending now happens at
  the prospective midpoint by default: experience is trended to the blend
  basis, blended with the complement **as stated**, and the blended rate is
  trended only from the basis to each projection period. Results change for
  any projection with credibility below 1 and a nonzero trend; full
  credibility results are unchanged.
- The real-dependency guard test no longer asserts an install-path substring
  ("site-packages"), which failed on Debian/Ubuntu layouts. It now verifies
  the imported module matches importlib's resolved spec.
- Unsegmented seasonality (a plain season/factor table with no segment
  lookups) now works in projections. `apply_seasonality` accepts a tidy
  per-segment table joined on `by` plus season, or a flat Series indexed by
  season; the projection previously always passed a DataFrame reconstructed
  from the expanded frame, which the unsegmented path rejects. It now passes
  the assumption's own table — as a flat Series when there are no segment
  columns — mirroring the deseasonalize step on the experience side.

### Added

- `ClaimProjection.complement_basis` — declares the cost level of the
  complement: `"prospective"` (default), `"experience"` (pre-0.5.0
  behaviour), or an explicit as-of date.
- `ClaimProjection.rate_loads` — flat per-exposure loads (for example a pooling
  charge) added to the projected rate as stated, after seasonality and
  outside the credibility blend. Loads register as assumptions, so scenarios
  can adjust them.
- `ProjectionHorizon.midpoint` — the mean period midpoint, used as the
  default blend basis.
- `trended_experience_rate` audit column between the experience rate and the
  blend.
- `examples/pooled_claims.py` — large-claim pooling composed ahead of
  `ClaimExperience` with `actuarialpy.pool_losses`, the charge carried as a
  rate load.
- Hand-pinned regression tests covering zero credibility, blend-basis
  invariance at full credibility, the legacy ordering as an explicit opt-in,
  partial-credibility levels, and load application.

All notable changes to this project are documented here.

## [Unreleased]

## [0.4.0] - 2026-07-07

### Changed

- Simplified the package root around claim, premium, expense, date, scenario,
  and result workflows.
- Moved the generic formula and roll-forward engine to
  `projectionmodels.advanced`.
- Moved actuarialpy-based estimation to the explicit
  `projectionmodels.integrations.actuarialpy` namespace.
- Rewrote the date-cohort and underwriting examples to use concrete workflows.
- Moved general, member-level, and generic sensitivity examples under
  `examples/advanced`.
- Kept temporary deprecated access to 0.3 advanced root names for migration.

### Added

- Public API surface regression tests.
- Compatibility testing for actuarialpy 0.41 and 0.42.
- Integration helper for removing selected seasonality before fitting trend.
- Concrete-workflow underwriting example combining claims, premium, and
  expenses without the generic model DSL.

## [0.3.0] - 2026-07-07

### Fixed

- Replaced the invalid `actuarialpy>=0.42.1` requirement with the tested compatibility range `actuarialpy>=0.41.0,<0.42.0`.
- Removed the session-wide fake `actuarialpy` fixture. The complete test suite now imports and exercises the installed package.
- Rebuilt completion tests and examples with valid triangles containing overlapping origin and development periods.
- Fixed row-wise `TrendAssumption.factor()` calls by coercing projection durations to a numeric vector accepted by the real `actuarialpy.trend_factor` API.
- Fixed ungrouped completion application by passing a development-indexed Series to `actuarialpy.apply_completion`.
- Fixed Bühlmann–Straub grouping for one or several risk keys by using stable integer risk identifiers instead of tuple-valued group labels.
- Fixed weighted experience midpoints under pandas datetime units; projected trend periods no longer start near 1970.
- Validate seasonality normalization separately for each assumption segment.
- Provide a clearer validation error when supplied experience cannot support completion-factor estimation.

### Added

- `PremiumProjection` for premium-rate roll-forwards using supplied membership.
- `RenewalRateActions` for one-time, effective-dated rate-action schedules.
- Recurring rate actions that apply at each record's renewal anniversary and persist in subsequent periods.
- Real integration assertions comparing projectionmodels wrappers with direct `actuarialpy` results.
- Regression tests for renewal timing, persistent premium changes, one-time schedules, repeated renewals, and partial-period exposure.
- `examples/renewal_rate_actions.py`.
- CI execution of every example and a clean built-wheel installation test.

### Changed

- Reframed the public documentation around concrete claim, premium, membership, expense, and renewal workflows. The general calculation engine is documented as an advanced API.
- Clarified that `from_experience()` constructors are thin `actuarialpy` adapters and do not replace experience-study diagnostics or exhibits.

## [0.2.1] - 2026-07-07

### Added

- Expanded tests and runnable examples.

### Changed

- Added automatic numerator and denominator resolution for recalculated metrics.

### Known issues

- This version used a session-wide fake `actuarialpy` module and declared an unavailable dependency version. It should not be used.

## [0.2.0] - 2026-07-07

### Changed

- Replaced the original health-group budgeting prototype with a general deterministic projection engine.

## [0.1.0] - 2026-01-01

- Initial prototype release.
