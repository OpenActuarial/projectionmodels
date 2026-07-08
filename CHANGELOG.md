# Changelog

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
