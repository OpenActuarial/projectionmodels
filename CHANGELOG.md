# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.2.1] - 2026-07-07

### Added

- Expanded the suite to 125 tests covering assumptions, adjustments, lifecycle dates, projection execution, result aggregation, claim workflows, expense workflows, and adapter failures.
- Added output-level regression tests for every runnable example.
- Added examples for actuarialpy-estimated assumptions, member-level projections, systematic sensitivities, and underwriting-ready results.
- Added a 95% coverage threshold to continuous integration.

### Changed

- Reworked all examples around a callable `run_example()` function while preserving direct script execution.
- `ProjectionResults.summarize()` now automatically resolves numerator and denominator dependencies when only a recalculated metric is requested.

## [0.2.0] - 2026-07-07

### Changed

- Replaced the original health-group budgeting prototype with a general deterministic projection engine.
- Projection grain is now selected with `projection_keys` and optional `component_keys`.
- Assumption lookup grain, scenario filters, and result-reporting grain are independent.

### Added

- `ProjectionData`, `ProjectionDataset`, and keyed supporting tables.
- Monthly, quarterly, and annual `ProjectionHorizon` objects.
- General `ProjectionModel`, `RollForward`, `Calculation`, `CashFlow`, and `Metric` APIs.
- Supplied or `actuarialpy`-estimated trend, seasonality, completion, and credibility assumptions.
- Claim-type projection workflow with supplied membership.
- PMPM, fixed, percent-of-premium, and percent-of-claims expense projections.
- Date cohorts, lifecycle dates, renewal-period flags, and partial-period exposure.
- Scenario adjustments and systematic sensitivities.
- Grain-aware result aggregation, scenario comparison, and assumption/adjustment audits.

### Removed

- `PMPMProjection`, `PremiumRollforward`, `GroupProjection`, and `BookProjection`.

## [0.1.0] - 2026-01-01

- Initial prototype release.
