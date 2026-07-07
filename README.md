# projectionmodels

[![CI](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/projectionmodels/actions/workflows/ci.yml)

Deterministic actuarial projections at a caller-selected tabular grain.

`projectionmodels` advances members, policies, groups, products, cohorts, rating cells, or other actuarial projection records through time. It coordinates the primitives in [`actuarialpy`](https://github.com/OpenActuarial/actuarialpy) without duplicating them.

## Design

The package keeps four grains independent:

- **Projection grain** — the records projected independently, selected with `projection_keys` and optional `component_keys`.
- **Assumption grain** — the columns used to look up trend, seasonality, credibility, completion, or another assumption.
- **Adjustment grain** — scenario filters selecting the records and periods to modify.
- **Reporting grain** — the columns supplied to `ProjectionResults.summarize()`.

For example, a model can project at `group_id × product_id × claim_type`, use trend at `product_id × claim_type`, apply an adjustment to one `group_id`, and report by product and year.

## Install

```bash
pip install projectionmodels
```

## General projection engine

```python
import pandas as pd
import projectionmodels as pm

records = pm.ProjectionData(
    pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "product_id": ["PPO", "HMO"],
            "current_members": [1_000.0, 600.0],
            "current_premium_pmpm": [525.0, 475.0],
        }
    ),
    projection_keys=["group_id", "product_id"],
)

horizon = pm.ProjectionHorizon(
    start="2027-01-01",
    periods=36,
    frequency="monthly",
)

model = pm.ProjectionModel(
    assumptions=pm.AssumptionSet(
        pm.Assumption("retention_rate", 0.995),
        pm.TrendAssumption.from_values("premium_trend", 0.05),
    ),
    roll_forwards=[
        pm.RollForward(
            "members",
            initial="current_members",
            formula=lambda x: x.prior("members") * x["retention_rate"],
            grain=["group_id", "product_id"],
        ),
        pm.RollForward(
            "premium_pmpm",
            initial="current_premium_pmpm",
            formula=lambda x: x.prior("premium_pmpm")
            * (1 + x["premium_trend"]) ** x.year_fraction,
            grain=["group_id", "product_id"],
        ),
    ],
    calculations=[
        pm.CashFlow(
            "premium",
            formula=lambda x: x["members"] * x["premium_pmpm"],
            reporting_role="revenue",
            grain=["group_id", "product_id"],
        )
    ],
)

results = model.project(records, horizon)
annual = results.summarize(by=["calendar_year", "product_id"])
```

## Claim projection by claim type

Claim experience may be completed, deseasonalized, credibility blended, trended, reseasonalized, and multiplied by supplied membership.

```python
experience = pm.ClaimExperience(
    data=claim_history,
    projection_keys=["group_id", "product_id"],
    claim_type_col="claim_type",
    date_col="incurred_month",
    claims_col="reported_claims",
    exposure_col="member_months",
    valuation_date="2026-12-31",
)

completion = pm.CompletionAssumption.from_experience(
    "claim_completion",
    claim_transactions,
    by=["product_id", "claim_type"],
    origin_col="incurred_month",
    valuation_col="paid_month",
    amount_col="paid_claims",
)

seasonality = pm.SeasonalityAssumption.from_experience(
    "claim_seasonality",
    claim_history,
    by=["product_id", "claim_type"],
    date_col="incurred_month",
    value_col="completed_claims",
    exposure_col="member_months",
)

trend = pm.TrendAssumption.from_experience(
    "claim_trend",
    deseasonalized_history,
    by=["product_id", "claim_type"],
    date_col="incurred_month",
    value_col="deseasonalized_claims",
    exposure_col="member_months",
)

credibility = pm.CredibilityAssumption.from_experience(
    "claim_credibility",
    base_experience,
    method="limited_fluctuation",
    by=["group_id", "product_id", "claim_type"],
    exposure_col="claim_count",
    full_credibility_standard=1_082,
)

projection = pm.ClaimProjection.from_experience(
    experience,
    completion=completion,
    seasonality=seasonality,
    trend=trend,
    credibility=credibility,
    complement=pm.Assumption(
        "manual_claim_rate",
        manual_rates,
        lookup=["product_id", "claim_type"],
        value_col="manual_claim_pmpm",
    ),
    membership=membership,
    horizon=pm.ProjectionHorizon("2027-01-01", periods=60),
)

results = projection.project()
```

`trend`, `seasonality`, `completion`, and `credibility` can instead be supplied directly with each class's `from_values()` or `from_weights()` constructor. Both paths create the same resolved assumption interface.

## Scenarios and adjustments

```python
adverse = pm.Scenario(
    "adverse",
    adjustments=[
        pm.Adjustment(
            name="Higher inpatient trend",
            target="claim_trend",
            method="add",
            value=0.02,
            filters={"claim_type": "inpatient"},
        ),
        pm.Adjustment(
            name="Group A premium change",
            target="premium_pmpm",
            method="multiply",
            value=1.08,
            filters={"group_id": "A"},
            effective_from="2027-07-01",
        ),
    ],
)

results = model.project(records, horizon, scenarios=[pm.Scenario("baseline"), adverse])
comparison = results.compare_scenarios(
    baseline="baseline",
    comparison="adverse",
    by=["calendar_year", "product_id"],
)
```

Adjustment methods are `set`, `add`, `multiply`, `floor`, and `cap`. Calculated outputs are not adjustable unless their definition explicitly sets `adjustable=True`.

## Membership and independently grained tables

Supporting tables declare their own keys and are broadcast with validated many-to-one joins.

```python
dataset = pm.ProjectionDataset(records)
dataset.add_table(
    "membership",
    membership,
    keys=["group_id", "product_id", "projection_period"],
)
```

Entity-level membership can be combined with claim-type rows without requiring the membership table to duplicate each claim type. Result measures carry grain metadata, so total member months are counted once when claim type is omitted from a summary.

## Expenses

`ExpenseProjection` supports:

- `pmpm`
- `fixed_monthly`
- `percent_premium`
- `percent_claims`

Each expense record can have its own trend and scenario adjustments.

## Dates, new business, and renewals

```python
records = pm.ProjectionData(
    frame,
    projection_keys=["group_id", "product_id"],
    dates=pm.ProjectionDates(
        entry_date="effective_date",
        exit_date="termination_date",
        renewal_date="next_renewal_date",
        exposure_timing="daily_prorated",
    ),
)

records = records.add_date_cohort(
    pm.DateCohort(
        name="business_origin",
        date_col="effective_date",
        split_date="2027-01-01",
        before_label="existing",
        on_or_after_label="new_business",
    )
)
```

Projected rows include `active_fraction`, `is_active`, `duration_month`, `duration_year`, and `is_renewal_period`. Date cohorts remain ordinary columns available to assumptions, scenarios, and result summaries.

## Results and exhibits

`projectionmodels` owns mathematical result aggregation and returns pandas DataFrames. It does not own exhibit formatting.

```python
summary = results.summarize(
    by=["scenario", "business_origin", "projection_period", "product_id"]
)
```

The summary can be passed to `experiencestudies.underwriting_summary` or another reporting layer. Ratios declared with `aggregation="recalculate"` are recomputed from summarized numerators and denominators rather than averaged. When a caller requests only the ratio, its numerator and denominator are summarized internally and omitted from the returned display unless they were also requested.

## Runnable examples

Every file in `examples/` exposes a `run_example()` function that returns its results and also runs directly as a script. The test suite executes both interfaces and checks numerical invariants.

| Example | Demonstrates |
|---|---|
| `general_projection.py` | General roll-forwards, a one-time renewal adjustment, scenario comparison, and audits |
| `health_claims.py` | Claim-type trends, seasonality, credibility, membership, and grain-aware PMPM summaries |
| `calculated_assumptions.py` | Completion, trend, seasonality, and credibility estimated through `actuarialpy` |
| `expenses.py` | PMPM, fixed, percent-of-premium, and percent-of-claims expenses |
| `date_cohorts.py` | Existing versus new business, partial-period exposure, and renewal flags |
| `member_level_projection.py` | Member-level expected-value projection with model-point weights |
| `sensitivity_analysis.py` | Systematic trend sensitivities and scenario comparisons |
| `underwriting_results.py` | Exhibit-ready premium, claims, expenses, margin, and loss-ratio output |

Run an example from the repository root:

```bash
PYTHONPATH=src python examples/health_claims.py
```

## Package boundary

- **actuarialpy** — actuarial mathematics and reusable primitives.
- **experiencestudies** — experience analyses and actuarial exhibits.
- **projectionmodels** — deterministic assumption orchestration, projection through time, scenarios, and result aggregation.
- **risksim** — stochastic simulation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest --cov=projectionmodels --cov-report=term-missing --cov-fail-under=95
ruff check src tests examples
python -m build
```

The API is alpha and intentionally does not preserve the original `GroupProjection` prototype.
