# projectionmodels

Focused actuarial projections of claims, premium, membership, and expenses.

The package is intentionally organized around concrete workflows. Most users
should not need to construct a calculation graph or define a custom state
engine.

## Installation

```bash
pip install projectionmodels
```

`projectionmodels` currently supports `actuarialpy>=0.41,<0.43` and Python
3.10–3.13.

## Public API

The package root contains the workflow objects most actuaries need:

```text
ClaimExperience        Prepare a base claim rate from experience
ClaimProjection        Project claim rates and claims by claim type
PremiumProjection      Roll premium forward, including renewal rate actions
RenewalRateActions     Supply effective-dated rate actions
ExpenseProjection      Project PMPM, fixed, premium-based, and claim-based expenses
ProjectionHorizon      Define monthly, quarterly, or annual projection periods
ProjectionDates        Define entry, exit, renewal, and experience date columns
DateCohort              Split records into existing/new or other date cohorts
Adjustment / Scenario  Run sensitivities and alternative assumptions
ProjectionResults      Summarize without averaging ratios or duplicating exposure
```

Lower-level modeling objects are available from `projectionmodels.advanced`,
but they are not part of the primary workflow.

## Premium at renewal

```python
import pandas as pd
import projectionmodels as pm

premium_data = pd.DataFrame(
    {
        "group_id": ["A", "B"],
        "renewal_date": pd.to_datetime(["2027-03-01", "2027-07-01"]),
        "current_premium_pmpm": [100.0, 100.0],
        "rate_action": [0.10, 0.20],
    }
)

periods = pd.period_range("2027-01", periods=12, freq="M").astype(str)
membership = pd.DataFrame(
    [
        {
            "group_id": group_id,
            "projection_period": period,
            "member_months": 1_000.0,
        }
        for group_id in ("A", "B")
        for period in periods
    ]
)

results = pm.PremiumProjection(
    premium_data=premium_data,
    projection_keys=["group_id"],
    membership=membership,
    horizon=pm.ProjectionHorizon("2027-01-01", periods=12),
    recurring_rate_action_col="rate_action",
).project()
```

Group A remains at $100 through February, increases to $110 in March, and
carries that rate forward. Group B increases to $120 in July.

For different actions at different renewals, provide an effective-dated table:

```python
actions = pm.RenewalRateActions(
    pd.DataFrame(
        {
            "group_id": ["A", "A", "B"],
            "effective_date": pd.to_datetime(
                ["2027-03-01", "2028-03-01", "2027-07-01"]
            ),
            "rate_action": [0.10, 0.06, 0.20],
        }
    ),
    projection_keys=["group_id"],
)
```

## Claims by claim type

```python
experience = pm.ClaimExperience(
    claims,
    projection_keys=["group_id", "product_id"],
    claim_type_col="claim_type",
    date_col="incurred_month",
    claims_col="reported_claims",
    exposure_col="member_months",
    valuation_date="2026-12-31",
)

projection = pm.ClaimProjection.from_experience(
    experience,
    membership=membership,
    horizon=pm.ProjectionHorizon("2027-01-01", periods=36),
    completion=completion,
    trend=trend,
    seasonality=seasonality,
    credibility=credibility,
    complement=manual_rates,
)

results = projection.project()
```

Trend, seasonality, completion, and credibility may be supplied directly as
assumption tables.

### Cost levels and pipeline order

The claim workflow evaluates, in order: complete → deseasonalize → trend the
experience rate to the blend basis → credibility blend → trend from the basis
to each projection period → reseasonalize → add `rate_loads` → multiply by
membership.

The complement is used **as stated**. By default the blend basis is the
prospective midpoint of the horizon (`complement_basis="prospective"`), the
level at which manual and book rates are conventionally quoted — so a
zero-credibility projection reproduces the complement rather than a trended
copy of it. Set `complement_basis="experience"` if your complement is quoted
at experience-period cost level, or pass an explicit as-of date. Because the
month arithmetic is exactly additive, results at full credibility are
identical under every basis.

`rate_loads` (for example a pooling charge) are added to the projected rate
as stated: flat across periods, after seasonality, outside the blend.

## Estimating assumptions with actuarialpy

Estimation is explicit and separate from projection execution:

```python
from projectionmodels.integrations.actuarialpy import (
    estimate_completion,
    estimate_credibility,
    estimate_seasonality,
    estimate_trend,
)

completion = estimate_completion(
    "claim_completion",
    payment_history,
    by=["claim_type"],
    origin_col="incurred_month",
    valuation_col="paid_month",
    amount_col="paid_claims",
)

seasonality = estimate_seasonality(
    "claim_seasonality",
    completed_history,
    by=["claim_type"],
    date_col="incurred_month",
    value_col="completed_claims",
    exposure_col="member_months",
)

trend = estimate_trend(
    "claim_trend",
    deseasonalized_history,
    by=["claim_type"],
    date_col="incurred_month",
    value_col="deseasonalized_claims",
    exposure_col="member_months",
)

credibility = estimate_credibility(
    "claim_credibility",
    experience_history,
    method="limited_fluctuation",
    by=["group_id", "claim_type"],
    exposure_col="claim_count",
    full_credibility_standard=2_000,
)
```

The returned assumptions retain indicated values and diagnostics. An actuary can
replace the indication while preserving the audit trail:

```python
selected_trend = trend.select(selected_table, note="2027 pricing selection")
```

## Expenses

`ExpenseProjection` supports:

- `pmpm`
- `fixed_monthly`
- `percent_premium`
- `percent_claims`

Each expense type may have its own trend and projection component.

## Date handling

`ProjectionDates` supports entry, exit, renewal, issue, and experience dates.
Records can be inactive before entry or after exit, and exposure can be whole-
period or daily-prorated.

`DateCohort` adds reportable classifications such as existing versus new
business:

```python
records = pm.DateCohort(
    "business_origin",
    "effective_date",
    split_date="2027-01-01",
    before_label="existing",
    on_or_after_label="new_business",
).apply(records)
```

## Results

```python
summary = results.summarize(
    by=["scenario", "product_id", "calendar_year"],
    measures=["member_months", "premium", "projected_claims", "claim_pmpm"],
)
```

`ProjectionResults` retains measure grain. It counts exposure once when claim
type is removed from a summary and recalculates PMPMs and loss ratios from their
summed numerators and denominators.

## Advanced models

Custom deterministic roll-forwards remain available, but are deliberately moved
out of the primary namespace:

```python
import projectionmodels.advanced as pma

model = pma.ProjectionModel(...)
```

Use this only when the claim, premium, and expense workflows are insufficient.
The advanced API remains provisional while the concrete workflows stabilize.

## Examples

Primary examples:

```text
examples/health_claims.py
examples/pooled_claims.py
examples/calculated_assumptions.py
examples/renewal_rate_actions.py
examples/date_cohorts.py
examples/expenses.py
examples/underwriting_results.py
```

Custom-engine examples are under `examples/advanced/`.

## Testing

The test suite imports the installed `actuarialpy`; it does not replace it with a
session-wide fake. CI runs the tests, every example, package builds, and a clean
wheel-install smoke test.
