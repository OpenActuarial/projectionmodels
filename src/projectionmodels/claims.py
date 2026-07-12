"""Claim experience preparation and claim-type projection workflow.

The projection pipeline evaluates, in order: complete reported claims to
ultimate, remove seasonality, trend the experience rate to the credibility
blend basis, blend with the complement **as stated**, trend the blended rate
from the basis to each projection period, reapply seasonality, add flat
``rate_loads``, and multiply by exposure. The blend basis defaults to the
prospective midpoint of the horizon — the level at which manual and book
rates are conventionally quoted — so a zero-credibility projection reproduces
the complement rather than a trended copy of it.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from actuarialpy import Experience, resolve_date, single_role, single_role_or_none

from .actuarialpy_adapter import actuarialpy_function
from .adjustments import Scenario
from .assumptions import (
    Assumption,
    AssumptionSet,
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)
from .calculations import Calculation, CashFlow, Metric
from .data import ProjectionData, ProjectionDataset, ProjectionDates
from .exceptions import ValidationError
from .horizon import ProjectionHorizon
from .model import ProjectionModel
from .results import ProjectionResults


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _weighted_midpoint(dates: pd.Series, weights: pd.Series) -> pd.Timestamp:
    valid = dates.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NaT
    values = (
        pd.to_datetime(dates.loc[valid])
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy(dtype=float)
    )
    w = weights.loc[valid].to_numpy(dtype=float)
    return pd.to_datetime(int(np.average(values, weights=w)), unit="ns")


def _months_between(base: Any, target: Any, index: pd.Index) -> pd.Series:
    """Calendar month gap, with day fractions, between two date-likes.

    Accepts scalars or Series and returns a Series aligned to ``index``. The
    arithmetic is exactly additive — ``gap(a, b) + gap(b, c) == gap(a, c)``
    for any intermediate ``b`` — which keeps full-credibility projections
    invariant to the choice of blend basis.
    """

    def as_series(value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return pd.to_datetime(value)
        return pd.Series(pd.Timestamp(value), index=index)

    base_dates = as_series(base)
    target_dates = as_series(target)
    return (
        (target_dates.dt.year - base_dates.dt.year) * 12
        + (target_dates.dt.month - base_dates.dt.month)
        + (target_dates.dt.day - base_dates.dt.day) / 30.4375
    )


def _coerce_completion(value):
    if value is None or isinstance(value, CompletionAssumption):
        return value
    if isinstance(value, (pd.Series, dict)):
        series = pd.Series(value)
        if series.index.name is None:
            series.index.name = "development_month"
        return CompletionAssumption.from_values("completion", series)
    raise ValidationError(
        "completion must be a CompletionAssumption, or a Series/mapping of "
        "completion factors keyed by development month"
    )


def _coerce_seasonality(value):
    if value is None or isinstance(value, SeasonalityAssumption):
        return value
    if isinstance(value, (pd.Series, dict)):
        series = pd.Series(value)
        table = pd.DataFrame({"season": series.index, "factor": series.to_numpy()})
        return SeasonalityAssumption.from_values("seasonality", table, factor_col="factor")
    raise ValidationError(
        "seasonality must be a SeasonalityAssumption, or a Series/mapping of "
        "factors keyed by season; use the assumption object for keyed tables"
    )


def _coerce_trend(value):
    if value is None or isinstance(value, TrendAssumption):
        return value
    if np.isscalar(value) or isinstance(value, (pd.Series, dict)):
        return TrendAssumption.from_values("claim_trend", value)
    raise ValidationError(
        "trend must be a TrendAssumption, a scalar annual rate, or a "
        "Series/mapping of rates; use the assumption object for keyed tables"
    )


def _coerce_credibility(value):
    if value is None or isinstance(value, CredibilityAssumption):
        return value
    if np.isscalar(value) or isinstance(value, (pd.Series, dict)):
        return CredibilityAssumption.from_weights("credibility", value)
    raise ValidationError(
        "credibility must be a CredibilityAssumption, a scalar weight, or a "
        "Series/mapping of weights; use the assumption object for keyed tables"
    )


def prepare_experience(
    exp: Experience,
    *,
    completion: CompletionAssumption | None = None,
    seasonality: SeasonalityAssumption | None = None,
    claims_col: str | None = None,
) -> pd.DataFrame:
    """Return experience with completed and deseasonalized claim columns.

    Takes the canonical :class:`actuarialpy.Experience` -- the bound expense,
    date, and exposure roles and the object's ``valuation_date`` fill the
    column plumbing -- and applies the projection assumptions in pipeline
    order (completion, then seasonality). The working column is carried to
    ``projectionmodels_adjusted_claims`` for :func:`base_rates`.
    """
    completion = _coerce_completion(completion)
    seasonality = _coerce_seasonality(seasonality)
    claims_col = claims_col if claims_col is not None else single_role(exp.expense, "expense")
    date_col = resolve_date(exp)
    valuation_date = exp.valuation_date

    work = exp.data.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    working_col = claims_col

    if completion is not None:
        if valuation_date is None and completion.development_col not in work.columns:
            raise ValidationError(
                "valuation_date is required to apply completion without a development column"
            )
        by = [
            column
            for column in completion.lookup
            if column != completion.development_col
        ]
        out_col = f"{claims_col}_completed"
        work = completion.apply(
            work,
            value_col=claims_col,
            date_col=(
                None if completion.development_col in work.columns else date_col
            ),
            valuation_date=valuation_date,
            development_col=(
                completion.development_col
                if completion.development_col in work.columns
                else None
            ),
            by=by or None,
            out_col=out_col,
        )
        working_col = out_col
    else:
        work[f"{claims_col}_completed"] = work[claims_col]
        working_col = f"{claims_col}_completed"

    if seasonality is not None:
        deseasonalize = actuarialpy_function("deseasonalize")
        by = [
            column
            for column in seasonality.lookup
            if column != seasonality.season_col
        ]
        out_col = f"{working_col}_deseasonalized"
        work = deseasonalize(
            work,
            seasonality.values,
            date_col=date_col,
            value_col=working_col,
            freq=seasonality.frequency,
            by=by or None,
            factor_col=seasonality.value_col or seasonality.name,
            season_name=seasonality.season_col,
            out_col=out_col,
        )
        working_col = out_col
    else:
        work[f"{working_col}_deseasonalized"] = work[working_col]
        working_col = f"{working_col}_deseasonalized"

    work["projectionmodels_adjusted_claims"] = work[working_col]
    return work


def base_rates(
    exp: Experience,
    *,
    grain: str | Iterable[str] | None = None,
    completion: CompletionAssumption | None = None,
    seasonality: SeasonalityAssumption | None = None,
    complement: Assumption | Any | None = None,
    extra_record_cols: Iterable[str] = (),
    claims_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate prepared experience to one row per projection record.

    ``grain`` names the record columns (projection keys plus the claim-type
    dimension) and defaults to the ``dimensions`` bound on the Experience.
    Produces ``experience_claim_rate``, the exposure-weighted
    ``experience_midpoint``, and -- when given -- ``complement_claim_rate``:
    the columns :class:`ClaimProjection` consumes.
    """
    grain_cols = list(_as_tuple(grain)) if grain is not None else list(exp.dimensions)
    if not grain_cols:
        raise ValidationError(
            "no record grain: bind dimensions=... on the Experience or pass grain=[...]"
        )
    exposure_col = single_role(exp.exposure, "exposure")
    date_col = resolve_date(exp)

    work = prepare_experience(
        exp, completion=completion, seasonality=seasonality, claims_col=claims_col
    )
    group_columns = grain_cols
    extra = list(extra_record_cols)
    for column in extra:
        if column not in work.columns:
            raise ValidationError(f"extra record column {column!r} is missing")
        variation = work.groupby(group_columns, dropna=False)[column].nunique(
            dropna=False
        )
        if (variation > 1).any():
            raise ValidationError(
                f"extra record column {column!r} is not constant within a projection record"
            )

    grouped = work.groupby(group_columns, dropna=False, sort=False)
    output = grouped.agg(
        adjusted_claims=("projectionmodels_adjusted_claims", "sum"),
        experience_exposure=(exposure_col, "sum"),
    ).reset_index()
    for column in extra:
        output = output.merge(
            grouped[column].first().rename(column).reset_index(),
            on=group_columns,
            how="left",
            validate="one_to_one",
        )

    midpoints = []
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_tuple, strict=True))
        row["experience_midpoint"] = _weighted_midpoint(
            part[date_col], part[exposure_col]
        )
        midpoints.append(row)
    output = output.merge(
        pd.DataFrame(midpoints),
        on=group_columns,
        how="left",
        validate="one_to_one",
    )
    per_exposure = actuarialpy_function("per_exposure")
    output["experience_claim_rate"] = per_exposure(
        output["adjusted_claims"], output["experience_exposure"]
    )

    if complement is not None:
        if isinstance(complement, Assumption):
            output["complement_claim_rate"] = complement.resolve(output)
        elif np.isscalar(complement):
            output["complement_claim_rate"] = complement
        else:
            raise ValidationError(
                "complement must be an Assumption or scalar; use Assumption for keyed tables"
            )
    return output


@dataclass
class ClaimProjection:
    """Project credibility-blended claim rates onto supplied exposure.

    Exposure is whatever unit the book uses — member-months, policy
    months, earned car-years — supplied by projection key and period and
    named with ``exposure_col``.

    Pipeline, in order:

    1. ``experience_claim_rate`` is trended from each record's
       ``experience_midpoint`` to the blend basis
       (``trended_experience_rate``).
    2. The trended experience is credibility blended with
       ``complement_claim_rate`` **as stated** (``credible_claim_rate``).
    3. The blended rate is trended from the blend basis to each period's
       midpoint (``trended_claim_rate``).
    4. Seasonality redistributes within the year and ``rate_loads`` are added,
       flat and outside the blend (``projected_claim_rate``).
    5. Rates are multiplied by exposure (``projected_claims``).

    Cost levels — ``complement_basis`` declares the level at which the
    complement is quoted:

    * ``"prospective"`` (default): the horizon's mean period midpoint, the
      conventional level for manual and book rates. Zero credibility
      therefore reproduces the complement as stated.
    * ``"experience"``: the record's experience midpoint, so the complement is
      trended alongside experience (the pre-0.5.0 behaviour).
    * an explicit date: any other as-of level.

    Because the calendar month arithmetic is exactly additive, projections at
    full credibility are identical under every basis.

    ``rate_loads`` (for example a pooling charge) are Assumptions or scalars
    quoted at prospective level; they are added to the projected rate as
    stated, per period, after seasonality, and are not credibility weighted.
    """

    base_rates: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    claim_type_col: str
    exposure: pd.DataFrame
    horizon: ProjectionHorizon
    trend: TrendAssumption
    seasonality: SeasonalityAssumption | None = None
    credibility: CredibilityAssumption | None = None
    complement_basis: str | pd.Timestamp = "prospective"
    rate_loads: Any = ()
    exposure_col: str = "exposure"
    exposure_period_col: str = "projection_period"
    dates: ProjectionDates | None = None
    additional_assumptions: tuple[Assumption, ...] | list[Assumption] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        self.projection_keys = _as_tuple(self.projection_keys)
        keys = [*self.projection_keys, self.claim_type_col]
        required = keys + ["experience_claim_rate", "experience_midpoint"]
        missing = [column for column in required if column not in self.base_rates.columns]
        if missing:
            raise ValidationError(f"base_rates is missing columns: {missing}")
        exposure_keys = [
            *self.projection_keys,
            self.exposure_period_col,
            self.exposure_col,
        ]
        missing_exposure = [
            column for column in exposure_keys if column not in self.exposure.columns
        ]
        if missing_exposure:
            raise ValidationError(
                f"exposure is missing columns: {missing_exposure}"
            )
        if self.credibility is not None and "complement_claim_rate" not in self.base_rates:
            raise ValidationError(
                "credibility requires complement_claim_rate in base_rates"
            )

        if isinstance(self.complement_basis, str):
            if self.complement_basis not in {"prospective", "experience"}:
                try:
                    self.complement_basis = pd.Timestamp(self.complement_basis)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        "complement_basis must be 'prospective', 'experience', or a date"
                    ) from exc
        else:
            self.complement_basis = pd.Timestamp(self.complement_basis)

        raw_loads = self.rate_loads
        if raw_loads is None:
            raw_loads = ()
        if isinstance(raw_loads, Assumption) or np.isscalar(raw_loads):
            raw_loads = (raw_loads,)
        loads: list[Assumption] = []
        for position, load in enumerate(raw_loads, start=1):
            if isinstance(load, Assumption):
                loads.append(load)
            elif np.isscalar(load):
                loads.append(Assumption(f"rate_load_{position}", float(load)))
            else:
                raise ValidationError(
                    "rate_loads entries must be Assumption objects or scalars"
                )
        self.rate_loads = tuple(loads)

    def _model(self) -> ProjectionModel:
        assumptions = AssumptionSet(self.trend)
        if self.seasonality is not None:
            assumptions.add(self.seasonality)
        if self.credibility is not None:
            assumptions.add(self.credibility)
        for item in self.additional_assumptions:
            assumptions.add(item)
        for load in self.rate_loads:
            assumptions.add(load)

        record_grain = self.projection_keys + (self.claim_type_col,)
        entity_grain = self.projection_keys

        if isinstance(self.complement_basis, pd.Timestamp):
            blend_basis: pd.Timestamp | None = self.complement_basis
        elif self.complement_basis == "experience":
            blend_basis = None
        else:  # "prospective"
            blend_basis = self.horizon.midpoint

        def trended_experience(context):
            if blend_basis is None:
                return context["experience_claim_rate"]
            trend_factor = actuarialpy_function("trend_factor")
            months = _months_between(
                context["experience_midpoint"], blend_basis, context.frame.index
            )
            return context["experience_claim_rate"] * trend_factor(
                context[self.trend.name], months
            )

        def credible_rate(context):
            observed = context["trended_experience_rate"]
            if self.credibility is None:
                return observed
            blend = actuarialpy_function("credibility_weighted_estimate")
            return blend(
                observed,
                context["complement_claim_rate"],
                context[self.credibility.name],
            )

        def trended_rate(context):
            trend_factor = actuarialpy_function("trend_factor")
            base = (
                context["experience_midpoint"] if blend_basis is None else blend_basis
            )
            months = _months_between(
                base, context["period_midpoint"], context.frame.index
            )
            return context["credible_claim_rate"] * trend_factor(
                context[self.trend.name], months
            )

        def seasonal_rate(context):
            if self.seasonality is None:
                return context["trended_claim_rate"]
            apply_seasonality = actuarialpy_function("apply_seasonality")
            by = [
                column
                for column in self.seasonality.lookup
                if column != self.seasonality.season_col
            ]
            # actuarialpy's contract: factors are either a tidy per-segment
            # table joined on by + season, or a flat Series indexed by season
            # when there are no segment columns. Use the assumption's own
            # table — the same one deseasonalize consumed on the experience
            # side — rather than reconstructing factors from the expanded
            # frame, which only ever contains the horizon's seasons.
            factor_col = self.seasonality.value_col or self.seasonality.name
            if by:
                factors = self.seasonality.values
            else:
                factors = self.seasonality.values.set_index(
                    self.seasonality.season_col
                )[factor_col]
            applied = apply_seasonality(
                context.frame.assign(
                    __projectionmodels_trended_rate__=context["trended_claim_rate"]
                ),
                factors,
                date_col="period_start",
                value_col="__projectionmodels_trended_rate__",
                freq=self.seasonality.frequency,
                by=by or None,
                factor_col=factor_col,
                season_name=self.seasonality.season_col,
                out_col="__projectionmodels_projected_rate__",
            )
            return applied["__projectionmodels_projected_rate__"]

        def projected_rate(context):
            value = seasonal_rate(context)
            for load in self.rate_loads:
                value = value + context[load.name]
            return value

        calculations = [
            Calculation(
                "trended_experience_rate",
                formula=trended_experience,
                aggregation="mean",
                grain=record_grain,
            ),
            Calculation(
                "credible_claim_rate",
                formula=credible_rate,
                aggregation="mean",
                grain=record_grain,
                depends_on=("trended_experience_rate",),
            ),
            Calculation(
                "trended_claim_rate",
                formula=trended_rate,
                aggregation="mean",
                grain=record_grain,
                depends_on=("credible_claim_rate",),
            ),
            Calculation(
                "projected_claim_rate",
                formula=projected_rate,
                aggregation="mean",
                grain=record_grain,
                depends_on=("trended_claim_rate",),
            ),
            Calculation(
                self.exposure_col,
                formula=lambda c: c[self.exposure_col] * c["active_fraction"],
                aggregation="sum",
                grain=entity_grain,
                reporting_role="exposure",
            ),
            CashFlow(
                "projected_claims",
                formula=lambda c: c["projected_claim_rate"]
                * c[self.exposure_col],
                aggregation="sum",
                grain=record_grain,
                reporting_role="loss",
                depends_on=("projected_claim_rate", self.exposure_col),
            ),
            Metric(
                "claims_per_exposure",
                formula=lambda c: c["projected_claims"] / c[self.exposure_col],
                aggregation="recalculate",
                numerator="projected_claims",
                denominator=self.exposure_col,
                grain=record_grain,
                depends_on=("projected_claims", self.exposure_col),
            ),
        ]
        return ProjectionModel(assumptions=assumptions, calculations=calculations)

    def project(
        self,
        *,
        scenarios: Scenario | Iterable[Scenario] | None = None,
    ) -> ProjectionResults:
        records = ProjectionData(
            self.base_rates.copy(),
            projection_keys=self.projection_keys,
            component_keys=[self.claim_type_col],
            dates=self.dates,
        )
        dataset = ProjectionDataset(records)
        exposure = self.exposure.rename(
            columns={self.exposure_period_col: "projection_period"}
        )
        dataset.add_table(
            "exposure",
            exposure,
            keys=[*self.projection_keys, "projection_period"],
        )
        return self._model().project(
            dataset,
            self.horizon,
            scenarios=scenarios,
        )

def project(
    exp: Experience,
    *,
    exposure: pd.DataFrame,
    horizon: ProjectionHorizon,
    trend: TrendAssumption,
    seasonality: SeasonalityAssumption | None = None,
    credibility: CredibilityAssumption | None = None,
    completion: CompletionAssumption | None = None,
    complement: Assumption | Any | None = None,
    complement_basis: str | pd.Timestamp = "prospective",
    rate_loads: Any = (),
    grain: str | Iterable[str] | None = None,
    claim_type: str | None = None,
    extra_record_cols: Iterable[str] = (),
    claims_col: str | None = None,
    exposure_col: str | None = None,
    exposure_period_col: str = "projection_period",
    dates: ProjectionDates | None = None,
) -> ClaimProjection:
    """Build a :class:`ClaimProjection` from the canonical Experience.

    The single projection entrypoint: takes an :class:`actuarialpy.Experience`
    plus only the concepts this package owns -- the assumptions, the horizon,
    and the prospective exposure. Named parameters *are* the pipeline phases;
    ordering (complete, deseasonalize, aggregate, trend / blend) is fixed and
    never inferred from a list.

    ``grain`` defaults to the Experience's bound ``dimensions``. Of the grain
    columns, the ones present in ``exposure`` are the projection keys (rates
    join to exposure on them); the one absent is the claim-type dimension
    (rates vary by it, exposure does not). Pass ``claim_type=`` when the split
    is ambiguous.
    """
    from actuarialpy import ExperienceSet

    if isinstance(exp, ExperienceSet):
        exp = exp.tab
    # A wide Experience (built with a wide_by= Source spec) melts itself:
    # the recorded pivot has one structural inverse.
    if exp.pivots:
        recorded = {p.by: p for p in exp.pivots}
        if claim_type is not None and claim_type in recorded:
            chosen = recorded[claim_type]
        elif len(recorded) == 1:
            chosen = next(iter(recorded.values()))
        else:
            raise ValidationError(
                f"multiple recorded pivots {sorted(recorded)}; pass claim_type= "
                "naming the one to project by"
            )
        if claims_col is not None and claims_col != chosen.value:
            raise ValidationError(
                f"claims_col={claims_col!r} is not the recorded pivot's value "
                f"column ({chosen.value!r}); melt or reshape explicitly before "
                "projecting a non-pivot measure"
            )
        other_expense = [c for c in exp.expense if c not in chosen.columns]
        if other_expense:
            warnings.warn(
                f"claim projection uses the {chosen.by!r} pivot columns "
                f"{list(chosen.columns)}; other expense columns {other_expense} "
                "are excluded",
                stacklevel=2,
            )
        exp = exp.melt(chosen.by)
        claim_type = chosen.by
        claims_col = chosen.value

    if isinstance(completion, str):
        raise ValidationError(
            "completion cannot be estimated from the bound history alone -- it "
            "needs origin/valuation development columns. Call "
            "projectionmodels.integrations.actuarialpy.estimate_completion(...) "
            "and pass the resulting assumption."
        )
    if isinstance(seasonality, str):
        if seasonality != "estimate":
            raise ValidationError(
                f"unknown seasonality sentinel {seasonality!r}; use 'estimate', "
                "a SeasonalityAssumption, or a Series/mapping of factors"
            )
        from .integrations.actuarialpy import estimate_seasonality

        season_value = claims_col if claims_col is not None else single_role(exp.expense, "expense")
        season_by = [claim_type] if claim_type is not None and claim_type in exp.data.columns else None
        seasonality = estimate_seasonality(
            "seasonality",
            exp.data,
            date_col=resolve_date(exp),
            value_col=season_value,
            exposure_col=single_role_or_none(exp.exposure),
            by=season_by,
        )

    trend = _coerce_trend(trend)
    completion = _coerce_completion(completion)
    seasonality = _coerce_seasonality(seasonality)
    credibility = _coerce_credibility(credibility)
    if exposure_col is None:
        bound = single_role(exp.exposure, "exposure")
        if bound in exposure.columns:
            exposure_col = bound
        elif "exposure" in exposure.columns:
            exposure_col = "exposure"
        else:
            raise ValidationError(
                f"no exposure column found in the exposure frame (looked for the "
                f"bound role {bound!r} and 'exposure'); pass exposure_col=... -- "
                f"the frame has columns {list(exposure.columns)}"
            )
    grain_cols = list(_as_tuple(grain)) if grain is not None else list(exp.dimensions)
    if not grain_cols:
        raise ValidationError(
            "no record grain: bind dimensions=... on the Experience or pass grain=[...]"
        )
    if claim_type is None:
        absent = [column for column in grain_cols if column not in exposure.columns]
        if len(absent) == 1:
            claim_type = absent[0]
        elif not absent:
            raise ValidationError(
                "every grain column appears in exposure; pass claim_type= to name "
                "the rate dimension exposure does not vary by"
            )
        else:
            raise ValidationError(
                f"grain columns {absent} are absent from exposure; pass claim_type= "
                "to name the claim-type dimension (the others must join to exposure)"
            )
    elif claim_type not in grain_cols:
        raise ValidationError(f"claim_type {claim_type!r} is not a grain column {grain_cols}")
    projection_keys = [column for column in grain_cols if column != claim_type]
    if not projection_keys:
        raise ValidationError(
            "grain needs at least one projection key besides the claim-type dimension"
        )

    rates = base_rates(
        exp,
        grain=[*projection_keys, claim_type],
        completion=completion,
        seasonality=seasonality,
        complement=complement,
        extra_record_cols=extra_record_cols,
        claims_col=claims_col,
    )
    return ClaimProjection(
        base_rates=rates,
        projection_keys=projection_keys,
        claim_type_col=claim_type,
        exposure=exposure,
        horizon=horizon,
        trend=trend,
        seasonality=seasonality,
        credibility=credibility,
        complement_basis=complement_basis,
        rate_loads=rate_loads,
        exposure_col=exposure_col,
        exposure_period_col=exposure_period_col,
        dates=dates,
    )
