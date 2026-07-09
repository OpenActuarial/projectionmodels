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

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class ClaimExperience:
    """Historical claims used to establish projected base rates.

    The input may contain one row per month, claim transaction, or another
    experience grain. ``to_base_rates`` develops immature claims, removes
    seasonality, aggregates to projection record + claim type, and calculates a
    per-exposure experience rate.
    """

    data: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    claim_type_col: str
    date_col: str
    claims_col: str
    exposure_col: str
    valuation_date: Any | None = None

    def __post_init__(self) -> None:
        keys = _as_tuple(self.projection_keys)
        object.__setattr__(self, "projection_keys", keys)
        required = [
            *keys,
            self.claim_type_col,
            self.date_col,
            self.claims_col,
            self.exposure_col,
        ]
        missing = [column for column in required if column not in self.data.columns]
        if missing:
            raise ValidationError(f"claim experience is missing columns: {missing}")

    @property
    def record_keys(self) -> tuple[str, ...]:
        return self.projection_keys + (self.claim_type_col,)

    def prepare(
        self,
        *,
        completion: CompletionAssumption | None = None,
        seasonality: SeasonalityAssumption | None = None,
    ) -> pd.DataFrame:
        """Return experience with completed and deseasonalized claim columns."""

        work = self.data.copy()
        work[self.date_col] = pd.to_datetime(work[self.date_col])
        working_col = self.claims_col

        if completion is not None:
            if self.valuation_date is None and completion.development_col not in work.columns:
                raise ValidationError(
                    "valuation_date is required to apply completion without a development column"
                )
            by = [
                column
                for column in completion.lookup
                if column != completion.development_col
            ]
            out_col = f"{self.claims_col}_completed"
            work = completion.apply(
                work,
                value_col=self.claims_col,
                date_col=(
                    None if completion.development_col in work.columns else self.date_col
                ),
                valuation_date=self.valuation_date,
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
            work[f"{self.claims_col}_completed"] = work[self.claims_col]
            working_col = f"{self.claims_col}_completed"

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
                date_col=self.date_col,
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

    def to_base_rates(
        self,
        *,
        completion: CompletionAssumption | None = None,
        seasonality: SeasonalityAssumption | None = None,
        complement: Assumption | Any | None = None,
        extra_record_cols: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Aggregate prepared experience to one row per record and claim type."""

        work = self.prepare(completion=completion, seasonality=seasonality)
        group_columns = list(self.record_keys)
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
            experience_exposure=(self.exposure_col, "sum"),
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
                part[self.date_col], part[self.exposure_col]
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

    @classmethod
    def from_experience(
        cls,
        experience: ClaimExperience,
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
        extra_record_cols: Iterable[str] = (),
        exposure_col: str = "exposure",
        exposure_period_col: str = "projection_period",
        dates: ProjectionDates | None = None,
    ) -> ClaimProjection:
        base_rates = experience.to_base_rates(
            completion=completion,
            seasonality=seasonality,
            complement=complement,
            extra_record_cols=extra_record_cols,
        )
        return cls(
            base_rates=base_rates,
            projection_keys=experience.projection_keys,
            claim_type_col=experience.claim_type_col,
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
