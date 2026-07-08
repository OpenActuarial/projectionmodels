"""Claim experience preparation and claim-type projection workflow."""

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
    """Project claim rates by entity and claim type onto supplied membership."""

    base_rates: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    claim_type_col: str
    membership: pd.DataFrame
    horizon: ProjectionHorizon
    trend: TrendAssumption
    seasonality: SeasonalityAssumption | None = None
    credibility: CredibilityAssumption | None = None
    membership_col: str = "member_months"
    membership_period_col: str = "projection_period"
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
        membership_keys = [
            *self.projection_keys,
            self.membership_period_col,
            self.membership_col,
        ]
        missing_membership = [
            column for column in membership_keys if column not in self.membership.columns
        ]
        if missing_membership:
            raise ValidationError(
                f"membership is missing columns: {missing_membership}"
            )
        if self.credibility is not None and "complement_claim_rate" not in self.base_rates:
            raise ValidationError(
                "credibility requires complement_claim_rate in base_rates"
            )

    @classmethod
    def from_experience(
        cls,
        experience: ClaimExperience,
        *,
        membership: pd.DataFrame,
        horizon: ProjectionHorizon,
        trend: TrendAssumption,
        seasonality: SeasonalityAssumption | None = None,
        credibility: CredibilityAssumption | None = None,
        completion: CompletionAssumption | None = None,
        complement: Assumption | Any | None = None,
        extra_record_cols: Iterable[str] = (),
        membership_col: str = "member_months",
        membership_period_col: str = "projection_period",
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
            membership=membership,
            horizon=horizon,
            trend=trend,
            seasonality=seasonality,
            credibility=credibility,
            membership_col=membership_col,
            membership_period_col=membership_period_col,
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

        record_grain = self.projection_keys + (self.claim_type_col,)
        entity_grain = self.projection_keys

        def credible_rate(context):
            observed = context["experience_claim_rate"]
            if self.credibility is None:
                return observed
            blend = actuarialpy_function("credibility_weighted_estimate")
            return blend(
                observed,
                context["complement_claim_rate"],
                context[self.credibility.name],
            )

        def trend_months(context):
            base = pd.to_datetime(context["experience_midpoint"])
            target = pd.to_datetime(context["period_midpoint"])
            return (
                (target.dt.year - base.dt.year) * 12
                + (target.dt.month - base.dt.month)
                + (target.dt.day - base.dt.day) / 30.4375
            )

        def trended_rate(context):
            trend_factor = actuarialpy_function("trend_factor")
            months = trend_months(context)
            return context["credible_claim_rate"] * trend_factor(
                context[self.trend.name], months
            )

        def seasonal_rate(context):
            if self.seasonality is None:
                return context["trended_claim_rate"]
            apply_seasonality = actuarialpy_function("apply_seasonality")
            lookup = list(self.seasonality.lookup)
            by = [
                column
                for column in lookup
                if column != self.seasonality.season_col
            ]
            factor_table = context.frame.loc[
                :, list(dict.fromkeys(lookup + [self.seasonality.name]))
            ].drop_duplicates(lookup)
            applied = apply_seasonality(
                context.frame.assign(
                    __projectionmodels_trended_rate__=context["trended_claim_rate"]
                ),
                factor_table,
                date_col="period_start",
                value_col="__projectionmodels_trended_rate__",
                freq=self.seasonality.frequency,
                by=by or None,
                factor_col=self.seasonality.name,
                season_name=self.seasonality.season_col,
                out_col="__projectionmodels_projected_rate__",
            )
            return applied["__projectionmodels_projected_rate__"]

        calculations = [
            Calculation(
                "credible_claim_rate",
                formula=credible_rate,
                aggregation="mean",
                grain=record_grain,
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
                formula=seasonal_rate,
                aggregation="mean",
                grain=record_grain,
                depends_on=("trended_claim_rate",),
            ),
            Calculation(
                self.membership_col,
                formula=lambda c: c[self.membership_col] * c["active_fraction"],
                aggregation="sum",
                grain=entity_grain,
                reporting_role="exposure",
            ),
            CashFlow(
                "projected_claims",
                formula=lambda c: c["projected_claim_rate"]
                * c[self.membership_col],
                aggregation="sum",
                grain=record_grain,
                reporting_role="loss",
                depends_on=("projected_claim_rate", self.membership_col),
            ),
            Metric(
                "claim_pmpm",
                formula=lambda c: c["projected_claims"] / c[self.membership_col],
                aggregation="recalculate",
                numerator="projected_claims",
                denominator=self.membership_col,
                grain=record_grain,
                depends_on=("projected_claims", self.membership_col),
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
        membership = self.membership.rename(
            columns={self.membership_period_col: "projection_period"}
        )
        dataset.add_table(
            "membership",
            membership,
            keys=[*self.projection_keys, "projection_period"],
        )
        return self._model().project(
            dataset,
            self.horizon,
            scenarios=scenarios,
        )
