"""Premium roll-forward workflow with renewal-effective rate actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import pandas as pd

from .adjustments import Scenario
from .calculations import Calculation, CashFlow, RollForward
from .data import ProjectionData, ProjectionDataset, ProjectionDates
from .exceptions import ValidationError
from .horizon import ProjectionHorizon
from .model import ProjectionModel
from .results import ProjectionResults


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class RenewalRateActions:
    """One-time rate actions keyed by projection record and effective date.

    ``rate_action`` values are decimal changes: ``0.10`` means a 10% increase.
    Each action is applied once in the projection period containing its effective
    date. Use a rate-action column on :class:`PremiumProjection` instead when the
    same action should recur at every renewal anniversary.
    """

    frame: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    effective_date_col: str = "effective_date"
    rate_action_col: str = "rate_action"

    def __post_init__(self) -> None:
        keys = _as_tuple(self.projection_keys)
        object.__setattr__(self, "projection_keys", keys)
        required = [*keys, self.effective_date_col, self.rate_action_col]
        missing = [column for column in required if column not in self.frame.columns]
        if missing:
            raise ValidationError(f"rate actions are missing columns: {missing}")
        dates = pd.to_datetime(self.frame[self.effective_date_col], errors="coerce")
        if dates.isna().any():
            raise ValidationError("rate-action effective dates must be valid dates")
        if self.frame[self.rate_action_col].isna().any():
            raise ValidationError("rate actions must not be missing")

    def to_projection_table(self, horizon: ProjectionHorizon) -> pd.DataFrame:
        """Return actions keyed to the horizon's projection periods."""

        table = self.frame.loc[
            :, [*self.projection_keys, self.effective_date_col, self.rate_action_col]
        ].copy()
        table["projection_period"] = pd.to_datetime(
            table[self.effective_date_col]
        ).dt.to_period(horizon.period_frequency).astype(str)
        keys = [*self.projection_keys, "projection_period"]
        if table.duplicated(keys).any():
            duplicated = table.loc[table.duplicated(keys, keep=False), keys].head()
            raise ValidationError(
                "rate actions must be unique by projection keys and projection period; "
                f"examples:\n{duplicated}"
            )
        return table.rename(columns={self.rate_action_col: "scheduled_rate_action"})


@dataclass
class PremiumProjection:
    """Project premium rates and premium using supplied exposure
    (member-months, policy months, and so on — see ``exposure_col``).

    A recurring action column is applied in every period marked
    ``is_renewal_period``. A :class:`RenewalRateActions` schedule is applied once
    in the period containing each supplied effective date. The adjusted rate is
    carried forward to all subsequent periods.
    """

    premium_data: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    exposure: pd.DataFrame
    horizon: ProjectionHorizon
    current_rate_col: str = "current_premium_rate"
    exposure_col: str = "exposure"
    exposure_period_col: str = "projection_period"
    renewal_date_col: str = "renewal_date"
    recurring_rate_action_col: str | None = None
    rate_actions: RenewalRateActions | None = None
    dates: ProjectionDates | None = None

    def __post_init__(self) -> None:
        self.projection_keys = _as_tuple(self.projection_keys)
        required = [*self.projection_keys, self.current_rate_col]
        if self.recurring_rate_action_col is not None:
            required.extend([self.renewal_date_col, self.recurring_rate_action_col])
        missing = [column for column in required if column not in self.premium_data.columns]
        if missing:
            raise ValidationError(f"premium_data is missing columns: {missing}")
        if self.premium_data.duplicated(list(self.projection_keys)).any():
            raise ValidationError("premium_data must be unique at projection_keys")

        exposure_required = [
            *self.projection_keys,
            self.exposure_period_col,
            self.exposure_col,
        ]
        missing_exposure = [
            column for column in exposure_required if column not in self.exposure.columns
        ]
        if missing_exposure:
            raise ValidationError(
                f"exposure is missing columns: {missing_exposure}"
            )
        if self.exposure.duplicated(
            [*self.projection_keys, self.exposure_period_col]
        ).any():
            raise ValidationError(
                "exposure must be unique by projection keys and projection period"
            )
        if self.rate_actions is not None and tuple(self.rate_actions.projection_keys) != tuple(
            self.projection_keys
        ):
            raise ValidationError(
                "rate action projection_keys must match premium projection_keys"
            )

        if self.dates is None and self.renewal_date_col in self.premium_data.columns:
            self.dates = ProjectionDates(renewal_date=self.renewal_date_col)
        elif (
            self.dates is not None
            and self.dates.renewal_date is None
            and self.renewal_date_col in self.premium_data.columns
        ):
            self.dates = replace(self.dates, renewal_date=self.renewal_date_col)

    def _model(self) -> ProjectionModel:
        grain = tuple(self.projection_keys)

        def premium_rate(context):
            rate = context.prior("projected_premium_rate").astype(float)
            if self.recurring_rate_action_col is not None:
                action = context[self.recurring_rate_action_col].fillna(0.0).astype(float)
                rate = rate * (1.0 + action.where(context["is_renewal_period"], 0.0))
            if "scheduled_rate_action" in context.frame.columns:
                action = context["scheduled_rate_action"].fillna(0.0).astype(float)
                rate = rate * (1.0 + action)
            return rate

        return ProjectionModel(
            roll_forwards=[
                RollForward(
                    "projected_premium_rate",
                    initial=self.current_rate_col,
                    formula=premium_rate,
                    aggregation="mean",
                    grain=grain,
                )
            ],
            calculations=[
                Calculation(
                    self.exposure_col,
                    formula=lambda c: c[self.exposure_col] * c["active_fraction"],
                    aggregation="sum",
                    grain=grain,
                    reporting_role="exposure",
                ),
                CashFlow(
                    "premium",
                    formula=lambda c: c["projected_premium_rate"] * c[self.exposure_col],
                    aggregation="sum",
                    grain=grain,
                    reporting_role="revenue",
                    depends_on=("projected_premium_rate", self.exposure_col),
                ),
            ],
        )

    def project(
        self,
        *,
        scenarios: Scenario | Iterable[Scenario] | None = None,
    ) -> ProjectionResults:
        records = ProjectionData(
            self.premium_data.copy(),
            projection_keys=self.projection_keys,
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
        if self.rate_actions is not None:
            actions = self.rate_actions.to_projection_table(self.horizon)
            dataset.add_table(
                "rate_actions",
                actions.loc[
                    :, [*self.projection_keys, "projection_period", "scheduled_rate_action"]
                ],
                keys=[*self.projection_keys, "projection_period"],
            )
        return self._model().project(dataset, self.horizon, scenarios=scenarios)
