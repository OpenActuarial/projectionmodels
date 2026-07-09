"""Expense projection workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .actuarialpy_adapter import actuarialpy_function
from .adjustments import Scenario
from .assumptions import AssumptionSet, TrendAssumption
from .calculations import Calculation, CashFlow
from .data import ProjectionData, ProjectionDataset, ProjectionDates
from .exceptions import ValidationError
from .horizon import ProjectionHorizon
from .model import ProjectionModel
from .results import ProjectionResults


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass
class ExpenseProjection:
    """Project expenses with per-exposure, fixed, premium, or claims bases."""

    expenses: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    expense_type_col: str
    base_value_col: str
    basis_col: str
    base_date_col: str
    horizon: ProjectionHorizon
    trend: TrendAssumption
    exposure: pd.DataFrame | None = None
    premium: pd.DataFrame | None = None
    claims: pd.DataFrame | None = None
    exposure_col: str = "exposure"
    premium_col: str = "premium"
    claims_col: str = "projected_claims"
    dates: ProjectionDates | None = None

    def __post_init__(self) -> None:
        self.projection_keys = _as_tuple(self.projection_keys)
        required = [
            *self.projection_keys,
            self.expense_type_col,
            self.base_value_col,
            self.basis_col,
            self.base_date_col,
        ]
        missing = [column for column in required if column not in self.expenses.columns]
        if missing:
            raise ValidationError(f"expenses is missing columns: {missing}")
        allowed = {"per_exposure", "fixed_monthly", "percent_premium", "percent_claims"}
        unknown = sorted(set(self.expenses[self.basis_col].dropna()) - allowed)
        if unknown:
            raise ValidationError(f"unknown expense bases: {unknown}")

    def _model(self) -> ProjectionModel:
        record_grain = self.projection_keys + (self.expense_type_col,)

        def trend_months(context):
            base = pd.to_datetime(context[self.base_date_col])
            target = pd.to_datetime(context["period_midpoint"])
            return (
                (target.dt.year - base.dt.year) * 12
                + (target.dt.month - base.dt.month)
                + (target.dt.day - base.dt.day) / 30.4375
            )

        def rate(context):
            factor = actuarialpy_function("trend_factor")(
                context[self.trend.name], trend_months(context)
            )
            return context[self.base_value_col] * factor

        def expense(context):
            basis = context[self.basis_col]
            projected_rate = context["projected_expense_rate"]
            result = pd.Series(np.nan, index=context.frame.index, dtype=float)
            mask = basis.eq("per_exposure")
            if mask.any():
                result.loc[mask] = (
                    projected_rate.loc[mask]
                    * context[self.exposure_col].loc[mask]
                    * context["active_fraction"].loc[mask]
                )
            mask = basis.eq("fixed_monthly")
            if mask.any():
                result.loc[mask] = (
                    projected_rate.loc[mask]
                    * context["active_fraction"].loc[mask]
                )
            mask = basis.eq("percent_premium")
            if mask.any():
                result.loc[mask] = (
                    projected_rate.loc[mask] * context[self.premium_col].loc[mask]
                )
            mask = basis.eq("percent_claims")
            if mask.any():
                result.loc[mask] = (
                    projected_rate.loc[mask] * context[self.claims_col].loc[mask]
                )
            return result

        return ProjectionModel(
            assumptions=AssumptionSet(self.trend),
            calculations=[
                Calculation(
                    "projected_expense_rate",
                    formula=rate,
                    aggregation="mean",
                    grain=record_grain,
                ),
                CashFlow(
                    "projected_expense",
                    formula=expense,
                    aggregation="sum",
                    grain=record_grain,
                    reporting_role="expense",
                    depends_on=("projected_expense_rate",),
                ),
            ],
        )

    def project(
        self,
        *,
        scenarios: Scenario | Iterable[Scenario] | None = None,
    ) -> ProjectionResults:
        records = ProjectionData(
            self.expenses.copy(),
            projection_keys=self.projection_keys,
            component_keys=[self.expense_type_col],
            dates=self.dates,
        )
        dataset = ProjectionDataset(records)
        for name, table, value_col in (
            ("exposure", self.exposure, self.exposure_col),
            ("premium", self.premium, self.premium_col),
            ("claims", self.claims, self.claims_col),
        ):
            if table is not None:
                required = [*self.projection_keys, "projection_period", value_col]
                missing = [column for column in required if column not in table.columns]
                if missing:
                    raise ValidationError(f"{name} table is missing columns: {missing}")
                dataset.add_table(
                    name,
                    table,
                    keys=[*self.projection_keys, "projection_period"],
                )
        bases = set(self.expenses[self.basis_col])
        if "per_exposure" in bases and self.exposure is None:
            raise ValidationError("Per-exposure expenses require an exposure table")
        if "percent_premium" in bases and self.premium is None:
            raise ValidationError("percent_premium expenses require a premium table")
        if "percent_claims" in bases and self.claims is None:
            raise ValidationError("percent_claims expenses require a claims table")
        return self._model().project(dataset, self.horizon, scenarios=scenarios)
