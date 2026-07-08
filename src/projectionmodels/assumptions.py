"""Supplied and actuarialpy-estimated projection assumptions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .actuarialpy_adapter import actuarialpy_function, require_actuarialpy
from .exceptions import AssumptionResolutionError, ValidationError


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _selection_table(
    selection: Any,
    *,
    lookup: tuple[str, ...],
    value_col: str,
) -> pd.DataFrame:
    if np.isscalar(selection):
        if lookup:
            raise ValidationError("a scalar selection cannot be keyed by lookup columns")
        return pd.DataFrame({value_col: [selection]})
    if isinstance(selection, pd.Series):
        if lookup and selection.index.names == list(lookup):
            return selection.rename(value_col).reset_index()
        if len(lookup) == 1 and selection.index.name == lookup[0]:
            return selection.rename(value_col).reset_index()
        return selection.rename(value_col).to_frame().reset_index(drop=True)
    if isinstance(selection, pd.DataFrame):
        if value_col not in selection.columns:
            raise ValidationError(f"selection does not contain {value_col!r}")
        return selection.copy()
    raise TypeError("selection must be scalar, Series, or DataFrame")


@dataclass(frozen=True)
class Assumption:
    """A named assumption resolved by explicit lookup fields.

    ``values`` may be a scalar, Series, or DataFrame. DataFrame assumptions are
    joined many-to-one on ``lookup`` and must contain ``value_col``.
    """

    name: str
    values: Any
    lookup: tuple[str, ...] | list[str] = field(default_factory=tuple)
    value_col: str | None = None
    source: str = "supplied"
    indicated_values: Any | None = None
    diagnostics: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lookup = _as_tuple(self.lookup)
        object.__setattr__(self, "lookup", lookup)
        if isinstance(self.values, pd.DataFrame):
            value_col = self.value_col or self.name
            object.__setattr__(self, "value_col", value_col)
            missing = [c for c in lookup + (value_col,) if c not in self.values.columns]
            if missing:
                raise ValidationError(
                    f"assumption {self.name!r} is missing columns: {missing}"
                )
            if lookup and self.values.duplicated(list(lookup)).any():
                raise ValidationError(
                    f"assumption {self.name!r} has duplicate lookup keys"
                )
        elif self.value_col is None:
            object.__setattr__(self, "value_col", self.name)

    @property
    def selected_values(self) -> Any:
        return self.values

    def resolve(
        self,
        frame: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> pd.Series:
        """Resolve assumption values onto ``frame`` in its original row order."""

        if np.isscalar(self.values):
            return pd.Series(self.values, index=frame.index, name=self.name)

        if isinstance(self.values, pd.Series):
            series = self.values
            if not self.lookup and len(series) == len(frame):
                return pd.Series(series.to_numpy(), index=frame.index, name=self.name)
            if len(self.lookup) == 1 and series.index.name == self.lookup[0]:
                mapped = frame[self.lookup[0]].map(series)
                mapped.name = self.name
                if strict and mapped.isna().any():
                    self._raise_missing(frame, mapped)
                return mapped
            table = series.rename(self.value_col or self.name).reset_index()
            return replace(self, values=table).resolve(frame, strict=strict)

        if not isinstance(self.values, pd.DataFrame):
            array = np.asarray(self.values)
            if array.ndim == 1 and len(array) == len(frame) and not self.lookup:
                return pd.Series(array, index=frame.index, name=self.name)
            raise AssumptionResolutionError(
                f"cannot resolve assumption {self.name!r} from {type(self.values).__name__}"
            )

        value_col = self.value_col or self.name
        if not self.lookup:
            if len(self.values) != 1:
                raise AssumptionResolutionError(
                    f"unkeyed assumption {self.name!r} must contain exactly one row"
                )
            return pd.Series(
                self.values.iloc[0][value_col], index=frame.index, name=self.name
            )

        missing_columns = [column for column in self.lookup if column not in frame.columns]
        if missing_columns:
            raise AssumptionResolutionError(
                f"projection rows lack lookup columns for {self.name!r}: {missing_columns}"
            )
        left = frame.loc[:, list(self.lookup)].copy()
        left["__row_order__"] = np.arange(len(left))
        merged = left.merge(
            self.values.loc[:, list(self.lookup) + [value_col]],
            on=list(self.lookup),
            how="left",
            validate="many_to_one",
            sort=False,
        ).sort_values("__row_order__")
        resolved = pd.Series(merged[value_col].to_numpy(), index=frame.index, name=self.name)
        if strict and resolved.isna().any():
            self._raise_missing(frame, resolved)
        return resolved

    def _raise_missing(self, frame: pd.DataFrame, resolved: pd.Series) -> None:
        if self.lookup:
            examples = frame.loc[resolved.isna(), list(self.lookup)].drop_duplicates().head()
            detail = f"; unmatched keys include:\n{examples}"
        else:
            detail = ""
        raise AssumptionResolutionError(
            f"assumption {self.name!r} has missing resolved values{detail}"
        )

    def select(
        self,
        selection: Any,
        *,
        lookup: str | Iterable[str] | None = None,
        value_col: str | None = None,
        note: str | None = None,
    ) -> Assumption:
        """Replace indicated values with an actuarial selection while retaining audit data."""

        selected_lookup = _as_tuple(lookup) if lookup is not None else tuple(self.lookup)
        selected_col = value_col or self.value_col or self.name
        table = _selection_table(selection, lookup=selected_lookup, value_col=selected_col)
        metadata = dict(self.metadata)
        if note is not None:
            metadata["selection_note"] = note
        return replace(
            self,
            values=table if isinstance(selection, (pd.Series, pd.DataFrame)) else selection,
            lookup=selected_lookup,
            value_col=selected_col,
            source=(
                "actuarialpy_estimate_with_selection"
                if self.source.startswith("actuarialpy")
                else "supplied_selection"
            ),
            indicated_values=(
                self.indicated_values if self.indicated_values is not None else self.values
            ),
            metadata=metadata,
        )

    def audit_frame(self) -> pd.DataFrame:
        """Return a compact assumption-audit table."""

        return pd.DataFrame(
            [
                {
                    "name": self.name,
                    "source": self.source,
                    "lookup": ", ".join(self.lookup),
                    "value_column": self.value_col,
                    **dict(self.metadata),
                }
            ]
        )


@dataclass(frozen=True)
class TrendAssumption(Assumption):
    """Annual trend rate, supplied or fitted with :func:`actuarialpy.fit_trend`."""

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Any,
        *,
        lookup: str | Iterable[str] | None = None,
        rate_col: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrendAssumption:
        return cls(
            name=name,
            values=values,
            lookup=_as_tuple(lookup),
            value_col=rate_col or name,
            metadata=metadata or {},
        )

    @classmethod
    def from_experience(
        cls,
        name: str,
        experience: pd.DataFrame,
        *,
        date_col: str,
        value_col: str,
        exposure_col: str | None = None,
        by: str | Iterable[str] | None = None,
        freq: str = "M",
        min_periods: int = 3,
        confidence: float = 0.95,
    ) -> TrendAssumption:
        fit_trend = actuarialpy_function("fit_trend")
        groups = _as_tuple(by)
        rows: list[dict[str, Any]] = []
        grouped: Any
        if groups:
            grouped = experience.groupby(list(groups), dropna=False, sort=True)
        else:
            grouped = [((), experience)]
        for key, part in grouped:
            fit = fit_trend(
                part,
                date_col=date_col,
                value_col=value_col,
                exposure_col=exposure_col,
                freq=freq,
                min_periods=min_periods,
                confidence=confidence,
            )
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(groups, key_tuple, strict=True)) if groups else {}
            annual_trend = float(fit.annual_trend)
            row.update(
                {
                    name: annual_trend,
                    "indicated_trend": annual_trend,
                    "r_squared": getattr(fit, "r_squared", np.nan),
                    "ci_low": getattr(fit, "ci_low", np.nan),
                    "ci_high": getattr(fit, "ci_high", np.nan),
                    "n_periods": getattr(fit, "n_periods", np.nan),
                }
            )
            rows.append(row)
        values = pd.DataFrame(rows)
        diagnostics = values.copy()
        return cls(
            name=name,
            values=values.loc[:, list(groups) + [name]],
            lookup=groups,
            value_col=name,
            source="actuarialpy_estimate",
            indicated_values=values,
            diagnostics=diagnostics,
            metadata={
                "method": "log_linear",
                "date_col": date_col,
                "value_col": value_col,
                "exposure_col": exposure_col,
                "frequency": freq,
            },
        )

    def factor(self, frame: pd.DataFrame, months: Any) -> pd.Series:
        """Resolve annual rates and return factors for scalar or row-wise months."""

        rates = self.resolve(frame)
        trend_factor = actuarialpy_function("trend_factor")
        if np.isscalar(months):
            resolved_months: Any = float(months)
        else:
            resolved_months = np.asarray(months, dtype=float)
            if resolved_months.ndim != 1 or len(resolved_months) != len(frame):
                raise ValidationError(
                    "months must be scalar or contain one value per projection row"
                )
        values = trend_factor(rates, resolved_months)
        return pd.Series(values, index=frame.index, name=f"{self.name}_factor")


@dataclass(frozen=True)
class SeasonalityAssumption(Assumption):
    """Normalized seasonal multipliers."""

    season_col: str = "season"
    frequency: str = "M"

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Any,
        *,
        lookup: str | Iterable[str] | None = None,
        season_col: str = "season",
        factor_col: str | None = None,
        frequency: str = "M",
    ) -> SeasonalityAssumption:
        lookup_fields = _as_tuple(lookup)
        if season_col not in lookup_fields:
            lookup_fields += (season_col,)
        return cls(
            name=name,
            values=values,
            lookup=lookup_fields,
            value_col=factor_col or name,
            season_col=season_col,
            frequency=frequency,
        )

    @classmethod
    def from_experience(
        cls,
        name: str,
        experience: pd.DataFrame,
        *,
        date_col: str,
        value_col: str,
        exposure_col: str | None = None,
        by: str | Iterable[str] | None = None,
        freq: str = "M",
        method: str = "ratio_to_moving_average",
        aggregate: str = "mean",
        min_years: int = 2,
        season_col: str = "season",
    ) -> SeasonalityAssumption:
        groups = _as_tuple(by)
        if groups:
            function = actuarialpy_function("seasonality_factors_by")
            values = function(
                experience,
                groupby=list(groups),
                date_col=date_col,
                value_col=value_col,
                exposure_col=exposure_col,
                freq=freq,
                method=method,
                aggregate=aggregate,
                min_years=min_years,
                season_name=season_col,
            ).rename(columns={"seasonal_factor": name})
        else:
            function = actuarialpy_function("seasonality_factors")
            factors = function(
                experience,
                date_col=date_col,
                value_col=value_col,
                exposure_col=exposure_col,
                freq=freq,
                method=method,
                aggregate=aggregate,
                min_years=min_years,
            )
            values = factors.rename(name).rename_axis(season_col).reset_index()
        lookup = groups + (season_col,)
        if groups:
            means = values.groupby(list(groups), dropna=False)[name].mean()
            invalid = means.loc[~np.isclose(means.to_numpy(dtype=float), 1.0, atol=1e-6)]
            if not invalid.empty:
                raise ValidationError(
                    "calculated seasonality factors are not normalized to 1.0 for "
                    f"these segments: {invalid.to_dict()}"
                )
        else:
            mean_factor = float(values[name].mean())
            if not np.isclose(mean_factor, 1.0, atol=1e-6):
                raise ValidationError(
                    f"calculated seasonality factors have mean {mean_factor:.8f}, not 1.0"
                )
        return cls(
            name=name,
            values=values.loc[:, list(lookup) + [name]],
            lookup=lookup,
            value_col=name,
            source="actuarialpy_estimate",
            indicated_values=values,
            diagnostics=values.copy(),
            metadata={
                "method": method,
                "frequency": freq,
                "date_col": date_col,
                "value_col": value_col,
                "exposure_col": exposure_col,
            },
            season_col=season_col,
            frequency=freq,
        )


@dataclass(frozen=True)
class CompletionAssumption(Assumption):
    """Claim completion factors in the divide convention, supplied or estimated."""

    development_col: str = "development_month"

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Any,
        *,
        lookup: str | Iterable[str] | None = None,
        development_col: str = "development_month",
        factor_col: str = "completion_factor",
    ) -> CompletionAssumption:
        lookup_fields = _as_tuple(lookup)
        if development_col not in lookup_fields:
            lookup_fields += (development_col,)
        return cls(
            name=name,
            values=values,
            lookup=lookup_fields,
            value_col=factor_col,
            development_col=development_col,
        )

    @classmethod
    def from_experience(
        cls,
        name: str,
        experience: pd.DataFrame,
        *,
        origin_col: str,
        valuation_col: str,
        amount_col: str,
        by: str | Iterable[str] | None = None,
        cumulative: bool = True,
        method: str = "volume",
        tail: float = 1.0,
        on_insufficient: str = "raise",
        development_col: str = "development_month",
    ) -> CompletionAssumption:
        groups = _as_tuple(by)
        try:
            if groups:
                function = actuarialpy_function("completion_factors_by")
                values = function(
                    experience,
                    groupby=list(groups),
                    origin_col=origin_col,
                    valuation_col=valuation_col,
                    amount_col=amount_col,
                    cumulative=cumulative,
                    method=method,
                    tail=tail,
                    on_insufficient=on_insufficient,
                    development_name=development_col,
                )
            else:
                make_triangle = actuarialpy_function("make_completion_triangle")
                completion_factors = actuarialpy_function("completion_factors")
                triangle = make_triangle(
                    experience,
                    origin_col=origin_col,
                    valuation_col=valuation_col,
                    amount_col=amount_col,
                    cumulative=cumulative,
                )
                factors = completion_factors(triangle, method=method, tail=tail)
                values = (
                    factors.rename("completion_factor")
                    .rename_axis(development_col)
                    .reset_index()
                )
        except ValueError as exc:
            raise ValidationError(
                "unable to estimate completion factors from the supplied experience; "
                "provide a triangle with at least two overlapping origin and "
                f"development periods, or change on_insufficient. Details: {exc}"
            ) from exc
        lookup = groups + (development_col,)
        return cls(
            name=name,
            values=values.loc[:, list(lookup) + ["completion_factor"]],
            lookup=lookup,
            value_col="completion_factor",
            source="actuarialpy_estimate",
            indicated_values=values,
            diagnostics=values.copy(),
            metadata={
                "method": method,
                "tail": tail,
                "origin_col": origin_col,
                "valuation_col": valuation_col,
                "amount_col": amount_col,
            },
            development_col=development_col,
        )

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        value_col: str,
        date_col: str | None = None,
        valuation_date: Any | None = None,
        development_col: str | None = None,
        by: str | Iterable[str] | None = None,
        out_col: str | None = None,
    ) -> pd.DataFrame:
        function = actuarialpy_function("apply_completion")
        factor_col = self.value_col or "completion_factor"
        by_fields = _as_tuple(by)
        factors = self.values
        if not by_fields and isinstance(factors, pd.DataFrame):
            if factors.duplicated([self.development_col]).any():
                raise ValidationError(
                    "ungrouped completion factors must be unique by development period"
                )
            factors = factors.set_index(self.development_col)[factor_col]
        return function(
            frame,
            factors,
            value_col=value_col,
            date_col=date_col,
            valuation_date=valuation_date,
            development_col=development_col,
            by=list(by_fields) or None,
            factor_col=factor_col,
            development_name=self.development_col,
            out_col=out_col,
        )


@dataclass(frozen=True)
class CredibilityAssumption(Assumption):
    """Credibility weights supplied or estimated by actuarialpy."""

    @classmethod
    def from_weights(
        cls,
        name: str,
        values: Any,
        *,
        lookup: str | Iterable[str] | None = None,
        weight_col: str | None = None,
    ) -> CredibilityAssumption:
        return cls(
            name=name,
            values=values,
            lookup=_as_tuple(lookup),
            value_col=weight_col or name,
        )

    @classmethod
    def from_experience(
        cls,
        name: str,
        experience: pd.DataFrame,
        *,
        method: str,
        by: str | Iterable[str],
        exposure_col: str | None = None,
        full_credibility_standard: float | None = None,
        value_col: str | None = None,
        period_col: str | None = None,
        weight_col: str | None = None,
    ) -> CredibilityAssumption:
        groups = _as_tuple(by)
        if not groups:
            raise ValidationError("credibility estimation requires at least one risk key")
        method_key = method.lower().replace("-", "_")

        if method_key in {"limited_fluctuation", "limited_fluctuation_z"}:
            if exposure_col is None or full_credibility_standard is None:
                raise ValidationError(
                    "limited fluctuation requires exposure_col and full_credibility_standard"
                )
            grouped = (
                experience.groupby(list(groups), dropna=False)[exposure_col]
                .sum()
                .rename("credibility_exposure")
                .reset_index()
            )
            z_function = actuarialpy_function("limited_fluctuation_z")
            grouped[name] = z_function(
                grouped["credibility_exposure"], full_credibility_standard
            )
            diagnostics = grouped.copy()
            metadata = {
                "method": "limited_fluctuation",
                "full_credibility_standard": full_credibility_standard,
                "exposure_col": exposure_col,
            }
        elif method_key == "buhlmann_straub":
            if value_col is None or weight_col is None:
                raise ValidationError(
                    "Buhlmann-Straub requires value_col and weight_col"
                )
            ap = require_actuarialpy()
            work = experience.copy()
            risk_col = "__projectionmodels_risk__"
            risk_map = work.loc[:, list(groups)].drop_duplicates().reset_index(drop=True)
            risk_map[risk_col] = np.arange(len(risk_map), dtype=int)
            work = work.merge(
                risk_map,
                on=list(groups),
                how="left",
                validate="many_to_one",
            )
            model = ap.BuhlmannStraub.from_frame(
                work,
                group=risk_col,
                value=value_col,
                weight=weight_col,
                period=period_col,
            )
            z = np.asarray(model.z(model.weights), dtype=float)
            estimates = np.asarray(
                model.premium(model.risk_means_, model.weights), dtype=float
            )
            grouped = pd.DataFrame(
                {
                    risk_col: np.asarray(model.groups_, dtype=int),
                    name: z,
                    "credibility_estimate": estimates,
                    "risk_mean": model.risk_means_,
                    "credibility_exposure": model.weights,
                }
            ).merge(risk_map, on=risk_col, how="left", validate="one_to_one")
            grouped = grouped.loc[:, list(groups) + [name, "credibility_estimate", "risk_mean", "credibility_exposure"]]
            diagnostics = grouped.copy()
            metadata = {
                "method": "buhlmann_straub",
                "value_col": value_col,
                "weight_col": weight_col,
                "period_col": period_col,
                "overall_mean": float(model.overall_mean),
                "epv": float(model.epv),
                "vhm": float(model.vhm),
            }
        elif method_key == "buhlmann":
            if value_col is None or period_col is None:
                raise ValidationError("Buhlmann requires value_col and period_col")
            ap = require_actuarialpy()
            pivot = experience.pivot_table(
                index=list(groups),
                columns=period_col,
                values=value_col,
                aggfunc="mean",
            )
            if pivot.isna().any().any():
                raise ValidationError(
                    "Buhlmann requires the same complete period set for every risk; "
                    "use Buhlmann-Straub for unequal histories"
                )
            model = ap.Buhlmann.fit(pivot.to_numpy())
            grouped = pivot.reset_index().loc[:, list(groups)]
            grouped[name] = float(model.z)
            grouped["risk_mean"] = pivot.mean(axis=1).to_numpy()
            grouped["credibility_estimate"] = model.premium(
                grouped["risk_mean"].to_numpy()
            )
            diagnostics = grouped.copy()
            metadata = {
                "method": "buhlmann",
                "value_col": value_col,
                "period_col": period_col,
                "overall_mean": float(model.overall_mean),
                "epv": float(model.epv),
                "vhm": float(model.vhm),
            }
        else:
            raise ValidationError(
                "method must be limited_fluctuation, buhlmann, or buhlmann_straub"
            )

        return cls(
            name=name,
            values=grouped.loc[:, list(groups) + [name]],
            lookup=groups,
            value_col=name,
            source="actuarialpy_estimate",
            indicated_values=grouped,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def blend(self, observed: Any, complement: Any, frame: pd.DataFrame) -> pd.Series:
        function = actuarialpy_function("credibility_weighted_estimate")
        z = self.resolve(frame)
        blended = function(observed, complement, z)
        return pd.Series(blended, index=frame.index, name=f"{self.name}_estimate")


@dataclass
class AssumptionSet:
    """A validated collection of named assumptions."""

    assumptions: dict[str, Assumption] = field(default_factory=dict)

    def __init__(self, *items: Assumption, **named: Assumption):
        self.assumptions = {}
        for item in items:
            self.add(item)
        for name, item in named.items():
            if item.name != name:
                item = replace(item, name=name)
            self.add(item)

    def add(self, assumption: Assumption) -> AssumptionSet:
        if assumption.name in self.assumptions:
            raise ValidationError(f"assumption {assumption.name!r} already exists")
        self.assumptions[assumption.name] = assumption
        return self

    def __iter__(self):
        return iter(self.assumptions.values())

    def __getitem__(self, name: str) -> Assumption:
        return self.assumptions[name]

    def resolve(self, frame: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
        result = frame.copy()
        for assumption in self:
            result[assumption.name] = assumption.resolve(result, strict=strict)
        return result

    def audit_frame(self) -> pd.DataFrame:
        if not self.assumptions:
            return pd.DataFrame(columns=["name", "source", "lookup", "value_column"])
        return pd.concat(
            [assumption.audit_frame() for assumption in self], ignore_index=True
        )
