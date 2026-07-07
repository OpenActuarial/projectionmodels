"""Projection records, supporting tables, and date classifications."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import ValidationError
from .horizon import ProjectionHorizon


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{label} is missing columns: {missing}")


@dataclass(frozen=True)
class ProjectionDates:
    """Column roles for lifecycle and actuarial dates."""

    entry_date: str | None = None
    exit_date: str | None = None
    renewal_date: str | None = None
    issue_date: str | None = None
    experience_start: str | None = None
    experience_end: str | None = None
    exposure_timing: str = "whole_period"

    def columns(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.entry_date,
                self.exit_date,
                self.renewal_date,
                self.issue_date,
                self.experience_start,
                self.experience_end,
            )
            if value is not None
        )

    def validate(self, frame: pd.DataFrame) -> None:
        _require_columns(frame, self.columns(), label="projection data")
        if self.exposure_timing not in {"whole_period", "daily_prorated"}:
            raise ValidationError(
                "exposure_timing must be 'whole_period' or 'daily_prorated'"
            )


@dataclass(frozen=True)
class DateCohort:
    """Create a reportable classification from a date column."""

    name: str
    date_col: str
    split_date: Any | None = None
    before_label: str = "before"
    on_or_after_label: str = "on_or_after"
    frequency: str | None = None
    breaks: tuple[Any, ...] | None = None
    labels: tuple[str, ...] | None = None
    include_lowest: bool = True

    def __post_init__(self) -> None:
        modes = sum(
            [self.split_date is not None, self.frequency is not None, self.breaks is not None]
        )
        if modes != 1:
            raise ValidationError(
                "DateCohort requires exactly one of split_date, frequency, or breaks"
            )
        if self.breaks is not None:
            if len(self.breaks) < 2:
                raise ValidationError("breaks must contain at least two dates")
            if self.labels is not None and len(self.labels) != len(self.breaks) - 1:
                raise ValidationError("labels must have len(breaks) - 1 entries")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, [self.date_col], label="date cohort input")
        result = frame.copy()
        dates = pd.to_datetime(result[self.date_col], errors="coerce")
        if self.split_date is not None:
            split = pd.Timestamp(self.split_date)
            result[self.name] = np.where(
                dates < split, self.before_label, self.on_or_after_label
            )
            result.loc[dates.isna(), self.name] = pd.NA
            return result
        if self.frequency is not None:
            aliases = {
                "month": "M",
                "monthly": "M",
                "quarter": "Q",
                "quarterly": "Q",
                "year": "Y",
                "annual": "Y",
                "yearly": "Y",
            }
            frequency = aliases.get(self.frequency.lower(), self.frequency)
            result[self.name] = dates.dt.to_period(frequency).astype("string")
            return result
        assert self.breaks is not None
        result[self.name] = pd.cut(
            dates,
            bins=pd.to_datetime(list(self.breaks)),
            labels=list(self.labels) if self.labels else None,
            include_lowest=self.include_lowest,
            right=False,
        )
        return result


@dataclass(frozen=True)
class ProjectionTable:
    """A named table that can be joined to projection rows by explicit keys."""

    name: str
    frame: pd.DataFrame
    keys: tuple[str, ...] | list[str]
    allow_duplicates: bool = False

    def __post_init__(self) -> None:
        keys = _as_tuple(self.keys)
        object.__setattr__(self, "keys", keys)
        if not keys:
            raise ValidationError("ProjectionTable keys must not be empty")
        _require_columns(self.frame, keys, label=f"table {self.name!r}")
        if not self.allow_duplicates and self.frame.duplicated(list(keys)).any():
            duplicated = self.frame.loc[
                self.frame.duplicated(list(keys), keep=False), list(keys)
            ].head()
            raise ValidationError(
                f"table {self.name!r} has duplicate keys; examples:\n{duplicated}"
            )


@dataclass(frozen=True)
class ProjectionData:
    """Starting projection records at a caller-selected actuarial grain.

    ``projection_keys`` identify the entity being projected. ``component_keys``
    identify repeated components of an entity, such as claim type or coverage.
    The combined keys must be unique.
    """

    frame: pd.DataFrame
    projection_keys: tuple[str, ...] | list[str]
    component_keys: tuple[str, ...] | list[str] = field(default_factory=tuple)
    record_weight: str | None = None
    dates: ProjectionDates | None = None

    def __post_init__(self) -> None:
        projection_keys = _as_tuple(self.projection_keys)
        component_keys = _as_tuple(self.component_keys)
        object.__setattr__(self, "projection_keys", projection_keys)
        object.__setattr__(self, "component_keys", component_keys)
        if not projection_keys:
            raise ValidationError("projection_keys must not be empty")
        keys = projection_keys + component_keys
        _require_columns(self.frame, keys, label="projection data")
        if len(set(keys)) != len(keys):
            raise ValidationError("projection_keys and component_keys must be distinct")
        if self.frame.duplicated(list(keys)).any():
            duplicated = self.frame.loc[
                self.frame.duplicated(list(keys), keep=False), list(keys)
            ].head()
            raise ValidationError(
                "projection records must be unique at projection_keys + "
                f"component_keys; examples:\n{duplicated}"
            )
        if self.record_weight is not None:
            _require_columns(self.frame, [self.record_weight], label="projection data")
            if (self.frame[self.record_weight] < 0).any():
                raise ValidationError("record_weight must be nonnegative")
        if self.dates is not None:
            self.dates.validate(self.frame)

    @property
    def record_keys(self) -> tuple[str, ...]:
        return self.projection_keys + self.component_keys

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(column for column in self.frame.columns if column not in self.record_keys)

    def add_date_cohort(self, cohort: DateCohort) -> "ProjectionData":
        return replace(self, frame=cohort.apply(self.frame))

    def expand(self, horizon: ProjectionHorizon) -> pd.DataFrame:
        """Cross projection records with the horizon and add lifecycle fields."""

        periods = horizon.to_frame()
        left = self.frame.reset_index(drop=True)
        n_records = len(left)
        n_periods = len(periods)
        expanded = left.iloc[np.repeat(np.arange(n_records), n_periods)].reset_index(
            drop=True
        )
        repeated_periods = periods.iloc[
            np.tile(np.arange(n_periods), n_records)
        ].reset_index(drop=True)
        expanded = pd.concat([expanded, repeated_periods], axis=1)
        return self._add_lifecycle_fields(expanded)

    def _add_lifecycle_fields(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["active_fraction"] = 1.0
        result["is_active"] = True
        result["duration_month"] = pd.NA
        result["duration_year"] = pd.NA
        result["is_renewal_period"] = False

        dates = self.dates
        if dates is None:
            return result

        if dates.entry_date is not None:
            entry = pd.to_datetime(result[dates.entry_date], errors="coerce")
            start = pd.to_datetime(result["period_start"])
            end = pd.to_datetime(result["period_end"])
            months = (start.dt.year - entry.dt.year) * 12 + (start.dt.month - entry.dt.month)
            result["duration_month"] = months.where(entry.notna())
            result["duration_year"] = (months // 12).where(entry.notna())
        else:
            entry = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

        if dates.exit_date is not None:
            exit_date = pd.to_datetime(result[dates.exit_date], errors="coerce")
        else:
            exit_date = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

        if dates.entry_date is not None or dates.exit_date is not None:
            period_start = pd.to_datetime(result["period_start"])
            period_end = pd.to_datetime(result["period_end"])
            if dates.exposure_timing == "whole_period":
                active = (entry.isna() | (entry <= period_end)) & (
                    exit_date.isna() | (exit_date >= period_start)
                )
                result["is_active"] = active
                result["active_fraction"] = active.astype(float)
            else:
                fractions: list[float] = []
                for ps, pe, ent, ext in zip(
                    period_start,
                    period_end,
                    entry,
                    exit_date,
                    strict=True,
                ):
                    effective_start = ps if pd.isna(ent) else max(ps, ent)
                    effective_end = pe if pd.isna(ext) else min(pe, ext)
                    if effective_end < effective_start:
                        fractions.append(0.0)
                    else:
                        active_days = (effective_end - effective_start).days + 1
                        period_days = (pe - ps).days + 1
                        fractions.append(active_days / period_days)
                result["active_fraction"] = fractions
                result["is_active"] = result["active_fraction"] > 0

        if dates.renewal_date is not None:
            renewal = pd.to_datetime(result[dates.renewal_date], errors="coerce")
            period_start = pd.to_datetime(result["period_start"])
            period_end = pd.to_datetime(result["period_end"])
            flags: list[bool] = []
            for ps, pe, renewal_date in zip(
                period_start, period_end, renewal, strict=True
            ):
                if pd.isna(renewal_date):
                    flags.append(False)
                    continue
                year = ps.year
                day = min(renewal_date.day, pd.Timestamp(year, renewal_date.month, 1).days_in_month)
                anniversary = pd.Timestamp(year, renewal_date.month, day)
                if anniversary < renewal_date:
                    flags.append(False)
                else:
                    flags.append(ps <= anniversary <= pe)
            result["is_renewal_period"] = flags
        return result


@dataclass
class ProjectionDataset:
    """Projection records plus named tables at independently declared grains."""

    records: ProjectionData
    tables: dict[str, ProjectionTable] = field(default_factory=dict)

    def add_table(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        keys: str | Iterable[str],
        allow_duplicates: bool = False,
    ) -> "ProjectionDataset":
        if name in self.tables:
            raise ValidationError(f"table {name!r} already exists")
        self.tables[name] = ProjectionTable(
            name=name,
            frame=frame.copy(),
            keys=_as_tuple(keys),
            allow_duplicates=allow_duplicates,
        )
        return self

    def get_table(self, name: str) -> ProjectionTable:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise ValidationError(f"table {name!r} is not present") from exc

    def merge_tables(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Join every table whose keys are available in ``frame``.

        Tables are many-to-one by default, so entity-level or period-level values
        may be broadcast to finer claim-type or coverage rows without multiplying
        the underlying table.
        """

        result = frame.copy()
        for table in self.tables.values():
            missing = [key for key in table.keys if key not in result.columns]
            if missing:
                raise ValidationError(
                    f"cannot join table {table.name!r}; projection rows lack keys {missing}"
                )
            overlapping = [
                column
                for column in table.frame.columns
                if column in result.columns and column not in table.keys
            ]
            if overlapping:
                raise ValidationError(
                    f"table {table.name!r} would overwrite columns {overlapping}"
                )
            validate = "many_to_many" if table.allow_duplicates else "many_to_one"
            result = result.merge(
                table.frame,
                on=list(table.keys),
                how="left",
                validate=validate,
                sort=False,
            )
        return result
