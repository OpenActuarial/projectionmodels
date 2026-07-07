"""Projection horizon and period construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .exceptions import ValidationError

_FREQUENCIES = {
    "monthly": ("MS", "M", 1.0 / 12.0),
    "month": ("MS", "M", 1.0 / 12.0),
    "m": ("MS", "M", 1.0 / 12.0),
    "quarterly": ("QS", "Q", 0.25),
    "quarter": ("QS", "Q", 0.25),
    "q": ("QS", "Q", 0.25),
    "annual": ("YS", "Y", 1.0),
    "yearly": ("YS", "Y", 1.0),
    "year": ("YS", "Y", 1.0),
    "y": ("YS", "Y", 1.0),
}


@dataclass(frozen=True)
class ProjectionHorizon:
    """A deterministic projection timeline.

    Parameters
    ----------
    start:
        First projection date. It is normalized to the beginning of the
        containing month, quarter, or year.
    periods:
        Number of projection periods. Supply either ``periods`` or ``end``.
    end:
        Last date to include. Supply either ``periods`` or ``end``.
    frequency:
        ``"monthly"``, ``"quarterly"``, or ``"annual"``.
    """

    start: Any
    periods: int | None = None
    end: Any | None = None
    frequency: str = "monthly"

    def __post_init__(self) -> None:
        key = str(self.frequency).lower()
        if key not in _FREQUENCIES:
            raise ValidationError(
                "frequency must be monthly, quarterly, or annual"
            )
        if (self.periods is None) == (self.end is None):
            raise ValidationError("supply exactly one of periods or end")
        if self.periods is not None and self.periods <= 0:
            raise ValidationError("periods must be positive")

    @property
    def _spec(self) -> tuple[str, str, float]:
        return _FREQUENCIES[str(self.frequency).lower()]

    @property
    def pandas_frequency(self) -> str:
        return self._spec[0]

    @property
    def period_frequency(self) -> str:
        return self._spec[1]

    @property
    def year_fraction(self) -> float:
        return self._spec[2]

    @property
    def normalized_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start).to_period(self.period_frequency).start_time

    def to_frame(self) -> pd.DataFrame:
        """Return one row per projection period."""

        start = self.normalized_start
        if self.periods is not None:
            starts = pd.date_range(
                start=start, periods=self.periods, freq=self.pandas_frequency
            )
        else:
            end = pd.Timestamp(self.end).to_period(self.period_frequency).start_time
            if end < start:
                raise ValidationError("end must not precede start")
            starts = pd.date_range(start=start, end=end, freq=self.pandas_frequency)

        periods = starts.to_period(self.period_frequency)
        frame = pd.DataFrame(
            {
                "projection_index": range(len(starts)),
                "projection_period": periods.astype(str),
                "period_start": starts,
                "period_end": periods.end_time.normalize(),
                "period_midpoint": starts
                + (periods.end_time.normalize() - starts) / 2,
                "calendar_year": starts.year,
                "calendar_quarter": starts.quarter,
                "calendar_month": starts.month,
                "year_fraction": self.year_fraction,
            }
        )
        if self.period_frequency == "M":
            frame["season"] = frame["calendar_month"]
        elif self.period_frequency == "Q":
            frame["season"] = frame["calendar_quarter"]
        else:
            frame["season"] = 1
        return frame

    def __len__(self) -> int:
        return len(self.to_frame())
