"""Explicit adapters that estimate projection assumptions with actuarialpy.

The core projection workflows consume selected assumptions.  These helpers make
assumption estimation available without making it appear to be part of the
projection engine itself.  Each function returns the same assumption container
used by the workflows, preserving indicated values, diagnostics, and later
actuarial selections.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..assumptions import (
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)


def estimate_trend(
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
    """Estimate an annual trend assumption with ``actuarialpy.fit_trend``."""

    return TrendAssumption.from_experience(
        name,
        experience,
        date_col=date_col,
        value_col=value_col,
        exposure_col=exposure_col,
        by=by,
        freq=freq,
        min_periods=min_periods,
        confidence=confidence,
    )


def estimate_seasonality(
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
    """Estimate normalized seasonal factors with actuarialpy."""

    return SeasonalityAssumption.from_experience(
        name,
        experience,
        date_col=date_col,
        value_col=value_col,
        exposure_col=exposure_col,
        by=by,
        freq=freq,
        method=method,
        aggregate=aggregate,
        min_years=min_years,
        season_col=season_col,
    )


def estimate_completion(
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
    """Estimate completion factors from an overlapping development triangle."""

    return CompletionAssumption.from_experience(
        name,
        experience,
        origin_col=origin_col,
        valuation_col=valuation_col,
        amount_col=amount_col,
        by=by,
        cumulative=cumulative,
        method=method,
        tail=tail,
        on_insufficient=on_insufficient,
        development_col=development_col,
    )


def estimate_credibility(
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
    """Estimate limited-fluctuation, Bühlmann, or Bühlmann–Straub credibility."""

    return CredibilityAssumption.from_experience(
        name,
        experience,
        method=method,
        by=by,
        exposure_col=exposure_col,
        full_credibility_standard=full_credibility_standard,
        value_col=value_col,
        period_col=period_col,
        weight_col=weight_col,
    )


def remove_seasonality(
    frame: pd.DataFrame,
    assumption: SeasonalityAssumption,
    *,
    date_col: str,
    value_col: str,
    by: str | Iterable[str] | None = None,
    out_col: str = "deseasonalized_value",
) -> pd.DataFrame:
    """Remove selected seasonal factors from an experience series."""

    from ..actuarialpy_adapter import actuarialpy_function

    function = actuarialpy_function("deseasonalize")
    groups = [by] if isinstance(by, str) else list(by or ())
    return function(
        frame,
        assumption.values,
        date_col=date_col,
        value_col=value_col,
        freq=assumption.frequency,
        by=groups or None,
        factor_col=assumption.value_col or assumption.name,
        season_name=assumption.season_col,
        out_col=out_col,
    )
