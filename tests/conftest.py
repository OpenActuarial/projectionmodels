from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def fake_actuarialpy(monkeypatch):
    module = types.ModuleType("actuarialpy")

    def safe_divide(a, b):
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            left = pd.Series(a) if not isinstance(a, pd.Series) else a
            right = pd.Series(b, index=left.index) if not isinstance(b, pd.Series) else b
            return left / right.replace(0, np.nan)
        return np.divide(a, b)

    module.per_exposure = safe_divide
    module.trend_factor = lambda annual_trend, months: (1 + annual_trend) ** (
        np.asarray(months, dtype=float) / 12.0
    )
    module.credibility_weighted_estimate = (
        lambda observed, complement, z: z * observed + (1 - z) * complement
    )
    module.limited_fluctuation_z = lambda exposure, standard: np.minimum(
        1.0, np.sqrt(np.maximum(np.asarray(exposure, dtype=float), 0) / standard)
    )

    def _season_values(values, freq):
        dates = pd.to_datetime(values)
        if freq.upper().startswith("M"):
            return dates.dt.month if isinstance(dates, pd.Series) else dates.month
        if freq.upper().startswith("Q"):
            return dates.dt.quarter if isinstance(dates, pd.Series) else dates.quarter
        return np.ones(len(dates), dtype=int)

    def _factor_rows(df, factors, date_col, freq, by, factor_col, season_name):
        seasons = pd.Series(_season_values(df[date_col], freq), index=df.index)
        if isinstance(factors, pd.Series):
            return seasons.map(factors)
        keys = ([] if by is None else ([by] if isinstance(by, str) else list(by)))
        left = df.loc[:, keys].copy()
        left[season_name] = seasons.to_numpy()
        left["__row__"] = np.arange(len(left))
        merged = left.merge(
            factors.loc[:, keys + [season_name, factor_col]],
            on=keys + [season_name],
            how="left",
            validate="many_to_one",
        ).sort_values("__row__")
        return pd.Series(merged[factor_col].to_numpy(), index=df.index)

    def deseasonalize(
        df,
        factors,
        *,
        date_col,
        value_col,
        freq="M",
        by=None,
        factor_col="seasonal_factor",
        season_name="season",
        out_col=None,
        copy=True,
    ):
        result = df.copy() if copy else df
        factor = _factor_rows(
            result, factors, date_col, freq, by, factor_col, season_name
        )
        result[out_col or f"{value_col}_deseasonalized"] = result[value_col] / factor
        return result

    def apply_seasonality(
        df,
        factors,
        *,
        date_col,
        value_col,
        freq="M",
        by=None,
        factor_col="seasonal_factor",
        season_name="season",
        out_col=None,
        copy=True,
    ):
        result = df.copy() if copy else df
        factor = _factor_rows(
            result, factors, date_col, freq, by, factor_col, season_name
        )
        result[out_col or f"{value_col}_seasonalized"] = result[value_col] * factor
        return result

    module.deseasonalize = deseasonalize
    module.apply_seasonality = apply_seasonality

    def apply_completion(
        df,
        factors,
        *,
        value_col,
        date_col=None,
        valuation_date=None,
        development_col=None,
        by=None,
        factor_col="completion_factor",
        development_name="development_month",
        out_col=None,
        copy=True,
    ):
        result = df.copy() if copy else df
        if development_col is not None:
            development = result[development_col]
        else:
            valuation = pd.Timestamp(valuation_date)
            origin = pd.to_datetime(result[date_col])
            development = (valuation.year - origin.dt.year) * 12 + (
                valuation.month - origin.dt.month
            )
        if isinstance(factors, pd.Series):
            factor = development.map(factors)
        else:
            keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
            left = result.loc[:, keys].copy()
            left[development_name] = development.to_numpy()
            left["__row__"] = np.arange(len(left))
            merged = left.merge(
                factors.loc[:, keys + [development_name, factor_col]],
                on=keys + [development_name],
                how="left",
                validate="many_to_one",
            ).sort_values("__row__")
            factor = pd.Series(merged[factor_col].to_numpy(), index=result.index)
        result[out_col or f"{value_col}_completed"] = result[value_col] / factor
        return result

    module.apply_completion = apply_completion

    def completion_factors_by(
        df,
        *,
        groupby,
        origin_col,
        valuation_col,
        amount_col,
        cumulative=True,
        method="volume",
        tail=1.0,
        on_insufficient="raise",
        warn=True,
        development_name="development_month",
    ):
        groups = [groupby] if isinstance(groupby, str) else list(groupby)
        rows = []
        for key in df[groups].drop_duplicates().itertuples(index=False, name=None):
            for development in range(25):
                rows.append(
                    {
                        **dict(zip(groups, key, strict=True)),
                        development_name: development,
                        "completion_factor": min(1.0, 0.5 + development / 24),
                    }
                )
        return pd.DataFrame(rows)

    module.completion_factors_by = completion_factors_by

    def make_completion_triangle(*args, **kwargs):
        return pd.DataFrame([[50.0, 75.0], [60.0, 90.0]])

    def completion_factors(*args, **kwargs):
        return pd.Series([0.75, 1.0], index=pd.Index([0, 1], name="development_month"))

    module.make_completion_triangle = make_completion_triangle
    module.completion_factors = completion_factors

    @dataclass
    class Fit:
        annual_trend: float = 0.12
        r_squared: float = 0.9
        ci_low: float = 0.08
        ci_high: float = 0.16
        n_periods: int = 12

    module.fit_trend = lambda *args, **kwargs: Fit()

    def seasonality_factors(*args, **kwargs):
        return pd.Series(
            np.ones(12), index=pd.Index(range(1, 13), name="season"), name="seasonal_factor"
        )

    def seasonality_factors_by(df, *, groupby, season_name="season", **kwargs):
        groups = [groupby] if isinstance(groupby, str) else list(groupby)
        rows = []
        for key in df[groups].drop_duplicates().itertuples(index=False, name=None):
            for season in range(1, 13):
                rows.append({**dict(zip(groups, key)), season_name: season, "seasonal_factor": 1.0})
        return pd.DataFrame(rows)

    module.seasonality_factors = seasonality_factors
    module.seasonality_factors_by = seasonality_factors_by


    class BuhlmannModel:
        def __init__(self, matrix):
            self.matrix = np.asarray(matrix, dtype=float)
            self.risk_means = self.matrix.mean(axis=1)
            self.overall_mean = float(self.risk_means.mean())
            self.epv = 2.0
            self.vhm = 1.0
            self.z = 0.6

        def premium(self, risk_means):
            risk_means = np.asarray(risk_means, dtype=float)
            return self.z * risk_means + (1 - self.z) * self.overall_mean

    class Buhlmann:
        @staticmethod
        def fit(matrix):
            return BuhlmannModel(matrix)

    class BuhlmannStraubModel:
        def __init__(self, frame, group, value, weight):
            grouped = frame.groupby(group, sort=True)
            self.groups_ = list(grouped.groups)
            self.weights = grouped[weight].sum().to_numpy(dtype=float)
            weighted_sum = grouped.apply(
                lambda part: np.average(part[value], weights=part[weight]),
                include_groups=False,
            )
            self.risk_means_ = weighted_sum.to_numpy(dtype=float)
            self.overall_mean = float(
                np.average(self.risk_means_, weights=self.weights)
            )
            self.epv = 3.0
            self.vhm = 1.5

        def z(self, weights):
            weights = np.asarray(weights, dtype=float)
            return weights / (weights + 10.0)

        def premium(self, risk_means, weights):
            z = self.z(weights)
            return z * np.asarray(risk_means, dtype=float) + (1 - z) * self.overall_mean

    class BuhlmannStraub:
        @staticmethod
        def from_frame(frame, *, group, value, weight, period=None):
            return BuhlmannStraubModel(frame, group, value, weight)

    module.Buhlmann = Buhlmann
    module.BuhlmannStraub = BuhlmannStraub

    monkeypatch.setitem(sys.modules, "actuarialpy", module)
    yield module
