import actuarialpy as ap
import pandas as pd
import pytest

from projectionmodels.integrations.actuarialpy import (
    estimate_completion,
    estimate_credibility,
    estimate_seasonality,
    estimate_trend,
)


def test_actuarialpy_estimated_assumptions_match_real_library():
    history = pd.DataFrame(
        {
            "claim_type": ["ip"] * 36 + ["op"] * 36,
            "month": list(pd.date_range("2024-01-01", periods=36, freq="MS")) * 2,
            "claims": [100.0 * (1.006**i) for i in range(36)]
            + [50.0 * (1.010**i) for i in range(36)],
            "exposure": [10.0] * 72,
        }
    )

    trend = estimate_trend(
        "claim_trend",
        history,
        by=["claim_type"],
        date_col="month",
        value_col="claims",
        exposure_col="exposure",
    )
    seasonality = estimate_seasonality(
        "claim_seasonality",
        history,
        by=["claim_type"],
        date_col="month",
        value_col="claims",
        exposure_col="exposure",
    )
    credibility = estimate_credibility(
        "claim_credibility",
        history,
        method="limited_fluctuation",
        by=["claim_type"],
        exposure_col="exposure",
        full_credibility_standard=400.0,
    )

    projection_rows = pd.DataFrame(
        {"claim_type": ["ip", "op"], "season": [1, 2]}
    )
    expected_trend = [
        ap.fit_trend(
            history.loc[history["claim_type"] == claim_type],
            date_col="month",
            value_col="claims",
            exposure_col="exposure",
        ).annual_trend
        for claim_type in ("ip", "op")
    ]
    expected_seasonality_table = ap.seasonality_factors_by(
        history,
        groupby=["claim_type"],
        date_col="month",
        value_col="claims",
        exposure_col="exposure",
        season_name="season",
    )
    expected_seasonality = (
        projection_rows.merge(
            expected_seasonality_table,
            on=["claim_type", "season"],
            how="left",
            validate="one_to_one",
        )["seasonal_factor"]
        .tolist()
    )
    expected_z = ap.limited_fluctuation_z(360.0, 400.0)

    assert trend.resolve(projection_rows).tolist() == pytest.approx(expected_trend)
    assert seasonality.resolve(projection_rows).tolist() == pytest.approx(
        expected_seasonality
    )
    assert credibility.resolve(projection_rows).tolist() == pytest.approx(
        [expected_z, expected_z]
    )
    assert trend.source == "actuarialpy_estimate"
    assert seasonality.source == "actuarialpy_estimate"
    assert credibility.source == "actuarialpy_estimate"


def test_actuarialpy_estimated_completion_can_be_applied(completion_transactions):
    completion = estimate_completion(
        "claim_completion",
        completion_transactions,
        by=["claim_type"],
        origin_col="incurred_month",
        valuation_col="paid_month",
        amount_col="paid_claims",
    )
    expected = ap.completion_factors_by(
        completion_transactions,
        groupby=["claim_type"],
        origin_col="incurred_month",
        valuation_col="paid_month",
        amount_col="paid_claims",
    )
    pd.testing.assert_frame_equal(
        completion.values.reset_index(drop=True),
        expected.reset_index(drop=True),
    )

    observed = pd.DataFrame(
        {
            "claim_type": ["ip", "op"],
            "development_month": [0, 2],
            "reported_claims": [50.0, 75.0],
        }
    )
    completed = completion.apply(
        observed,
        value_col="reported_claims",
        development_col="development_month",
        by=["claim_type"],
        out_col="ultimate_claims",
    )
    assert completed["ultimate_claims"].tolist() == pytest.approx([100.0, 75.0])
    assert completion.source == "actuarialpy_estimate"


def test_tests_use_the_installed_actuarialpy_package():
    import importlib.metadata
    import importlib.util
    import re
    from pathlib import Path

    # The installed actuarialpy must satisfy the floor this package declares --
    # derived from the pin, not hardcoded, so bumping the floor can't strand
    # this guard on a stale range.
    requirement = next(
        r for r in importlib.metadata.requires("projectionmodels")
        if r.startswith("actuarialpy")
    )
    floor = re.search(r">=\s*([0-9][0-9.]*)", requirement).group(1)
    installed = tuple(int(p) for p in ap.__version__.split(".")[:3])
    minimum = tuple(int(p) for p in floor.split(".")[:3])
    assert installed >= minimum, (
        f"installed actuarialpy {ap.__version__} is below the declared floor {floor}"
    )
    assert ap.__file__ is not None
    # The imported module must be the one the import system resolves (an
    # injected fake in sys.modules would fail this) and must not resolve from
    # THIS repo's own source tree -- i.e. a vendored or stray copy under src/.
    # A venv created inside the checkout dir is fine (the min-deps CI job does
    # exactly that); an installed package lives in site-packages, not src/.
    spec = importlib.util.find_spec("actuarialpy")
    assert spec is not None and spec.origin is not None
    ap_path = Path(ap.__file__).resolve()
    assert ap_path == Path(spec.origin).resolve()
    repo_src = (Path(__file__).resolve().parent.parent / "src").resolve()
    assert not ap_path.is_relative_to(repo_src), (
        f"actuarialpy resolved from this repository's source tree: {ap_path}"
    )
