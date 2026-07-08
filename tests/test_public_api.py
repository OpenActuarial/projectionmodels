from __future__ import annotations

import warnings

import projectionmodels as pm
import projectionmodels.advanced as advanced
from projectionmodels.integrations import actuarialpy as apx


def test_package_root_is_focused_on_concrete_workflows():
    expected = {
        "Adjustment",
        "Assumption",
        "ClaimExperience",
        "ClaimProjection",
        "CompletionAssumption",
        "CredibilityAssumption",
        "DateCohort",
        "ExpenseProjection",
        "PremiumProjection",
        "ProjectionDates",
        "ProjectionHorizon",
        "ProjectionModelsError",
        "ProjectionResults",
        "RenewalRateActions",
        "Scenario",
        "SeasonalityAssumption",
        "TrendAssumption",
        "ValidationError",
    }
    assert set(pm.__all__) == expected
    assert "ProjectionModel" not in pm.__all__
    assert "Calculation" not in pm.__all__
    assert "ProjectionData" not in pm.__all__


def test_advanced_engine_has_an_explicit_namespace():
    assert advanced.ProjectionModel is not None
    assert advanced.ProjectionData is not None
    assert advanced.RollForward is not None


def test_old_advanced_root_names_warn_during_migration():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = pm.ProjectionModel
    assert value is advanced.ProjectionModel
    assert any(item.category is DeprecationWarning for item in caught)


def test_actuarialpy_estimation_is_explicitly_namespaced():
    assert callable(apx.estimate_completion)
    assert callable(apx.estimate_seasonality)
    assert callable(apx.estimate_trend)
    assert callable(apx.estimate_credibility)
    assert callable(apx.remove_seasonality)
