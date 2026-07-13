"""Specialized assumptions reject out-of-domain supplied values at construction.

Previously any subclass could be built with an arbitrary payload -- a completion
factor above 1, a negative credibility weight, a -100% trend -- and the bad value
only surfaced (if at all) far downstream. Each type now validates its domain when
constructed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projectionmodels.assumptions import (
    CompletionAssumption,
    CredibilityAssumption,
    SeasonalityAssumption,
    TrendAssumption,
)
from projectionmodels.exceptions import ValidationError


# --------------------------------------------------------------------------- #
# completion factors: (0, 1]
# --------------------------------------------------------------------------- #
def test_completion_accepts_valid_and_rejects_out_of_range():
    CompletionAssumption.from_values("cf", 0.8)  # ok
    CompletionAssumption.from_values("cf", 1.0)  # boundary ok
    for bad in (1.5, 0.0, -0.2, np.nan):
        with pytest.raises(ValidationError, match="completion"):
            CompletionAssumption.from_values("cf", bad)


def test_completion_validates_dataframe_column():
    good = pd.DataFrame({"development_month": [12, 24], "completion_factor": [0.5, 0.9]})
    CompletionAssumption.from_values("cf", good)
    bad = pd.DataFrame({"development_month": [12, 24], "completion_factor": [0.5, 1.4]})
    with pytest.raises(ValidationError, match="completion"):
        CompletionAssumption.from_values("cf", bad)


# --------------------------------------------------------------------------- #
# credibility weights: [0, 1]
# --------------------------------------------------------------------------- #
def test_credibility_bounds():
    CredibilityAssumption.from_weights("z", 0.0)
    CredibilityAssumption.from_weights("z", 1.0)
    for bad in (-0.01, 1.01, np.nan):
        with pytest.raises(ValidationError, match="credibility"):
            CredibilityAssumption.from_weights("z", bad)


# --------------------------------------------------------------------------- #
# trend rate: finite, > -1
# --------------------------------------------------------------------------- #
def test_trend_rate_domain():
    TrendAssumption.from_values("t", 0.07)
    TrendAssumption.from_values("t", -0.5)
    for bad in (-1.0, -2.0, np.inf, np.nan):
        with pytest.raises(ValidationError, match="trend"):
            TrendAssumption.from_values("t", bad)


# --------------------------------------------------------------------------- #
# seasonality multipliers: finite, positive
# --------------------------------------------------------------------------- #
def test_seasonality_positive():
    SeasonalityAssumption.from_values("s", pd.Series([0.9, 1.1, 1.0], index=[1, 2, 3]))
    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError, match="seasonality"):
            SeasonalityAssumption.from_values("s", bad)


def test_base_assumption_has_no_domain():
    # a plain named value can be anything -- only the specialized types constrain
    from projectionmodels.assumptions import Assumption

    Assumption("raw", -999.0)  # no raise
    Assumption("raw", 42.0)
