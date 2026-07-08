"""Shared fixtures for tests that exercise the real actuarialpy dependency."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def completion_transactions() -> pd.DataFrame:
    """A valid incremental payment triangle for two claim types.

    Each segment has overlapping origins at development months 0, 1, and 2,
    producing completion factors 0.5, 0.8, and 1.0.
    """

    rows: list[dict[str, object]] = []
    for claim_type, scale in (("ip", 1.0), ("op", 0.6)):
        origins = pd.date_range("2024-01-01", periods=4, freq="MS")
        for origin_index, origin in enumerate(origins):
            maximum_development = 2 if origin_index <= 1 else 1 if origin_index == 2 else 0
            origin_scale = 1.0 + 0.2 * origin_index
            for development, payment in enumerate((50.0, 30.0, 20.0)):
                if development > maximum_development:
                    continue
                rows.append(
                    {
                        "claim_type": claim_type,
                        "incurred_month": origin,
                        "paid_month": origin + pd.DateOffset(months=development),
                        "paid_claims": payment * origin_scale * scale,
                    }
                )
    return pd.DataFrame(rows)
