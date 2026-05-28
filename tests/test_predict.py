import pytest
import pandas as pd

from power_forecast.models.predict import _validate_feature_schema


def test_validate_feature_schema_passes_when_required_features_exist():
    df = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01"],
            "demand_mwh": [100.0],
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )

    _validate_feature_schema(df, ["feature_a", "feature_b"])


def test_validate_feature_schema_fails_when_feature_missing():
    df = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01"],
            "demand_mwh": [100.0],
            "feature_a": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing model feature columns"):
        _validate_feature_schema(df, ["feature_a", "feature_b"])
