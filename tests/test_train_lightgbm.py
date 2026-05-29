import pandas as pd

from power_forecast.models.train_compare import get_feature_columns


def test_get_feature_columns_excludes_timestamp_and_target():
    df = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01"],
            "demand_mwh": [100.0],
            "temperature_2m": [20.0],
            "demand_lag_1h": [95.0],
        }
    )

    features = get_feature_columns(
        df=df,
        timestamp_column="timestamp_utc",
        target_column="demand_mwh",
    )

    assert "timestamp_utc" not in features
    assert "demand_mwh" not in features
    assert "temperature_2m" in features
    assert "demand_lag_1h" in features
