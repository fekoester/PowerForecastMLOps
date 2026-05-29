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
    
def test_get_feature_columns_forecast_safe_filters_short_lags_and_rolls():
    df = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01"],
            "demand_mwh": [100.0],
            "temperature_2m": [20.0],
            "hour": [1],
            "demand_lag_1h": [99.0],
            "demand_lag_24h": [95.0],
            "demand_lag_48h": [93.0],
            "demand_lag_168h": [90.0],
            "demand_roll_mean_24h": [96.0],
        }
    )

    features = get_feature_columns(
        df=df,
        timestamp_column="timestamp_utc",
        target_column="demand_mwh",
        forecast_safe_features=True,
        allowed_lag_hours=[24, 48, 168],
    )

    assert "temperature_2m" in features
    assert "hour" in features

    assert "demand_lag_1h" not in features
    assert "demand_lag_24h" in features
    assert "demand_lag_48h" in features
    assert "demand_lag_168h" in features
    assert "demand_roll_mean_24h" not in features
