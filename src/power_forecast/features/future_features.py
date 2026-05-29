from __future__ import annotations

from pathlib import Path

import pandas as pd

from power_forecast.data.weather_forecast import (
    fetch_open_meteo_forecast,
    save_weather_forecast,
)
from power_forecast.features.build_features import (
    _add_calendar_features,
    _add_cyclic_calendar_features,
    _add_weather_features,
)


def _add_future_origin_rolling_features(
    future: pd.DataFrame,
    history: pd.Series,
    windows_hours: list[int],
) -> pd.DataFrame:
    out = future.copy()

    for window in windows_hours:
        means = []
        stds = []
        mins = []
        maxs = []

        for ts in out["timestamp_utc"]:
            end_ts = ts - pd.Timedelta(hours=24)
            start_ts = end_ts - pd.Timedelta(hours=window - 1)

            values = history.loc[start_ts:end_ts]

            if len(values) < window:
                raise RuntimeError(
                    f"Not enough history for demand_origin_roll_{window}h at {ts}. "
                    f"Needed {window}, got {len(values)}."
                )

            means.append(float(values.mean()))
            stds.append(float(values.std()))
            mins.append(float(values.min()))
            maxs.append(float(values.max()))

        out[f"demand_origin_roll_mean_{window}h"] = means
        out[f"demand_origin_roll_std_{window}h"] = stds
        out[f"demand_origin_roll_min_{window}h"] = mins
        out[f"demand_origin_roll_max_{window}h"] = maxs

    return out


def _add_future_same_hour_features(
    future: pd.DataFrame,
    history: pd.Series,
    windows_days: list[int],
) -> pd.DataFrame:
    out = future.copy()

    for window_days in windows_days:
        means = []
        stds = []
        mins = []
        maxs = []

        for ts in out["timestamp_utc"]:
            values = []

            for day in range(1, window_days + 1):
                lag_ts = ts - pd.Timedelta(hours=24 * day)
                if lag_ts not in history.index:
                    raise RuntimeError(
                        f"Missing history for demand_same_hour_{window_days}d at {ts}: {lag_ts}"
                    )
                values.append(float(history.loc[lag_ts]))

            values_s = pd.Series(values)
            means.append(float(values_s.mean()))
            stds.append(float(values_s.std()))
            mins.append(float(values_s.min()))
            maxs.append(float(values_s.max()))

        out[f"demand_same_hour_mean_{window_days}d"] = means
        out[f"demand_same_hour_std_{window_days}d"] = stds
        out[f"demand_same_hour_min_{window_days}d"] = mins
        out[f"demand_same_hour_max_{window_days}d"] = maxs

    return out


def build_future_24h_features(
    eia_path: str | Path,
    weather_forecast_output_path: str | Path,
    latitude: float,
    longitude: float,
    timezone_name: str,
    base_temperature_c: float,
    allowed_lag_hours: list[int],
    same_hour_windows_days: list[int],
    origin_rolling_windows_hours: list[int],
    output_path: str | Path,
    use_cyclic_calendar_features: bool,
) -> pd.DataFrame:
    eia = pd.read_csv(eia_path)
    eia["timestamp_utc"] = pd.to_datetime(eia["timestamp_utc"], utc=True)
    eia["demand_mwh"] = pd.to_numeric(eia["demand_mwh"], errors="coerce")
    eia = eia.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")

    latest_known_timestamp = eia["timestamp_utc"].max()

    future_timestamps = pd.date_range(
        start=latest_known_timestamp + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        tz="UTC",
    )

    weather_forecast = fetch_open_meteo_forecast(
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        forecast_days=3,
    )
    save_weather_forecast(weather_forecast, weather_forecast_output_path)

    future = pd.DataFrame({"timestamp_utc": future_timestamps})
    future = future.merge(
        weather_forecast.drop(columns=["timestamp_local"]),
        on="timestamp_utc",
        how="left",
    )

    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]

    future[weather_cols] = future[weather_cols].interpolate(limit_direction="both")

    if future[weather_cols].isna().any().any():
        missing = future[weather_cols].isna().sum().to_dict()
        raise RuntimeError(f"Future weather forecast has missing values: {missing}")

    future = _add_calendar_features(future, timezone_name=timezone_name)

    if use_cyclic_calendar_features:
        future = _add_cyclic_calendar_features(future)

    future = _add_weather_features(future, base_temperature_c=base_temperature_c)

    history = eia.set_index("timestamp_utc")["demand_mwh"]

    for lag in allowed_lag_hours:
        values = []

        for ts in future["timestamp_utc"]:
            lag_ts = ts - pd.Timedelta(hours=lag)

            if lag_ts not in history.index:
                raise RuntimeError(
                    f"Cannot create demand_lag_{lag}h for {ts}: "
                    f"missing historical demand at {lag_ts}"
                )

            values.append(float(history.loc[lag_ts]))

        future[f"demand_lag_{lag}h"] = values

    future = _add_future_origin_rolling_features(
        future=future,
        history=history,
        windows_hours=origin_rolling_windows_hours,
    )

    future = _add_future_same_hour_features(
        future=future,
        history=history,
        windows_days=same_hour_windows_days,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    future.to_csv(output_path, index=False)

    return future
