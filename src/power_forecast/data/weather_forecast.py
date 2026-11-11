from __future__ import annotations

from pathlib import Path

import pandas as pd

from power_forecast.data.locations import combine_locations, get_with_retry

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo_forecast(
    latitude: float,
    longitude: float,
    timezone_name: str,
    forecast_days: int = 3,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "forecast_days": forecast_days,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
        ],
    }

    payload = get_with_retry(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    hourly = payload["hourly"]

    df = pd.DataFrame(
        {
            "timestamp_local": hourly["time"],
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "shortwave_radiation": hourly["shortwave_radiation"],
        }
    )

    ts_local = pd.to_datetime(df["timestamp_local"])
    ts_local = ts_local.dt.tz_localize(
        timezone_name,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    df["timestamp_utc"] = ts_local.dt.tz_convert("UTC")

    df = df[
        [
            "timestamp_utc",
            "timestamp_local",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
        ]
    ].copy()

    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    return df


def fetch_open_meteo_forecast_locations(
    locations: list[dict],
    timezone_name: str,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Forecast for several locations, combined like the historical weather."""
    frames = {
        loc["slug"]: fetch_open_meteo_forecast(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            timezone_name=timezone_name,
            forecast_days=forecast_days,
        )
        for loc in locations
    }
    if len(locations) == 1:
        return frames[locations[0]["slug"]]
    combined = combine_locations(
        {slug: frame.drop(columns=["timestamp_local"]) for slug, frame in frames.items()},
        locations,
        key="timestamp_utc",
    )
    local = frames[locations[0]["slug"]][["timestamp_utc", "timestamp_local"]]
    return local.merge(combined, on="timestamp_utc", how="left")


def save_weather_forecast(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
