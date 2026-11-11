from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from power_forecast.data.locations import combine_locations, get_with_retry

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_open_meteo_weather_locations(
    locations: list[dict],
    timezone_name: str,
    lookback_days: int,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fetch hourly history for several locations and store their population-weighted mean."""
    frames = {
        loc["slug"]: fetch_open_meteo_weather(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            timezone_name=timezone_name,
            lookback_days=lookback_days,
            output_path=None,
        )
        for loc in locations
    }
    df = combine_locations(frames, locations, key="timestamp_local")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    timezone_name: str,
    lookback_days: int,
    output_path: str | Path | None,
) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo (written to ``output_path`` if given)."""
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
        ],
        "timezone": timezone_name,
    }

    payload = get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"Open-Meteo returned no hourly data. Payload keys: {payload.keys()}")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_local"})

    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    for col in df.columns:
        if col != "timestamp_local":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp_local").reset_index(drop=True)

    if output_path is None:
        return df
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
