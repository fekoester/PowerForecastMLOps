"""Weather locations and population-weighted aggregation shared by history and forecast ingestion."""
from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
import requests

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def weather_locations(weather_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns [{name, slug, latitude, longitude, weight}] from the weather config.

    Uses ``weather.locations`` if present, otherwise the single ``latitude``/``longitude``
    (backwards compatible with older configs).
    """
    locations = weather_cfg.get("locations")
    if not locations:
        return [
            {
                "name": str(weather_cfg.get("location_name", "location")),
                "slug": _slug(str(weather_cfg.get("location_name", "location"))),
                "latitude": float(weather_cfg["latitude"]),
                "longitude": float(weather_cfg["longitude"]),
                "weight": 1.0,
            }
        ]
    out = []
    for loc in locations:
        out.append(
            {
                "name": str(loc["name"]),
                "slug": _slug(str(loc["name"])),
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "weight": float(loc.get("weight", 1.0)),
            }
        )
    if sum(loc["weight"] for loc in out) <= 0:
        raise ValueError("weather.locations weights must sum to a positive number")
    return out


def location_weights(weather_cfg: dict[str, Any]) -> dict[str, float]:
    """Weights by slug, as used for per-location degree-hours."""
    locs = weather_locations(weather_cfg)
    return {loc["slug"]: loc["weight"] for loc in locs} if len(locs) > 1 else {}


def get_with_retry(url: str, params: dict[str, Any], timeout: int, attempts: int = 5) -> dict:
    """GET with exponential backoff on rate limits (429) and transient server errors."""
    delay = 5.0
    for attempt in range(1, attempts + 1):
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


def combine_locations(
    frames: dict[str, pd.DataFrame],
    locations: list[dict[str, Any]],
    key: str,
) -> pd.DataFrame:
    """Population-weighted mean of every weather variable across locations.

    With a single location the frame is returned unchanged. With several, the result also keeps
    each location's temperature as ``temperature_2m__<slug>``.
    """
    if len(locations) == 1:
        return frames[locations[0]["slug"]]

    total = sum(loc["weight"] for loc in locations)
    base = frames[locations[0]["slug"]][[key]].copy()
    for variable in WEATHER_VARIABLES:
        weighted = None
        for loc in locations:
            series = frames[loc["slug"]].set_index(key)[variable] * (loc["weight"] / total)
            weighted = series if weighted is None else weighted.add(series)
        base = base.merge(weighted.rename(variable).reset_index(), on=key, how="left")
    for loc in locations:
        temps = frames[loc["slug"]][[key, "temperature_2m"]].rename(
            columns={"temperature_2m": f"temperature_2m__{loc['slug']}"}
        )
        base = base.merge(temps, on=key, how="left")
    return base
