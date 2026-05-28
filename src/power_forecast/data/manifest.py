from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def dataframe_summary(df: pd.DataFrame, timestamp_col: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
    }

    if timestamp_col and timestamp_col in df.columns and len(df) > 0:
        ts = pd.to_datetime(df[timestamp_col])
        summary["min_timestamp"] = str(ts.min())
        summary["max_timestamp"] = str(ts.max())

    return summary


def write_manifest(
    output_path: str | Path,
    eia_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.get("project", {}),
        "sources": {
            "eia": {
                "respondent": config["eia"]["respondent"],
                "type": config["eia"]["type"],
                "summary": dataframe_summary(eia_df, "timestamp_utc"),
            },
            "open_meteo": {
                "location_name": config["weather"]["location_name"],
                "latitude": config["weather"]["latitude"],
                "longitude": config["weather"]["longitude"],
                "timezone": config["weather"]["timezone"],
                "summary": dataframe_summary(weather_df, "timestamp_local"),
            },
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
