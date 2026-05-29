from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd



def _load_validation_status(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation report not found: {path}. Run `make validate` before `make features`."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    return str(report.get("status", "missing"))


def _prepare_eia(eia_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(eia_path)

    required = ["timestamp_utc", "demand_mwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"EIA data missing required columns: {missing}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")

    df = df[["timestamp_utc", "demand_mwh"]].copy()
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    return df


def _prepare_weather(weather_path: str | Path, timezone_name: str) -> pd.DataFrame:
    df = pd.read_csv(weather_path)

    required = [
        "timestamp_local",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Weather data missing required columns: {missing}")

    # Open-Meteo gives local timestamps without timezone offset.
    # We localize them to the configured timezone, then convert to UTC
    # so they align with EIA timestamps.
    ts_local = pd.to_datetime(df["timestamp_local"])
    ts_local = ts_local.dt.tz_localize(timezone_name, ambiguous="infer", nonexistent="shift_forward")
    df["timestamp_utc"] = ts_local.dt.tz_convert("UTC")

    numeric_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["timestamp_utc", *numeric_cols]].copy()
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    return df


def _add_calendar_features(df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    out = df.copy()

    local_ts = out["timestamp_utc"].dt.tz_convert(timezone_name)

    out["hour"] = local_ts.dt.hour
    out["day_of_week"] = local_ts.dt.dayofweek
    out["day_of_month"] = local_ts.dt.day
    out["month"] = local_ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    return out


def _add_weather_features(df: pd.DataFrame, base_temperature_c: float) -> pd.DataFrame:
    out = df.copy()

    temp = out["temperature_2m"]
    out["cooling_degree"] = (temp - base_temperature_c).clip(lower=0)
    out["heating_degree"] = (base_temperature_c - temp).clip(lower=0)

    return out


def _add_lag_features(df: pd.DataFrame, lags_hours: list[int]) -> pd.DataFrame:
    out = df.copy()

    for lag in lags_hours:
        out[f"demand_lag_{lag}h"] = out["demand_mwh"].shift(lag)

    return out


def _add_rolling_features(df: pd.DataFrame, windows_hours: list[int]) -> pd.DataFrame:
    out = df.copy()

    # Critical leakage-prevention rule:
    # roll over shifted target, not current target.
    shifted = out["demand_mwh"].shift(1)

    for window in windows_hours:
        out[f"demand_roll_mean_{window}h"] = shifted.rolling(window=window).mean()
        out[f"demand_roll_std_{window}h"] = shifted.rolling(window=window).std()
        out[f"demand_roll_min_{window}h"] = shifted.rolling(window=window).min()
        out[f"demand_roll_max_{window}h"] = shifted.rolling(window=window).max()

    return out

def _add_origin_rolling_features(df: pd.DataFrame, windows_hours: list[int]) -> pd.DataFrame:
    """Rolling demand features available one day before the target timestamp.

    For target time t, this uses demand history ending at t-24h.
    That makes it safe for direct next-24h/day-ahead forecasting.
    """
    out = df.copy()

    shifted_by_day = out["demand_mwh"].shift(24)

    for window in windows_hours:
        out[f"demand_origin_roll_mean_{window}h"] = shifted_by_day.rolling(window=window).mean()
        out[f"demand_origin_roll_std_{window}h"] = shifted_by_day.rolling(window=window).std()
        out[f"demand_origin_roll_min_{window}h"] = shifted_by_day.rolling(window=window).min()
        out[f"demand_origin_roll_max_{window}h"] = shifted_by_day.rolling(window=window).max()

    return out


def _add_same_hour_history_features(df: pd.DataFrame, windows_days: list[int]) -> pd.DataFrame:
    """Same-hour historical demand statistics.

    For target time t and window 7d, this uses:
    demand(t-24h), demand(t-48h), ..., demand(t-168h).
    """
    out = df.copy()

    for window_days in windows_days:
        lagged_same_hour = [
            out["demand_mwh"].shift(24 * day)
            for day in range(1, window_days + 1)
        ]

        lagged_df = pd.concat(lagged_same_hour, axis=1)

        out[f"demand_same_hour_mean_{window_days}d"] = lagged_df.mean(axis=1)
        out[f"demand_same_hour_std_{window_days}d"] = lagged_df.std(axis=1)
        out[f"demand_same_hour_min_{window_days}d"] = lagged_df.min(axis=1)
        out[f"demand_same_hour_max_{window_days}d"] = lagged_df.max(axis=1)

    return out


def build_features(
    eia_path: str | Path,
    weather_path: str | Path,
    validation_report_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    timezone_name: str,
    lags_hours: list[int],
    rolling_windows_hours: list[int],
    same_hour_windows_days: list[int],
    origin_rolling_windows_hours: list[int],
    base_temperature_c: float,
) -> pd.DataFrame:
    validation_status = _load_validation_status(validation_report_path)
    if validation_status == "fail":
        raise RuntimeError(
            f"Raw data validation failed in {validation_report_path}. "
            "Refusing to build features."
        )

    eia = _prepare_eia(eia_path)
    weather = _prepare_weather(weather_path, timezone_name=timezone_name)

    df = pd.merge(eia, weather, on="timestamp_utc", how="left")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df = _add_calendar_features(df, timezone_name=timezone_name)
    df = _add_weather_features(df, base_temperature_c=base_temperature_c)
    df = _add_lag_features(df, lags_hours=lags_hours)
    df = _add_rolling_features(df, windows_hours=rolling_windows_hours)
    df = _add_origin_rolling_features(df, windows_hours=origin_rolling_windows_hours)
    df = _add_same_hour_history_features(df, windows_days=same_hour_windows_days)

    # Keep rows where all model features are known.
    # This drops the first 168 hours because lag_168h / rolling_168h need history.
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    after_drop = len(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    feature_columns = [c for c in df.columns if c not in ["timestamp_utc", "demand_mwh"]]

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "n_rows_before_dropna": int(before_drop),
        "n_rows_after_dropna": int(after_drop),
        "n_dropped_rows": int(before_drop - after_drop),
        "n_features": int(len(feature_columns)),
        "target_column": "demand_mwh",
        "timestamp_column": "timestamp_utc",
        "feature_columns": feature_columns,
        "min_timestamp": str(df["timestamp_utc"].min()) if len(df) else None,
        "max_timestamp": str(df["timestamp_utc"].max()) if len(df) else None,
        "lags_hours": lags_hours,
        "rolling_windows_hours": rolling_windows_hours,
        "same_hour_windows_days": same_hour_windows_days,
        "origin_rolling_windows_hours": origin_rolling_windows_hours,
        "base_temperature_c": base_temperature_c,
        "leakage_note": (
            "Lag features use shifted demand. Rolling features are computed from "
            "demand_mwh.shift(1), so current target is not used as an input."
        ),
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return df
