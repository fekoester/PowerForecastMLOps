from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from power_forecast.features.future_features import build_future_24h_features
from power_forecast.models.inference import apply_intervals, predict_with_history
from power_forecast.models.predict import (
    _get_models_from_bundle,
    _shade_intervals,
    _validate_feature_schema,
)


def run_future_24h_forecast(
    model_path: str | Path,
    eia_path: str | Path,
    locations: list[dict],
    timezone_name: str,
    base_temperature_c: float,
    allowed_lag_hours: list[int],
    same_hour_windows_days: list[int],
    origin_rolling_windows_hours: list[int],
    features_output_path: str | Path,
    weather_forecast_output_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    figure_path: str | Path,
    timestamp_column: str,
    prediction_prefix: str,
    use_cyclic_calendar_features: bool,
    use_holiday_features: bool = False,
    location_weights: dict[str, float] | None = None,
    history_features_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    models = _get_models_from_bundle(bundle)
    best_model_name = bundle.get("best_model_name", bundle.get("model_name", "unknown"))
    feature_columns = list(bundle["feature_columns"])

    future_features = build_future_24h_features(
        eia_path=eia_path,
        weather_forecast_output_path=weather_forecast_output_path,
        locations=locations,
        timezone_name=timezone_name,
        base_temperature_c=base_temperature_c,
        allowed_lag_hours=allowed_lag_hours,
        same_hour_windows_days=same_hour_windows_days,
        origin_rolling_windows_hours=origin_rolling_windows_hours,
        output_path=features_output_path,
        use_cyclic_calendar_features=use_cyclic_calendar_features,
        use_holiday_features=use_holiday_features,
        location_weights=location_weights,
    )

    _validate_feature_schema(future_features, feature_columns)

    # Recent feature rows warm up stateful models (ESN) before the forecast horizon.
    history = None
    if history_features_path is not None and Path(history_features_path).exists():
        history = pd.read_csv(history_features_path)
        if all(c in history.columns for c in feature_columns):
            history = history[feature_columns]
        else:
            history = None

    forecast_df = future_features[[timestamp_column]].copy()
    prediction_columns_by_model = {}

    for model_name, model in models.items():
        col = f"{prediction_prefix}_{model_name}"
        prediction_columns_by_model[model_name] = col
        forecast_df[col] = predict_with_history(
            model, future_features[feature_columns], X_history=history
        )

    if best_model_name not in prediction_columns_by_model:
        raise RuntimeError(f"Best model {best_model_name} not available in model bundle.")

    forecast_df[f"{prediction_prefix}_best"] = forecast_df[
        prediction_columns_by_model[best_model_name]
    ]
    interval_bounds = apply_intervals(
        forecast_df[f"{prediction_prefix}_best"].to_numpy(),
        future_features["hour"].to_numpy() if "hour" in future_features.columns else None,
        bundle.get("prediction_intervals", {}).get(best_model_name),
    )
    for key, values in interval_bounds.items():
        forecast_df[f"{prediction_prefix}_best_{key}"] = values

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)

    _plot_future_forecast(
        forecast_df=forecast_df,
        timestamp_column=timestamp_column,
        prediction_columns_by_model=prediction_columns_by_model,
        best_model_name=best_model_name,
        figure_path=figure_path,
        interval_bounds=interval_bounds,
    )

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "model_trained_at_utc": bundle.get("trained_at_utc"),
        "best_model_name": best_model_name,
        "available_models": list(models.keys()),
        "prediction_columns_by_model": prediction_columns_by_model,
        "interval_columns": [f"{prediction_prefix}_best_{k}" for k in interval_bounds],
        "features_output_path": str(features_output_path),
        "weather_forecast_output_path": str(weather_forecast_output_path),
        "output_path": str(output_path),
        "figure_path": str(figure_path),
        "horizon_hours": int(len(forecast_df)),
        "min_timestamp": str(forecast_df[timestamp_column].min()),
        "max_timestamp": str(forecast_df[timestamp_column].max()),
        "note": (
            "Future forecast uses weather forecast data, calendar features, and demand-history "
            "features available before the forecast horizon. No actual future demand is used."
        ),
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _plot_future_forecast(
    forecast_df: pd.DataFrame,
    timestamp_column: str,
    prediction_columns_by_model: dict[str, str],
    best_model_name: str,
    figure_path: str | Path,
    interval_bounds: dict | None = None,
) -> None:
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    _shade_intervals(forecast_df[timestamp_column], interval_bounds)

    for model_name, col in prediction_columns_by_model.items():
        linewidth = 2.8 if model_name == best_model_name else 1.7
        label = f"{model_name}" + (" selected" if model_name == best_model_name else "")

        plt.plot(
            forecast_df[timestamp_column],
            forecast_df[col],
            marker="o",
            linewidth=linewidth,
            label=label,
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Forecast demand (MWh)")
    plt.title("Next 24h demand forecast by model")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
