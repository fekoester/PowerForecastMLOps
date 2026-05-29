from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from power_forecast.models.metrics import bias, mae, mape, rmse


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run `make train` first.")

    bundle = joblib.load(model_path)

    required_keys = [
        "model",
        "feature_columns",
        "target_column",
        "timestamp_column",
        "trained_at_utc",
        "model_config",
    ]
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise ValueError(f"Model bundle missing required keys: {missing}")

    return bundle


def _validate_feature_schema(df: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input data missing model feature columns: {missing}")


def _get_models_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return all available final models.

    New bundles contain bundle["models"].
    Older bundles only contain bundle["model"].
    """
    if "models" in bundle and isinstance(bundle["models"], dict):
        return bundle["models"]

    model_name = bundle.get("model_name", "model")
    return {model_name: bundle["model"]}


def _safe_prediction_column(model_name: str) -> str:
    return f"prediction_mwh_{model_name}"


def run_batch_prediction(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    figure_path: str | Path,
    timestamp_column: str,
    target_column: str,
    prediction_column: str,
    n_latest_rows: int,
) -> dict[str, Any]:
    bundle = load_model_bundle(model_path)
    models = _get_models_from_bundle(bundle)
    best_model_name = bundle.get("best_model_name", bundle.get("model_name", "unknown"))
    feature_columns = list(bundle["feature_columns"])

    df = pd.read_csv(input_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    _validate_feature_schema(df, feature_columns)

    if len(df) < n_latest_rows:
        raise ValueError(
            f"Not enough rows for prediction: requested {n_latest_rows}, available {len(df)}"
        )

    pred_df = df.tail(n_latest_rows).copy()

    per_model_metrics: dict[str, dict[str, float]] = {}
    prediction_columns_by_model: dict[str, str] = {}

    for model_name, model in models.items():
        col = _safe_prediction_column(model_name)
        prediction_columns_by_model[model_name] = col
        pred_df[col] = model.predict(pred_df[feature_columns])

    if best_model_name not in prediction_columns_by_model:
        raise RuntimeError(
            f"Best model {best_model_name} not found in available prediction models: "
            f"{list(prediction_columns_by_model)}"
        )

    best_prediction_col = prediction_columns_by_model[best_model_name]
    pred_df[prediction_column] = pred_df[best_prediction_col]

    has_actuals = target_column in pred_df.columns and pred_df[target_column].notna().all()

    best_metrics = None
    if has_actuals:
        for model_name, col in prediction_columns_by_model.items():
            per_model_metrics[model_name] = {
                "mae": mae(pred_df[target_column], pred_df[col]),
                "rmse": rmse(pred_df[target_column], pred_df[col]),
                "mape": mape(pred_df[target_column], pred_df[col]),
                "bias": bias(pred_df[target_column], pred_df[col]),
            }

        best_metrics = per_model_metrics[best_model_name]

        pred_df["error"] = pred_df[prediction_column] - pred_df[target_column]
        pred_df["absolute_error"] = pred_df["error"].abs()
        pred_df["absolute_percentage_error"] = (
            pred_df["absolute_error"] / pred_df[target_column].abs()
        )

    output_columns = [
        timestamp_column,
        prediction_column,
        *prediction_columns_by_model.values(),
    ]

    # Avoid duplicate prediction column if best prediction column already appears.
    output_columns = list(dict.fromkeys(output_columns))

    if has_actuals:
        output_columns.extend(
            [
                target_column,
                "error",
                "absolute_error",
                "absolute_percentage_error",
            ]
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df[output_columns].to_csv(output_path, index=False)

    _plot_latest_predictions(
        pred_df=pred_df,
        timestamp_column=timestamp_column,
        target_column=target_column,
        prediction_columns_by_model=prediction_columns_by_model,
        figure_path=figure_path,
        has_actuals=has_actuals,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "figure_path": str(figure_path),
        "model_trained_at_utc": bundle.get("trained_at_utc"),
        "model_name": best_model_name,
        "best_model_name": best_model_name,
        "available_models": list(models.keys()),
        "prediction_columns_by_model": prediction_columns_by_model,
        "n_prediction_rows": int(len(pred_df)),
        "min_timestamp": str(pred_df[timestamp_column].min()),
        "max_timestamp": str(pred_df[timestamp_column].max()),
        "feature_count": int(len(feature_columns)),
        "prediction_column": prediction_column,
        "has_actuals": bool(has_actuals),
        "metrics": best_metrics,
        "per_model_metrics": per_model_metrics,
        "note": (
            "This prediction job scores the latest known feature rows for all candidate models. "
            "It validates model feature schema before inference and writes prediction metadata."
        ),
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _plot_latest_predictions(
    pred_df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    prediction_columns_by_model: dict[str, str],
    figure_path: str | Path,
    has_actuals: bool,
) -> None:
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    if has_actuals:
        plt.plot(
            pred_df[timestamp_column],
            pred_df[target_column],
            marker="o",
            linewidth=2.4,
            label="Actual",
        )

    for model_name, col in prediction_columns_by_model.items():
        plt.plot(
            pred_df[timestamp_column],
            pred_df[col],
            marker="o",
            linewidth=1.7,
            label=model_name,
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Demand (MWh)")
    plt.title("Latest demand predictions by model")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
