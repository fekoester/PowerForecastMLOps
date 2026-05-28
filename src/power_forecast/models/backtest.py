from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from power_forecast.models.metrics import bias, mae, mape, rmse


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def make_walk_forward_folds(
    timestamps: pd.Series,
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
) -> list[Fold]:
    ts = pd.to_datetime(timestamps, utc=True).sort_values()

    data_start = ts.min()
    data_end = ts.max()

    folds: list[Fold] = []

    train_start = data_start
    valid_start = data_start + pd.Timedelta(days=min_train_days)
    fold_id = 1

    while True:
        valid_end = valid_start + pd.Timedelta(days=validation_window_days)
        train_end = valid_start

        if valid_end > data_end:
            break

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

        valid_start = valid_start + pd.Timedelta(days=step_days)
        fold_id += 1

    return folds


def _evaluate_one_baseline(
    df_valid: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> dict[str, float]:
    y_true = df_valid[target_column]
    y_pred = df_valid[prediction_column]

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def run_baseline_backtest(
    features_path: str | Path,
    output_path: str | Path,
    figure_path: str | Path,
    timestamp_column: str,
    target_column: str,
    baselines: dict[str, str],
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
) -> dict[str, Any]:
    df = pd.read_csv(features_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    folds = make_walk_forward_folds(
        timestamps=df[timestamp_column],
        min_train_days=min_train_days,
        validation_window_days=validation_window_days,
        step_days=step_days,
    )

    if not folds:
        raise RuntimeError(
            "No walk-forward folds could be created. "
            "Use more data or reduce min_train_days/validation_window_days."
        )

    fold_results: list[dict[str, Any]] = []

    for fold in folds:
        valid_mask = (df[timestamp_column] >= fold.valid_start) & (
            df[timestamp_column] < fold.valid_end
        )
        df_valid = df.loc[valid_mask].copy()

        if df_valid.empty:
            continue

        baseline_metrics = {}

        for baseline_name, prediction_column in baselines.items():
            if prediction_column not in df_valid.columns:
                raise ValueError(
                    f"Baseline column not found: {prediction_column} for {baseline_name}"
                )

            baseline_metrics[baseline_name] = _evaluate_one_baseline(
                df_valid=df_valid,
                target_column=target_column,
                prediction_column=prediction_column,
            )

        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "valid_start": str(fold.valid_start),
                "valid_end": str(fold.valid_end),
                "n_valid_rows": int(len(df_valid)),
                "metrics": baseline_metrics,
            }
        )

    aggregate = _aggregate_fold_results(fold_results=fold_results, baselines=baselines)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "n_rows": int(len(df)),
        "n_folds": int(len(fold_results)),
        "min_train_days": int(min_train_days),
        "validation_window_days": int(validation_window_days),
        "step_days": int(step_days),
        "baselines": baselines,
        "aggregate_metrics": aggregate,
        "fold_results": fold_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_baseline_mae(aggregate=aggregate, figure_path=figure_path)

    return report


def _aggregate_fold_results(
    fold_results: list[dict[str, Any]],
    baselines: dict[str, str],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}

    for baseline_name in baselines:
        metric_rows = []

        for fold in fold_results:
            metric_rows.append(fold["metrics"][baseline_name])

        metrics_df = pd.DataFrame(metric_rows)

        aggregate[baseline_name] = {
            "mae_mean": float(metrics_df["mae"].mean()),
            "mae_std": float(metrics_df["mae"].std(ddof=0)),
            "rmse_mean": float(metrics_df["rmse"].mean()),
            "rmse_std": float(metrics_df["rmse"].std(ddof=0)),
            "mape_mean": float(metrics_df["mape"].mean()),
            "mape_std": float(metrics_df["mape"].std(ddof=0)),
            "bias_mean": float(metrics_df["bias"].mean()),
            "bias_std": float(metrics_df["bias"].std(ddof=0)),
        }

    best_by_mae = min(aggregate.items(), key=lambda item: item[1]["mae_mean"])
    aggregate["_best_by_mae"] = {
        "name": best_by_mae[0],
        "mae_mean": best_by_mae[1]["mae_mean"],
        "rmse_mean": best_by_mae[1]["rmse_mean"],
    }

    return aggregate


def _plot_baseline_mae(aggregate: dict[str, Any], figure_path: str | Path) -> None:
    rows = [
        {"baseline": name, "mae": metrics["mae_mean"]}
        for name, metrics in aggregate.items()
        if not name.startswith("_")
    ]
    plot_df = pd.DataFrame(rows).sort_values("mae", ascending=True)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["baseline"], plot_df["mae"])
    plt.xlabel("Mean validation MAE")
    plt.ylabel("Baseline")
    plt.title("Walk-forward baseline performance")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
