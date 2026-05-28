from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

from power_forecast.models.backtest import make_walk_forward_folds
from power_forecast.models.metrics import bias, mae, mape, rmse


def _load_best_baseline(backtest_path: str | Path) -> dict[str, Any]:
    path = Path(backtest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline backtest report not found: {path}. Run `make backtest` first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    return report["aggregate_metrics"]["_best_by_mae"]


def _get_feature_columns(df: pd.DataFrame, timestamp_column: str, target_column: str) -> list[str]:
    excluded = {timestamp_column, target_column}
    return [c for c in df.columns if c not in excluded]


def _make_model(model_config: dict[str, Any]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        num_leaves=int(model_config["num_leaves"]),
        max_depth=int(model_config["max_depth"]),
        subsample=float(model_config["subsample"]),
        colsample_bytree=float(model_config["colsample_bytree"]),
        random_state=int(model_config["random_state"]),
        verbosity=-1,
    )


def train_lightgbm_walk_forward(
    features_path: str | Path,
    baseline_backtest_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    feature_importance_path: str | Path,
    timestamp_column: str,
    target_column: str,
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    df = pd.read_csv(features_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    feature_columns = _get_feature_columns(
        df=df,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    folds = make_walk_forward_folds(
        timestamps=df[timestamp_column],
        min_train_days=min_train_days,
        validation_window_days=validation_window_days,
        step_days=step_days,
    )

    if not folds:
        raise RuntimeError("No folds available. Run with more data or shorter windows.")

    fold_results: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = (df[timestamp_column] >= fold.train_start) & (
            df[timestamp_column] < fold.train_end
        )
        valid_mask = (df[timestamp_column] >= fold.valid_start) & (
            df[timestamp_column] < fold.valid_end
        )

        train_df = df.loc[train_mask].copy()
        valid_df = df.loc[valid_mask].copy()

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_valid = valid_df[feature_columns]
        y_valid = valid_df[target_column]

        model = _make_model(model_config)
        model.fit(X_train, y_train)

        pred = model.predict(X_valid)

        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "valid_start": str(fold.valid_start),
                "valid_end": str(fold.valid_end),
                "n_train_rows": int(len(train_df)),
                "n_valid_rows": int(len(valid_df)),
                "metrics": {
                    "mae": mae(y_valid, pred),
                    "rmse": rmse(y_valid, pred),
                    "mape": mape(y_valid, pred),
                    "bias": bias(y_valid, pred),
                },
            }
        )

    metrics_df = pd.DataFrame([fold["metrics"] for fold in fold_results])

    aggregate = {
        "mae_mean": float(metrics_df["mae"].mean()),
        "mae_std": float(metrics_df["mae"].std(ddof=0)),
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std(ddof=0)),
        "mape_mean": float(metrics_df["mape"].mean()),
        "mape_std": float(metrics_df["mape"].std(ddof=0)),
        "bias_mean": float(metrics_df["bias"].mean()),
        "bias_std": float(metrics_df["bias"].std(ddof=0)),
    }

    best_baseline = _load_best_baseline(baseline_backtest_path)
    baseline_mae = float(best_baseline["mae_mean"])
    model_mae = float(aggregate["mae_mean"])

    comparison = {
        "best_baseline_name": best_baseline["name"],
        "best_baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "mae_improvement": baseline_mae - model_mae,
        "mae_improvement_pct": 100.0 * (baseline_mae - model_mae) / baseline_mae,
        "beats_baseline": model_mae < baseline_mae,
    }

    # Train final model on all currently available data.
    final_model = _make_model(model_config)
    final_model.fit(df[feature_columns], df[target_column])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": model_config,
        "backtest_aggregate": aggregate,
        "baseline_comparison": comparison,
    }
    joblib.dump(model_bundle, model_path)

    _plot_feature_importance(
        model=final_model,
        feature_columns=feature_columns,
        figure_path=feature_importance_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "model_path": str(model_path),
        "feature_importance_path": str(feature_importance_path),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_config": model_config,
        "aggregate_metrics": aggregate,
        "baseline_comparison": comparison,
        "fold_results": fold_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _plot_feature_importance(
    model: lgb.LGBMRegressor,
    feature_columns: list[str],
    figure_path: str | Path,
    top_k: int = 25,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance = importance.head(top_k).sort_values("importance", ascending=True)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.barh(importance["feature"], importance["importance"])
    plt.xlabel("LightGBM feature importance")
    plt.ylabel("Feature")
    plt.title("Top LightGBM feature importances")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
