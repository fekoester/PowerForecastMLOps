from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from power_forecast.models.backtest import make_walk_forward_folds
from power_forecast.models.metrics import bias, mae, mape, rmse
from power_forecast.models.model_zoo import enabled_models, make_model


def _load_best_baseline(backtest_path: str | Path) -> dict[str, Any]:
    path = Path(backtest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline backtest report not found: {path}. Run `make backtest` first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    return report["aggregate_metrics"]["_best_by_mae"]


def get_feature_columns(df: pd.DataFrame, timestamp_column: str, target_column: str) -> list[str]:
    excluded = {timestamp_column, target_column}
    return [c for c in df.columns if c not in excluded]


def _evaluate_model_on_folds(
    df: pd.DataFrame,
    model_name: str,
    model_config: dict[str, Any],
    feature_columns: list[str],
    timestamp_column: str,
    target_column: str,
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
) -> dict[str, Any]:
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

        model = make_model(model_name, model_config)

        # MLP may warn about convergence. For this lightweight daily pipeline,
        # we record performance rather than failing the run.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
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

    return {
        "model_name": model_name,
        "aggregate_metrics": aggregate,
        "fold_results": fold_results,
    }


def train_and_compare_models(
    features_path: str | Path,
    baseline_backtest_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    feature_importance_path: str | Path,
    model_comparison_path: str | Path,
    timestamp_column: str,
    target_column: str,
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
    models_config: dict[str, dict[str, Any]],
    model_selection_metric: str = "mae",
) -> dict[str, Any]:
    if model_selection_metric != "mae":
        raise ValueError("Currently only model_selection_metric='mae' is supported.")

    df = pd.read_csv(features_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    feature_columns = get_feature_columns(
        df=df,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    active_models = enabled_models(models_config)
    if not active_models:
        raise RuntimeError("No enabled models found in train.models config.")

    model_results: dict[str, Any] = {}

    for model_name, model_config in active_models.items():
        result = _evaluate_model_on_folds(
            df=df,
            model_name=model_name,
            model_config=model_config,
            feature_columns=feature_columns,
            timestamp_column=timestamp_column,
            target_column=target_column,
            min_train_days=min_train_days,
            validation_window_days=validation_window_days,
            step_days=step_days,
        )
        model_results[model_name] = result

    best_model_name = min(
        model_results.keys(),
        key=lambda name: model_results[name]["aggregate_metrics"]["mae_mean"],
    )

    best_model_config = active_models[best_model_name]
    best_model_metrics = model_results[best_model_name]["aggregate_metrics"]

    best_baseline = _load_best_baseline(baseline_backtest_path)
    baseline_mae = float(best_baseline["mae_mean"])
    best_model_mae = float(best_model_metrics["mae_mean"])

    comparison = {
        "best_baseline_name": best_baseline["name"],
        "best_baseline_mae": baseline_mae,
        "best_model_name": best_model_name,
        "best_model_mae": best_model_mae,
        "mae_improvement": baseline_mae - best_model_mae,
        "mae_improvement_pct": 100.0 * (baseline_mae - best_model_mae) / baseline_mae,
        "beats_baseline": best_model_mae < baseline_mae,
    }

    # Train final selected model on all currently available data.
    final_model = make_model(best_model_name, best_model_config)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_model.fit(df[feature_columns], df[target_column])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": final_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": best_model_config,
        "all_model_configs": active_models,
        "backtest_aggregate": best_model_metrics,
        "baseline_comparison": comparison,
    }
    joblib.dump(model_bundle, model_path)

    _plot_model_comparison(
        model_results=model_results,
        best_baseline=best_baseline,
        figure_path=model_comparison_path,
    )

    _plot_feature_importance_or_placeholder(
        model=final_model,
        model_name=best_model_name,
        feature_columns=feature_columns,
        figure_path=feature_importance_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "model_path": str(model_path),
        "feature_importance_path": str(feature_importance_path),
        "model_comparison_path": str(model_comparison_path),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_selection_metric": model_selection_metric,
        "models": model_results,
        "best_model": {
            "name": best_model_name,
            "aggregate_metrics": best_model_metrics,
        },
        "aggregate_metrics": best_model_metrics,
        "baseline_comparison": comparison,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _plot_model_comparison(
    model_results: dict[str, Any],
    best_baseline: dict[str, Any],
    figure_path: str | Path,
) -> None:
    rows = []

    for model_name, result in model_results.items():
        rows.append(
            {
                "name": model_name,
                "mae": result["aggregate_metrics"]["mae_mean"],
            }
        )

    rows.append(
        {
            "name": f"baseline: {best_baseline['name']}",
            "mae": float(best_baseline["mae_mean"]),
        }
    )

    plot_df = pd.DataFrame(rows).sort_values("mae", ascending=True)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["name"], plot_df["mae"])
    plt.xlabel("Mean walk-forward MAE")
    plt.ylabel("Model")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()


def _extract_feature_importances(model):
    """Extract feature importances from plain or wrapped tree models."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # TransformedTargetRegressor after fit stores fitted estimator in regressor_
    regressor = getattr(model, "regressor_", None)
    if regressor is not None:
        if hasattr(regressor, "feature_importances_"):
            return regressor.feature_importances_

        # Pipeline case: Pipeline([("x_scaler", ...), ("model", LightGBM)])
        if hasattr(regressor, "named_steps") and "model" in regressor.named_steps:
            inner_model = regressor.named_steps["model"]
            if hasattr(inner_model, "feature_importances_"):
                return inner_model.feature_importances_

    return None


def _plot_feature_importance_or_placeholder(
    model,
    model_name: str,
    feature_columns: list[str],
    figure_path: str | Path,
    top_k: int = 25,
) -> None:
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    importances = _extract_feature_importances(model)

    if importances is not None:
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        importance = importance.head(top_k).sort_values("importance", ascending=True)

        plt.figure(figsize=(9, 7))
        plt.barh(importance["feature"], importance["importance"])
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title(f"Top feature importances: {model_name}")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160)
        plt.close()
        return

    plt.figure(figsize=(9, 4))
    plt.text(
        0.5,
        0.5,
        f"No native feature importance available for selected model: {model_name}",
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
