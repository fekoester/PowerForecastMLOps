from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from power_forecast.models.backtest import make_walk_forward_folds
from power_forecast.models.ensemble import LinearEnsembleRegressor
from power_forecast.models.inference import interval_table, predict_with_history
from power_forecast.models.metrics import bias, mae, mape, rmse
from power_forecast.models.model_zoo import enabled_models, make_model

DEFAULT_ENSEMBLE_HOLDOUT_DAYS = 30


def _load_best_baseline(backtest_path: str | Path) -> dict[str, Any]:
    path = Path(backtest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline backtest report not found: {path}. Run `make backtest` first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    return report["aggregate_metrics"]["_best_by_mae"]

def _make_recency_weights(
    timestamps: pd.Series,
    reference_time: pd.Timestamp,
    half_life_days: float,
    min_weight: float,
) -> pd.Series:
    """Exponential recency weights.

    Most recent rows get weight close to 1.
    Older rows decay exponentially with the configured half-life.
    """
    age_days = (reference_time - timestamps).dt.total_seconds() / 86400.0
    weights = 0.5 ** (age_days / half_life_days)
    weights = weights.clip(lower=min_weight, upper=1.0)
    return weights
    
    
def _candidate_name(model_name: str, training_window_days: int) -> str:
    return f"{model_name}_{training_window_days}d"


def _base_model_name(candidate_name: str) -> str:
    # "lightgbm_90d" -> "lightgbm"
    return candidate_name.rsplit("_", maxsplit=1)[0]


def _recent_training_subset(
    train_df: pd.DataFrame,
    timestamp_column: str,
    training_window_days: int,
) -> pd.DataFrame:
    """Keep only the most recent N days inside a walk-forward training fold."""
    train_end = train_df[timestamp_column].max()
    train_start = train_end - pd.Timedelta(days=training_window_days)

    subset = train_df[train_df[timestamp_column] > train_start].copy()

    if subset.empty:
        raise RuntimeError(
            f"Training subset is empty for training_window_days={training_window_days}."
        )

    return subset

def get_feature_columns(
    df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    forecast_safe_features: bool = False,
    allowed_lag_hours: list[int] | None = None,
) -> list[str]:
    excluded = {timestamp_column, target_column}
    columns = [c for c in df.columns if c not in excluded]

    if not forecast_safe_features:
        return columns

    allowed_lag_hours = allowed_lag_hours or [24, 48, 168]
    allowed_lag_columns = {f"demand_lag_{h}h" for h in allowed_lag_hours}

    safe_columns = []

    for col in columns:
        if col.startswith("demand_lag_"):
            if col in allowed_lag_columns:
                safe_columns.append(col)
            continue

        if col.startswith("demand_roll_"):
            continue

        if col.startswith("demand_origin_roll_"):
            safe_columns.append(col)
            continue

        if col.startswith("demand_same_hour_"):
            safe_columns.append(col)
            continue

        safe_columns.append(col)

    return safe_columns


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
    training_window_days: int,
    recency_weighting_config: dict[str, Any] | None = None,
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
    residuals: dict[str, list] = {"residual": [], "hour": []}

    for fold in folds:
        train_mask = (df[timestamp_column] >= fold.train_start) & (
            df[timestamp_column] < fold.train_end
        )
        valid_mask = (df[timestamp_column] >= fold.valid_start) & (
            df[timestamp_column] < fold.valid_end
        )

        train_df_full = df.loc[train_mask].copy()
        train_df = _recent_training_subset(
            train_df=train_df_full,
            timestamp_column=timestamp_column,
            training_window_days=training_window_days,
        )
        valid_df = df.loc[valid_mask].copy()

        model = _fit_candidate_model(
            df=train_df_full,
            model_name=model_name,
            model_config=model_config,
            feature_columns=feature_columns,
            timestamp_column=timestamp_column,
            target_column=target_column,
            training_window_days=training_window_days,
            recency_weighting_config=recency_weighting_config,
        )
        y_valid = valid_df[target_column]
        # Stateful models (ESN) are warmed up on the end of the training fold.
        pred = predict_with_history(
            model, valid_df[feature_columns], X_history=train_df_full[feature_columns]
        )
        _collect_residuals(residuals, valid_df, y_valid, pred)

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
        "training_window_days": training_window_days,
        "candidate_name": _candidate_name(model_name, training_window_days),
        "aggregate_metrics": aggregate,
        "fold_results": fold_results,
        # Out-of-sample residuals for prediction intervals (removed before writing JSON).
        "_residuals": residuals,
    }


def _collect_residuals(
    residuals: dict[str, list], valid_df: pd.DataFrame, y_true, y_pred
) -> None:
    residuals["residual"].extend((np.asarray(y_true, float) - np.asarray(y_pred, float)).tolist())
    hours = valid_df["hour"] if "hour" in valid_df.columns else pd.Series([-1] * len(valid_df))
    residuals["hour"].extend(hours.astype(int).tolist())



def _aggregate_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, float]:
    metrics_df = pd.DataFrame([fold["metrics"] for fold in fold_results])

    return {
        "mae_mean": float(metrics_df["mae"].mean()),
        "mae_std": float(metrics_df["mae"].std(ddof=0)),
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std(ddof=0)),
        "mape_mean": float(metrics_df["mape"].mean()),
        "mape_std": float(metrics_df["mape"].std(ddof=0)),
        "bias_mean": float(metrics_df["bias"].mean()),
        "bias_std": float(metrics_df["bias"].std(ddof=0)),
    }


def _metrics_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def _fit_candidate_model(
    df: pd.DataFrame,
    model_name: str,
    model_config: dict[str, Any],
    feature_columns: list[str],
    timestamp_column: str,
    target_column: str,
    training_window_days: int,
    recency_weighting_config: dict[str, Any] | None = None,
):
    train_df = _recent_training_subset(
        train_df=df,
        timestamp_column=timestamp_column,
        training_window_days=int(training_window_days),
    )

    model = make_model(model_name, model_config)

    sample_weight = None
    if recency_weighting_config and bool(recency_weighting_config.get("enabled", False)):
        apply_to = set(recency_weighting_config.get("apply_to", []))
        candidate = _candidate_name(model_name, int(training_window_days))

        if model_name in apply_to or candidate in apply_to:
            sample_weight = _make_recency_weights(
                timestamps=train_df[timestamp_column],
                reference_time=train_df[timestamp_column].max(),
                half_life_days=float(recency_weighting_config["half_life_days"]),
                min_weight=float(recency_weighting_config.get("min_weight", 0.0)),
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if sample_weight is not None:
            model.fit(
                train_df[feature_columns],
                train_df[target_column],
                sample_weight=sample_weight,
            )
        else:
            model.fit(
                train_df[feature_columns],
                train_df[target_column],
            )

    return model

def _evaluate_linear_ensemble_on_folds(
    df: pd.DataFrame,
    active_models: dict[str, dict[str, Any]],
    training_windows_days: list[int],
    feature_columns: list[str],
    timestamp_column: str,
    target_column: str,
    min_train_days: int,
    validation_window_days: int,
    step_days: int,
    recency_weighting_config: dict[str, Any] | None,
    ensemble_config: dict[str, Any],
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    residuals: dict[str, list] = {"residual": [], "hour": []}

    ensemble_name = str(ensemble_config.get("name", "ensemble_linear"))
    alpha = float(ensemble_config.get("alpha", 1.0))
    positive = bool(ensemble_config.get("positive", False))

    folds = make_walk_forward_folds(
        timestamps=df[timestamp_column],
        min_train_days=min_train_days,
        validation_window_days=validation_window_days,
        step_days=step_days,
    )

    if not folds:
        raise RuntimeError("No folds available for ensemble. Run with more data or shorter windows.")

    for fold in folds:
        train_mask = (df[timestamp_column] >= fold.train_start) & (
            df[timestamp_column] < fold.train_end
        )
        valid_mask = (df[timestamp_column] >= fold.valid_start) & (
            df[timestamp_column] < fold.valid_end
        )

        train_df_full = df.loc[train_mask].copy()
        valid_df = df.loc[valid_mask].copy()

        ensemble = _fit_stacked_ensemble(
            df=train_df_full,
            active_models=active_models,
            training_windows_days=training_windows_days,
            feature_columns=feature_columns,
            timestamp_column=timestamp_column,
            target_column=target_column,
            recency_weighting_config=recency_weighting_config,
            ensemble_config=ensemble_config,
        )

        y_true = valid_df[target_column]
        y_pred = ensemble.predict(
            valid_df[feature_columns], X_history=train_df_full[feature_columns]
        )
        _collect_residuals(residuals, valid_df, y_true, y_pred)

        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "valid_start": str(fold.valid_start),
                "valid_end": str(fold.valid_end),
                "n_train_rows": int(len(train_df_full)),
                "n_valid_rows": int(len(valid_df)),
                "metrics": _metrics_dict(y_true, y_pred),
                "coefficients": ensemble.coefficients(),
                "intercept": ensemble.intercept(),
            }
        )

    aggregate = _aggregate_fold_metrics(fold_results)

    all_coefficient_names = sorted(
        {
            name
            for fold in fold_results
            for name in fold.get("coefficients", {}).keys()
        }
    )

    mean_coefficients = {}
    for name in all_coefficient_names:
        values = [
            float(fold.get("coefficients", {}).get(name, 0.0))
            for fold in fold_results
        ]
        mean_coefficients[name] = float(np.mean(values))

    return {
        "model_name": ensemble_name,
        "candidate_name": ensemble_name,
        "training_window_days": None,
        "aggregate_metrics": aggregate,
        "fold_results": fold_results,
        "ensemble": {
            "type": "linear_stacking",
            "meta_fit": "time-based holdout (out-of-sample base predictions)",
            "holdout_days": int(ensemble_config.get("holdout_days", DEFAULT_ENSEMBLE_HOLDOUT_DAYS)),
            "alpha": alpha,
            "positive": positive,
            "mean_coefficients": mean_coefficients,
        },
        "_residuals": residuals,
    }


def _fit_stacked_ensemble(
    df: pd.DataFrame,
    active_models: dict[str, dict[str, Any]],
    training_windows_days: list[int],
    feature_columns: list[str],
    timestamp_column: str,
    target_column: str,
    recency_weighting_config: dict[str, Any] | None,
    ensemble_config: dict[str, Any],
    final_base_models: dict[str, Any] | None = None,
) -> LinearEnsembleRegressor:
    """Stacking without in-sample leakage.

    1. Fit every base candidate on data before the last ``holdout_days``.
    2. Fit the meta-model on their predictions for the held-out days (out-of-sample).
    3. Use base candidates fitted on all of ``df`` (``final_base_models`` if already trained)
       together with that meta-model.
    """
    holdout_days = int(ensemble_config.get("holdout_days", DEFAULT_ENSEMBLE_HOLDOUT_DAYS))
    cutoff = df[timestamp_column].max() - pd.Timedelta(days=holdout_days)
    early = df[df[timestamp_column] <= cutoff]
    holdout = df[df[timestamp_column] > cutoff]
    if early.empty or holdout.empty:
        raise RuntimeError(
            f"Not enough data for an ensemble holdout of {holdout_days} days; "
            "reduce train.ensemble.holdout_days."
        )

    def fit_all(train: pd.DataFrame) -> dict[str, Any]:
        return {
            _candidate_name(name, int(window)): _fit_candidate_model(
                df=train,
                model_name=name,
                model_config=config,
                feature_columns=feature_columns,
                timestamp_column=timestamp_column,
                target_column=target_column,
                training_window_days=int(window),
                recency_weighting_config=recency_weighting_config,
            )
            for name, config in active_models.items()
            for window in training_windows_days
        }

    early_models = fit_all(early)
    meta_X = pd.DataFrame(
        {
            name: predict_with_history(
                model, holdout[feature_columns], X_history=early[feature_columns]
            )
            for name, model in early_models.items()
        },
        index=holdout.index,
    )
    ensemble = LinearEnsembleRegressor(
        base_models=final_base_models if final_base_models is not None else fit_all(df),
        feature_columns=feature_columns,
        alpha=float(ensemble_config.get("alpha", 1.0)),
        positive=bool(ensemble_config.get("positive", False)),
    )
    ensemble.fit_from_predictions(meta_X, holdout[target_column])
    return ensemble

def _is_ensemble_name(model_name: str) -> bool:
    return model_name == "ensemble_linear" or model_name.startswith("ensemble_")

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
    forecast_safe_features: bool = False,
    allowed_lag_hours: list[int] | None = None,
    recency_weighting_config: dict[str, Any] | None = None,
    training_windows_days: list[int] | None = None,
    selected_models: list[str] | None = None,
    selected_windows: list[int] | None = None,
    ensemble_config: dict[str, Any] | None = None,
    prediction_intervals_config: dict[str, Any] | None = None,
    monitoring_holdout_hours: int = 0,
) -> dict[str, Any]:
    """Walk-forward comparison of all candidates, then final training of every candidate.

    ``monitoring_holdout_hours``: the final models are trained without the latest N hours, so the
    "latest 24h" monitoring prediction is a genuine out-of-sample check instead of scoring rows the
    models were fitted on.
    """
    if model_selection_metric != "mae":
        raise ValueError("Currently only model_selection_metric='mae' is supported.")

    df = pd.read_csv(features_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    feature_columns = get_feature_columns(
        df=df,
        timestamp_column=timestamp_column,
        target_column=target_column,
        forecast_safe_features=forecast_safe_features,
        allowed_lag_hours=allowed_lag_hours,
    )

    active_models = enabled_models(models_config)
    if not active_models:
        raise RuntimeError("No enabled models found in train.models config.")

    if selected_models:
        selected_model_set = set(selected_models)
        unknown_models = selected_model_set - set(active_models.keys())
        if unknown_models:
            raise RuntimeError(
                f"Requested models are not enabled or not configured: {sorted(unknown_models)}. "
                f"Available enabled models: {sorted(active_models.keys())}"
            )

        active_models = {
            name: cfg
            for name, cfg in active_models.items()
            if name in selected_model_set
        }

    training_windows_days = training_windows_days or [1095]

    if selected_windows:
        configured_windows = set(int(w) for w in training_windows_days)
        selected_window_set = set(int(w) for w in selected_windows)
        unknown_windows = selected_window_set - configured_windows

        if unknown_windows:
            raise RuntimeError(
                f"Requested training windows are not configured: {sorted(unknown_windows)}. "
                f"Configured windows: {sorted(configured_windows)}"
            )

        training_windows_days = [
            int(w)
            for w in training_windows_days
            if int(w) in selected_window_set
        ]

    model_results: dict[str, Any] = {}

    for model_name, model_config in active_models.items():
        for training_window_days in training_windows_days:
            candidate = _candidate_name(model_name, training_window_days)

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
                training_window_days=int(training_window_days),
                recency_weighting_config=recency_weighting_config,
            )
            model_results[candidate] = result

    if ensemble_config and bool(ensemble_config.get("enabled", False)):
        ensemble_name = str(ensemble_config.get("name", "ensemble_linear"))

        if len(model_results) >= 2:
            ensemble_result = _evaluate_linear_ensemble_on_folds(
                df=df,
                active_models=active_models,
                training_windows_days=[int(w) for w in training_windows_days],
                feature_columns=feature_columns,
                timestamp_column=timestamp_column,
                target_column=target_column,
                min_train_days=min_train_days,
                validation_window_days=validation_window_days,
                step_days=step_days,
                recency_weighting_config=recency_weighting_config,
                ensemble_config=ensemble_config,
            )
            model_results[ensemble_name] = ensemble_result

    best_model_name = min(
        model_results.keys(),
        key=lambda name: model_results[name]["aggregate_metrics"]["mae_mean"],
    )
    if _is_ensemble_name(best_model_name):
        best_base_model_name = best_model_name
        best_training_window_days = None
        best_model_config = ensemble_config or {"enabled": True}
    else:
        best_base_model_name = _base_model_name(best_model_name)
        best_training_window_days = int(
            best_model_name.rsplit("_", maxsplit=1)[1].removesuffix("d")
        )
        best_model_config = active_models[best_base_model_name]
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

    # Train final versions of all candidate models on all available data except the latest
    # monitoring window (kept out-of-sample for the realized latest-24h check).
    final_df = df.iloc[: len(df) - monitoring_holdout_hours] if monitoring_holdout_hours > 0 else df
    final_models = {}

    for model_name, model_config in active_models.items():
        for training_window_days in training_windows_days:
            candidate = _candidate_name(model_name, int(training_window_days))
            final_models[candidate] = _fit_candidate_model(
                df=final_df,
                model_name=model_name,
                model_config=model_config,
                feature_columns=feature_columns,
                timestamp_column=timestamp_column,
                target_column=target_column,
                training_window_days=int(training_window_days),
                recency_weighting_config=recency_weighting_config,
            )


    ensemble_coefficients = {}

    if ensemble_config and bool(ensemble_config.get("enabled", False)):
        ensemble_name = str(ensemble_config.get("name", "ensemble_linear"))

        if len(final_models) >= 2:
            ensemble_model = _fit_stacked_ensemble(
                df=final_df,
                active_models=active_models,
                training_windows_days=[int(w) for w in training_windows_days],
                feature_columns=feature_columns,
                timestamp_column=timestamp_column,
                target_column=target_column,
                recency_weighting_config=recency_weighting_config,
                ensemble_config=ensemble_config,
                final_base_models=dict(final_models),
            )
            final_models[ensemble_name] = ensemble_model
            ensemble_coefficients = ensemble_model.coefficients()
            
    selected_model = final_models[best_model_name]

    # Split-conformal prediction intervals from the out-of-sample walk-forward residuals.
    intervals_cfg = prediction_intervals_config or {"enabled": True}
    prediction_intervals: dict[str, Any] = {}
    for name, result in model_results.items():
        residuals = result.pop("_residuals", None)
        if residuals and bool(intervals_cfg.get("enabled", True)):
            hours = np.asarray(residuals["hour"])
            prediction_intervals[name] = interval_table(
                residuals=np.asarray(residuals["residual"]),
                hours=hours if bool(intervals_cfg.get("by_hour", True)) else None,
                levels=[float(x) for x in intervals_cfg.get("levels", [0.8, 0.95])],
            )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    trained_at_utc = datetime.now(UTC).isoformat()

    model_bundle = {
        # Backward-compatible best model access
        "model": selected_model,
        "model_name": best_model_name,

        # New multi-model access
        "models": final_models,
        "best_model_name": best_model_name,
        "best_base_model_name": best_base_model_name,
        "best_training_window_days": best_training_window_days,
        "training_windows_days": training_windows_days,

        "feature_columns": feature_columns,
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "trained_at_utc": trained_at_utc,
        "final_training_end": str(final_df[timestamp_column].max()),
        "monitoring_holdout_hours": int(monitoring_holdout_hours),
        "forecast_safe_features": forecast_safe_features,
        "allowed_lag_hours": allowed_lag_hours or [24, 48, 168],
        "model_config": best_model_config,
        "all_model_configs": active_models,
        "backtest_aggregate": best_model_metrics,
        "all_backtest_results": model_results,
        "baseline_comparison": comparison,
        "recency_weighting": recency_weighting_config or {"enabled": False},
        
        "ensemble_config": ensemble_config or {"enabled": False},
        "ensemble_coefficients": ensemble_coefficients,
        "prediction_intervals": prediction_intervals,
    }
    joblib.dump(model_bundle, model_path)

    _plot_model_comparison(
        model_results=model_results,
        best_baseline=best_baseline,
        figure_path=model_comparison_path,
    )

    _plot_feature_importance_or_placeholder(
        model=selected_model,
        model_name=best_model_name,
        feature_columns=feature_columns,
        figure_path=feature_importance_path,
    )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "features_path": str(features_path),
        "model_path": str(model_path),
        "feature_importance_path": str(feature_importance_path),
        "model_comparison_path": str(model_comparison_path),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "n_rows": int(len(df)),
        "final_training_end": str(final_df[timestamp_column].max()),
        "monitoring_holdout_hours": int(monitoring_holdout_hours),
        "training_windows_days": training_windows_days,
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_selection_metric": model_selection_metric,
        "forecast_safe_features": forecast_safe_features,
        "allowed_lag_hours": allowed_lag_hours or [24, 48, 168],
        "models": model_results,
        "best_model": {
            "name": best_model_name,
            "base_model_name": best_base_model_name,
            "training_window_days": best_training_window_days,
            "aggregate_metrics": best_model_metrics,
        },
        "aggregate_metrics": best_model_metrics,
        "baseline_comparison": comparison,
        "recency_weighting": recency_weighting_config or {"enabled": False},
        
        "ensemble_config": ensemble_config or {"enabled": False},
        "ensemble_coefficients": ensemble_coefficients,
        "prediction_intervals": {
            "method": "split-conformal on walk-forward validation residuals, by local hour",
            "best_model": prediction_intervals.get(best_model_name, {}).get("overall"),
        },
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
