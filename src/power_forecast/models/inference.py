"""Inference helpers shared by validation, batch prediction and the future forecast.

* ESN warm-up: the echo state network carries a recurrent state across rows. Predicting a
  24-row window on its own starts that state at zero, so the first predictions are made by an
  "empty" reservoir. ``predict_with_history`` runs the model over recent history first and
  returns only the predictions for the requested rows.
* Prediction intervals: split-conformal intervals from walk-forward validation residuals, per
  local hour of day (errors are larger at the daily peak than at night).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from power_forecast.models.esn import EchoStateRegressor

DEFAULT_WARMUP_ROWS = 336  # two weeks of hourly rows


def needs_history(model: Any) -> bool:
    """True if the model (or anything it wraps) is stateful across rows."""
    if isinstance(model, EchoStateRegressor):
        return True
    for attribute in ("regressor_", "regressor"):
        inner = getattr(model, attribute, None)
        if inner is not None and needs_history(inner):
            return True
    steps = getattr(model, "steps", None)
    if steps and any(needs_history(step) for _, step in steps):
        return True
    base_models = getattr(model, "base_models", None)
    if isinstance(base_models, dict):
        return any(needs_history(m) for m in base_models.values())
    return False


def predict_with_history(
    model: Any,
    X: pd.DataFrame,
    X_history: pd.DataFrame | None = None,
    warmup_rows: int = DEFAULT_WARMUP_ROWS,
) -> np.ndarray:
    """Predicts ``X``; stateful models are first run over the last ``warmup_rows`` of history."""
    if hasattr(model, "base_models"):  # ensembles route history to their base models
        return np.asarray(model.predict(X, X_history=X_history))
    if X_history is None or len(X_history) == 0 or not needs_history(model):
        return np.asarray(model.predict(X))
    context = X_history.tail(warmup_rows)
    combined = pd.concat([context, X], axis=0)
    return np.asarray(model.predict(combined))[len(context):]


def interval_table(
    residuals: np.ndarray,
    hours: np.ndarray | None,
    levels: list[float],
    min_samples_per_hour: int = 30,
) -> dict[str, Any]:
    """Residual quantiles (y - prediction) for each central interval level.

    Returns {"levels": [...], "overall": {level: [lo, hi]}, "by_hour": {hour: {level: [lo, hi]}}}.
    Hours with too few residuals fall back to the overall quantiles.
    """
    residuals = np.asarray(residuals, dtype=float)
    mask = np.isfinite(residuals)
    residuals = residuals[mask]

    def quantiles(values: np.ndarray) -> dict[str, list[float]]:
        out = {}
        for level in levels:
            alpha = (1.0 - level) / 2.0
            out[str(level)] = [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))]
        return out

    table: dict[str, Any] = {"levels": [float(x) for x in levels], "overall": quantiles(residuals),
                             "n_residuals": int(residuals.size), "by_hour": {}}
    if hours is not None:
        hours = np.asarray(hours)[mask]
        for hour in range(24):
            values = residuals[hours == hour]
            if values.size >= min_samples_per_hour:
                table["by_hour"][str(hour)] = quantiles(values)
    return table


def apply_intervals(
    predictions: np.ndarray,
    hours: np.ndarray | None,
    table: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Returns {"lower_80": ..., "upper_80": ...} arrays for every level in ``table``."""
    if not table:
        return {}
    predictions = np.asarray(predictions, dtype=float)
    out: dict[str, np.ndarray] = {}
    for level in table["levels"]:
        key = str(level)
        lo = np.full(predictions.shape, table["overall"][key][0])
        hi = np.full(predictions.shape, table["overall"][key][1])
        if hours is not None and table.get("by_hour"):
            for i, hour in enumerate(np.asarray(hours)):
                bounds = table["by_hour"].get(str(int(hour)))
                if bounds:
                    lo[i], hi[i] = bounds[key]
        pct = int(round(100 * float(level)))
        out[f"lower_{pct}"] = predictions + lo
        out[f"upper_{pct}"] = predictions + hi
    return out


def interval_coverage(actuals: np.ndarray, bounds: dict[str, np.ndarray]) -> dict[str, float]:
    """Fraction of actuals inside each interval, e.g. {"coverage_80": 0.79}."""
    actuals = np.asarray(actuals, dtype=float)
    coverage = {}
    for key, lower in bounds.items():
        if not key.startswith("lower_"):
            continue
        pct = key.removeprefix("lower_")
        upper = bounds[f"upper_{pct}"]
        coverage[f"coverage_{pct}"] = float(np.mean((actuals >= lower) & (actuals <= upper)))
    return coverage
