from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _metric(metrics: dict[str, Any], base_name: str) -> float | None:
    """Read either raw metric names or aggregate metric names.

    Latest-window metrics use:
      mae, rmse, mape, bias

    Walk-forward aggregate metrics use:
      mae_mean, rmse_mean, mape_mean, bias_mean
    """
    if base_name in metrics:
        return metrics.get(base_name)
    return metrics.get(f"{base_name}_mean")


def _safe_metric_for_sort(metrics: dict[str, Any], base_name: str) -> float:
    value = _metric(metrics, base_name)
    if value is None:
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _health_status(
    latest_mae: float | None,
    training_mae: float,
    best_baseline_mae: float,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if latest_mae is None:
        return "unknown", ["Latest predictions do not contain actuals yet, so realized error is unknown."]

    ratio_vs_training = latest_mae / training_mae
    ratio_vs_baseline = latest_mae / best_baseline_mae

    if ratio_vs_training >= thresholds["degraded_mae_ratio_vs_training"]:
        warnings.append(
            f"Latest MAE is {ratio_vs_training:.2f}x the walk-forward training MAE."
        )
    elif ratio_vs_training >= thresholds["watch_mae_ratio_vs_training"]:
        warnings.append(
            f"Latest MAE is {ratio_vs_training:.2f}x the walk-forward training MAE."
        )

    if ratio_vs_baseline >= thresholds["degraded_mae_ratio_vs_baseline"]:
        warnings.append(
            f"Latest MAE is worse than or equal to the best baseline "
            f"({ratio_vs_baseline:.2f}x baseline MAE)."
        )
    elif ratio_vs_baseline >= thresholds["watch_mae_ratio_vs_baseline"]:
        warnings.append(
            f"Latest MAE is close to the best baseline "
            f"({ratio_vs_baseline:.2f}x baseline MAE)."
        )

    if ratio_vs_training >= thresholds["degraded_mae_ratio_vs_training"]:
        return "degraded", warnings
    if ratio_vs_baseline >= thresholds["degraded_mae_ratio_vs_baseline"]:
        return "degraded", warnings
    if warnings:
        return "watch", warnings

    return "healthy", warnings


def _split_candidate_name(model_name: str) -> tuple[str, str]:
    """Split candidate name like 'lightgbm_180d' into ('lightgbm', '180d').

    Falls back to (model_name, "") for names without a training-window suffix.
    """
    if model_name.endswith("d") and "_" in model_name:
        base, window = model_name.rsplit("_", maxsplit=1)
        return base, window
    return model_name, ""


def _model_sort_key(
    model_name: str,
    per_model_training: dict[str, dict[str, Any]],
    per_model_latest: dict[str, dict[str, Any]],
) -> tuple[float, float, str]:
    train_metrics = per_model_training.get(model_name, {})
    latest_metrics = per_model_latest.get(model_name, {})
    return (
        _safe_metric_for_sort(train_metrics, "mae"),
        _safe_metric_for_sort(latest_metrics, "mae"),
        model_name,
    )


def build_monitoring_report(
    prediction_summary_path: str | Path,
    train_report_path: str | Path,
    baseline_report_path: str | Path,
    predictions_path: str | Path,
    summary_path: str | Path,
    markdown_report_path: str | Path,
    html_report_path: str | Path,
    thresholds: dict[str, float],
    future_forecast_summary_path: str | Path | None = None,
    future_forecast_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction_summary = _load_json(prediction_summary_path)
    train_report = _load_json(train_report_path)
    baseline_report = _load_json(baseline_report_path)

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)

    future_forecast_summary = None
    future_forecast = None

    if future_forecast_summary_path is not None and Path(future_forecast_summary_path).exists():
        future_forecast_summary = _load_json(future_forecast_summary_path)

    if future_forecast_path is not None and Path(future_forecast_path).exists():
        future_forecast = pd.read_csv(future_forecast_path)

    latest_metrics = prediction_summary.get("metrics") or {}
    per_model_latest_metrics = prediction_summary.get("per_model_metrics") or {}

    latest_window_winner = None
    if per_model_latest_metrics:
        latest_window_winner = min(
            per_model_latest_metrics.keys(),
            key=lambda name: per_model_latest_metrics[name].get("mae", float("inf")),
        )

    latest_mae = latest_metrics.get("mae")
    latest_rmse = latest_metrics.get("rmse")
    latest_mape = latest_metrics.get("mape")
    latest_bias = latest_metrics.get("bias")

    training_mae = float(train_report["aggregate_metrics"]["mae_mean"])
    training_rmse = float(train_report["aggregate_metrics"]["rmse_mean"])
    training_mape = float(train_report["aggregate_metrics"]["mape_mean"])

    train_models = train_report.get("models", {})
    per_model_training_metrics = {
        model_name: model_result.get("aggregate_metrics", {})
        for model_name, model_result in train_models.items()
    }

    best_baseline = baseline_report["aggregate_metrics"]["_best_by_mae"]
    best_baseline_name = str(best_baseline["name"])
    best_baseline_mae = float(best_baseline["mae_mean"])
    best_baseline_rmse = float(best_baseline["rmse_mean"])

    health, warnings = _health_status(
        latest_mae=latest_mae,
        training_mae=training_mae,
        best_baseline_mae=best_baseline_mae,
        thresholds=thresholds,
    )

    if latest_mae is not None:
        ratio_vs_training = latest_mae / training_mae
        ratio_vs_baseline = latest_mae / best_baseline_mae
    else:
        ratio_vs_training = None
        ratio_vs_baseline = None

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "health_status": health,
        "warnings": warnings,
        "latest_prediction_window": {
            "min_timestamp": prediction_summary.get("min_timestamp"),
            "max_timestamp": prediction_summary.get("max_timestamp"),
            "n_rows": prediction_summary.get("n_prediction_rows"),
            "has_actuals": prediction_summary.get("has_actuals"),
        },
        "model": {
            "model_path": prediction_summary.get("model_path"),
            "model_name": prediction_summary.get("model_name", "unknown"),
            "model_trained_at_utc": prediction_summary.get("model_trained_at_utc"),
            "feature_count": prediction_summary.get("feature_count"),
        },
        "latest_window_winner": latest_window_winner,
        "latest_metrics": {
            "mae": latest_mae,
            "rmse": latest_rmse,
            "mape": latest_mape,
            "bias": latest_bias,
        },
        "per_model_latest_metrics": per_model_latest_metrics,
        "walk_forward_training_metrics": {
            "mae": training_mae,
            "rmse": training_rmse,
            "mape": training_mape,
        },
        "per_model_training_metrics": per_model_training_metrics,
        "best_baseline": {
            "name": best_baseline_name,
            "mae": best_baseline_mae,
            "rmse": best_baseline_rmse,
        },
        "future_forecast": future_forecast_summary,
        "ratios": {
            "latest_mae_vs_training_mae": ratio_vs_training,
            "latest_mae_vs_best_baseline_mae": ratio_vs_baseline,
        },
        "input_files": {
            "prediction_summary_path": str(prediction_summary_path),
            "train_report_path": str(train_report_path),
            "baseline_report_path": str(baseline_report_path),
            "predictions_path": str(predictions_path),
            "future_forecast_path": str(future_forecast_path)
            if future_forecast_path is not None
            else None,
        },
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = _render_markdown_report(summary, predictions)
    markdown_path = Path(markdown_report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    html_report = _render_html_dashboard(summary, predictions, future_forecast)
    html_path = Path(html_report_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_report, encoding="utf-8")

    return summary


def _render_markdown_report(summary: dict[str, Any], predictions: pd.DataFrame) -> str:
    health = summary["health_status"]
    warnings = summary["warnings"]

    latest = summary["latest_metrics"]
    training = summary["walk_forward_training_metrics"]
    baseline = summary["best_baseline"]
    ratios = summary["ratios"]
    window = summary["latest_prediction_window"]
    model = summary["model"]
    latest_winner = summary.get("latest_window_winner", "N/A")

    per_model_latest = summary.get("per_model_latest_metrics", {})
    per_model_training = summary.get("per_model_training_metrics", {})

    warning_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "- None"

    model_rows = []
    all_model_names = sorted(
        set(per_model_training.keys()) | set(per_model_latest.keys()),
        key=lambda name: _model_sort_key(name, per_model_training, per_model_latest),
    )

    for model_name in all_model_names:
        train_m = per_model_training.get(model_name, {})
        latest_m = per_model_latest.get(model_name, {})
        roles = []
        if model_name == model["model_name"]:
            roles.append("production")
        if model_name == latest_winner:
            roles.append("latest winner")

        model_rows.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    _format_float(_metric(train_m, "mae")),
                    _format_pct(_metric(train_m, "mape")),
                    _format_float(_metric(latest_m, "mae")),
                    _format_pct(_metric(latest_m, "mape")),
                    ", ".join(roles),
                ]
            )
            + " |"
        )

    model_table = "\n".join(
        [
            "| Model | Walk-forward MAE | Walk-forward MAPE | Latest 24h MAE | Latest 24h MAPE | Role |",
            "|---|---:|---:|---:|---:|---|",
            *model_rows,
        ]
    )

    table_rows = []
    cols = list(predictions.columns)
    show_cols = [
        c
        for c in [
            "timestamp_utc",
            "prediction_mwh",
            "demand_mwh",
            "error",
            "absolute_error",
            "absolute_percentage_error",
        ]
        if c in cols
    ]

    for _, row in predictions.tail(10).iterrows():
        cells = []
        for col in show_cols:
            value = row[col]
            if isinstance(value, float):
                if col == "absolute_percentage_error":
                    cells.append(f"{100 * value:.2f}%")
                else:
                    cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        table_rows.append("| " + " | ".join(cells) + " |")

    table_header = "| " + " | ".join(show_cols) + " |"
    table_sep = "| " + " | ".join(["---"] * len(show_cols)) + " |"
    latest_table = "\n".join([table_header, table_sep, *table_rows])

    return f"""# Power Forecast Monitoring Report

Generated at: `{summary["created_at_utc"]}`

## Health status

**Status:** `{health}`

## Warnings

{warning_text}

## Latest prediction window

| Item | Value |
|---|---:|
| Min timestamp | `{window["min_timestamp"]}` |
| Max timestamp | `{window["max_timestamp"]}` |
| Rows | {window["n_rows"]} |
| Has actuals | {window["has_actuals"]} |

## Model selection

| Item | Value |
|---|---:|
| Production model | `{model["model_name"]}` |
| Latest 24h winner | `{latest_winner}` |
| Model path | `{model["model_path"]}` |
| Trained at | `{model["model_trained_at_utc"]}` |
| Feature count | {model["feature_count"]} |

## Production model latest realized performance

| Metric | Value |
|---|---:|
| MAE | {_format_float(latest["mae"])} |
| RMSE | {_format_float(latest["rmse"])} |
| MAPE | {_format_pct(latest["mape"])} |
| Bias | {_format_float(latest["bias"])} |

## Model comparison

{model_table}

## Reference performance

| Reference | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Production model walk-forward average | {_format_float(training["mae"])} | {_format_float(training["rmse"])} | {_format_pct(training["mape"])} |
| Best baseline: {baseline["name"]} | {_format_float(baseline["mae"])} | {_format_float(baseline["rmse"])} | N/A |

## Ratios

| Ratio | Value |
|---|---:|
| Latest MAE / training MAE | {_format_float(ratios["latest_mae_vs_training_mae"], 3)} |
| Latest MAE / best baseline MAE | {_format_float(ratios["latest_mae_vs_best_baseline_mae"], 3)} |

## Latest prediction samples

{latest_table}

## Interpretation

The production model is selected by average walk-forward validation MAE. The latest 24h winner is the model with the lowest realized MAE on the latest known prediction window. These two can differ because shorter training windows may adapt better to the current regime, while longer training windows may be more robust on average.
"""


def _render_html_report(markdown: str) -> str:
    lines = markdown.splitlines()
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Power Forecast Monitoring Report</title>",
        "<style>",
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; max-width: 1100px; line-height: 1.5; }",
        "h1, h2 { color: #222; }",
        "code { background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }",
        "table { border-collapse: collapse; margin: 16px 0; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f4f4f4; }",
        "tr:nth-child(even) { background: #fafafa; }",
        ".status { font-size: 1.1em; font-weight: 700; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    in_table = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            continue

        if stripped.startswith("# "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]

            if all(set(c) <= {"-", ":"} for c in cells):
                continue

            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"

            row_html = "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
            html_lines.append(row_html)
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

            escaped = html.escape(stripped)
            escaped = escaped.replace("`", "")
            html_lines.append(f"<p>{escaped}</p>")

    if in_table:
        html_lines.append("</table>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def _model_description(model_name: str) -> dict[str, str]:
    base_model_name, window_label = _split_candidate_name(model_name)

    descriptions = {
        "lightgbm": {
            "title": "LightGBM",
            "style": "Gradient-boosted decision-tree model",
            "description": (
                "LightGBM builds an ensemble of decision trees sequentially. Each tree focuses on "
                "the residual errors left by previous trees. It is strong for tabular data, handles "
                "nonlinear interactions well, and usually performs very well on structured forecasting features."
            ),
            "strengths": (
                "Strong tabular baseline, nonlinear feature interactions, robust with small-to-medium datasets, "
                "fast CPU training."
            ),
            "caveats": (
                "Less naturally sequential than recurrent models; relies heavily on explicit lag and rolling features."
            ),
        },
        "mlp": {
            "title": "MLP Neural Network",
            "style": "Feed-forward neural network",
            "description": (
                "The MLP receives engineered demand, calendar, and weather features and learns nonlinear "
                "combinations through dense hidden layers. Inputs and targets are standardized during training."
            ),
            "strengths": (
                "Flexible nonlinear function approximator; useful comparison against tree and reservoir models."
            ),
            "caveats": (
                "Can be sensitive to scaling, regularization, early stopping, and small-data regimes."
            ),
        },
        "esn": {
            "title": "Echo State Network / Reservoir Computer",
            "style": "Fixed recurrent reservoir with trained linear readout",
            "description": (
                "The ESN projects the feature sequence through a fixed random recurrent reservoir. Only the "
                "readout is trained. This creates a nonlinear temporal state representation while keeping "
                "training lightweight and stable."
            ),
            "strengths": (
                "Very fast training, strong temporal inductive bias, useful for autocorrelated dynamical systems."
            ),
            "caveats": (
                "Performance can depend on reservoir hyperparameters and random seed; feature importances are less direct."
            ),
        },
    }

    desc = descriptions.get(
        base_model_name,
        {
            "title": model_name,
            "style": "Model",
            "description": "No description available.",
            "strengths": "N/A",
            "caveats": "N/A",
        },
    )

    if window_label:
        desc = dict(desc)
        desc["title"] = f'{desc["title"]} ({window_label} training window)'
        desc["description"] = (
            desc["description"]
            + f" This candidate is trained only on the most recent {window_label} of available training data."
        )

    return desc


def _build_prediction_sample_table(
    predictions: pd.DataFrame,
    production_model: str,
    latest_winner: str | None,
) -> tuple[str, str]:
    show_cols = ["timestamp_utc"]

    if "demand_mwh" in predictions.columns:
        show_cols.append("demand_mwh")

    production_col = f"prediction_mwh_{production_model}"
    if production_col in predictions.columns:
        show_cols.append(production_col)

    if latest_winner is not None:
        latest_col = f"prediction_mwh_{latest_winner}"
        if latest_col in predictions.columns and latest_col not in show_cols:
            show_cols.append(latest_col)

    for col in ["prediction_mwh", "error", "absolute_error", "absolute_percentage_error"]:
        if col in predictions.columns and col not in show_cols:
            show_cols.append(col)

    prediction_rows = []
    for _, row in predictions.tail(10).iterrows():
        cells = []
        for col in show_cols:
            value = row[col]
            if isinstance(value, float):
                if col == "absolute_percentage_error":
                    cells.append(f"<td>{100 * value:.2f}%</td>")
                else:
                    cells.append(f"<td>{value:.2f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        prediction_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_header = "".join(f"<th>{html.escape(col)}</th>" for col in show_cols)
    table_body = "".join(prediction_rows)
    return table_header, table_body


def _render_html_dashboard(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    future_forecast: pd.DataFrame | None = None,
) -> str:
    health = summary["health_status"]
    warnings = summary["warnings"]

    latest = summary["latest_metrics"]
    baseline = summary["best_baseline"]
    ratios = summary["ratios"]
    window = summary["latest_prediction_window"]
    model = summary["model"]

    production_model = str(model.get("model_name", "unknown"))
    latest_winner = summary.get("latest_window_winner")

    per_model_latest = summary.get("per_model_latest_metrics", {})
    per_model_training = summary.get("per_model_training_metrics", {})

    all_model_names = sorted(
        set(per_model_training.keys()) | set(per_model_latest.keys()),
        key=lambda name: _model_sort_key(name, per_model_training, per_model_latest),
    )

    health_class = {
        "healthy": "healthy",
        "watch": "watch",
        "degraded": "degraded",
        "unknown": "unknown",
    }.get(health, "unknown")

    warning_items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    if not warning_items:
        warning_items = "<li>None</li>"

    generated_at = html.escape(str(summary["created_at_utc"]))

    model_metric_rows = []
    for model_name in all_model_names:
        train_m = per_model_training.get(model_name, {})
        latest_m = per_model_latest.get(model_name, {})

        roles = []
        if model_name == production_model:
            roles.append("production")
        if model_name == latest_winner:
            roles.append("latest")
        selected = " ".join(roles)

        model_metric_rows.append(
            f"""
            <tr>
              <td><strong>{html.escape(model_name)}</strong> <span class="model-tags">{html.escape(selected)}</span></td>
              <td>{_format_float(_metric(train_m, "mae"))}</td>
              <td>{_format_float(_metric(train_m, "rmse"))}</td>
              <td>{_format_pct(_metric(train_m, "mape"))}</td>
              <td>{_format_float(_metric(train_m, "bias"))}</td>
              <td>{_format_float(_metric(latest_m, "mae"))}</td>
              <td>{_format_float(_metric(latest_m, "rmse"))}</td>
              <td>{_format_pct(_metric(latest_m, "mape"))}</td>
              <td>{_format_float(_metric(latest_m, "bias"))}</td>
            </tr>
            """
        )

    model_detail_cards = []
    for model_name in all_model_names:
        desc = _model_description(model_name)
        train_m = per_model_training.get(model_name, {})
        latest_m = per_model_latest.get(model_name, {})

        badges = []
        if model_name == production_model:
            badges.append('<span class="mini-badge selected">production model</span>')
        if model_name == latest_winner:
            badges.append('<span class="mini-badge selected">latest 24h winner</span>')

        selected_badge = " ".join(badges)

        model_detail_cards.append(
            f"""
            <div class="model-card">
              <div class="model-card-header">
                <div>
                  <h3>{html.escape(desc["title"])}</h3>
                  <p class="model-style">{html.escape(desc["style"])}</p>
                </div>
                {selected_badge}
              </div>
              <p>{html.escape(desc["description"])}</p>

              <div class="model-grid">
                <div>
                  <h4>Walk-forward performance</h4>
                  <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>MAE</td><td>{_format_float(_metric(train_m, "mae"))}</td></tr>
                    <tr><td>RMSE</td><td>{_format_float(_metric(train_m, "rmse"))}</td></tr>
                    <tr><td>MAPE</td><td>{_format_pct(_metric(train_m, "mape"))}</td></tr>
                    <tr><td>Bias</td><td>{_format_float(_metric(train_m, "bias"))}</td></tr>
                  </table>
                </div>
                <div>
                  <h4>Latest 24h performance</h4>
                  <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>MAE</td><td>{_format_float(_metric(latest_m, "mae"))}</td></tr>
                    <tr><td>RMSE</td><td>{_format_float(_metric(latest_m, "rmse"))}</td></tr>
                    <tr><td>MAPE</td><td>{_format_pct(_metric(latest_m, "mape"))}</td></tr>
                    <tr><td>Bias</td><td>{_format_float(_metric(latest_m, "bias"))}</td></tr>
                  </table>
                </div>
              </div>

              <div class="model-notes">
                <p><strong>Strengths:</strong> {html.escape(desc["strengths"])}</p>
                <p><strong>Caveats:</strong> {html.escape(desc["caveats"])}</p>
              </div>
            </div>
            """
        )

    table_header, prediction_rows = _build_prediction_sample_table(
        predictions=predictions,
        production_model=production_model,
        latest_winner=latest_winner,
    )

    future_summary = summary.get("future_forecast") or {}
    future_window_text = "N/A"
    if future_summary:
        future_window_text = (
            f"{html.escape(str(future_summary.get('min_timestamp', 'N/A')))} → "
            f"{html.escape(str(future_summary.get('max_timestamp', 'N/A')))}"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PowerForecastMLOps Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #07111f;
      --panel: #101827;
      --panel2: #162033;
      --panel3: #1e293b;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --line: #334155;
      --good: #22c55e;
      --watch: #f59e0b;
      --bad: #ef4444;
      --unknown: #94a3b8;
      --accent: #38bdf8;
      --accent2: #a78bfa;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 34rem),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 30rem),
        linear-gradient(180deg, #020617 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.55;
    }}

    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 36px 22px 64px;
    }}

    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .hero {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: start;
      margin-bottom: 20px;
    }}

    .title {{
      margin: 0;
      font-size: 2.25rem;
      letter-spacing: -0.04em;
    }}

    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      max-width: 920px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 9px 15px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.82rem;
      border: 1px solid var(--line);
      background: var(--panel);
    }}

    .badge.healthy {{
      color: var(--good);
      border-color: rgba(34, 197, 94, 0.5);
      background: rgba(34, 197, 94, 0.09);
    }}

    .badge.watch {{
      color: var(--watch);
      border-color: rgba(245, 158, 11, 0.5);
      background: rgba(245, 158, 11, 0.09);
    }}

    .badge.degraded {{
      color: var(--bad);
      border-color: rgba(239, 68, 68, 0.5);
      background: rgba(239, 68, 68, 0.09);
    }}

    .badge.unknown {{
      color: var(--unknown);
      border-color: rgba(148, 163, 184, 0.5);
      background: rgba(148, 163, 184, 0.09);
    }}

    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 26px 0 18px;
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 10px 0;
      backdrop-filter: blur(12px);
    }}

    .tab-button {{
      border: 1px solid var(--line);
      color: var(--muted);
      background: rgba(16, 24, 39, 0.88);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 700;
    }}

    .tab-button.active {{
      color: #ffffff;
      border-color: rgba(56, 189, 248, 0.7);
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(167, 139, 250, 0.18));
    }}

    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}

    .card {{
      background: rgba(16, 24, 39, 0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.25);
    }}

    .card .label {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 6px;
    }}

    .card .value {{
      font-size: 1.55rem;
      font-weight: 850;
      letter-spacing: -0.04em;
      word-break: break-word;
    }}

    .card .small {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.88rem;
    }}

    section, .section {{
      margin-top: 22px;
      background: rgba(16, 24, 39, 0.74);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.18);
    }}

    h2 {{
      margin: 0 0 14px;
      letter-spacing: -0.02em;
    }}

    h3 {{
      margin-bottom: 8px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      font-size: 0.94rem;
    }}

    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}

    th {{
      color: var(--muted);
      font-weight: 800;
      background: rgba(30, 41, 59, 0.72);
    }}

    .sortable-table th {{
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}

    .sortable-table th:hover {{
      color: #ffffff;
      background: rgba(56, 189, 248, 0.16);
    }}

    .sort-indicator {{
      color: var(--accent);
      font-size: 0.78rem;
      margin-left: 6px;
    }}

    tr:last-child td {{ border-bottom: none; }}

    code {{
      background: rgba(148, 163, 184, 0.12);
      padding: 2px 5px;
      border-radius: 5px;
      color: #f8fafc;
    }}

    .pipeline {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
    }}

    .step {{
      padding: 8px 11px;
      border: 1px solid var(--line);
      background: rgba(30, 41, 59, 0.8);
      border-radius: 999px;
      color: var(--text);
      font-size: 0.9rem;
    }}

    .arrow {{ color: var(--muted); }}

    .plot {{
      background: #ffffff;
      border-radius: 14px;
      padding: 12px;
      margin-top: 10px;
    }}

    .plot img {{
      width: 100%;
      display: block;
      border-radius: 8px;
    }}

    .note {{
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .model-tags {{
      color: var(--accent);
      font-size: 0.82rem;
      margin-left: 6px;
      white-space: nowrap;
    }}

    .model-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      background: rgba(30, 41, 59, 0.48);
      margin-bottom: 18px;
    }}

    .model-card-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }}

    .model-card h3 {{
      margin: 0;
    }}

    .model-style {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.94rem;
    }}

    .model-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 12px;
    }}

    .model-notes {{
      margin-top: 12px;
      color: var(--muted);
    }}

    .mini-badge {{
      display: inline-flex;
      white-space: nowrap;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-left: 6px;
    }}

    .mini-badge.selected {{
      color: var(--good);
      border: 1px solid rgba(34, 197, 94, 0.5);
      background: rgba(34, 197, 94, 0.10);
    }}

    .scroll-table {{
      overflow-x: auto;
    }}

    @media (max-width: 1120px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}

    @media (max-width: 960px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .model-grid {{ grid-template-columns: 1fr; }}
    }}

    @media (max-width: 620px) {{
      .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 26px 14px 48px; }}
      .title {{ font-size: 1.7rem; }}
    }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <div>
      <h1 class="title">PowerForecastMLOps Dashboard</h1>
      <p class="subtitle">
        Live electricity-demand forecasting system using public EIA demand data and Open-Meteo weather data.
        The pipeline compares LightGBM, an MLP neural network, and an Echo State Network across multiple
        recent training windows under the same forecast-safe walk-forward protocol.
      </p>
      <p class="note">Generated at: <code>{generated_at}</code></p>
    </div>
    <div>
      <span class="badge {health_class}">{html.escape(health)}</span>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-button active" onclick="openTab(event, 'overview')">Overview</button>
    <button class="tab-button" onclick="openTab(event, 'models')">Models</button>
    <button class="tab-button" onclick="openTab(event, 'data')">Data & Pipeline</button>
    <button class="tab-button" onclick="openTab(event, 'predictions')">Predictions</button>
    <button class="tab-button" onclick="openTab(event, 'diagnostics')">Diagnostics</button>
  </div>

  <div id="overview" class="tab-panel active">
    <div class="grid">
      <div class="card">
        <div class="label">Production model</div>
        <div class="value">{html.escape(production_model)}</div>
        <div class="small">Selected by walk-forward MAE</div>
      </div>
      <div class="card">
        <div class="label">Latest 24h winner</div>
        <div class="value">{html.escape(str(latest_winner or "N/A"))}</div>
        <div class="small">Best realized MAE on shown window</div>
      </div>
      <div class="card">
        <div class="label">Production latest MAE</div>
        <div class="value">{_format_float(latest["mae"])}</div>
        <div class="small">MWh over latest 24h window</div>
      </div>
      <div class="card">
        <div class="label">Production latest MAPE</div>
        <div class="value">{_format_pct(latest["mape"])}</div>
        <div class="small">Realized latest-window error</div>
      </div>
      <div class="card">
        <div class="label">Production / baseline</div>
        <div class="value">{_format_float(ratios["latest_mae_vs_best_baseline_mae"], 3)}×</div>
        <div class="small">Latest MAE / best baseline MAE</div>
      </div>
    </div>

    <p class="note">
      The production model is selected by average walk-forward validation MAE. The latest 24h winner is the
      candidate with the lowest realized MAE on the current monitoring window. These can differ because shorter
      training windows may adapt better to the current regime, while longer windows may be more robust on average.
    </p>

    <section>
      <h2>Model comparison</h2>
      <p class="note">Click a column header to sort ascending; click it again to sort descending.</p>
      <div class="scroll-table">
        <table class="sortable-table">
          <tr>
            <th onclick="sortTable(this, 0, 'text')">Model</th>
            <th onclick="sortTable(this, 1, 'number')">Walk-forward MAE</th>
            <th onclick="sortTable(this, 2, 'number')">Walk-forward RMSE</th>
            <th onclick="sortTable(this, 3, 'number')">Walk-forward MAPE</th>
            <th onclick="sortTable(this, 4, 'number')">Walk-forward Bias</th>
            <th onclick="sortTable(this, 5, 'number')">Latest MAE</th>
            <th onclick="sortTable(this, 6, 'number')">Latest RMSE</th>
            <th onclick="sortTable(this, 7, 'number')">Latest MAPE</th>
            <th onclick="sortTable(this, 8, 'number')">Latest Bias</th>
          </tr>
          {''.join(model_metric_rows)}
          <tr>
            <td><strong>Best baseline: {html.escape(str(baseline["name"]))}</strong></td>
            <td>{_format_float(baseline["mae"])}</td>
            <td>{_format_float(baseline["rmse"])}</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
          </tr>
        </table>
      </div>
    </section>

    <section>
      <h2>Latest 24h prediction</h2>
      <p class="note">The plot compares actual demand with each model candidate's prediction over the latest 24 known hours.</p>
      <div class="plot"><img src="figures/latest_predictions.png" alt="Latest predictions by model"></div>
    </section>

    <section>
      <h2>Next 24h forecast</h2>
      <p class="note">
        Forecast window: <code>{future_window_text}</code>. This uses weather forecast data, calendar features,
        and demand-history features available before the forecast horizon.
      </p>
      <div class="plot"><img src="figures/future_24h_forecast.png" alt="Next 24h forecast"></div>
    </section>

    <section>
      <h2>Warnings</h2>
      <ul>{warning_items}</ul>
    </section>
  </div>

  <div id="models" class="tab-panel">
    <section>
      <h2>Model details</h2>
      <p class="note">
        Each candidate is evaluated using the same forecast-safe feature table and walk-forward validation protocol.
        The suffix, such as <code>30d</code> or <code>1095d</code>, denotes how much recent training data was used.
      </p>
      {''.join(model_detail_cards)}
    </section>
  </div>

  <div id="data" class="tab-panel">
    <section>
      <h2>Dataset</h2>
      <table>
        <tr><th>Item</th><th>Value</th></tr>
        <tr><td>Demand source</td><td>EIA Open Data API, California ISO hourly demand</td></tr>
        <tr><td>Weather source</td><td>Open-Meteo historical and forecast weather, Los Angeles proxy location</td></tr>
        <tr><td>Latest prediction window start</td><td><code>{html.escape(str(window["min_timestamp"]))}</code></td></tr>
        <tr><td>Latest prediction window end</td><td><code>{html.escape(str(window["max_timestamp"]))}</code></td></tr>
        <tr><td>Rows in latest prediction window</td><td>{window["n_rows"]}</td></tr>
        <tr><td>Feature count</td><td>{model["feature_count"]}</td></tr>
      </table>
    </section>

    <section>
      <h2>Pipeline overview</h2>
      <div class="pipeline">
        <span class="step">EIA + Open-Meteo APIs</span>
        <span class="arrow">→</span>
        <span class="step">Raw data validation</span>
        <span class="arrow">→</span>
        <span class="step">Forecast-safe features</span>
        <span class="arrow">→</span>
        <span class="step">Walk-forward backtest</span>
        <span class="arrow">→</span>
        <span class="step">Windowed model comparison</span>
        <span class="arrow">→</span>
        <span class="step">Latest 24h prediction</span>
        <span class="arrow">→</span>
        <span class="step">Next 24h forecast</span>
        <span class="arrow">→</span>
        <span class="step">Monitoring report</span>
        <span class="arrow">→</span>
        <span class="step">GitHub Pages</span>
      </div>
    </section>
  </div>

  <div id="predictions" class="tab-panel">
    <section>
      <h2>Next 24h future forecast</h2>
      <p class="note">
        This is a true future forecast. Actual demand is not known yet, so no error metrics are shown.
      </p>
      <div class="plot"><img src="figures/future_24h_forecast.png" alt="Next 24h future forecast"></div>
    </section>

    <section>
      <h2>Latest prediction samples</h2>
      <p class="note">
        This compact table shows actual demand, the production-model prediction, and the latest-window-winner prediction.
        The full per-model comparison is shown in the overview table.
      </p>
      <div class="scroll-table">
        <table>
          <tr>{table_header}</tr>
          {prediction_rows}
        </table>
      </div>
    </section>
  </div>

  <div id="diagnostics" class="tab-panel">
    <section>
      <h2>Model comparison plot</h2>
      <div class="plot"><img src="figures/model_comparison.png" alt="Model comparison plot"></div>
    </section>

    <section>
      <h2>Feature importance / explanation</h2>
      <p class="note">
        Tree-based models expose native feature importance. Neural and reservoir models do not provide the same direct interpretation,
        so the report shows a placeholder when the selected model has no native feature importances.
      </p>
      <div class="plot"><img src="figures/lightgbm_feature_importance.png" alt="Feature importance plot"></div>
    </section>

    <section>
      <h2>Baseline comparison</h2>
      <div class="plot"><img src="figures/backtest_baselines.png" alt="Baseline comparison plot"></div>
    </section>

    <section>
      <h2>Model metadata</h2>
      <table>
        <tr><th>Item</th><th>Value</th></tr>
        <tr><td>Production model</td><td><code>{html.escape(production_model)}</code></td></tr>
        <tr><td>Latest 24h winner</td><td><code>{html.escape(str(latest_winner or "N/A"))}</code></td></tr>
        <tr><td>Model path</td><td><code>{html.escape(str(model["model_path"]))}</code></td></tr>
        <tr><td>Trained at</td><td><code>{html.escape(str(model["model_trained_at_utc"]))}</code></td></tr>
        <tr><td>Production latest MAE / training MAE</td><td>{_format_float(ratios["latest_mae_vs_training_mae"], 3)}</td></tr>
        <tr><td>Production latest MAE / best baseline MAE</td><td>{_format_float(ratios["latest_mae_vs_best_baseline_mae"], 3)}</td></tr>
      </table>
    </section>
  </div>
</main>

<script>
function openTab(event, tabId) {{
  const panels = document.querySelectorAll('.tab-panel');
  panels.forEach(panel => panel.classList.remove('active'));

  const buttons = document.querySelectorAll('.tab-button');
  buttons.forEach(button => button.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');
  event.currentTarget.classList.add('active');
}}

function parseCellValue(text, type) {{
  const cleaned = text
    .replace(/production/g, '')
    .replace(/latest/g, '')
    .replace(/,/g, '')
    .replace(/%/g, '')
    .replace(/×/g, '')
    .trim();

  if (cleaned === '' || cleaned === 'N/A') {{
    return type === 'number' ? Number.POSITIVE_INFINITY : '';
  }}

  if (type === 'number') {{
    const value = Number.parseFloat(cleaned);
    return Number.isNaN(value) ? Number.POSITIVE_INFINITY : value;
  }}

  return cleaned.toLowerCase();
}}

function clearSortIndicators(table) {{
  table.querySelectorAll('th').forEach(th => {{
    const indicator = th.querySelector('.sort-indicator');
    if (indicator) {{
      indicator.remove();
    }}
  }});
}}

function sortTable(headerCell, columnIndex, type) {{
  const table = headerCell.closest('table');
  const tbody = table.tBodies[0] || table;
  const rows = Array.from(table.querySelectorAll('tr')).slice(1);

  const currentDirection = headerCell.dataset.sortDirection || 'none';
  const nextDirection = currentDirection === 'asc' ? 'desc' : 'asc';

  table.querySelectorAll('th').forEach(th => {{
    th.dataset.sortDirection = 'none';
  }});
  headerCell.dataset.sortDirection = nextDirection;

  const sortableRows = [];
  const pinnedRows = [];

  rows.forEach(row => {{
    const cells = row.querySelectorAll('td');
    const firstCell = cells[0] ? cells[0].innerText.toLowerCase() : '';

    if (firstCell.includes('best baseline')) {{
      pinnedRows.push(row);
    }} else {{
      sortableRows.push(row);
    }}
  }});

  sortableRows.sort((a, b) => {{
    const aText = a.children[columnIndex]?.innerText || '';
    const bText = b.children[columnIndex]?.innerText || '';

    const aValue = parseCellValue(aText, type);
    const bValue = parseCellValue(bText, type);

    if (aValue < bValue) return nextDirection === 'asc' ? -1 : 1;
    if (aValue > bValue) return nextDirection === 'asc' ? 1 : -1;
    return 0;
  }});

  clearSortIndicators(table);

  const indicator = document.createElement('span');
  indicator.className = 'sort-indicator';
  indicator.textContent = nextDirection === 'asc' ? '▲' : '▼';
  headerCell.appendChild(indicator);

  sortableRows.concat(pinnedRows).forEach(row => tbody.appendChild(row));
}}
</script>
</body>
</html>
"""
