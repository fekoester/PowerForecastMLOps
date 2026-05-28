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
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.{digits}f}%"


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


def build_monitoring_report(
    prediction_summary_path: str | Path,
    train_report_path: str | Path,
    baseline_report_path: str | Path,
    predictions_path: str | Path,
    summary_path: str | Path,
    markdown_report_path: str | Path,
    html_report_path: str | Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    prediction_summary = _load_json(prediction_summary_path)
    train_report = _load_json(train_report_path)
    baseline_report = _load_json(baseline_report_path)

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    latest_metrics = prediction_summary.get("metrics") or {}

    latest_mae = latest_metrics.get("mae")
    latest_rmse = latest_metrics.get("rmse")
    latest_mape = latest_metrics.get("mape")
    latest_bias = latest_metrics.get("bias")

    training_mae = float(train_report["aggregate_metrics"]["mae_mean"])
    training_rmse = float(train_report["aggregate_metrics"]["rmse_mean"])
    training_mape = float(train_report["aggregate_metrics"]["mape_mean"])

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
            "model_trained_at_utc": prediction_summary.get("model_trained_at_utc"),
            "feature_count": prediction_summary.get("feature_count"),
        },
        "latest_metrics": {
            "mae": latest_mae,
            "rmse": latest_rmse,
            "mape": latest_mape,
            "bias": latest_bias,
        },
        "walk_forward_training_metrics": {
            "mae": training_mae,
            "rmse": training_rmse,
            "mape": training_mape,
        },
        "best_baseline": {
            "name": best_baseline_name,
            "mae": best_baseline_mae,
            "rmse": best_baseline_rmse,
        },
        "ratios": {
            "latest_mae_vs_training_mae": ratio_vs_training,
            "latest_mae_vs_best_baseline_mae": ratio_vs_baseline,
        },
        "input_files": {
            "prediction_summary_path": str(prediction_summary_path),
            "train_report_path": str(train_report_path),
            "baseline_report_path": str(baseline_report_path),
            "predictions_path": str(predictions_path),
        },
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = _render_markdown_report(summary, predictions)
    markdown_path = Path(markdown_report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    html_report = _render_html_report(markdown)
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

    warning_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "- None"

    # Show only a compact table of the latest predictions.
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

## Model

| Item | Value |
|---|---:|
| Model path | `{model["model_path"]}` |
| Trained at | `{model["model_trained_at_utc"]}` |
| Feature count | {model["feature_count"]} |

## Latest realized performance

| Metric | Value |
|---|---:|
| MAE | {_format_float(latest["mae"])} |
| RMSE | {_format_float(latest["rmse"])} |
| MAPE | {_format_pct(latest["mape"])} |
| Bias | {_format_float(latest["bias"])} |

## Reference performance

| Reference | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Walk-forward LightGBM average | {_format_float(training["mae"])} | {_format_float(training["rmse"])} | {_format_pct(training["mape"])} |
| Best baseline: {baseline["name"]} | {_format_float(baseline["mae"])} | {_format_float(baseline["rmse"])} | N/A |

## Ratios

| Ratio | Value |
|---|---:|
| Latest MAE / training MAE | {_format_float(ratios["latest_mae_vs_training_mae"], 3)} |
| Latest MAE / best baseline MAE | {_format_float(ratios["latest_mae_vs_best_baseline_mae"], 3)} |

## Latest prediction samples

{latest_table}

## Interpretation

This report checks whether the latest prediction window is consistent with the model's walk-forward validation performance and whether it remains competitive with the strongest naive baseline. A `healthy` status does not mean the model is perfect; it means there is no obvious degradation signal in the latest available window.
"""


def _render_html_report(markdown: str) -> str:
    # Minimal Markdown-to-HTML renderer.
    # This intentionally avoids adding another dependency.
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

            # separator row like |---|---|
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
