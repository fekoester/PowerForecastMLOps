from __future__ import annotations

from pathlib import Path
import shutil

from rich.console import Console
from rich.table import Table

from power_forecast.monitoring.report import build_monitoring_report
from power_forecast.utils.config import load_yaml

console = Console()


def _format_metric(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.{digits}f}%"


def run_monitor_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)
    monitor_cfg = config["monitor"]

    console.print("[bold]Starting monitoring report pipeline[/bold]")

    summary = build_monitoring_report(
        prediction_summary_path=monitor_cfg["prediction_summary_path"],
        train_report_path=monitor_cfg["train_report_path"],
        baseline_report_path=monitor_cfg["baseline_report_path"],
        predictions_path=monitor_cfg["predictions_path"],
        summary_path=monitor_cfg["summary_path"],
        markdown_report_path=monitor_cfg["markdown_report_path"],
        html_report_path=monitor_cfg["html_report_path"],
        thresholds=dict(monitor_cfg["degradation_thresholds"]),
        future_forecast_summary_path=monitor_cfg.get("future_forecast_summary_path"),
        future_forecast_path=monitor_cfg.get("future_forecast_path"),
    )

    local_figures_dir = Path("reports/monitoring/figures")
    local_figures_dir.mkdir(parents=True, exist_ok=True)

    source_figures_dir = Path("reports/figures")
    for figure_path in source_figures_dir.glob("*.png"):
        shutil.copy2(figure_path, local_figures_dir / figure_path.name)

    latest = summary["latest_metrics"]
    ratios = summary["ratios"]
    production_model = summary.get("model", {}).get("model_name", "unknown")
    latest_winner = summary.get("latest_window_winner", "unknown")

    table = Table(title="Monitoring Summary")
    table.add_column("Item")
    table.add_column("Value", justify="right")

    table.add_row("Health status", summary["health_status"])
    table.add_row("Production model", str(production_model))
    table.add_row("Latest 24h winner", str(latest_winner))
    table.add_row("Latest MAE", _format_metric(latest.get("mae")))
    table.add_row(
        "Latest MAE / training MAE",
        _format_metric(ratios.get("latest_mae_vs_training_mae"), digits=3),
    )
    table.add_row(
        "Latest MAE / baseline MAE",
        _format_metric(ratios.get("latest_mae_vs_best_baseline_mae"), digits=3),
    )
    table.add_row("Warnings", str(len(summary["warnings"])))
    table.add_row("Markdown report", monitor_cfg["markdown_report_path"])
    table.add_row("HTML report", monitor_cfg["html_report_path"])

    console.print(table)

    per_model_latest = summary.get("per_model_latest_metrics", {})

    if per_model_latest:
        latest_table = Table(title="Latest 24h Model Performance")
        latest_table.add_column("Model")
        latest_table.add_column("MAE", justify="right")
        latest_table.add_column("RMSE", justify="right")
        latest_table.add_column("MAPE", justify="right")
        latest_table.add_column("Bias", justify="right")
        latest_table.add_column("Role", justify="right")

        rows = sorted(
            per_model_latest.items(),
            key=lambda item: item[1].get("mae", float("inf")),
        )

        for model_name, metrics in rows:
            roles = []
            if model_name == production_model:
                roles.append("production")
            if model_name == latest_winner:
                roles.append("latest winner")

            latest_table.add_row(
                str(model_name),
                _format_metric(metrics.get("mae")),
                _format_metric(metrics.get("rmse")),
                _format_pct(metrics.get("mape")),
                _format_metric(metrics.get("bias")),
                ", ".join(roles),
            )

        console.print(latest_table)

    if summary["warnings"]:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    else:
        console.print("[green]No monitoring warnings.[/green]")
