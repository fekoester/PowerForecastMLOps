from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.monitoring.report import build_monitoring_report
from power_forecast.utils.config import load_yaml

console = Console()


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
    )

    table = Table(title="Monitoring Summary")
    table.add_column("Item")
    table.add_column("Value", justify="right")

    latest = summary["latest_metrics"]
    ratios = summary["ratios"]

    table.add_row("Health status", summary["health_status"])
    table.add_row("Latest MAE", f"{latest['mae']:.2f}" if latest["mae"] is not None else "N/A")
    table.add_row(
        "Latest MAE / training MAE",
        f"{ratios['latest_mae_vs_training_mae']:.3f}"
        if ratios["latest_mae_vs_training_mae"] is not None
        else "N/A",
    )
    table.add_row(
        "Latest MAE / baseline MAE",
        f"{ratios['latest_mae_vs_best_baseline_mae']:.3f}"
        if ratios["latest_mae_vs_best_baseline_mae"] is not None
        else "N/A",
    )
    table.add_row("Warnings", str(len(summary["warnings"])))
    table.add_row("Markdown report", monitor_cfg["markdown_report_path"])
    table.add_row("HTML report", monitor_cfg["html_report_path"])

    console.print(table)

    if summary["warnings"]:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    else:
        console.print("[green]No monitoring warnings.[/green]")
