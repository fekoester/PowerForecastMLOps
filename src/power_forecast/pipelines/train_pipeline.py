from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.models.train_lightgbm import train_lightgbm_walk_forward
from power_forecast.utils.config import load_yaml

console = Console()


def run_train_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)

    train_cfg = config["train"]
    features_cfg = config["features"]
    backtest_cfg = config["backtest"]

    console.print("[bold]Starting LightGBM training pipeline[/bold]")

    report = train_lightgbm_walk_forward(
        features_path=features_cfg["output_path"],
        baseline_backtest_path=backtest_cfg["output_path"],
        output_path=train_cfg["output_path"],
        model_path=train_cfg["model_path"],
        feature_importance_path=train_cfg["feature_importance_path"],
        timestamp_column=train_cfg["timestamp_column"],
        target_column=train_cfg["target_column"],
        min_train_days=int(train_cfg["min_train_days"]),
        validation_window_days=int(train_cfg["validation_window_days"]),
        step_days=int(train_cfg["step_days"]),
        model_config=dict(train_cfg["model"]),
    )

    metrics = report["aggregate_metrics"]
    comparison = report["baseline_comparison"]

    table = Table(title="LightGBM Walk-Forward Training Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Model MAE", f"{metrics['mae_mean']:.2f}")
    table.add_row("Model RMSE", f"{metrics['rmse_mean']:.2f}")
    table.add_row("Model MAPE", f"{100 * metrics['mape_mean']:.2f}%")
    table.add_row("Model Bias", f"{metrics['bias_mean']:.2f}")
    table.add_row("Best baseline", comparison["best_baseline_name"])
    table.add_row("Best baseline MAE", f"{comparison['best_baseline_mae']:.2f}")
    table.add_row("MAE improvement", f"{comparison['mae_improvement']:.2f}")
    table.add_row("MAE improvement %", f"{comparison['mae_improvement_pct']:.2f}%")
    table.add_row("Beats baseline", str(comparison["beats_baseline"]))

    console.print(table)

    if comparison["beats_baseline"]:
        console.print("[green]Model beats the best baseline.[/green]")
    else:
        console.print("[yellow]Model does not beat the best baseline yet.[/yellow]")

    console.print(f"Training report written: [bold]{train_cfg['output_path']}[/bold]")
    console.print(f"Model saved: [bold]{train_cfg['model_path']}[/bold]")
    console.print(f"Feature importance figure: [bold]{train_cfg['feature_importance_path']}[/bold]")
