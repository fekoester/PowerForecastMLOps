from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.models.train_compare import train_and_compare_models
from power_forecast.utils.config import load_yaml

console = Console()


def run_train_pipeline(
    config_path: str,
    selected_models: list[str] | None = None,
    selected_windows: list[int] | None = None,
) -> None:
    config = load_yaml(config_path)

    train_cfg = config["train"]
    features_cfg = config["features"]
    backtest_cfg = config["backtest"]

    console.print("[bold]Starting model comparison training pipeline[/bold]")

    report = train_and_compare_models(
        features_path=features_cfg["output_path"],
        baseline_backtest_path=backtest_cfg["output_path"],
        output_path=train_cfg["output_path"],
        model_path=train_cfg["model_path"],
        feature_importance_path=train_cfg["feature_importance_path"],
        model_comparison_path=train_cfg["model_comparison_path"],
        timestamp_column=train_cfg["timestamp_column"],
        target_column=train_cfg["target_column"],
        min_train_days=int(train_cfg["min_train_days"]),
        validation_window_days=int(train_cfg["validation_window_days"]),
        step_days=int(train_cfg["step_days"]),
        models_config=dict(train_cfg["models"]),
        model_selection_metric=str(train_cfg["model_selection_metric"]),
        forecast_safe_features=bool(train_cfg.get("forecast_safe_features", False)),
        allowed_lag_hours=list(train_cfg.get("allowed_lag_hours", [24, 48, 168])),
        recency_weighting_config=dict(train_cfg.get("recency_weighting", {"enabled": False})),
        training_windows_days=list(train_cfg.get("training_windows_days", [1095])),
        selected_models=selected_models,
        selected_windows=selected_windows,
    )
    
    if selected_models:
        console.print(f"[cyan]Training selected model families:[/cyan] {', '.join(selected_models)}")
    if selected_windows:
        console.print(f"[cyan]Training selected windows:[/cyan] {', '.join(str(w) for w in selected_windows)}")

    table = Table(title="Model Comparison Summary")
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("Bias", justify="right")

    model_names = sorted(
        report["models"].keys(),
        key=lambda name: report["models"][name]["aggregate_metrics"]["mae_mean"],
    )

    for name in model_names:
        metrics = report["models"][name]["aggregate_metrics"]
        table.add_row(
            name,
            f"{metrics['mae_mean']:.2f}",
            f"{metrics['rmse_mean']:.2f}",
            f"{100 * metrics['mape_mean']:.2f}%",
            f"{metrics['bias_mean']:.2f}",
        )

    comparison = report["baseline_comparison"]
    table.add_row(
        f"baseline: {comparison['best_baseline_name']}",
        f"{comparison['best_baseline_mae']:.2f}",
        "",
        "",
        "",
    )

    console.print(table)

    console.print(
        f"[green]Best model:[/green] {comparison['best_model_name']} "
        f"(MAE={comparison['best_model_mae']:.2f})"
    )
    console.print(
        f"Improvement over best baseline: "
        f"{comparison['mae_improvement']:.2f} "
        f"({comparison['mae_improvement_pct']:.2f}%)"
    )

    if comparison["beats_baseline"]:
        console.print("[green]Best model beats the best baseline.[/green]")
    else:
        console.print("[yellow]Best model does not beat the best baseline yet.[/yellow]")

    console.print(f"Training report written: [bold]{train_cfg['output_path']}[/bold]")
    console.print(f"Model saved: [bold]{train_cfg['model_path']}[/bold]")
    console.print(f"Model comparison figure: [bold]{train_cfg['model_comparison_path']}[/bold]")
    console.print(f"Feature importance figure: [bold]{train_cfg['feature_importance_path']}[/bold]")
