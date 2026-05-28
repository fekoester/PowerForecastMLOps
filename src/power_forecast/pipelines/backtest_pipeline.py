from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.models.backtest import run_baseline_backtest
from power_forecast.utils.config import load_yaml

console = Console()


def run_backtest_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)

    backtest_cfg = config["backtest"]
    features_cfg = config["features"]

    console.print("[bold]Starting baseline backtest[/bold]")

    report = run_baseline_backtest(
        features_path=features_cfg["output_path"],
        output_path=backtest_cfg["output_path"],
        figure_path=backtest_cfg["figure_path"],
        timestamp_column=backtest_cfg["timestamp_column"],
        target_column=backtest_cfg["target_column"],
        baselines=dict(backtest_cfg["baselines"]),
        min_train_days=int(backtest_cfg["min_train_days"]),
        validation_window_days=int(backtest_cfg["validation_window_days"]),
        step_days=int(backtest_cfg["step_days"]),
    )

    table = Table(title="Baseline Backtest Summary")
    table.add_column("Baseline")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("Bias", justify="right")

    aggregate = report["aggregate_metrics"]

    baseline_names = [name for name in aggregate.keys() if not name.startswith("_")]
    baseline_names = sorted(baseline_names, key=lambda name: aggregate[name]["mae_mean"])

    for name in baseline_names:
        metrics = aggregate[name]
        table.add_row(
            name,
            f"{metrics['mae_mean']:.2f}",
            f"{metrics['rmse_mean']:.2f}",
            f"{100 * metrics['mape_mean']:.2f}%",
            f"{metrics['bias_mean']:.2f}",
        )

    best = aggregate["_best_by_mae"]

    console.print(table)
    console.print(
        f"[green]Best baseline by MAE:[/green] {best['name']} "
        f"(MAE={best['mae_mean']:.2f}, RMSE={best['rmse_mean']:.2f})"
    )
    console.print(f"Backtest report written: [bold]{backtest_cfg['output_path']}[/bold]")
    console.print(f"Figure written: [bold]{backtest_cfg['figure_path']}[/bold]")
