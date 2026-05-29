from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.models.predict import run_batch_prediction
from power_forecast.utils.config import load_yaml

console = Console()


def run_predict_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)
    predict_cfg = config["predict"]

    console.print("[bold]Starting batch prediction pipeline[/bold]")

    summary = run_batch_prediction(
        model_path=predict_cfg["model_path"],
        input_path=predict_cfg["input_path"],
        output_path=predict_cfg["output_path"],
        summary_path=predict_cfg["summary_path"],
        figure_path=predict_cfg["figure_path"],
        timestamp_column=predict_cfg["timestamp_column"],
        target_column=predict_cfg["target_column"],
        prediction_column=predict_cfg["prediction_column"],
        n_latest_rows=int(predict_cfg["n_latest_rows"]),
    )

    table = Table(title="Batch Prediction Summary")
    table.add_column("Item")
    table.add_column("Value", justify="right")

    table.add_row("Rows predicted", str(summary["n_prediction_rows"]))
    table.add_row("Min timestamp", summary["min_timestamp"])
    table.add_row("Max timestamp", summary["max_timestamp"])
    table.add_row("Has actuals", str(summary["has_actuals"]))

    table.add_row("Best model", str(summary["best_model_name"]))
    table.add_row("Available models", ", ".join(summary["available_models"]))

    metrics = summary.get("metrics")
    if metrics is not None:
        table.add_row("Best MAE", f"{metrics['mae']:.2f}")
        table.add_row("Best RMSE", f"{metrics['rmse']:.2f}")
        table.add_row("Best MAPE", f"{100 * metrics['mape']:.2f}%")
        table.add_row("Best Bias", f"{metrics['bias']:.2f}")

    table.add_row("Prediction CSV", predict_cfg["output_path"])
    table.add_row("Summary JSON", predict_cfg["summary_path"])
    table.add_row("Figure", predict_cfg["figure_path"])

    console.print(table)
