from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.models.forecast import run_future_24h_forecast
from power_forecast.utils.config import load_yaml

console = Console()


def run_forecast_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)

    forecast_cfg = config["forecast"]
    train_cfg = config["train"]
    feature_cfg = config["features"]

    console.print("[bold]Starting future 24h forecast pipeline[/bold]")

    summary = run_future_24h_forecast(
        model_path=forecast_cfg["model_path"],
        eia_path=config["eia"]["output_path"],
        latitude=float(config["weather"]["latitude"]),
        longitude=float(config["weather"]["longitude"]),
        timezone_name=config["weather"]["timezone"],
        base_temperature_c=float(feature_cfg["base_temperature_c"]),
        allowed_lag_hours=list(train_cfg.get("allowed_lag_hours", [24, 48, 168])),
        same_hour_windows_days=list(feature_cfg["same_hour_windows_days"]),
        origin_rolling_windows_hours=list(feature_cfg["origin_rolling_windows_hours"]),
        features_output_path=forecast_cfg["features_output_path"],
        weather_forecast_output_path=forecast_cfg["weather_forecast_output_path"],
        output_path=forecast_cfg["output_path"],
        summary_path=forecast_cfg["summary_path"],
        figure_path=forecast_cfg["figure_path"],
        timestamp_column=forecast_cfg["timestamp_column"],
        prediction_prefix=forecast_cfg["prediction_prefix"],
        use_cyclic_calendar_features=bool(
            feature_cfg.get("use_cyclic_calendar_features", False)
        ),
    )

    table = Table(title="Future 24h Forecast Summary")
    table.add_column("Item")
    table.add_column("Value", justify="right")

    table.add_row("Best model", summary["best_model_name"])
    table.add_row("Available models", ", ".join(summary["available_models"]))
    table.add_row("Horizon hours", str(summary["horizon_hours"]))
    table.add_row("Min timestamp", summary["min_timestamp"])
    table.add_row("Max timestamp", summary["max_timestamp"])
    table.add_row("Forecast CSV", forecast_cfg["output_path"])
    table.add_row("Figure", forecast_cfg["figure_path"])

    console.print(table)
