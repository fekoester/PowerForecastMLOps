from __future__ import annotations

from rich.console import Console
from rich.table import Table

from power_forecast.features.build_features import build_features
from power_forecast.utils.config import load_yaml

console = Console()


def run_feature_pipeline(config_path: str) -> None:
    config = load_yaml(config_path)

    feature_cfg = config["features"]
    validation_cfg = config["validation"]

    console.print("[bold]Starting feature pipeline[/bold]")

    df = build_features(
        eia_path=config["eia"]["output_path"],
        weather_path=config["weather"]["output_path"],
        validation_report_path=validation_cfg["output_path"],
        output_path=feature_cfg["output_path"],
        summary_path=feature_cfg["summary_path"],
        timezone_name=config["weather"]["timezone"],
        lags_hours=list(feature_cfg["lags_hours"]),
        rolling_windows_hours=list(feature_cfg["rolling_windows_hours"]),
        same_hour_windows_days=list(feature_cfg["same_hour_windows_days"]),
        origin_rolling_windows_hours=list(feature_cfg["origin_rolling_windows_hours"]),
        base_temperature_c=float(feature_cfg["base_temperature_c"]),
        use_cyclic_calendar_features=bool(
            feature_cfg.get("use_cyclic_calendar_features", False)
        ),
    )

    feature_columns = [c for c in df.columns if c not in ["timestamp_utc", "demand_mwh"]]

    table = Table(title="Feature Engineering Summary")
    table.add_column("Item")
    table.add_column("Value")

    table.add_row("Rows", f"{len(df):,}")
    table.add_row("Features", str(len(feature_columns)))
    table.add_row("Min timestamp", str(df["timestamp_utc"].min()))
    table.add_row("Max timestamp", str(df["timestamp_utc"].max()))
    table.add_row("Output", feature_cfg["output_path"])
    table.add_row("Summary", feature_cfg["summary_path"])

    console.print(table)
