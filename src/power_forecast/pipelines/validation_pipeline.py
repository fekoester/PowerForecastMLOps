from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from power_forecast.data.validate import validate_dataset, write_validation_report
from power_forecast.utils.config import load_yaml

console = Console()


def run_validation(config_path: str) -> None:
    config = load_yaml(config_path)

    validation_cfg = config["validation"]

    console.print("[bold]Starting data validation[/bold]")

    eia_report = validate_dataset(
        path=config["eia"]["output_path"],
        dataset_name="eia",
        config=validation_cfg["eia"],
    )

    weather_report = validate_dataset(
        path=config["weather"]["output_path"],
        dataset_name="weather",
        config=validation_cfg["weather"],
    )

    overall_status = "pass"
    if eia_report["status"] == "fail" or weather_report["status"] == "fail":
        overall_status = "fail"
    elif eia_report["status"] == "warn" or weather_report["status"] == "warn":
        overall_status = "warn"

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "datasets": {
            "eia": eia_report,
            "weather": weather_report,
        },
    }

    output_path = validation_cfg["output_path"]
    write_validation_report(output_path, report)

    table = Table(title="Data Validation Summary")
    table.add_column("Dataset")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    table.add_column("Failed checks", justify="right")
    table.add_column("Warnings", justify="right")

    for name, dataset_report in report["datasets"].items():
        checks = dataset_report["checks"]
        failed = sum(1 for c in checks if c["status"] == "fail")
        warnings = sum(1 for c in checks if c["status"] == "warn")

        status = dataset_report["status"]
        style = "green" if status == "pass" else "yellow" if status == "warn" else "red"

        table.add_row(
            name,
            f"[{style}]{status}[/{style}]",
            str(dataset_report.get("n_rows", 0)),
            str(failed),
            str(warnings),
        )

    console.print(table)
    console.print(f"Validation report written: [bold]{output_path}[/bold]")

    if overall_status == "fail":
        raise RuntimeError("Data validation failed. See validation report for details.")
