import typer
from rich.console import Console

from power_forecast.pipelines.ingest_pipeline import run_ingestion
from power_forecast.pipelines.validation_pipeline import run_validation
from power_forecast.pipelines.feature_pipeline import run_feature_pipeline

app = typer.Typer(help="PowerForecastMLOps CLI")
console = Console()


@app.command()
def info() -> None:
    """Print project information."""
    console.print("[bold]PowerForecastMLOps[/bold]")
    console.print("Live ML forecasting platform for electricity demand.")


@app.command()
def ingest(config: str = "configs/data.yaml") -> None:
    """Ingest raw demand and weather data."""
    run_ingestion(config)

@app.command()
def validate(config: str = "configs/data.yaml") -> None:
    """Validate raw demand and weather data."""
    run_validation(config)

@app.command()
def features(config: str = "configs/data.yaml") -> None:
    """Build leakage-safe model features."""
    run_feature_pipeline(config)

if __name__ == "__main__":
    app()
