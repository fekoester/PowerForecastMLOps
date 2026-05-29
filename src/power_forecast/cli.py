import typer
from rich.console import Console

from power_forecast.pipelines.ingest_pipeline import run_ingestion
from power_forecast.pipelines.validation_pipeline import run_validation
from power_forecast.pipelines.feature_pipeline import run_feature_pipeline
from power_forecast.pipelines.backtest_pipeline import run_backtest_pipeline
from power_forecast.pipelines.train_pipeline import run_train_pipeline
from power_forecast.pipelines.predict_pipeline import run_predict_pipeline
from power_forecast.pipelines.monitor_pipeline import run_monitor_pipeline
from power_forecast.pipelines.forecast_pipeline import run_forecast_pipeline

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

@app.command()
def backtest(config: str = "configs/data.yaml") -> None:
    """Run walk-forward baseline backtests."""
    run_backtest_pipeline(config)

@app.command()
def train(config: str = "configs/data.yaml") -> None:
    """Train LightGBM model with walk-forward validation."""
    run_train_pipeline(config)

@app.command()
def predict(config: str = "configs/data.yaml") -> None:
    """Run batch inference using the saved model."""
    run_predict_pipeline(config)

@app.command()
def monitor(config: str = "configs/data.yaml") -> None:
    """Build monitoring reports for latest predictions."""
    run_monitor_pipeline(config)
    
@app.command()
def forecast(config: str = "configs/data.yaml") -> None:
    """Forecast the next 24 hours using forecast-safe features."""
    run_forecast_pipeline(config)

if __name__ == "__main__":
    app()
