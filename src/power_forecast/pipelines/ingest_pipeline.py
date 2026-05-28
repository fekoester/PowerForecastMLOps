from __future__ import annotations

from dotenv import load_dotenv
from rich.console import Console

from power_forecast.data.eia import fetch_eia_hourly_demand
from power_forecast.data.manifest import write_manifest
from power_forecast.data.weather import fetch_open_meteo_weather
from power_forecast.utils.config import load_yaml

console = Console()


def run_ingestion(config_path: str) -> None:
    load_dotenv()
    config = load_yaml(config_path)

    console.print("[bold]Starting ingestion pipeline[/bold]")

    eia_cfg = config["eia"]
    weather_cfg = config["weather"]

    console.print(
        f"Fetching EIA demand: respondent={eia_cfg['respondent']} "
        f"type={eia_cfg['type']} lookback_days={eia_cfg['lookback_days']}"
    )
    eia_df = fetch_eia_hourly_demand(
        respondent=eia_cfg["respondent"],
        demand_type=eia_cfg["type"],
        lookback_days=int(eia_cfg["lookback_days"]),
        output_path=eia_cfg["output_path"],
    )
    console.print(f"[green]Saved EIA demand rows:[/green] {len(eia_df):,}")

    console.print(
        f"Fetching weather: location={weather_cfg['location_name']} "
        f"lookback_days={weather_cfg['lookback_days']}"
    )
    weather_df = fetch_open_meteo_weather(
        latitude=float(weather_cfg["latitude"]),
        longitude=float(weather_cfg["longitude"]),
        timezone_name=weather_cfg["timezone"],
        lookback_days=int(weather_cfg["lookback_days"]),
        output_path=weather_cfg["output_path"],
    )
    console.print(f"[green]Saved weather rows:[/green] {len(weather_df):,}")

    manifest = write_manifest(
        output_path=config["manifest"]["output_path"],
        eia_df=eia_df,
        weather_df=weather_df,
        config=config,
    )

    console.print(f"[green]Manifest written:[/green] {config['manifest']['output_path']}")
    console.print(
        f"EIA time range: {manifest['sources']['eia']['summary'].get('min_timestamp')} "
        f"→ {manifest['sources']['eia']['summary'].get('max_timestamp')}"
    )
