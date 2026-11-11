"""End-to-end pipeline run on synthetic data (no network): validate -> features -> backtest ->
train (incl. ESN and the stacked ensemble) -> predict -> forecast -> monitor."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from power_forecast.data.locations import weather_locations

REPO = Path(__file__).resolve().parents[1]
DAYS = 150


def _synthetic_weather(timestamps_utc: pd.DatetimeIndex, locations, rng) -> pd.DataFrame:
    local = timestamps_utc.tz_convert("America/Los_Angeles").tz_localize(None)
    hour = local.hour.to_numpy()
    doy = local.dayofyear.to_numpy()
    frames = {}
    for i, loc in enumerate(locations):
        temp = 18 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + 6 * np.sin(2 * np.pi * (hour - 9) / 24)
        frames[loc["slug"]] = temp + i + rng.normal(0, 1, len(local))
    total = sum(loc["weight"] for loc in locations)
    df = pd.DataFrame({"timestamp_local": local})
    df["temperature_2m"] = sum(loc["weight"] * frames[loc["slug"]] for loc in locations) / total
    df["relative_humidity_2m"] = 50 + 10 * np.cos(2 * np.pi * hour / 24)
    df["wind_speed_10m"] = 10 + rng.random(len(df))
    df["shortwave_radiation"] = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    for loc in locations:
        df[f"temperature_2m__{loc['slug']}"] = frames[loc["slug"]]
    return df.drop_duplicates("timestamp_local")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config = yaml.safe_load((REPO / "configs" / "data_ci.yaml").read_text())
    config["train"]["min_train_days"] = 60
    config["backtest"]["min_train_days"] = 60
    config["train"]["models"]["mlp"]["max_iter"] = 20
    config["train"]["ensemble"]["enabled"] = True
    config["train"]["ensemble"]["holdout_days"] = 14
    config["validation"]["eia"]["min_rows"] = 100
    config["validation"]["weather"]["min_rows"] = 100
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(0)
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
    ts = pd.date_range(end=end, periods=DAYS * 24, freq="h", tz="UTC")
    locations = weather_locations(config["weather"])
    weather = _synthetic_weather(ts, locations, rng)

    local = ts.tz_convert("America/Los_Angeles")
    demand = (
        25000
        + 5000 * np.sin(2 * np.pi * (local.hour - 10) / 24)
        - 3000 * (local.dayofweek >= 5)
        + 300 * np.clip(weather["temperature_2m"].reindex(range(len(ts))).ffill().to_numpy() - 18, 0, None)
        + rng.normal(0, 400, len(ts))
    )
    eia = pd.DataFrame({"timestamp_utc": ts, "demand_mwh": demand, "region": "CISO", "demand_type": "D"})
    eia = eia.drop(index=[500, 501, 1700])  # missing hours, as in real EIA data
    Path("data/raw").mkdir(parents=True)
    eia.to_csv(config["eia"]["output_path"], index=False)
    weather.to_csv(config["weather"]["output_path"], index=False)

    future_ts = pd.date_range(end + pd.Timedelta(hours=-23), periods=96, freq="h", tz="UTC")
    forecast = _synthetic_weather(future_ts, locations, rng)
    forecast.insert(0, "timestamp_utc", future_ts)

    def fake_forecast(locations, timezone_name, forecast_days=3):
        return forecast.copy()

    monkeypatch.setattr(
        "power_forecast.features.future_features.fetch_open_meteo_forecast_locations", fake_forecast
    )
    return config


def test_full_pipeline_offline(workspace):
    from power_forecast.pipelines.backtest_pipeline import run_backtest_pipeline
    from power_forecast.pipelines.feature_pipeline import run_feature_pipeline
    from power_forecast.pipelines.forecast_pipeline import run_forecast_pipeline
    from power_forecast.pipelines.monitor_pipeline import run_monitor_pipeline
    from power_forecast.pipelines.predict_pipeline import run_predict_pipeline
    from power_forecast.pipelines.train_pipeline import run_train_pipeline
    from power_forecast.pipelines.validation_pipeline import run_validation

    config_path = "configs/data.yaml"
    run_validation(config_path)
    run_feature_pipeline(config_path)
    features = pd.read_csv(workspace["features"]["output_path"])
    for column in ["is_holiday", "is_workday", "cooling_degree_weighted", "temperature_2m__fresno"]:
        assert column in features.columns
    summary = json.loads(Path(workspace["features"]["summary_path"]).read_text())
    assert summary["n_missing_hours_in_source"] == 3

    run_backtest_pipeline(config_path)
    run_train_pipeline(config_path, selected_models=["lightgbm", "esn"], selected_windows=[30])
    bundle = joblib.load(workspace["train"]["model_path"])
    assert "ensemble_linear" in bundle["models"]
    assert bundle["prediction_intervals"][bundle["best_model_name"]]["by_hour"]

    run_predict_pipeline(config_path)
    prediction = json.loads(Path(workspace["predict"]["summary_path"]).read_text())
    # The latest monitoring window must be out-of-sample for the final models.
    assert pd.Timestamp(bundle["final_training_end"]) < pd.Timestamp(prediction["min_timestamp"])
    assert "coverage_80" in prediction["metrics"]
    assert prediction["metrics"]["mape"] < 0.1

    run_forecast_pipeline(config_path)
    forecast = pd.read_csv(workspace["forecast"]["output_path"])
    assert len(forecast) == 24
    assert (forecast["forecast_mwh_best_lower_95"] <= forecast["forecast_mwh_best"]).all()
    assert (forecast["forecast_mwh_best"] <= forecast["forecast_mwh_best_upper_95"]).all()

    run_monitor_pipeline(config_path)
    html = Path(workspace["monitor"]["html_report_path"]).read_text()
    assert "prediction intervals" in html
    shutil.rmtree("models", ignore_errors=True)
