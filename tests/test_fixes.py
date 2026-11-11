import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from power_forecast.data.locations import combine_locations, location_weights, weather_locations
from power_forecast.features.build_features import (
    _add_calendar_features,
    _add_lag_features,
    _add_weather_features,
    _to_hourly_grid,
)
from power_forecast.models.ensemble import LinearEnsembleRegressor
from power_forecast.models.esn import EchoStateRegressor
from power_forecast.models.inference import (
    apply_intervals,
    interval_coverage,
    interval_table,
    needs_history,
    predict_with_history,
)
from power_forecast.models.model_zoo import make_model


def test_lags_are_exact_hours_when_source_has_gaps():
    ts = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "demand_mwh": np.arange(60.0)}).drop(index=[30])
    grid = _add_lag_features(_to_hourly_grid(df), lags_hours=[24])
    row = grid[grid["timestamp_utc"] == ts[40]].iloc[0]
    assert row["demand_lag_24h"] == 16.0  # value at ts[16], not ts[15]
    assert grid["demand_mwh"].isna().sum() == 1


def test_holiday_features():
    ts = pd.to_datetime(["2026-07-03 19:00", "2026-07-04 19:00", "2026-07-06 19:00"], utc=True)
    out = _add_calendar_features(pd.DataFrame({"timestamp_utc": ts}), "UTC", use_holiday_features=True)
    # July 4th 2026 is a Saturday, so the observed federal holiday is Friday July 3rd.
    assert out["is_holiday"].tolist() == [1, 0, 0]
    assert out["is_day_after_holiday"].tolist() == [0, 1, 0]
    assert out["is_workday"].tolist() == [0, 0, 1]


def test_weighted_degree_hours_use_per_location_temperatures():
    df = pd.DataFrame({"temperature_2m": [20.0], "temperature_2m__a": [10.0], "temperature_2m__b": [30.0]})
    out = _add_weather_features(df, 18.0, location_weights={"a": 1.0, "b": 1.0})
    # mean temperature 20 -> 2 cooling degrees, but per location: (0 + 12) / 2 = 6.
    assert out["cooling_degree"].iloc[0] == 2.0
    assert out["cooling_degree_weighted"].iloc[0] == 6.0
    assert out["heating_degree_weighted"].iloc[0] == 4.0


def test_locations_config_and_weighting():
    single = {"location_name": "Los Angeles", "latitude": 1, "longitude": 2}
    assert len(weather_locations(single)) == 1 and location_weights(single) == {}
    cfg = {"locations": [{"name": "A", "latitude": 0, "longitude": 0, "weight": 3},
                         {"name": "B B", "latitude": 0, "longitude": 0, "weight": 1}]}
    locs = weather_locations(cfg)
    frame = lambda t: pd.DataFrame({"k": [0, 1], "temperature_2m": [t, t], "relative_humidity_2m": [0, 0],  # noqa: E731
                                     "wind_speed_10m": [0, 0], "shortwave_radiation": [0, 0]})
    out = combine_locations({"a": frame(10.0), "b_b": frame(30.0)}, locs, key="k")
    assert out["temperature_2m"].tolist() == [15.0, 15.0]
    assert out["temperature_2m__b_b"].tolist() == [30.0, 30.0]


def test_esn_warmup_matches_predicting_the_full_sequence():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)))
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=400)
    model = make_model("esn", {"reservoir_size": 32, "spectral_radius": 0.9, "input_scale": 0.5,
                               "leaking_rate": 0.3, "ridge_alpha": 1.0, "random_state": 0})
    model.fit(X.iloc[:300], y.iloc[:300])
    assert needs_history(model) and not needs_history(LinearRegression())
    full = model.predict(X.iloc[:324])[-24:]
    warm = predict_with_history(model, X.iloc[300:324], X.iloc[:300], warmup_rows=300)
    cold = model.predict(X.iloc[300:324])
    np.testing.assert_allclose(warm, full)
    assert np.abs(cold - full).max() > np.abs(warm - full).max()


def test_stacking_on_holdout_does_not_trust_the_memorizing_model():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=600)})
    y = 3 * X["x"] + rng.normal(scale=1.0, size=600)
    train, holdout = slice(0, 400), slice(400, 600)
    bases = {
        "memorizer": KNeighborsRegressor(n_neighbors=1).fit(X.iloc[train], y.iloc[train]),
        "linear": LinearRegression().fit(X.iloc[train], y.iloc[train]),
    }
    in_sample = LinearEnsembleRegressor(bases, ["x"], alpha=0.0).fit(X.iloc[train], y.iloc[train])
    held_out = LinearEnsembleRegressor(bases, ["x"], alpha=0.0).fit(X.iloc[holdout], y.iloc[holdout])
    assert in_sample.coefficients()["memorizer"] > 0.9  # in-sample stacking picks the memorizer
    assert held_out.coefficients()["linear"] > held_out.coefficients()["memorizer"]


def test_conformal_intervals_cover_at_nominal_rate():
    rng = np.random.default_rng(2)
    hours = np.tile(np.arange(24), 400)
    residuals = rng.normal(scale=1 + hours / 10, size=hours.size)  # larger errors later in the day
    table = interval_table(residuals, hours, levels=[0.8, 0.95])
    assert table["by_hour"]["23"]["0.8"][1] > table["by_hour"]["0"]["0.8"][1]
    test_hours = np.tile(np.arange(24), 200)
    actuals = rng.normal(scale=1 + test_hours / 10)
    bounds = apply_intervals(np.zeros(test_hours.size), test_hours, table)
    coverage = interval_coverage(actuals, bounds)
    assert abs(coverage["coverage_80"] - 0.8) < 0.03
    assert abs(coverage["coverage_95"] - 0.95) < 0.02


def test_esn_is_detected_inside_ensembles():
    esn = EchoStateRegressor(reservoir_size=4)
    assert needs_history(LinearEnsembleRegressor({"esn": esn}, ["x"]))
