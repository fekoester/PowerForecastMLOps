import pandas as pd

from power_forecast.features.build_features import _add_lag_features, _add_rolling_features


def test_lag_feature_uses_past_values():
    df = pd.DataFrame({"demand_mwh": [10, 20, 30, 40]})

    out = _add_lag_features(df, lags_hours=[1, 2])

    assert pd.isna(out.loc[0, "demand_lag_1h"])
    assert out.loc[1, "demand_lag_1h"] == 10
    assert out.loc[2, "demand_lag_1h"] == 20

    assert pd.isna(out.loc[0, "demand_lag_2h"])
    assert pd.isna(out.loc[1, "demand_lag_2h"])
    assert out.loc[2, "demand_lag_2h"] == 10


def test_rolling_feature_is_shifted_to_prevent_leakage():
    df = pd.DataFrame({"demand_mwh": [10, 20, 30, 40]})

    out = _add_rolling_features(df, windows_hours=[2])

    # At index 2, the leakage-safe rolling mean should use values [10, 20],
    # not [20, 30]. So it should equal 15, not 25.
    assert out.loc[2, "demand_roll_mean_2h"] == 15

    # At index 3, it should use [20, 30].
    assert out.loc[3, "demand_roll_mean_2h"] == 25
