import pandas as pd

from power_forecast.data.validate import validate_dataset


def test_validation_fails_for_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01 00:00:00+00:00"],
            # demand_mwh is intentionally missing
            "region": ["CISO"],
            "demand_type": ["D"],
        }
    ).to_csv(path, index=False)

    config = {
        "required_columns": ["timestamp_utc", "demand_mwh", "region", "demand_type"],
        "timestamp_column": "timestamp_utc",
        "numeric_columns": ["demand_mwh"],
        "min_rows": 1,
        "max_missing_rate": 0.05,
        "value_ranges": {"demand_mwh": {"min": 0, "max": 100000}},
    }

    report = validate_dataset(path, "eia", config)

    assert report["status"] == "fail"
    failed_checks = [c for c in report["checks"] if c["status"] == "fail"]
    assert any(c["name"] == "eia_required_columns" for c in failed_checks)


def test_validation_fails_for_negative_demand(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-01-01 00:00:00+00:00",
                "2026-01-01 01:00:00+00:00",
            ],
            "demand_mwh": [1000, -5],
            "region": ["CISO", "CISO"],
            "demand_type": ["D", "D"],
        }
    ).to_csv(path, index=False)

    config = {
        "required_columns": ["timestamp_utc", "demand_mwh", "region", "demand_type"],
        "timestamp_column": "timestamp_utc",
        "numeric_columns": ["demand_mwh"],
        "min_rows": 1,
        "max_missing_rate": 0.05,
        "value_ranges": {"demand_mwh": {"min": 0, "max": 100000}},
    }

    report = validate_dataset(path, "eia", config)

    assert report["status"] == "fail"
    failed_checks = [c for c in report["checks"] if c["status"] == "fail"]
    assert any(c["name"] == "eia_demand_mwh_range" for c in failed_checks)
