from power_forecast.monitoring.report import _health_status


def test_health_status_healthy_when_latest_error_is_good():
    status, warnings = _health_status(
        latest_mae=100.0,
        training_mae=200.0,
        best_baseline_mae=500.0,
        thresholds={
            "watch_mae_ratio_vs_training": 1.25,
            "degraded_mae_ratio_vs_training": 1.75,
            "watch_mae_ratio_vs_baseline": 0.90,
            "degraded_mae_ratio_vs_baseline": 1.00,
        },
    )

    assert status == "healthy"
    assert warnings == []


def test_health_status_degraded_when_worse_than_baseline():
    status, warnings = _health_status(
        latest_mae=600.0,
        training_mae=200.0,
        best_baseline_mae=500.0,
        thresholds={
            "watch_mae_ratio_vs_training": 1.25,
            "degraded_mae_ratio_vs_training": 1.75,
            "watch_mae_ratio_vs_baseline": 0.90,
            "degraded_mae_ratio_vs_baseline": 1.00,
        },
    )

    assert status == "degraded"
    assert len(warnings) > 0


def test_health_status_unknown_without_actuals():
    status, warnings = _health_status(
        latest_mae=None,
        training_mae=200.0,
        best_baseline_mae=500.0,
        thresholds={
            "watch_mae_ratio_vs_training": 1.25,
            "degraded_mae_ratio_vs_training": 1.75,
            "watch_mae_ratio_vs_baseline": 0.90,
            "degraded_mae_ratio_vs_baseline": 1.00,
        },
    )

    assert status == "unknown"
    assert len(warnings) == 1
