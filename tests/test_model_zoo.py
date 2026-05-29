from power_forecast.models.model_zoo import enabled_models, make_model


def test_enabled_models_filters_disabled_models():
    config = {
        "lightgbm": {"enabled": True},
        "mlp": {"enabled": False},
    }

    active = enabled_models(config)

    assert "lightgbm" in active
    assert "mlp" not in active


def test_make_esn_model():
    model = make_model(
        "esn",
        {
            "reservoir_size": 16,
            "spectral_radius": 0.9,
            "input_scale": 0.1,
            "leaking_rate": 0.5,
            "ridge_alpha": 1.0,
            "random_state": 42,
        },
    )

    assert model.reservoir_size == 16
