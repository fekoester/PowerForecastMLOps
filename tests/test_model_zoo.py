from sklearn.compose import TransformedTargetRegressor

from power_forecast.models.model_zoo import enabled_models, make_model


def test_enabled_models_filters_disabled_models():
    config = {
        "lightgbm": {"enabled": True},
        "mlp": {"enabled": False},
    }

    active = enabled_models(config)

    assert "lightgbm" in active
    assert "mlp" not in active


def test_make_esn_model_uses_target_scaling():
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

    assert isinstance(model, TransformedTargetRegressor)


def test_make_mlp_model_uses_target_scaling():
    model = make_model(
        "mlp",
        {
            "hidden_layer_sizes": [16],
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 10,
            "early_stopping": True,
            "random_state": 42,
        },
    )

    assert isinstance(model, TransformedTargetRegressor)


def test_make_lightgbm_model_uses_target_scaling():
    model = make_model(
        "lightgbm",
        {
            "n_estimators": 10,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        },
    )

    assert isinstance(model, TransformedTargetRegressor)
