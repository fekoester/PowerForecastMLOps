from __future__ import annotations

from typing import Any

import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from power_forecast.models.esn import EchoStateRegressor


def make_model(model_name: str, config: dict[str, Any]):
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            num_leaves=int(config["num_leaves"]),
            max_depth=int(config["max_depth"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            random_state=int(config["random_state"]),
            verbosity=-1,
        )

    if model_name == "mlp":
        hidden_layer_sizes = tuple(int(x) for x in config["hidden_layer_sizes"])

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=str(config["activation"]),
                        alpha=float(config["alpha"]),
                        learning_rate_init=float(config["learning_rate_init"]),
                        max_iter=int(config["max_iter"]),
                        early_stopping=bool(config["early_stopping"]),
                        random_state=int(config["random_state"]),
                    ),
                ),
            ]
        )

    if model_name == "esn":
        return EchoStateRegressor(
            reservoir_size=int(config["reservoir_size"]),
            spectral_radius=float(config["spectral_radius"]),
            input_scale=float(config["input_scale"]),
            leaking_rate=float(config["leaking_rate"]),
            ridge_alpha=float(config["ridge_alpha"]),
            random_state=int(config["random_state"]),
        )

    raise ValueError(f"Unknown model name: {model_name}")


def enabled_models(models_config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: cfg
        for name, cfg in models_config.items()
        if bool(cfg.get("enabled", False))
    }
