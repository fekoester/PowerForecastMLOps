from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class LinearEnsembleRegressor:
    """Linear stacking over already trained base models.

    The meta-model must be fit on predictions for data the base models did NOT see
    (``fit_from_predictions`` with a held-out window). Fitting it on in-sample predictions
    (``fit``) rewards whichever base model overfits its training data the most.
    """

    base_models: dict[str, Any]
    feature_columns: list[str]
    alpha: float = 1.0
    positive: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> LinearEnsembleRegressor:
        """Fits the meta-model on base-model predictions for ``X`` (use held-out rows)."""
        return self.fit_from_predictions(self._predict_base_models(X), y)

    def fit_from_predictions(
        self, meta_X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> LinearEnsembleRegressor:
        """Fits the meta-model on precomputed (out-of-sample) base-model predictions."""
        meta_X = meta_X[list(self.base_models.keys())]
        if self.alpha > 0:
            self.meta_model_ = Ridge(alpha=self.alpha, positive=self.positive)
        else:
            self.meta_model_ = LinearRegression(positive=self.positive)

        self.meta_model_.fit(meta_X, np.asarray(y, dtype=float))
        self.base_model_names_ = list(self.base_models.keys())
        return self

    def predict(self, X: pd.DataFrame, X_history: pd.DataFrame | None = None) -> np.ndarray:
        meta_X = self._predict_base_models(X, X_history)
        return self.meta_model_.predict(meta_X)

    def coefficients(self) -> dict[str, float]:
        coefs = getattr(self.meta_model_, "coef_", None)
        if coefs is None:
            return {}
        return {
            name: float(coef)
            for name, coef in zip(self.base_model_names_, coefs)
        }

    def intercept(self) -> float:
        return float(getattr(self.meta_model_, "intercept_", 0.0))

    def _predict_base_models(
        self, X: pd.DataFrame, X_history: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        from power_forecast.models.inference import predict_with_history

        history = None if X_history is None else X_history[self.feature_columns]
        rows = {
            model_name: predict_with_history(model, X[self.feature_columns], history)
            for model_name, model in self.base_models.items()
        }
        return pd.DataFrame(rows, index=X.index)
