from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class LinearEnsembleRegressor:
    """Linear ensemble over already trained base model predictions."""

    base_models: dict[str, Any]
    feature_columns: list[str]
    alpha: float = 1.0
    positive: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LinearEnsembleRegressor":
        meta_X = self._predict_base_models(X)

        if self.alpha > 0:
            self.meta_model_ = Ridge(alpha=self.alpha, positive=self.positive)
        else:
            self.meta_model_ = LinearRegression(positive=self.positive)

        self.meta_model_.fit(meta_X, np.asarray(y, dtype=float))
        self.base_model_names_ = list(self.base_models.keys())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = self._predict_base_models(X)
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

    def _predict_base_models(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = {}

        for model_name, model in self.base_models.items():
            rows[model_name] = model.predict(X[self.feature_columns])

        return pd.DataFrame(rows, index=X.index)
