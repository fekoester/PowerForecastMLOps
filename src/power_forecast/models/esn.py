from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class EchoStateRegressor(BaseEstimator, RegressorMixin):
    """Small Echo State Network / reservoir computer for regression.

    This is intentionally lightweight:
    - fixed random recurrent reservoir
    - only Ridge readout is trained
    - CPU-friendly for GitHub Actions
    """

    def __init__(
        self,
        reservoir_size: int = 128,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1,
        leaking_rate: float = 0.5,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        rng = np.random.default_rng(self.random_state)
        n_features = X_scaled.shape[1]

        self.W_in_ = rng.uniform(
            low=-self.input_scale,
            high=self.input_scale,
            size=(self.reservoir_size, n_features),
        )

        W = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(self.reservoir_size, self.reservoir_size),
        )

        # Scale recurrent matrix to desired spectral radius.
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius == 0:
            current_radius = 1.0
        self.W_res_ = W * (self.spectral_radius / current_radius)

        states = self._compute_states(X_scaled)

        # Concatenate original scaled inputs and reservoir state.
        readout_X = np.hstack([X_scaled, states])

        self.readout_ = Ridge(alpha=self.ridge_alpha)
        self.readout_.fit(readout_X, y_arr)

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X_arr)

        states = self._compute_states(X_scaled)
        readout_X = np.hstack([X_scaled, states])

        return self.readout_.predict(readout_X)

    def _compute_states(self, X_scaled: np.ndarray) -> np.ndarray:
        states = np.zeros((X_scaled.shape[0], self.reservoir_size), dtype=float)
        state = np.zeros(self.reservoir_size, dtype=float)

        for t, x_t in enumerate(X_scaled):
            pre_activation = self.W_in_ @ x_t + self.W_res_ @ state
            candidate = np.tanh(pre_activation)

            state = (1.0 - self.leaking_rate) * state + self.leaking_rate * candidate
            states[t] = state

        return states
