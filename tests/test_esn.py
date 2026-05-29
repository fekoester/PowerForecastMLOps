import numpy as np

from power_forecast.models.esn import EchoStateRegressor


def test_esn_fit_predict_shape():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5))
    y = rng.normal(size=50)

    model = EchoStateRegressor(
        reservoir_size=16,
        spectral_radius=0.9,
        input_scale=0.1,
        leaking_rate=0.5,
        ridge_alpha=1.0,
        random_state=42,
    )

    model.fit(X, y)
    pred = model.predict(X[:7])

    assert pred.shape == (7,)
