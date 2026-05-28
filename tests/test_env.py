import pytest

from power_forecast.utils.env import require_env_var


def test_require_env_var_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_REQUIRED_VAR", "abc123")

    assert require_env_var("TEST_REQUIRED_VAR") == "abc123"


def test_require_env_var_fails_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_REQUIRED_VAR", raising=False)

    with pytest.raises(RuntimeError, match="Required environment variable TEST_REQUIRED_VAR is not set"):
        require_env_var("TEST_REQUIRED_VAR")
