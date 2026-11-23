"""get_with_retry: transient failures are retried, permanent ones are not (no network needed)."""
from __future__ import annotations

import pytest
import requests

from power_forecast.data import locations


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(locations.time, "sleep", sleeps.append)
    return sleeps


def scripted_get(monkeypatch, outcomes):
    """Patch requests.get to return or raise the given outcomes in order."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        outcome = outcomes[len(calls)]
        calls.append(url)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(locations.requests, "get", fake_get)
    return calls


def test_retries_timeouts_and_connection_errors(monkeypatch, no_sleep):
    calls = scripted_get(
        monkeypatch,
        [
            requests.exceptions.ReadTimeout("handshake timed out"),
            requests.exceptions.ConnectionError("connection aborted"),
            FakeResponse(200, {"hourly": {}}),
        ],
    )
    assert locations.get_with_retry("https://example.test", params={}, timeout=1) == {"hourly": {}}
    assert len(calls) == 3
    assert no_sleep == [5.0, 10.0]


def test_retries_server_errors_then_succeeds(monkeypatch, no_sleep):
    calls = scripted_get(monkeypatch, [FakeResponse(503), FakeResponse(429), FakeResponse(200, {"ok": 1})])
    assert locations.get_with_retry("https://example.test", params={}, timeout=1) == {"ok": 1}
    assert len(calls) == 3


def test_gives_up_after_last_attempt(monkeypatch, no_sleep):
    calls = scripted_get(monkeypatch, [requests.exceptions.ReadTimeout("slow")] * 3)
    with pytest.raises(requests.exceptions.ReadTimeout):
        locations.get_with_retry("https://example.test", params={}, timeout=1, attempts=3)
    assert len(calls) == 3
    assert no_sleep == [5.0, 10.0]


def test_client_errors_are_not_retried(monkeypatch, no_sleep):
    calls = scripted_get(monkeypatch, [FakeResponse(401), FakeResponse(200)])
    with pytest.raises(requests.HTTPError):
        locations.get_with_retry("https://example.test", params={}, timeout=1)
    assert len(calls) == 1
    assert no_sleep == []
