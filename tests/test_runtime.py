"""Validate runtime configuration and HTTP failure behavior without a service."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType

import pytest

from geotrace.config import RecoveryConfig, RuntimeConfig, TransportConfig
from geotrace.logging_config import JsonFormatter, RunContext, configure_logging
from geotrace.model_client import RolloutError, VLLMClient
from geotrace.resilience import CircuitBreaker, CircuitOpenError, retry_delay
from geotrace.types import JsonRecord


def test_toml_settings_and_environment_override(tmp_path: Path) -> None:
    config = tmp_path / "runtime.toml"
    config.write_text(
        '[runtime]\nrun_id="from-file"\nreward_mode="outcome"\n'
        "[recovery]\nmax_recoveries=2\ninitial_backoff_seconds=0.1\n"
        "[transport]\nmax_attempts=4\ntemperature=0.2\nmax_tokens=123\n"
    )
    cfg = RuntimeConfig.from_env(
        {
            "GEOTRACE_CONFIG": str(config),
            "GEOTRACE_RUN_ID": "override",
            "GEOTRACE_HTTP_TIMEOUT": "7",
        }
    )
    assert cfg.run_id == "override"
    assert cfg.reward_mode == "outcome"
    assert cfg.recovery.max_recoveries == 2
    assert cfg.transport.max_attempts == 4
    assert cfg.transport.temperature == 0.2
    assert cfg.transport.max_tokens == 123
    assert cfg.transport.timeout_seconds == 7


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_timeout_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="invalid transport"):
        RuntimeConfig.from_env({"GEOTRACE_HTTP_TIMEOUT": value})


def test_invalid_config_tables_and_modes(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text('runtime="wrong type"\n')
    with pytest.raises(ValueError, match="must be a table"):
        RuntimeConfig.from_env({"GEOTRACE_CONFIG": str(path)})
    with pytest.raises(ValueError, match="reward_mode"):
        RuntimeConfig.from_env({"GEOTRACE_REWARD_MODE": "unknown"})
    with pytest.raises(ValueError, match="invalid recovery"):
        RecoveryConfig(max_recoveries=-1)
    with pytest.raises(ValueError, match="invalid transport"):
        TransportConfig(max_tokens=0)


def test_json_logs_keep_context_and_setup_is_idempotent() -> None:
    root = logging.getLogger()
    handlers, level = root.handlers.copy(), root.level
    try:
        configure_logging("INFO", "default-run")
        configure_logging("DEBUG", "changed-run")
        selected = [handler for handler in root.handlers if handler.name == "geotrace-json"]
        assert len(selected) == 1
        record = logging.makeLogRecord(
            {
                "name": "test",
                "levelname": "INFO",
                "msg": "optimizer_step",
                "step": 3,
                "run_id": "explicit-run",
                "metrics": {"loss": 0.2},
            }
        )
        RunContext("fallback").filter(record)
        formatted = json.loads(JsonFormatter().format(record))
        assert formatted["run_id"] == "explicit-run"
        assert formatted["step"] == 3
        assert formatted["metrics"] == {"loss": 0.2}
        assert formatted["timestamp"].endswith("+00:00")
        assert formatted["level"] == "INFO"
        missing = logging.makeLogRecord({"msg": "no identity"})
        selected[0].filter(missing)
        assert missing.run_id == "changed-run"
    finally:
        root.handlers = handlers
        root.setLevel(level)


def test_circuit_cooldown_and_exponential_delay() -> None:
    assert [retry_delay(attempt, 0.25, 1.0) for attempt in range(1, 6)] == [0.25, 0.5, 1, 1, 1]
    with pytest.raises(ValueError, match="attempt"):
        retry_delay(0, 1, 2)
    now = [10.0]
    breaker = CircuitBreaker(2, 5, clock=lambda: now[0])
    breaker.record_failure()
    breaker.before_request()
    breaker.record_failure()
    with pytest.raises(CircuitOpenError):
        breaker.before_request()
    now[0] += 5
    breaker.before_request()
    breaker.record_success()
    assert breaker.failures == 0
    assert breaker.opened_at is None


class Response:
    """Small context-managed urllib response for a deterministic HTTP fixture."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> Response:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


def test_http_retry_preserves_request_and_resets_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[JsonRecord] = []
    delays: list[float] = []
    envelope = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "gravity_forward", "arguments": '{"model": {}}'},
                        }
                    ]
                },
            }
        ]
    }

    def request(req: urllib.request.Request, timeout: float) -> Response:
        assert timeout == 2
        requests.append(json.loads(req.data or b"{}"))
        if len(requests) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)
        return Response(json.dumps(envelope).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    client = VLLMClient(
        "http://localhost:1234/v1",
        "local-model",
        transport=TransportConfig(
            timeout_seconds=2,
            max_attempts=2,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=0.2,
            temperature=0.3,
            max_tokens=33,
        ),
        sleeper=delays.append,
    )
    action = client.next_call([], [], seed=7)
    assert action["name"] == "gravity_forward"
    assert requests[0] == requests[1]
    assert requests[0]["max_tokens"] == 33
    assert requests[0]["temperature"] == 0.3
    assert delays == [0.1]
    assert client.circuit.failures == 0


def test_http_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def bad_request(req: urllib.request.Request, timeout: float) -> Response:
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 400, "invalid model", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    client = VLLMClient("http://localhost:1234/v1", "bad-model")
    with pytest.raises(RolloutError, match="HTTP 400"):
        client.next_call([], [], seed=7)
    assert len(calls) == 1
