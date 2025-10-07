"""vLLM HTTP transport with bounded retries and strict tool-action parsing."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from .config import RuntimeConfig, TransportConfig
from .constants import MAX_HTTP_RESPONSE_BYTES
from .resilience import CircuitBreaker, CircuitOpenError, retry_delay
from .types import JsonRecord

LOGGER = logging.getLogger(__name__)


class RolloutError(RuntimeError):
    """The model service failed independently of a sampled action's validity."""


class InvalidAction(ValueError):
    """A completed model sample failed the action protocol, not transport."""


class VLLMClient:
    """Request one structured model action without hiding endpoint failures."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float | None = None,
        *,
        transport: TransportConfig | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = transport or RuntimeConfig.from_env().transport
        selected_timeout = cfg.timeout_seconds if timeout is None else timeout
        if (
            not endpoint.startswith(("http://", "https://"))
            or not model
            or selected_timeout <= 0
            or not math.isfinite(selected_timeout)
        ):
            raise ValueError("set an HTTP endpoint, model ID and positive timeout")
        self.endpoint, self.model, self.timeout = endpoint.rstrip("/"), model, selected_timeout
        self.transport, self.sleeper = cfg, sleeper
        self.circuit = CircuitBreaker(cfg.circuit_failure_limit, cfg.circuit_reset_seconds)

    def _request(self, request: urllib.request.Request) -> bytes:
        for attempt in range(1, self.transport.max_attempts + 1):
            try:
                self.circuit.before_request()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = bytes(response.read(MAX_HTTP_RESPONSE_BYTES + 1))
                if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                    raise RolloutError(
                        "model response exceeds the byte limit. reduce response size"
                    )
                self.circuit.record_success()
                return raw
            except CircuitOpenError as error:
                raise RolloutError(str(error)) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if (
                    isinstance(error, urllib.error.HTTPError)
                    and error.code not in {408, 429}
                    and error.code < 500
                ):
                    raise RolloutError(
                        f"model endpoint returned HTTP {error.code}. check endpoint and model ID"
                    ) from error
                self.circuit.record_failure()
                if attempt == self.transport.max_attempts:
                    raise RolloutError(
                        f"model request failed after {attempt} attempts: {type(error).__name__}"
                    ) from error
                delay = retry_delay(
                    attempt,
                    self.transport.initial_backoff_seconds,
                    self.transport.max_backoff_seconds,
                )
                LOGGER.warning(
                    "model_request_retry",
                    extra={
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error_type": type(error).__name__,
                    },
                )
                self.sleeper(delay)
        raise RolloutError("model retry budget exhausted")

    def next_call(
        self, messages: list[JsonRecord], tools: list[JsonRecord], seed: int
    ) -> JsonRecord:
        """Fetch a complete tool call. Malformed model actions remain separate failures."""
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": self.transport.temperature,
            "seed": seed,
            "max_tokens": self.transport.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(body, allow_nan=False).encode(),
            headers=headers,
        )
        try:
            envelope = json.loads(self._request(request))
            choice = envelope["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise RolloutError(f"invalid model-service envelope: {type(error).__name__}") from error
        try:
            if choice.get("finish_reason") not in {"tool_calls", "stop"}:
                raise ValueError("incomplete tool action")
            calls = choice["message"]["tool_calls"]
            if len(calls) != 1:
                raise ValueError("one tool action is required")
            call = calls[0]
            return {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
                "id": call["id"],
            }
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise InvalidAction(
                f"model action violates the tool schema: {type(error).__name__}"
            ) from error
