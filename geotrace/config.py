"""Environment and TOML configuration for local and distributed workers."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_ENDPOINT, HTTP_MAX_TOKENS


@dataclass(frozen=True)
class RecoveryConfig:
    """Bounds on infrastructure recovery. Application errors are never retried."""

    max_recoveries: int = 3
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0
    circuit_failure_limit: int = 5

    def __post_init__(self) -> None:
        if (
            self.max_recoveries < 0
            or self.circuit_failure_limit < 1
            or not math.isfinite(self.initial_backoff_seconds)
            or not math.isfinite(self.max_backoff_seconds)
            or not 0 <= self.initial_backoff_seconds <= self.max_backoff_seconds
        ):
            raise ValueError(
                "invalid recovery limits. set nonnegative retries and bounded finite delays"
            )


@dataclass(frozen=True)
class TransportConfig:
    """HTTP retry limits and circuit timing for the model endpoint."""

    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    circuit_failure_limit: int = 5
    circuit_reset_seconds: float = 60.0
    temperature: float = 0.8
    max_tokens: int = HTTP_MAX_TOKENS

    def __post_init__(self) -> None:
        values = [
            self.timeout_seconds,
            self.initial_backoff_seconds,
            self.max_backoff_seconds,
            self.circuit_reset_seconds,
            self.temperature,
        ]
        if (
            not all(math.isfinite(value) for value in values)
            or self.max_attempts < 1
            or self.timeout_seconds <= 0
            or not 0 <= self.initial_backoff_seconds <= self.max_backoff_seconds
            or self.circuit_failure_limit < 1
            or self.circuit_reset_seconds <= 0
            or self.temperature < 0
            or self.max_tokens < 1
        ):
            raise ValueError(
                "invalid transport limits. use positive timeouts/attempts and finite bounded delays"
            )


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a table")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Read runtime paths and backend settings without importing GPU packages."""

    log_level: str = "INFO"
    run_id: str = "local"
    config_path: Path | None = None
    data_dir: Path = Path("outputs/verl_data")
    checkpoint_dir: Path = Path("outputs/grpo_checkpoints")
    prm_device: str = "cpu"
    reward_mode: str = "process"
    rollout_mode: str = "tool_agent"
    ray_address: str | None = None
    recovery: RecoveryConfig = RecoveryConfig()
    transport: TransportConfig = TransportConfig()

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> RuntimeConfig:
        """Load an optional GEOTRACE_CONFIG TOML file, then environment overrides."""
        env = os.environ if environment is None else environment
        config_path = Path(env["GEOTRACE_CONFIG"]) if env.get("GEOTRACE_CONFIG") else None
        document: dict[str, Any] = {}
        if config_path:
            with config_path.open("rb") as handle:
                document = tomllib.load(handle)
        runtime = _section(document, "runtime")
        recovery = _section(document, "recovery")
        transport = _section(document, "transport")
        rec = RecoveryConfig(
            max_recoveries=int(
                env.get("GEOTRACE_RECOVERY_RETRIES", recovery.get("max_recoveries", 3))
            ),
            initial_backoff_seconds=float(
                env.get("GEOTRACE_RECOVERY_BACKOFF", recovery.get("initial_backoff_seconds", 2.0))
            ),
            max_backoff_seconds=float(recovery.get("max_backoff_seconds", 30.0)),
            circuit_failure_limit=int(recovery.get("circuit_failure_limit", 5)),
        )
        http = TransportConfig(
            endpoint=env.get("VLLM_ENDPOINT", transport.get("endpoint", DEFAULT_ENDPOINT)),
            timeout_seconds=float(
                env.get("GEOTRACE_HTTP_TIMEOUT", transport.get("timeout_seconds", 120.0))
            ),
            max_attempts=int(transport.get("max_attempts", 3)),
            initial_backoff_seconds=float(transport.get("initial_backoff_seconds", 1.0)),
            max_backoff_seconds=float(transport.get("max_backoff_seconds", 30.0)),
            circuit_failure_limit=int(transport.get("circuit_failure_limit", 5)),
            circuit_reset_seconds=float(transport.get("circuit_reset_seconds", 60.0)),
            temperature=float(transport.get("temperature", 0.8)),
            max_tokens=int(transport.get("max_tokens", HTTP_MAX_TOKENS)),
        )
        if (
            rec.max_recoveries < 0
            or rec.initial_backoff_seconds < 0
            or rec.max_backoff_seconds < rec.initial_backoff_seconds
            or rec.circuit_failure_limit < 1
            or http.max_attempts < 1
            or http.timeout_seconds <= 0
            or http.initial_backoff_seconds < 0
            or http.max_backoff_seconds < http.initial_backoff_seconds
            or http.circuit_failure_limit < 1
            or http.circuit_reset_seconds <= 0
        ):
            raise ValueError("invalid recovery/HTTP limits. check GEOTRACE_CONFIG")
        result = cls(
            log_level=env.get("GEOTRACE_LOG_LEVEL", runtime.get("log_level", "INFO")).upper(),
            run_id=env.get("GEOTRACE_RUN_ID", runtime.get("run_id", "local")),
            config_path=config_path,
            data_dir=Path(
                env.get("GEOTRACE_DATA_DIR", runtime.get("data_dir", "outputs/verl_data"))
            ),
            checkpoint_dir=Path(
                env.get(
                    "GEOTRACE_CHECKPOINT_DIR",
                    runtime.get("checkpoint_dir", "outputs/grpo_checkpoints"),
                )
            ),
            prm_device=env.get("GEOTRACE_PRM_DEVICE", runtime.get("prm_device", "cpu")),
            reward_mode=env.get("GEOTRACE_REWARD_MODE", runtime.get("reward_mode", "process")),
            rollout_mode=env.get(
                "GEOTRACE_ROLLOUT_MODE", runtime.get("rollout_mode", "tool_agent")
            ),
            ray_address=env.get("GEOTRACE_RAY_ADDRESS") or None,
            recovery=rec,
            transport=http,
        )
        if result.reward_mode not in {"process", "outcome"} or result.rollout_mode not in {
            "tool_agent",
            "single_turn",
        }:
            raise ValueError(
                "reward_mode must be process/outcome and rollout_mode tool_agent/single_turn"
            )
        return result
