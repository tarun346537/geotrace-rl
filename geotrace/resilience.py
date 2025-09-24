"""Bounded transport backoff and endpoint circuit state."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class CircuitOpenError(RuntimeError):
    """The endpoint has failed repeatedly and its cooldown has not expired."""


def retry_delay(attempt: int, initial: float, maximum: float) -> float:
    """Cap exponential retry delay without allowing unbounded exponent growth."""
    if (
        attempt < 1
        or not math.isfinite(initial)
        or not math.isfinite(maximum)
        or not 0 <= initial <= maximum
    ):
        raise ValueError("retry delay needs attempt >=1 and 0<=initial<=maximum")
    return float(min(maximum, initial * 2 ** min(attempt - 1, 30)))


@dataclass
class CircuitBreaker:
    """Allow a trial request after cooldown and close again only on success."""

    failure_limit: int
    reset_seconds: float
    clock: Callable[[], float] = time.monotonic
    failures: int = field(default=0, init=False)
    opened_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (
            self.failure_limit < 1
            or not math.isfinite(self.reset_seconds)
            or self.reset_seconds <= 0
        ):
            raise ValueError("circuit needs a positive failure limit and reset duration")

    def before_request(self) -> None:
        """Reject requests while the endpoint's failure cooldown is active."""
        if self.opened_at is not None and self.clock() - self.opened_at < self.reset_seconds:
            raise CircuitOpenError(
                "model endpoint circuit is open. wait for cooldown or check the server"
            )

    def record_success(self) -> None:
        """A successful request closes the circuit and clears its failure count."""
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        """Open the circuit after repeated endpoint failures."""
        self.failures += 1
        if self.failures >= self.failure_limit:
            self.opened_at = self.clock()
