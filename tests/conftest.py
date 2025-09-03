"""Shared fixtures for gravity traces and recovery checks."""

from pathlib import Path

import pytest

from geotrace.config import RecoveryConfig


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    """Isolate the durable ledger and checkpoint store for each test."""
    return tmp_path / "runs"


@pytest.fixture
def fast_recovery() -> RecoveryConfig:
    """Exercise retries without imposing wall-clock backoff on unit tests."""
    return RecoveryConfig(initial_backoff_seconds=0, max_backoff_seconds=0)
