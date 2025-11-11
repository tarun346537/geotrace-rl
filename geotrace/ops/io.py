"""Atomic state writes and append-only telemetry IO."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..types import JsonRecord


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append(path: str | Path, value: JsonRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: str | Path, offset: int = 0) -> tuple[list[JsonRecord], int, bool]:
    """Return intact records and whether a crash left an incomplete last line."""
    path = Path(path)
    if not path.exists():
        return [], offset, False
    if path.stat().st_size < offset:
        raise ValueError("append-only telemetry file was truncated")
    records: list[JsonRecord] = []
    partial, cursor = False, offset
    with path.open("rb") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.endswith(b"\n"):
                partial = True
                break
            if line.strip():
                records.append(json.loads(line))
            cursor += len(line)
    return records, cursor, partial
