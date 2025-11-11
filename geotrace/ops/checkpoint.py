"""Immutable checkpoint payloads and verified manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..types import JsonRecord
from .io import _atomic, _json, _utc


class CheckpointStore:
    """Atomic immutable checkpoint bytes and a verified latest manifest.

    Training code serializes its entire model/optimizer/RNG state into bytes.
    Publishing a manifest happens only after the immutable payload is durable.
    One worker attempt may write this store at a time.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def latest(self, *, verify_payload: bool = True) -> JsonRecord | None:
        """Read the latest manifest and optionally verify its payload digest."""
        manifest = self.root / "latest.json"
        if not manifest.exists():
            return None
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("checkpoint manifest must be an object. inspect latest.json")
        filename = value["filename"]
        if Path(filename).name != filename:
            raise ValueError("checkpoint manifest contains an invalid filename")
        payload_path = self.root / filename
        if payload_path.stat().st_size != value["bytes"]:
            raise ValueError("checkpoint size mismatch. refusing to resume")
        if (
            verify_payload
            and hashlib.sha256(payload_path.read_bytes()).hexdigest() != value["sha256"]
        ):
            raise ValueError("checkpoint digest/size mismatch. refusing to resume")
        if not isinstance(value["step"], int) or value["step"] < 0 or value["attempt"] < 1:
            raise ValueError("invalid checkpoint step or attempt")
        return value

    def load(self) -> tuple[int, bytes] | None:
        """Read the committed step and serialized training state."""
        value = self.latest()
        return (
            None if value is None else (value["step"], (self.root / value["filename"]).read_bytes())
        )

    def save(self, *, step: int, attempt: int, payload: bytes) -> JsonRecord:
        """Publish immutable state before advancing the latest manifest."""
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or attempt < 1:
            raise ValueError("require a nonnegative integer step and positive attempt")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("a checkpoint must contain nonempty serialized bytes")
        digest = hashlib.sha256(payload).hexdigest()
        prior = self.latest(verify_payload=False)
        if prior and step < prior["step"]:
            raise ValueError("refusing a checkpoint that regresses committed progress")
        if prior and step == prior["step"]:
            if prior["sha256"] != digest:
                raise ValueError("conflicting checkpoint payload for an already committed step")
            return prior
        filename = f"step-{step:012d}-{digest}.bin"
        _atomic(self.root / filename, payload)
        value = {
            "step": step,
            "attempt": int(attempt),
            "filename": filename,
            "sha256": digest,
            "bytes": len(payload),
            "published_at_utc": _utc(),
        }
        _atomic(self.root / "latest.json", (_json(value) + "\n").encode())
        return value
