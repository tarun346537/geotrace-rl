"""Transactional run, event and metric accounting."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..types import JsonRecord
from .io import _json, _utc


class RunLedger:
    """Driver-owned SQLite state. Duplicate events/metrics cannot inflate counts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                  attempt INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS events (
                  run_id TEXT NOT NULL, event_key TEXT NOT NULL, kind TEXT NOT NULL,
                  attempt INTEGER NOT NULL, detail TEXT NOT NULL, at_utc TEXT NOT NULL,
                  PRIMARY KEY(run_id,event_key));
                CREATE TABLE IF NOT EXISTS metrics (
                  run_id TEXT NOT NULL, attempt INTEGER NOT NULL, step INTEGER NOT NULL,
                  record TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(run_id,attempt,step));
                CREATE TABLE IF NOT EXISTS cursors (
                  run_id TEXT NOT NULL, attempt INTEGER NOT NULL, stream TEXT NOT NULL,
                  byte_offset INTEGER NOT NULL, PRIMARY KEY(run_id,attempt,stream));
            """)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, run_id: str, spec: JsonRecord) -> None:
        """Bind a run ID to one immutable experiment specification."""
        if (
            not run_id
            or run_id in {".", ".."}
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in run_id
            )
        ):
            raise ValueError("run_id may contain only letters, digits, dot, underscore and hyphen")
        encoded = _json(spec)
        with self._db() as db:
            row = db.execute("SELECT spec FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row and row["spec"] != encoded:
                raise ValueError("run ID already belongs to a different specification")
            db.execute(
                "INSERT OR IGNORE INTO runs(run_id,spec,status) VALUES(?,?,'pending')",
                (run_id, encoded),
            )

    def state(self, run_id: str) -> JsonRecord:
        """Return the durable state of a registered run."""
        with self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def event(self, run_id: str, key: str, kind: str, attempt: int, detail: JsonRecord) -> bool:
        """Insert an event once by its idempotency key."""
        with self._db() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                (run_id, key, kind, attempt, _json(detail), _utc()),
            )
            return cursor.rowcount == 1

    def has_event(self, run_id: str, key: str) -> bool:
        """Check whether an event identity is already recorded."""
        with self._db() as db:
            return (
                db.execute(
                    "SELECT 1 FROM events WHERE run_id=? AND event_key=?", (run_id, key)
                ).fetchone()
                is not None
            )

    def begin(self, run_id: str) -> int:
        """Atomically start the next pending or interrupted attempt."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["status"] not in {"pending", "interrupted"}:
                raise ValueError(
                    "only pending/interrupted runs can begin. reconcile an abandoned running attempt explicitly"
                )
            attempt = row["attempt"] + 1
            db.execute(
                "UPDATE runs SET status='running',attempt=?,error=NULL WHERE run_id=?",
                (attempt, run_id),
            )
            db.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                (run_id, f"attempt:{attempt}:started", "attempt_started", attempt, "{}", _utc()),
            )
        return int(attempt)

    def interrupted(self, run_id: str, attempt: int, reason: str) -> bool:
        """Record confirmed loss once for the active attempt."""
        with self._db() as db:
            changed = db.execute(
                "UPDATE runs SET status='interrupted',error=? WHERE run_id=? AND attempt=? AND status='running'",
                (reason, run_id, attempt),
            ).rowcount
            if changed:
                db.execute(
                    "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                    (
                        run_id,
                        f"attempt:{attempt}:interrupted",
                        "interrupted",
                        attempt,
                        _json({"reason": reason}),
                        _utc(),
                    ),
                )
        return bool(changed)

    def fail(self, run_id: str, attempt: int, reason: str) -> None:
        """Close an active attempt after an application failure."""
        with self._db() as db:
            changed = db.execute(
                "UPDATE runs SET status='failed',error=? WHERE run_id=? AND attempt=? AND status='running'",
                (reason, run_id, attempt),
            ).rowcount
            if changed:
                db.execute(
                    "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                    (
                        run_id,
                        f"attempt:{attempt}:failed",
                        "failed",
                        attempt,
                        _json({"reason": reason}),
                        _utc(),
                    ),
                )

    def complete(self, run_id: str, attempt: int, result: JsonRecord) -> bool:
        """Persist one completed result without repeating completed work."""
        encoded = _json(result)
        with self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row["status"] == "completed":
                if row["result"] != encoded:
                    raise ValueError("conflicting result for completed run")
                return False
            if row["status"] != "running" or row["attempt"] != attempt:
                raise ValueError("only the current running attempt can complete")
            db.execute(
                "UPDATE runs SET status='completed',result=?,error=NULL WHERE run_id=?",
                (encoded, run_id),
            )
            db.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                (run_id, f"attempt:{attempt}:completed", "completed", attempt, "{}", _utc()),
            )
        return True

    def metric(self, row: JsonRecord) -> bool:
        """Reject conflicting metrics for an existing attempt and step."""
        encoded = _json({key: value for key, value in row.items() if key != "at_utc"})
        key = (row["run_id"], row["attempt"], row["step"])
        with self._db() as db:
            old = db.execute(
                "SELECT record FROM metrics WHERE run_id=? AND attempt=? AND step=?", key
            ).fetchone()
            if old and old["record"] != encoded:
                raise ValueError("conflicting metric record for the same attempt/global step")
            db.execute(
                "INSERT OR IGNORE INTO metrics(run_id,attempt,step,record) VALUES(?,?,?,?)",
                (*key, encoded),
            )
        return old is None

    def checkpoint(self, run_id: str, manifest: JsonRecord) -> None:
        """Attribute measured work to the checkpoint that committed it."""
        self.event(
            run_id,
            f"checkpoint:{manifest['step']}:{manifest['sha256']}",
            "checkpoint_committed",
            manifest["attempt"],
            manifest,
        )
        with self._db() as db:
            db.execute(
                "UPDATE metrics SET committed=1 WHERE run_id=? AND attempt=? AND step<=?",
                (run_id, manifest["attempt"], manifest["step"]),
            )

    def cursor(self, run_id: str, attempt: int, stream: str, new_offset: int | None = None) -> int:
        """Read or advance an append-only telemetry cursor."""
        with self._db() as db:
            if new_offset is not None:
                db.execute(
                    "INSERT INTO cursors VALUES(?,?,?,?) ON CONFLICT(run_id,attempt,stream) DO UPDATE SET byte_offset=excluded.byte_offset",
                    (run_id, attempt, stream, new_offset),
                )
            row = db.execute(
                "SELECT byte_offset FROM cursors WHERE run_id=? AND attempt=? AND stream=?",
                (run_id, attempt, stream),
            ).fetchone()
        return int(row["byte_offset"]) if row else 0

    def pending_alerts(self, run_id: str) -> list[JsonRecord]:
        """List alert requests without an SDK acknowledgment."""
        with self._db() as db:
            return [
                dict(row)
                for row in db.execute(
                    """
                SELECT e.* FROM events e WHERE e.run_id=? AND e.kind='alert_requested'
                AND NOT EXISTS (SELECT 1 FROM events d WHERE d.run_id=e.run_id AND d.event_key='delivered:'||e.event_key)
            """,
                    (run_id,),
                )
            ]

    def summary(self, run_id: str) -> JsonRecord:
        """Summarize measured attempts, committed steps and alert failures."""
        state = self.state(run_id)
        with self._db() as db:
            events = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY at_utc", (run_id,)
                )
            ]
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM metrics WHERE run_id=? ORDER BY attempt,step", (run_id,)
                )
            ]
        committed_by_step: dict[int, float]
        observed_seconds, committed_by_step, utilization_numerator, utilization_weight = (
            0.0,
            {},
            0.0,
            0.0,
        )
        for item in rows:
            record = json.loads(item["record"])
            seconds = record["elapsed_seconds"] * record["allocated_gpus"]
            observed_seconds += seconds
            if item["committed"]:
                committed_by_step.setdefault(record["step"], seconds)
            if "gpu_utilization_percent" in record["metrics"]:
                utilization_numerator += seconds * record["metrics"]["gpu_utilization_percent"]
                utilization_weight += seconds
        committed_seconds = sum(committed_by_step.values())
        return {
            "run_id": run_id,
            "status": state["status"],
            "attempts": state["attempt"],
            "interruptions": sum(e["kind"] == "interrupted" for e in events),
            "resumed_attempts": sum(e["kind"] == "checkpoint_restored" for e in events),
            "completed": state["status"] == "completed",
            "observed_optimizer_steps": len(rows),
            "observed_step_gpu_hours": observed_seconds / 3600,
            "checkpoint_committed_gpu_hours": committed_seconds / 3600,
            "checkpoint_committed_fraction": committed_seconds / observed_seconds
            if observed_seconds
            else None,
            "observed_gpu_utilization_percent": utilization_numerator / utilization_weight
            if utilization_weight
            else None,
            "alert_delivery_failures": sum(e["kind"] == "sink_failure" for e in events),
            "accounting_scope": "checkpoint-committed and repeated optimizer-step GPU time",
        }
