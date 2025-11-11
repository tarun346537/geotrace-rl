"""Inspect durable experiment health without starting a worker or model server."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from geotrace.logging_config import configure_logging
from geotrace.monitoring import CheckpointStore, RunLedger, gpu_time_accounting
from geotrace.types import JsonRecord


def run_health(
    directory: str | Path,
    run_id: str,
    *,
    verify_checkpoint: bool = False,
    reserved_gpu_hours: float | None = None,
    ledger_path: str | Path | None = None,
) -> JsonRecord:
    """Check ledger integrity and report checkpoint progress for one run."""
    root = Path(directory).resolve()
    database = Path(ledger_path) if ledger_path is not None else root / "runs.sqlite3"
    if not database.is_file():
        return {
            "healthy": False,
            "run_id": run_id,
            "error": "run ledger is missing. check --directory",
        }
    try:
        with closing(sqlite3.connect(database)) as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()
        if check is None or check[0] != "ok":
            raise sqlite3.DatabaseError("ledger integrity check failed")
        ledger = RunLedger(database)
        state = ledger.state(run_id)
        checkpoint_dir = root / run_id / "checkpoints"
        checkpoint = (
            CheckpointStore(checkpoint_dir).latest(verify_payload=verify_checkpoint)
            if checkpoint_dir.is_dir()
            else None
        )
        measurements = ledger.summary(run_id)
        accounting = (
            gpu_time_accounting(measurements["checkpoint_committed_gpu_hours"], reserved_gpu_hours)
            if reserved_gpu_hours is not None
            else None
        )
        return {
            "healthy": state["status"] != "failed",
            "needs_attention": state["status"] in {"failed", "interrupted"},
            "run_id": run_id,
            "status": state["status"],
            "checkpoint_step": checkpoint["step"] if checkpoint else None,
            "checkpoint_digest_verified": bool(checkpoint and verify_checkpoint),
            "measurements": measurements,
            "reservation_accounting": accounting,
        }
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError) as error:
        return {
            "healthy": False,
            "run_id": run_id,
            "error": f"{type(error).__name__}: {error}",
            "action": "inspect the durable run directory and checkpoint manifest",
        }


def main() -> int:
    """Exit nonzero for missing or failed run state, retaining a JSON log record."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger-file", type=Path)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument(
        "--reserved-gpu-hours",
        type=float,
        help="full GPU reservation time measured from the allocation ledger",
    )
    args = parser.parse_args()
    result = run_health(
        args.directory,
        args.run_id,
        verify_checkpoint=args.verify_checkpoint,
        reserved_gpu_hours=args.reserved_gpu_hours,
        ledger_path=args.ledger_file,
    )
    logging.getLogger(__name__).info(
        "experiment_health", extra={"health": result, "run_id": args.run_id}
    )
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
