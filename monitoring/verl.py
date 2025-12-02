"""Supervise native verl jobs with durable attempts, checkpoint pointers and alerts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from geotrace.config import RuntimeConfig
from geotrace.logging_config import configure_logging
from geotrace.monitoring import RecoverySupervisor, WandbSink
from geotrace.ops.verl_job import VerlJobBackend


def main() -> None:
    """Submit the launcher's native configuration to an existing Ray cluster."""
    cfg = RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    parser.add_argument("--wandb-project")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configure_logging(cfg.log_level, args.run_id)
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    if not overrides:
        parser.error("supply the native trainer overrides after --")
    # Credentials stay in the local tracker process and never enter the ledger.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("GEOTRACE_")
        and key not in {"GEOTRACE_ATTEMPT", "GEOTRACE_RUN_DIRECTORY", "GEOTRACE_ALLOCATED_GPUS"}
        and not any(word in key for word in ("KEY", "TOKEN", "SECRET"))
    }
    environment["GEOTRACE_RUN_ID"] = args.run_id
    environment["PYTHONPATH"] = str(Path.cwd())
    spec = {
        "overrides": overrides,
        "environment": environment,
        "working_directory": str(Path.cwd()),
        "python": sys.executable,
    }
    tracker = None
    if args.wandb_project:
        import wandb

        tracker = wandb.init(project=args.wandb_project, id=args.run_id, resume="allow")
    directory = args.directory.resolve()
    supervisor = RecoverySupervisor(
        directory,
        VerlJobBackend(args.address),
        sink=WandbSink(tracker) if tracker else None,
        recovery=cfg.recovery,
        ledger_path=args.ledger_file.resolve(),
    )
    try:
        result = supervisor.run(
            args.run_id, "geotrace.verl_driver", spec, resources={"num_cpus": 1, "num_gpus": 0}
        )
        logging.getLogger(__name__).info("native_training_complete", extra={"result": result})
    finally:
        try:
            report = supervisor.ledger.summary(args.run_id)
            (directory / f"{args.run_id}-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            if tracker:
                tracker.finish()


if __name__ == "__main__":
    main()
