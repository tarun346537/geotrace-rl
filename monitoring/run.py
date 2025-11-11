"""Run a checkpoint-aware worker locally or through Ray and write its ledger report."""

import argparse
import json
import logging
from pathlib import Path

from geotrace.logging_config import configure_logging
from geotrace.monitoring import LocalBackend, RayBackend, RecoverySupervisor, WandbSink


def main() -> None:
    """Execute a checkpoint-aware worker and retain its measured run report."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.json"))
    parser.add_argument("--directory", type=Path, default=Path("outputs/monitored-runs"))
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--ray-address")
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    tracker = None
    if args.wandb_project:
        import wandb

        tracker = wandb.init(project=args.wandb_project, id=spec["run_id"], resume="allow")
    supervisor = RecoverySupervisor(
        args.directory,
        RayBackend(address=args.ray_address) if args.ray else LocalBackend(),
        sink=WandbSink(tracker) if tracker else None,
        thresholds=spec.get("alert_thresholds"),
    )
    try:
        result = supervisor.run(
            spec["run_id"],
            spec["worker"],
            spec["worker_config"],
            max_recoveries=spec.get("max_recoveries", 3),
            resources=spec.get("resources"),
        )
        logging.getLogger(__name__).info(
            "monitored_run_complete",
            extra={"result": result, "monitoring": supervisor.reconcile(spec["run_id"])},
        )
    finally:
        args.directory.mkdir(parents=True, exist_ok=True)
        try:
            report = supervisor.ledger.summary(spec["run_id"])
            (args.directory / f"{spec['run_id']}-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            if tracker:
                tracker.finish()


if __name__ == "__main__":
    main()
