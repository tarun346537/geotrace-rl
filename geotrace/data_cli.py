"""Survey import and finite-grid labeling command-line entry points."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from .data import generate, import_survey, label_records, read_records, write_jsonl
from .logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run survey import, sample-data creation or Monte Carlo labeling."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("generate")
    make.add_argument("--surveys", type=int, default=30)
    make.add_argument("--seed", type=int, default=17)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--backend", choices=["point_mass", "simpeg_cube"], default="point_mass")
    make.add_argument("--cube-size-m", type=float, default=50)
    label = sub.add_parser("label")
    label.add_argument("--input", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument("--samples", type=int, default=128)
    label.add_argument("--seed", type=int, default=17)
    imported = sub.add_parser("import-survey")
    imported.add_argument("--survey", type=Path, required=True)
    imported.add_argument("--grid", type=Path, required=True)
    imported.add_argument("--survey-id", required=True)
    imported.add_argument("--source-url", help="Public survey release URL for source provenance")
    imported.add_argument("--split", choices=["train", "validation", "test"], required=True)
    imported.add_argument("--backend", choices=["point_mass", "simpeg_cube"], default="point_mass")
    imported.add_argument("--cube-size-m", type=float, default=50)
    imported.add_argument("--tolerance", type=float, default=0.035)
    imported.add_argument("--trace-count", type=int, default=8)
    imported.add_argument("--seed", type=int, default=17)
    imported.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        records = generate(args.surveys, args.seed, args.backend, args.cube_size_m)
    elif args.command == "label":
        records = label_records(read_records(args.input), args.samples, args.seed)
    else:
        records = import_survey(
            args.survey,
            args.grid,
            survey_id=args.survey_id,
            split=args.split,
            backend=args.backend,
            cube_size_m=args.cube_size_m,
            tolerance=args.tolerance,
            trace_count=args.trace_count,
            seed=args.seed,
            source_url=args.source_url,
        )
    write_jsonl(args.output, records)
    LOGGER.info(
        "survey_records_written",
        extra={
            "output": str(args.output),
            "traces": len(records),
            "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
