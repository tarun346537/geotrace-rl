"""Write small seeded gravity traces without downloading survey data."""

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    """Write the same survey/trace schema accepted by the training pipeline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geotrace.data import generate, write_jsonl
    from geotrace.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample-traces.jsonl"))
    parser.add_argument("--surveys", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--backend", choices=["point_mass", "simpeg_cube"], default="point_mass")
    args = parser.parse_args()
    records = generate(args.surveys, args.seed, args.backend)
    write_jsonl(args.output, records)
    logging.getLogger(__name__).info(
        "sample_traces_written",
        extra={"output": str(args.output), "traces": len(records), "seed": args.seed},
    )


if __name__ == "__main__":
    main()
