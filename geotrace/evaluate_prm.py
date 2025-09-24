"""Evaluate a saved PRM without refitting its threshold on the test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .logging_config import configure_logging
from .prm_provenance import read_provenance, validate_heldout
from .train_prm import evaluate, load_dataset, load_head
from .types import JsonRecord

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Evaluate a saved process head on its held-out survey split."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prm/evaluation_only.json"))
    args = parser.parse_args()
    report = evaluate_checkpoint(args.input, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("process_head_evaluated", extra={"metrics": report})


def evaluate_checkpoint(source: str | Path, checkpoint: str | Path) -> JsonRecord:
    """Score held-out surveys only after checking the saved training identities."""
    model, state = load_head(checkpoint)
    dataset = load_dataset(source)
    validate_heldout(read_provenance(source), state.get("survey_provenance"))
    if "threshold" not in state:
        raise ValueError("checkpoint lacks a validation-selected threshold")
    metrics, _ = evaluate(
        model, dataset["test"], state["mean"].numpy(), state["std"].numpy(), state["threshold"]
    )
    return {
        "kind": "local_prm_evaluation",
        "test": metrics,
        "survey_overlap_checked": True,
        "training_dataset_sha256": state["dataset_sha256"],
    }


if __name__ == "__main__":
    main()
