"""Evaluate an exported textual reward model on independent annotated surveys."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from .data import parse_task
from .logging_config import configure_logging
from .prm_provenance import survey_provenance, validate_heldout
from .rollouts import verify_steps
from .text_prm_data import Trace, first_error_metrics, load_traces
from .text_prm_model import TextRewardScorer, inference_settings

LOGGER = logging.getLogger(__name__)


def evaluate_saved(source: str | Path, model_directory: str | Path, device: str = "cpu") -> Trace:
    """Reuse the validation-selected threshold and reject training-survey reuse."""
    config = inference_settings(model_directory)
    traces = load_traces(source)
    validate_heldout(survey_provenance(traces), config.get("survey_provenance"))
    threshold = config.get("first_error_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or config.get("first_error_threshold_selected_on") != "validation"
    ):
        raise ValueError(
            "export lacks a validation-selected first-error threshold. train with independent annotations"
        )
    heldout = [trace for trace in traces if trace["split"] == "test"]
    if any(not verify_steps(parse_task(trace["task"]), trace["steps"]) for trace in heldout):
        raise ValueError(
            "evaluation trace has inconsistent tool results. use recorded simulator observations"
        )
    if any("error_label" not in step for trace in heldout for step in trace["steps"]):
        raise ValueError(
            "test prefixes need independent error_label annotations. supply boolean step-error labels"
        )
    scorer = TextRewardScorer(model_directory, device=device)
    scores = [
        value for trace in heldout for value in scorer.probabilities(trace["task"], trace["steps"])
    ]
    return {
        "kind": "text_prm_heldout_evaluation",
        "survey_overlap_checked": True,
        "training_dataset_sha256": config.get("dataset_sha256"),
        "evaluation_dataset_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "test": first_error_metrics(heldout, scores, float(threshold)),
    }


def main() -> None:
    """Write a held-out first-error report without fitting a new threshold."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/text-prm/heldout.json"))
    args = parser.parse_args()
    report = evaluate_saved(args.input, args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("text_process_head_evaluated", extra={"metrics": report})


if __name__ == "__main__":
    main()
