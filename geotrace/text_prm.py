"""Textual process rewards: schema helpers, inference and training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .prm_settings import Namespace, defaults
from .text_prm_data import Trace, examples, first_error_metrics, load_traces, prefix_text
from .text_prm_model import TextRewardScorer, inference_settings

__all__ = [
    "TextRewardScorer",
    "inference_settings",
    "prefix_text",
    "load_traces",
    "examples",
    "first_error_metrics",
    "train",
]


def train(args: Namespace) -> Trace:
    """Load the optional transformer trainer and fit labeled prefixes."""
    from .text_prm_training import train as fit

    return fit(args)


def main() -> None:
    """Train a transformer PRM from the configured data and model paths."""
    from .config import RuntimeConfig
    from .logging_config import configure_logging

    cfg, runtime = defaults("training"), RuntimeConfig.from_env()
    configure_logging(runtime.log_level, runtime.run_id)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True, help="base model ID or local model directory")
    parser.add_argument("--output", type=Path, default=runtime.checkpoint_dir / "text-prm")
    parser.add_argument("--epochs", type=float, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["learning_rate"])
    parser.add_argument("--max-length", type=int, default=cfg["max_length"])
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--checkpoint-steps", type=int, default=cfg["checkpoint_steps"])
    parser.add_argument("--patience", type=int, default=cfg["early_stopping_patience"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-id", default=runtime.run_id)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
