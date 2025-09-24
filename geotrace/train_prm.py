"""Feature-head training with resumable batches and validation stopping."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from .config import RuntimeConfig
from .data import FEATURE_NAMES
from .logging_config import configure_logging
from .prm_metrics import (
    FeatureSplit,
    ProcessRewardHead,
    Record,
    atomic_save,
    evaluate,
    f1_error,
    load_dataset,
    load_head,
)
from .prm_provenance import read_provenance
from .prm_settings import FeatureSettings, Namespace, defaults

__all__ = [
    "ProcessRewardHead",
    "load_dataset",
    "evaluate",
    "f1_error",
    "atomic_save",
    "load_head",
    "train",
]
LOGGER = logging.getLogger(__name__)


class Tracker(Protocol):
    """Run-level experiment sink accepted by the training loop."""

    def log(self, data: Mapping[str, Any], step: int) -> None:
        """Record metrics for a completed optimizer step."""
        ...

    def finish(self) -> None:
        """Flush and close the experiment run."""
        ...


@contextmanager
def _stop_signals() -> Iterator[list[bool]]:
    stopped = [False]
    handlers: dict[int, Any] = {}

    def _request_stop(signum: int, frame: FrameType | None) -> None:
        stopped[0] = True

    if threading.current_thread() is threading.main_thread():
        for event in (signal.SIGINT, signal.SIGTERM):
            handlers[event] = signal.signal(event, _request_stop)
    try:
        yield stopped
    finally:
        for signal_number, handler in handlers.items():
            signal.signal(signal_number, handler)


def _threshold(
    model: ProcessRewardHead,
    split: FeatureSplit,
    mean: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
) -> float:
    _, scores = evaluate(model, split, mean, std, 0.5)
    cfg = defaults("feature")
    candidates = [
        float(value)
        for value in np.linspace(cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_steps"])
    ]
    return max(candidates, key=lambda value: f1_error(scores, split["feasible"], value))


def _tracker(cfg: FeatureSettings, settings: Record) -> Tracker | None:
    if not cfg.wandb_project:
        return None
    import wandb

    return cast(
        Tracker,
        wandb.init(project=cfg.wandb_project, config=settings, resume="allow", id=cfg.run_id),
    )


def train(args: Namespace) -> Record:
    """Train from labeled surveys, preserving the last complete optimizer batch.

    Args:
        args: CLI settings or a programmatic namespace with the same fields.

    Returns:
        Validation and test metrics for the last resumable checkpoint.
    """
    cfg = FeatureSettings.from_args(args)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    splits = load_dataset(cfg.input)
    provenance = read_provenance(cfg.input)
    digest = hashlib.sha256(cfg.input.read_bytes()).hexdigest()
    training = splits["train"]
    mean = training["x"].mean(axis=0)
    std = np.maximum(training["x"].std(axis=0), cfg.std_floor)
    model = ProcessRewardHead(cfg.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    epoch, cursor, step = 0, 0, 0
    order = rng.permutation(len(training["x"]))
    best_loss, stale = float("inf"), 0
    best_model: dict[str, torch.Tensor] | None = None
    previous: Record | None = None
    settings: Record = {
        "hidden": cfg.hidden,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "min_delta": cfg.min_delta,
        "grad_clip": cfg.grad_clip,
        "std_floor": cfg.std_floor,
    }
    if cfg.resume:
        previous = cast(Record, torch.load(cfg.resume, map_location="cpu", weights_only=True))
        if previous["dataset_sha256"] != digest or previous["settings"] != settings:
            raise ValueError(
                "resume settings differ. use the same input, model and optimizer flags"
            )
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["torch_rng"])
        rng.bit_generator.state = previous["numpy_rng"]
        epoch, cursor, step = previous["epoch"], previous["cursor"], previous["step"]
        order = previous["permutation"].numpy()
        best_loss, stale = previous.get("best_loss", best_loss), previous.get("stale_epochs", 0)
        best_model = previous.get("best_model")
    x, y = torch.tensor((training["x"] - mean) / std), torch.tensor(training["y"])
    cfg.output.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        atomic_save(cfg.output / "checkpoint.pt", previous)
    tracker = _tracker(cfg, settings)
    try:
        with _stop_signals() as stopped:
            while (
                epoch < cfg.epochs
                and not stopped[0]
                and stale < cfg.patience
                and (not cfg.max_steps or step < cfg.max_steps)
            ):
                model.train()
                indices = order[cursor : cursor + cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite loss. check targets and reduce the learning rate")
                torch.autograd.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                cursor += len(indices)
                step += 1
                if cursor >= len(order):
                    epoch, cursor = epoch + 1, 0
                    order = rng.permutation(len(training["x"]))
                    validation, _ = evaluate(model, splits["validation"], mean, std, 0.5)
                    current = float(validation["soft_target_mse"])
                    if current < best_loss - cfg.min_delta:
                        best_loss, stale = current, 0
                        best_model = {
                            key: value.detach().clone() for key, value in model.state_dict().items()
                        }
                    else:
                        stale += 1
                row: Record = {
                    "kind": "training_step",
                    "step": step,
                    "completed_epochs": epoch,
                    "loss": float(loss.detach()),
                    "dataset_sha256": digest,
                    "run_id": cfg.run_id,
                }
                with (cfg.output / "steps.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                LOGGER.info("feature_batch_complete", extra={"metrics": row})
                if tracker is not None:
                    tracker.log(row, step=step)
                atomic_save(
                    cfg.output / "checkpoint.pt",
                    {
                        "kind": "feature_process_reward_model",
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "hidden": cfg.hidden,
                        "feature_names": list(FEATURE_NAMES),
                        "settings": settings,
                        "mean": torch.tensor(mean),
                        "std": torch.tensor(std),
                        "epoch": epoch,
                        "cursor": cursor,
                        "step": step,
                        "permutation": torch.tensor(order),
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": rng.bit_generator.state,
                        "dataset_sha256": digest,
                        "survey_provenance": provenance,
                        "best_loss": best_loss,
                        "stale_epochs": stale,
                        "best_model": best_model,
                    },
                )
        threshold = _threshold(model, splits["validation"], mean, std)
        validation, _ = evaluate(model, splits["validation"], mean, std, threshold)
        test, _ = evaluate(model, splits["test"], mean, std, threshold)
        report: Record = {
            "kind": "synthetic_evaluation" if cfg.synthetic else "evaluation",
            "dataset_sha256": digest,
            "completed_epochs": epoch,
            "steps": step,
            "early_stopped": stale >= cfg.patience,
            "threshold_selected_on": "validation_only",
            "validation": validation,
            "test": test,
        }
        (cfg.output / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        state = cast(
            Record, torch.load(cfg.output / "checkpoint.pt", map_location="cpu", weights_only=True)
        )
        state["threshold"] = threshold
        state["survey_provenance"] = provenance
        atomic_save(cfg.output / "checkpoint.pt", state)
        if best_model is not None:
            best_head = ProcessRewardHead(cfg.hidden)
            best_head.load_state_dict(best_model)
            atomic_save(
                cfg.output / "best_model.pt",
                {
                    "model": best_model,
                    "hidden": cfg.hidden,
                    "mean": torch.tensor(mean),
                    "std": torch.tensor(std),
                    "threshold": _threshold(best_head, splits["validation"], mean, std),
                    "dataset_sha256": digest,
                    "survey_provenance": provenance,
                    "selection": "validation_soft_target_mse",
                },
            )
        LOGGER.info("feature_training_complete", extra={"metrics": report})
        return report
    finally:
        if tracker is not None:
            tracker.finish()


def main() -> None:
    """Run feature-head training from explicit data and runtime settings."""
    cfg, runtime = defaults("feature"), RuntimeConfig.from_env()
    configure_logging(runtime.log_level, runtime.run_id)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=runtime.checkpoint_dir / "feature-prm")
    for flag, kind in (("epochs", int), ("hidden", int), ("batch-size", int), ("seed", int)):
        parser.add_argument("--" + flag, type=kind, default=cfg[flag.replace("-", "_")])
    parser.add_argument("--lr", type=float, default=cfg["learning_rate"])
    parser.add_argument("--patience", type=int, default=cfg["early_stopping_patience"])
    parser.add_argument("--min-delta", type=float, default=cfg["min_delta"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-id", default=runtime.run_id)
    parser.add_argument("--synthetic", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
