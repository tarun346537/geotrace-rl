"""Process-reward training defaults and command-line settings."""

from __future__ import annotations

import argparse
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

Namespace = argparse.Namespace | SimpleNamespace
DEFAULT_CONFIG = Path(__file__).with_name("prm_defaults.toml")


def defaults(section: str) -> dict[str, Any]:
    """Load the selected training section from the configured TOML file."""
    path = Path(os.environ.get("GEOTRACE_PRM_CONFIG", str(DEFAULT_CONFIG)))
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{section}] in {path}. select a PRM config")
    return dict(value)


@dataclass(frozen=True)
class FeatureSettings:
    """Validated feature-head settings shared by the CLI and recovery tests."""

    input: Path
    output: Path
    epochs: int
    hidden: int
    batch_size: int
    lr: float
    seed: int
    max_steps: int
    patience: int
    min_delta: float
    grad_clip: float
    std_floor: float
    resume: Path | None
    wandb_project: str | None
    run_id: str | None
    synthetic: bool

    @classmethod
    def from_args(cls, args: Namespace) -> FeatureSettings:
        """Apply file defaults to optional programmatic arguments."""
        cfg = defaults("feature")
        settings = cls(
            Path(args.input),
            Path(args.output),
            int(args.epochs),
            int(args.hidden),
            int(args.batch_size),
            float(args.lr),
            int(args.seed),
            int(args.max_steps),
            int(getattr(args, "patience", cfg["early_stopping_patience"])),
            float(getattr(args, "min_delta", cfg["min_delta"])),
            float(cfg["grad_clip"]),
            float(cfg["std_floor"]),
            Path(args.resume) if args.resume else None,
            getattr(args, "wandb_project", None),
            getattr(args, "run_id", None),
            bool(getattr(args, "synthetic", False)),
        )
        if min(settings.epochs, settings.hidden, settings.batch_size, settings.patience) < 1:
            raise ValueError("training sizes and patience must be positive. check PRM config")
        if not all(
            math.isfinite(value)
            for value in (settings.lr, settings.min_delta, settings.grad_clip, settings.std_floor)
        ):
            raise ValueError("training rates and tolerances must be finite. check PRM config")
        if (
            min(settings.lr, settings.grad_clip, settings.std_floor) <= 0
            or settings.max_steps < 0
            or settings.min_delta < 0
        ):
            raise ValueError("invalid rate, step limit, tolerance or scale. check PRM config")
        return settings
