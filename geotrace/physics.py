"""Analytic point-mass gravity in SI units with output in microGal."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from .constants import (
    CUBE_SIZE_M,
    GRAVITATIONAL_CONSTANT,
    MICROGAL_PER_MS2,
    PARAMETERS,
    SUCCESS_TOLERANCE,
)
from .types import FloatArray

G = GRAVITATIONAL_CONSTANT


def forward(x_m: ArrayLike, *, source_x_m: float, depth_m: float, mass_kg: float) -> FloatArray:
    """Downward vertical acceleration from one positive point-mass anomaly.

    Receivers lie on z=0. The source lies at positive downward depth.
    One microGal is 1e-8 m/s². The response uses the analytic point-mass equation.
    """
    x = np.asarray(x_m, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("receiver locations must be a nonempty finite vector")
    if not np.all(np.isfinite([source_x_m, depth_m, mass_kg])):
        raise ValueError("source parameters must be finite")
    if depth_m <= 0 or mass_kg <= 0:
        raise ValueError("this positive-anomaly task requires positive depth and mass")
    radius_squared = (x - source_x_m) ** 2 + depth_m**2
    return G * mass_kg * depth_m / radius_squared**1.5 * MICROGAL_PER_MS2


def relative_rmse(prediction: ArrayLike, observations: ArrayLike) -> float:
    """Normalize prediction RMSE by the RMS of a nonzero observed survey."""
    p, y = np.asarray(prediction, dtype=float), np.asarray(observations, dtype=float)
    if p.shape != y.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("predictions and observations must be matching nonempty vectors")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError("predictions and observations must be finite")
    scale = float(np.sqrt(np.mean(y * y)))
    if scale == 0:
        raise ValueError("relative RMSE is undefined for an all-zero survey")
    return float(np.sqrt(np.mean((p - y) ** 2)) / scale)


@dataclass(frozen=True)
class GravityTask:
    """A finite-parameter gravity inverse problem with an explicit forward backend."""

    receivers_m: FloatArray
    observations_microgal: FloatArray
    grid: Mapping[str, tuple[float, ...]]
    success_tolerance: float = SUCCESS_TOLERANCE
    forward_backend: str = "point_mass"
    cube_size_m: float = CUBE_SIZE_M

    def __post_init__(self) -> None:
        x = np.array(self.receivers_m, dtype=float, copy=True)
        y = np.array(self.observations_microgal, dtype=float, copy=True)
        relative_rmse(y, y)
        if x.shape != y.shape or not np.all(np.isfinite(x)):
            raise ValueError("receiver and observation vectors must match and be finite")
        if set(self.grid) != set(PARAMETERS):
            raise ValueError("grid must specify x_m, depth_m, mass_kg")
        copied_grid = {key: tuple(float(v) for v in self.grid[key]) for key in PARAMETERS}
        for key, values in copied_grid.items():
            if not values or len(set(values)) != len(values) or not np.all(np.isfinite(values)):
                raise ValueError("each parameter grid must contain unique finite values")
            if key != "x_m" and min(values) <= 0:
                raise ValueError("depth and mass grid entries must be positive")
        if not np.isfinite(self.success_tolerance) or self.success_tolerance <= 0:
            raise ValueError("success tolerance must be finite and positive")
        if self.forward_backend not in {"point_mass", "simpeg_cube"}:
            raise ValueError("forward_backend must be point_mass or simpeg_cube")
        if not np.isfinite(self.cube_size_m) or self.cube_size_m <= 0:
            raise ValueError("cube_size_m must be finite and positive")
        if (
            self.forward_backend == "simpeg_cube"
            and min(copied_grid["depth_m"]) <= self.cube_size_m / 2
        ):
            raise ValueError("SimPEG cube must remain strictly below the receiver surface")
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "receivers_m", x)
        object.__setattr__(self, "observations_microgal", y)
        object.__setattr__(self, "grid", copied_grid)

    def validate_prefix(self, prefix: Mapping[str, float]) -> None:
        """Reject unknown or out-of-grid parameter assignments."""
        if not set(prefix) <= set(PARAMETERS):
            raise ValueError("unknown parameter in prefix")
        for key, value in prefix.items():
            if value not in self.grid[key]:
                raise ValueError(f"parameter {key} is outside the task grid")

    def prediction(self, model: Mapping[str, float]) -> FloatArray:
        """Evaluate a complete model using this task's selected solver."""
        self.validate_prefix(model)
        if set(model) != set(PARAMETERS):
            raise ValueError("forward prediction requires a complete model")
        if self.forward_backend == "simpeg_cube":
            from .simpeg_adapter import cube_forward

            return cube_forward(
                self.receivers_m,
                source_x_m=model["x_m"],
                depth_m=model["depth_m"],
                mass_kg=model["mass_kg"],
                cube_size_m=self.cube_size_m,
            )
        return forward(
            self.receivers_m,
            source_x_m=model["x_m"],
            depth_m=model["depth_m"],
            mass_kg=model["mass_kg"],
        )

    def successful(self, model: Mapping[str, float]) -> bool:
        """Test the configured relative-RMSE outcome threshold."""
        return (
            relative_rmse(self.prediction(model), self.observations_microgal)
            <= self.success_tolerance
        )

    def completions(self, prefix: Mapping[str, float]) -> Iterator[dict[str, float]]:
        """Enumerate finite-grid completions for an evaluation oracle."""
        self.validate_prefix(prefix)
        remaining = [key for key in PARAMETERS if key not in prefix]
        for values in product(*(self.grid[key] for key in remaining)):
            yield {**prefix, **dict(zip(remaining, values, strict=True))}


def synthetic_task() -> tuple[GravityTask, dict[str, float]]:
    """Small noiseless finite-grid inverse problem without external observations."""
    grid = {
        "x_m": (-100.0, -50.0, 0.0, 50.0, 100.0),
        "depth_m": (100.0, 150.0, 200.0, 250.0),
        "mass_kg": (6e9, 8e9, 1e10, 1.2e10, 1.4e10),
    }
    truth = {"x_m": 0.0, "depth_m": 150.0, "mass_kg": 1e10}
    x = np.linspace(-400.0, 400.0, 25, dtype=np.float64)
    y = forward(x, source_x_m=truth["x_m"], depth_m=truth["depth_m"], mass_kg=truth["mass_kg"])
    return GravityTask(x, y, grid), truth
