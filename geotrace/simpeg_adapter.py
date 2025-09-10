"""Actual optional SimPEG 3-D density gravity adapter, with explicit units."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .constants import CUBE_SIZE_M, KGM3_PER_GCC, MICROGAL_PER_MILLIGAL
from .logging_config import configure_logging
from .types import FloatArray

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _unit_density_cube(
    receivers_tuple: tuple[float, ...], source_x_m: float, depth_m: float, cube_size_m: float
) -> FloatArray:
    xyz = np.column_stack(
        [receivers_tuple, np.zeros(len(receivers_tuple)), np.zeros(len(receivers_tuple))]
    )
    response = predict_density_gravity(
        widths_m=[[cube_size_m], [cube_size_m], [cube_size_m]],
        origin_m=[source_x_m - cube_size_m / 2, -cube_size_m / 2, -depth_m - cube_size_m / 2],
        receivers_xyz_m=xyz,
        density_gcc=[1.0],
    )
    response.setflags(write=False)
    return response


def cube_forward(
    x_m: ArrayLike,
    *,
    source_x_m: float,
    depth_m: float,
    mass_kg: float,
    cube_size_m: float = CUBE_SIZE_M,
) -> FloatArray:
    """Single finite cube for the same x/depth/mass assignment task.

    Unlike the analytic point-mass backend, this computes the exact integral
    response of a uniform-density cube using SimPEG. mass / volume converts
    kg/m³ to g/cm³ by dividing by 1000. Cached unit-density responses are
    linear in density and never reused across different survey geometries.
    """
    x = np.asarray(x_m, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("receiver locations must be a finite nonempty vector")
    if not np.all(np.isfinite([source_x_m, depth_m, mass_kg, cube_size_m])):
        raise ValueError("cube parameters must be finite")
    if cube_size_m <= 0 or mass_kg <= 0 or depth_m <= cube_size_m / 2:
        raise ValueError("require positive mass/size and a fully buried cube")
    density_gcc = mass_kg / cube_size_m**3 / KGM3_PER_GCC
    return (
        _unit_density_cube(tuple(x.tolist()), float(source_x_m), float(depth_m), float(cube_size_m))
        * density_gcc
    )


def predict_density_gravity(
    *,
    widths_m: list[ArrayLike],
    origin_m: ArrayLike,
    receivers_xyz_m: ArrayLike,
    density_gcc: ArrayLike,
) -> FloatArray:
    """Return DOWNWARD gravity in microGal from a 3-D density contrast mesh.

    SimPEG uses z upward, density in g/cm³, and gz in mGal (upward).
    The sign reversal and x1000 conversion here match the core demo's
    downward-positive microGal convention. No spherical point mass is used.
    """
    from discretize import TensorMesh
    from simpeg import maps
    from simpeg.potential_fields import gravity

    if len(widths_m) != 3 or any(
        np.asarray(w).ndim != 1
        or np.asarray(w).size == 0
        or not np.all(np.isfinite(w))
        or np.any(np.asarray(w) <= 0)
        for w in widths_m
    ):
        raise ValueError("widths_m must contain three positive finite cell-width vectors")
    origin = np.asarray(origin_m, dtype=float)
    receivers = np.asarray(receivers_xyz_m, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_m must contain three finite coordinates")
    if (
        receivers.ndim != 2
        or receivers.shape[1] != 3
        or len(receivers) == 0
        or not np.all(np.isfinite(receivers))
    ):
        raise ValueError("receivers_xyz_m must be nonempty finite [receivers,3] coordinates")
    mesh = TensorMesh([np.asarray(w, dtype=float) for w in widths_m], origin=origin)
    density = np.asarray(density_gcc, dtype=float)
    if density.shape != (mesh.n_cells,) or not np.all(np.isfinite(density)):
        raise ValueError("density_gcc must contain one finite contrast per mesh cell")
    receiver = gravity.receivers.Point(receivers, components=["gz"])
    source = gravity.sources.SourceField(receiver_list=[receiver])
    survey = gravity.survey.Survey(source)
    simulation = gravity.simulation.Simulation3DIntegral(
        mesh,
        survey=survey,
        rhoMap=maps.IdentityMap(nP=mesh.n_cells),
        store_sensitivities="forward_only",
    )
    return -MICROGAL_PER_MILLIGAL * np.asarray(simulation.dpred(density), dtype=float)


def main() -> None:
    """Run a density-mesh gravity forward calculation from supplied inputs."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey", type=Path, required=True, help="CSV with x_m,y_m,z_m columns (z upward)"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="NPZ with hx,hy,hz,origin_m,density_gcc"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.survey.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    xyz = np.array([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows])
    with np.load(args.model, allow_pickle=False) as model:
        prediction = predict_density_gravity(
            widths_m=[model[key] for key in ("hx", "hy", "hz")],
            origin_m=model["origin_m"],
            receivers_xyz_m=xyz,
            density_gcc=model["density_gcc"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "kind": "simpeg_forward_run",
                "component": "downward_gz",
                "units": "microGal",
                "prediction": prediction.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "simpeg_prediction_written",
        extra={"receivers": len(prediction), "output": str(args.output)},
    )


if __name__ == "__main__":
    main()
