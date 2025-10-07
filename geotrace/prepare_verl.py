"""Export grouped survey prompts to the verl 0.5.x parquet schema."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data import read_records
from .logging_config import configure_logging
from .types import JsonRecord

LOGGER = logging.getLogger(__name__)


def export(
    input_path: str | Path, output_directory: str | Path, mode: str = "single_turn"
) -> dict[str, int]:
    """Export one policy prompt per survey with fixed split membership."""
    import pandas as pd

    if mode not in {"single_turn", "tool_agent"}:
        raise ValueError("mode must be single_turn or tool_agent")
    records = read_records(input_path)
    by_split: dict[str, list[JsonRecord]] = {split: [] for split in ("train", "validation", "test")}
    seen = set()
    instruction = (
        "Infer a positive gravity source from this survey using its specified forward_backend. Units: x_m/depth_m in meters, mass_kg in kg, "
        "downward gravity in microGal. Choose one value from each supplied parameter grid. "
        'Return ONLY JSON with assignments [["x_m",value],["depth_m",value],["mass_kg",value]] '
        "and claimed_prediction_microgal, an array of independently computed gravity values at every receiver. "
        "For point_mass, use G=6.67430e-11 and g=G*mass*depth/((receiver_x-source_x)^2+depth^2)^(3/2)*1e8. "
        "For simpeg_cube, use the finite uniform cube integral with the supplied cube_size_m, centered at source x, y=0, z=-depth. "
        "density contrast is mass/(cube_size_m^3)/1000 g/cm^3 and receivers have y=z=0.\nSurvey: "
    )
    for record in records:
        if record["survey_id"] in seen:
            continue
        seen.add(record["survey_id"])
        row: JsonRecord = {
            "data_source": "geotrace",
            "prompt": [{"role": "user", "content": instruction + json.dumps(record["task"])}],
            "ability": "geophysical_parameter_reasoning",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "task": record["task"],
                "survey_id": record["survey_id"],
                "provenance": record["provenance"],
            },
        }
        if mode == "tool_agent":
            from .agent_loop import AGENT_INSTRUCTION

            row["prompt"] = [
                {"role": "system", "content": AGENT_INSTRUCTION},
                {"role": "user", "content": json.dumps(record["task"], allow_nan=False)},
            ]
            row["agent_name"] = "geotrace_tool"
        by_split[record["split"]].append(row)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        if not rows:
            raise ValueError(f"the {split} split is empty")
        pd.DataFrame(rows).to_parquet(output / f"{split}.parquet", index=False)
    return {split: len(rows) for split, rows in by_split.items()}


def main() -> None:
    """Write training, validation and test parquet files."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/verl_data"))
    parser.add_argument("--mode", choices=["single_turn", "tool_agent"], default="tool_agent")
    args = parser.parse_args()
    LOGGER.info(
        "verl_prompts_written", extra={"surveys": export(args.input, args.output, args.mode)}
    )


if __name__ == "__main__":
    main()
