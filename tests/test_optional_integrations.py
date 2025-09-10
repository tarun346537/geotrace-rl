import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geotrace.data import generate, parse_task, task_dict, write_jsonl
from geotrace.physics import GravityTask, forward
from geotrace.process_rewards import monte_carlo_label, verify_tool_output


@unittest.skipUnless(
    importlib.util.find_spec("simpeg"), "Install .[simpeg] for actual solver integration"
)
class SimpegTests(unittest.TestCase):
    def test_finite_cube_backend_is_used_by_labeler_and_verifier(self) -> None:
        from geotrace.simpeg_adapter import cube_forward

        x = np.array([-400.0, 0.0, 400.0])
        truth = {"x_m": 0.0, "depth_m": 150.0, "mass_kg": 1e10}
        y = cube_forward(x, source_x_m=0, depth_m=150, mass_kg=1e10)
        point = forward(x, source_x_m=0, depth_m=150, mass_kg=1e10)
        self.assertTrue(np.all(y > 0))
        np.testing.assert_allclose(y, point, rtol=0.02)
        self.assertFalse(np.allclose(y, point))
        task = GravityTask(
            x, y, {key: (value,) for key, value in truth.items()}, forward_backend="simpeg_cube"
        )
        restored = parse_task(task_dict(task))
        np.testing.assert_array_equal(restored.prediction(truth), y)
        self.assertTrue(verify_tool_output(restored, truth, y))
        self.assertEqual(
            monte_carlo_label(restored, {}, samples=4, rng=np.random.default_rng(1)).probability, 1
        )


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
    "Install .[data] for parquet export",
)
class ParquetTests(unittest.TestCase):
    def test_tool_agent_export_contains_plain_trusted_task_and_agent_route(self) -> None:
        import json

        import pandas as pd

        from geotrace.prepare_verl import export

        local = Path(__file__).resolve().parents[1] / "outputs/test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            directory = Path(directory)
            source = directory / "traces.jsonl"
            write_jsonl(source, generate(10))
            export(source, directory / "parquet", "tool_agent")
            frame = pd.read_parquet(directory / "parquet/train.parquet")
            self.assertEqual(set(frame["agent_name"]), {"geotrace_tool"})
            for _, row in frame.iterrows():
                self.assertEqual(
                    json.loads(row["prompt"][-1]["content"])["forward_backend"], "point_mass"
                )
                self.assertIn("<END_STEP>", row["prompt"][0]["content"])

    def test_export_preserves_grouped_survey_splits_and_trusted_tasks(self) -> None:
        import pandas as pd

        from geotrace.prepare_verl import export

        local = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            directory = Path(directory)
            source = directory / "traces.jsonl"
            write_jsonl(source, generate(10))
            counts = export(source, directory / "parquet")
            self.assertEqual(counts, {"train": 6, "validation": 2, "test": 2})
            groups = []
            for split in counts:
                frame = pd.read_parquet(directory / "parquet" / f"{split}.parquet")
                groups.append({row["survey_id"] for row in frame["extra_info"]})
                self.assertTrue(
                    all(
                        row["task"]["forward_backend"] == "point_mass"
                        for row in frame["extra_info"]
                    )
                )
            self.assertFalse(
                groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
            )


if __name__ == "__main__":
    unittest.main()
