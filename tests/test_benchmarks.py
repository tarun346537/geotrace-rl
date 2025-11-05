import copy
import json
import tempfile
import unittest
from pathlib import Path

from geotrace.benchmarks import compare, compare_memory, evaluate
from geotrace.data import task_dict, write_jsonl
from geotrace.physics import forward, synthetic_task
from geotrace.rollouts import execute_tool


class BenchmarkTests(unittest.TestCase):
    def test_reward_hacking_is_checked_against_independent_outcome(self) -> None:
        task, truth = synthetic_task()
        wrong = dict(truth, x_m=100.0)
        x = [-175.0, -75.0, 25.0, 125.0, 225.0]
        y = forward(x, source_x_m=truth["x_m"], depth_m=truth["depth_m"], mass_kg=truth["mass_kg"])
        records = []
        for index, model in enumerate((truth, wrong)):
            arguments = {"reasoning": "Submit the proposed source", "model": model}
            records.append(
                {
                    "trace_id": f"trace-{index}",
                    "evaluation_id": f"case-{index}",
                    "survey_id": f"survey-{index}",
                    "split": "test",
                    "task": task_dict(task),
                    "assigned_reward": 1.0,
                    "heldout_survey": {"receivers_m": x, "observations_microgal": y.tolist()},
                    "steps": [
                        {
                            "reasoning": arguments["reasoning"],
                            "tool_call": {"name": "submit_model", "arguments": arguments},
                            "tool_result": execute_tool(task, "submit_model", arguments),
                        }
                    ],
                }
            )
        local = Path(__file__).resolve().parents[1] / "outputs/test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            source = Path(directory) / "rollouts.jsonl"
            write_jsonl(source, records)
            result = evaluate(source)
        self.assertEqual(result["reward_hacking_rate_all_rollouts"], 0.5)
        self.assertEqual(result["high_reward_failed_verification"], 1)
        self.assertGreater(result["mean_heldout_relative_rmse"], 0)
        self.assertNotIn("heldout_survey", json.dumps(records[0]["task"]))

    def test_policy_comparison_requires_identical_heldout_cases(self) -> None:
        report = {
            "cases": {"a": {"case_sha256": "same"}},
            "rollouts": 1,
            "high_reward_threshold": 0.5,
            "mean_heldout_relative_rmse": 2.0,
            "reward_hacking_rate_all_rollouts": 0.2,
        }
        other = copy.deepcopy(report)
        other["mean_heldout_relative_rmse"] = 1.0
        self.assertEqual(compare(report, other)["relative_prediction_error_reduction"], 0.5)
        other["cases"]["a"]["case_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "exact same"):
            compare(report, other)

    def test_invalid_model_is_counted_without_favorable_error_selection(self) -> None:
        task, truth = synthetic_task()
        record = {
            "trace_id": "invalid",
            "evaluation_id": "case",
            "survey_id": "survey",
            "split": "test",
            "task": task_dict(task),
            "assigned_reward": 1.0,
            "heldout_survey": {
                "receivers_m": (task.receivers_m + 1).tolist(),
                "observations_microgal": task.observations_microgal.tolist(),
            },
            "steps": [
                {
                    "reasoning": "An invalid model",
                    "tool_call": {"name": "submit_model", "arguments": []},
                    "tool_result": {},
                }
            ],
        }
        local = Path(__file__).resolve().parents[1] / "outputs/test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            source = Path(directory) / "invalid.jsonl"
            write_jsonl(source, [record])
            report = evaluate(source)
        self.assertEqual(report["reward_hacking_rate_all_rollouts"], 1.0)
        self.assertEqual(report["valid_prediction_rollouts"], 0)
        self.assertIsNone(report["mean_heldout_relative_rmse"])

    def test_memory_comparison_requires_matching_workload(self) -> None:
        first = {
            "workload": {"model": "test", "tokens": 1024, "batch_size": 4},
            "peak_allocated_bytes_per_device": [100, 100],
        }
        second = copy.deepcopy(first)
        second["peak_allocated_bytes_per_device"] = [75, 75]
        self.assertEqual(compare_memory(first, second)["relative_peak_memory_reduction"], 0.25)
        second["workload"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "must match"):
            compare_memory(first, second)


if __name__ == "__main__":
    unittest.main()
