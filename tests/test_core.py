import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geotrace.data import generate, label_records, read_records, write_jsonl
from geotrace.grpo import clipped_objective, group_relative_advantages
from geotrace.physics import forward, synthetic_task
from geotrace.process_rewards import (
    exact_completion_probability,
    first_irrecoverable_step,
    monte_carlo_label,
    verify_tool_output,
)
from geotrace.verl_reward import compute_score


class PhysicsTests(unittest.TestCase):
    def test_symmetry_scaling_and_depth_decay(self) -> None:
        p = forward([-10, 0, 10], source_x_m=0, depth_m=100, mass_kg=1e10)
        self.assertAlmostEqual(p[0], p[2])
        np.testing.assert_allclose(
            forward([-10, 0, 10], source_x_m=0, depth_m=100, mass_kg=2e10), 2 * p
        )
        deeper = forward([0], source_x_m=0, depth_m=200, mass_kg=1e10)
        self.assertAlmostEqual(deeper[0], p[1] / 4)

    def test_physics_verifier_does_not_accept_forged_low_signal(self) -> None:
        task, truth = synthetic_task()
        self.assertTrue(verify_tool_output(task, truth, task.prediction(truth)))
        self.assertFalse(verify_tool_output(task, truth, np.zeros(25)))
        self.assertFalse(verify_tool_output(task, truth, [np.nan] * 25))

    def test_negative_depth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            forward([0], source_x_m=0, depth_m=-1, mass_kg=1)


class LabelTests(unittest.TestCase):
    def test_monte_carlo_agrees_with_small_exact_grid(self) -> None:
        task, _ = synthetic_task()
        prefix = {"x_m": 0, "depth_m": 150}
        exact = exact_completion_probability(task, prefix)
        sampled = monte_carlo_label(task, prefix, samples=3000, rng=np.random.default_rng(17))
        self.assertAlmostEqual(sampled.probability, exact, delta=0.03)
        self.assertLess(sampled.wilson_lower, exact)
        self.assertGreater(sampled.wilson_upper, exact)

    def test_first_error_requires_irrecoverable_prefix(self) -> None:
        task, truth = synthetic_task()
        self.assertIsNone(first_irrecoverable_step(task, list(truth.items())))
        self.assertEqual(
            first_irrecoverable_step(task, [("x_m", 100), ("depth_m", 150), ("mass_kg", 1e10)]), 0
        )
        with self.assertRaises(ValueError):
            first_irrecoverable_step(task, [("x_m", 0), ("x_m", 50)])


class ObjectiveTests(unittest.TestCase):
    def test_equal_reward_group_has_zero_advantage(self) -> None:
        np.testing.assert_array_equal(group_relative_advantages([[2, 2, 2]]), [[0, 0, 0]])

    def test_clipping_and_padding(self) -> None:
        old = np.log(np.full((2, 2), 0.1))
        current = old + np.log([[2.0, 9.0], [0.1, 0.1]])
        result = clipped_objective(current, old, current, [1, -1], [[1, 0], [1, 1]], kl_beta=0)
        # Positive advantage capped at 1.2, negative advantage capped at -0.8.
        self.assertAlmostEqual(result["objective"], (1.2 - 0.8) / 2)
        self.assertEqual(result["active_tokens"], 3)

    def test_kl_penalty_reduces_objective(self) -> None:
        old = np.log(np.full((2, 2), 0.1))
        current = old + 0.2
        low = clipped_objective(current, old, old, [1, -1], np.ones((2, 2)), kl_beta=0)
        high = clipped_objective(current, old, old, [1, -1], np.ones((2, 2)), kl_beta=1)
        self.assertGreater(high["mean_sampled_kl"], 0)
        self.assertLess(high["objective"], low["objective"])


class DataAndRewardTests(unittest.TestCase):
    def test_group_leakage_rejected(self) -> None:
        records = generate(10)
        records[1]["split"] = "train"
        local = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            path = Path(directory) / "traces.jsonl"
            write_jsonl(path, records)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_records(path)

    def test_label_pipeline_has_observable_features_and_provenance(self) -> None:
        records = label_records(generate(10), samples=4)
        self.assertEqual(len(records), 40)
        self.assertEqual(records[0]["provenance"]["kind"], "synthetic_fixture")
        self.assertEqual(len(records[0]["labeled_steps"][0]["features"]), 11)

    def test_verl_reward_rejects_malformed_and_forged_results(self) -> None:
        from geotrace.data import task_dict

        task, truth = synthetic_task()
        metadata = {"task": task_dict(task)}
        valid = json.dumps(
            {
                "assignments": list(truth.items()),
                "claimed_prediction_microgal": task.prediction(truth).tolist(),
            }
        )
        self.assertGreater(compute_score("geotrace", valid, "", metadata)["score"], 0.5)
        self.assertEqual(compute_score("geotrace", "not json", "", metadata)["score"], -1)
        forged = json.dumps(
            {"assignments": list(truth.items()), "claimed_prediction_microgal": [0] * 25}
        )
        self.assertEqual(compute_score("geotrace", forged, "", metadata)["score"], -1)


if __name__ == "__main__":
    unittest.main()
