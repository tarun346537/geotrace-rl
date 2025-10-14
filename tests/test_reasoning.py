import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geotrace.agent_loop import (
    END_STEP,
    TOOL_CLOSE,
    TOOL_OPEN,
    load_pruning_scorer,
    parse_protocol,
    run_token_loop,
)
from geotrace.data import task_dict
from geotrace.physics import synthetic_task
from geotrace.rollouts import RolloutError, execute_tool, label_trace, rollout, verify_steps
from geotrace.text_prm import first_error_metrics, inference_settings, prefix_text
from geotrace.types import JsonRecord
from geotrace.verl_reward import compute_agent_score


class Policy:
    model = "controlled-test-policy"

    def __init__(self, model: dict[str, float]) -> None:
        self.parameters, self.calls = model, 0

    def next_call(
        self, messages: list[JsonRecord], tools: list[JsonRecord], seed: int
    ) -> JsonRecord:
        self.calls += 1
        name = "gravity_forward" if len(messages) == 2 else "submit_model"
        return {
            "name": name,
            "id": f"call-{self.calls}",
            "arguments": {
                "reasoning": "Test this source against the survey.",
                "model": self.parameters,
            },
        }


class CharacterTokenizer:
    def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
        return [7, 8, 9]

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        return "".join(chr(token) for token in tokens)


class ReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task, self.truth = synthetic_task()
        self.task_value = task_dict(self.task)

    def test_actual_tool_loop_and_model_continuation_labels(self) -> None:
        policy = Policy(self.truth)
        generated = rollout(self.task_value, policy)
        self.assertTrue(generated["success"])
        self.assertEqual(len(generated["steps"]), 2)
        labeled = label_trace({"task": self.task_value, **generated}, policy, samples=5)
        self.assertTrue(all(step["mc_label"]["successes"] == 5 for step in labeled["steps"]))
        self.assertEqual(labeled["labeling"]["method"], "model_policy_monte_carlo")
        self.assertFalse(labeled["labeling"]["pruning_during_labeling"])

    def test_early_pruning_prevents_further_generation(self) -> None:
        class BadPrefix:
            def probabilities(self, task: JsonRecord, steps: list[JsonRecord]) -> list[float]:
                return [0.0] * len(steps)

        policy = Policy(self.truth)
        generated = rollout(self.task_value, policy, scorer=BadPrefix(), prune_threshold=0.1)
        self.assertEqual(generated["termination"], "process_reward_pruned")
        self.assertEqual(policy.calls, 1)

    def test_outcome_ablation_does_not_load_prm_even_when_checkpoint_is_exported(self) -> None:
        loaded = []

        def loader(path: str, device: str) -> object:
            loaded.append((path, device))
            return object()

        environment = {
            "GEOTRACE_REWARD_MODE": "outcome",
            "GEOTRACE_TEXT_PRM": "exported-prm",
            "GEOTRACE_PRM_DEVICE": "cuda",
        }
        self.assertIsNone(load_pruning_scorer(environment, loader))
        self.assertEqual(loaded, [])
        environment["GEOTRACE_REWARD_MODE"] = "process"
        self.assertIsNotNone(load_pruning_scorer(environment, loader))
        self.assertEqual(loaded, [("exported-prm", "cuda")])

    def test_invalid_mc_actions_count_as_failures_but_transport_errors_propagate(self) -> None:
        generated = rollout(self.task_value, Policy(self.truth))

        class BadModel(Policy):
            def next_call(
                self, messages: list[JsonRecord], tools: list[JsonRecord], seed: int
            ) -> JsonRecord:
                value = super().next_call(messages, tools, seed)
                value["arguments"]["model"] = dict(self.parameters, depth_m=-1)
                return value

        result = label_trace(
            {"task": self.task_value, **generated}, BadModel(self.truth), samples=3
        )
        self.assertEqual(result["steps"][0]["mc_label"]["successes"], 0)
        self.assertEqual(result["steps"][1]["mc_label"]["successes"], 3)

        class Offline(Policy):
            def next_call(self, *args: object) -> JsonRecord:
                raise RolloutError("controlled server outage")

        with self.assertRaises(RolloutError):
            label_trace({"task": self.task_value, **generated}, Offline(self.truth), samples=3)

    def test_physics_verifier_rejects_forged_success_and_prediction(self) -> None:
        generated = rollout(self.task_value, Policy(self.truth))
        self.assertTrue(verify_steps(self.task, generated["steps"]))
        generated["steps"][0]["tool_result"]["prediction_microgal"][0] += 1.0
        self.assertFalse(verify_steps(self.task, generated["steps"]))

    def test_text_inputs_do_not_contain_targets_or_error_annotations(self) -> None:
        generated = rollout(self.task_value, Policy(self.truth))
        generated["steps"][0]["mc_label"] = {"probability": 0.123456789}
        generated["steps"][0]["error_label"] = True
        text = prefix_text(self.task_value, generated["steps"])
        self.assertNotIn("mc_label", text)
        self.assertNotIn("error_label", text)
        self.assertNotIn(".123456789", text)

    def test_inference_reuses_training_context_contract(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs/test-temp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            Path(directory, "prm_config.json").write_text(
                json.dumps({"max_length": 128, "truncation_side": "left"})
            )
            self.assertEqual(inference_settings(directory)["max_length"], 128)
            with self.assertRaisesRegex(ValueError, "must match"):
                inference_settings(directory, 64)

    def test_wrong_first_error_location_counts_as_fp_and_fn(self) -> None:
        traces = [
            {"steps": [{"error_label": False}, {"error_label": True}]},
            {"steps": [{"error_label": False}, {"error_label": True}]},
        ]
        result = first_error_metrics(traces, [0.1, 0.1, 0.9, 0.1], 0.5)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["first_error_f1"], 0.5)

    def test_token_identity_and_tool_observation_loss_mask(self) -> None:
        tokenizer = CharacterTokenizer()
        actions = [
            {"reasoning": "Check source", "tool": "gravity_forward", "model": self.truth},
            {"reasoning": "Submit tested source", "tool": "submit_model", "model": self.truth},
        ]
        tokens = [tokenizer.encode(json.dumps(action) + END_STEP) for action in actions]
        prompts = []

        async def generate(prompt_ids: list[int], sampling: JsonRecord) -> list[int]:
            prompts.append(prompt_ids.copy())
            return tokens[len(prompts) - 1]

        result = asyncio.run(
            run_token_loop(tokenizer, generate, [], self.task_value, {}, response_length=10000)
        )
        self.assertEqual(result.termination, "submitted")
        self.assertEqual(
            [
                token
                for token, mask in zip(result.response_ids, result.response_mask, strict=True)
                if mask
            ],
            tokens[0] + tokens[1],
        )
        self.assertIn(0, result.response_mask)
        self.assertEqual(prompts[1][: 3 + len(tokens[0])], [7, 8, 9] + tokens[0])
        protocol = tokenizer.decode(result.response_ids)
        self.assertEqual(len(parse_protocol(self.task_value, protocol)), 2)
        with patch.dict(os.environ, {"GEOTRACE_REWARD_MODE": "outcome"}):
            self.assertEqual(
                compute_agent_score("geotrace", protocol, "", {"task": self.task_value})["score"],
                1.0,
            )

    def test_protocol_rejects_fabricated_tool_observation(self) -> None:
        action = {"reasoning": "Check source", "tool": "gravity_forward", "model": self.truth}
        observation = execute_tool(
            self.task, action["tool"], {"reasoning": action["reasoning"], "model": self.truth}
        )
        observation["success"] = False
        text = json.dumps(action) + END_STEP + TOOL_OPEN + json.dumps(observation) + TOOL_CLOSE
        with self.assertRaisesRegex(ValueError, "violates"):
            parse_protocol(self.task_value, text)

    def test_empty_followup_ends_on_model_token_so_ppo_keeps_penalty(self) -> None:
        tokenizer = CharacterTokenizer()
        action = {"reasoning": "Check source", "tool": "gravity_forward", "model": self.truth}
        generated = tokenizer.encode(json.dumps(action) + END_STEP)
        calls = []

        async def generate(prompt: list[int], sampling: JsonRecord) -> list[int]:
            calls.append(prompt)
            return generated if len(calls) == 1 else []

        result = asyncio.run(
            run_token_loop(tokenizer, generate, [], self.task_value, {}, response_length=10000)
        )
        self.assertEqual(result.termination, "empty_generation")
        self.assertEqual(result.response_ids, generated)
        self.assertEqual(result.response_mask[-1], 1)
        self.assertEqual(
            len(parse_protocol(self.task_value, tokenizer.decode(result.response_ids))), 1
        )


if __name__ == "__main__":
    unittest.main()
