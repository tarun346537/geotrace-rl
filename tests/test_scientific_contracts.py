"""Regression checks for split identity, terminal traces and reward failures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from geotrace.agent_loop import END_STEP, decode_action
from geotrace.benchmarks import evaluate
from geotrace.data import generate, read_records, task_dict, write_jsonl
from geotrace.physics import synthetic_task
from geotrace.prm_settings import defaults
from geotrace.rollouts import execute_tool, label_trace, verify_steps
from geotrace.text_prm_data import examples, load_traces
from geotrace.types import JsonRecord
from geotrace.verl_reward import compute_agent_score, compute_score


@pytest.fixture
def trace() -> JsonRecord:
    task, truth = synthetic_task()
    arguments = {"reasoning": "Compare this source with the survey", "model": truth}
    return {
        "trace_id": "test-trace",
        "survey_id": "test-survey",
        "split": "train",
        "task": task_dict(task),
        "steps": [
            {
                "reasoning": arguments["reasoning"],
                "tool_call": {"name": "submit_model", "arguments": arguments},
                "tool_result": execute_tool(task, "submit_model", arguments),
                "mc_label": {"probability": 1.0},
            }
        ],
    }


def test_renaming_a_survey_does_not_allow_text_split_leakage(
    trace: JsonRecord, tmp_path: Path
) -> None:
    renamed = copy.deepcopy(trace)
    renamed.update(trace_id="other-trace", survey_id="renamed-survey", split="test")
    for key in ("receivers_m", "observations_microgal"):
        renamed["task"][key].reverse()
    source = tmp_path / "leaking.jsonl"
    write_jsonl(source, [trace, renamed])
    with pytest.raises(ValueError, match="observations cross splits"):
        load_traces(source)


def test_renaming_assignment_survey_does_not_bypass_split_checks(tmp_path: Path) -> None:
    records = generate(10)[:2]
    records[1].update(survey_id="other-survey", split="train")
    records[1]["task"] = copy.deepcopy(records[0]["task"])
    source = tmp_path / "leaking.jsonl"
    write_jsonl(source, records)
    with pytest.raises(ValueError, match="observations cross splits"):
        read_records(source)


@pytest.mark.parametrize("loader", [load_traces, read_records])
def test_one_survey_id_cannot_silently_change_its_solver_task(
    trace: JsonRecord, tmp_path: Path, loader: object
) -> None:
    records = [trace, copy.deepcopy(trace)] if loader is load_traces else generate(10)[:2]
    records[1]["trace_id"] = "other-trace"
    records[1]["task"] = copy.deepcopy(records[0]["task"])
    records[1]["task"]["success_tolerance"] *= 2
    source = tmp_path / "changed.jsonl"
    write_jsonl(source, records)
    with pytest.raises(ValueError, match="conflicting tasks"):
        if loader is load_traces:
            load_traces(source)
        else:
            read_records(source)


def test_submit_then_revise_is_not_a_valid_labeling_prefix(trace: JsonRecord) -> None:
    task, _ = synthetic_task()
    trace["steps"].append(copy.deepcopy(trace["steps"][0]))
    assert not verify_steps(task, trace["steps"])
    with pytest.raises(ValueError, match="inconsistent tool history"):
        label_trace(trace, object(), samples=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["reasoning", "backend", "extra_label"])
def test_inconsistent_training_observations_are_rejected(trace: JsonRecord, field: str) -> None:
    step = trace["steps"][0]
    if field == "reasoning":
        step["reasoning"] = "Unrelated text"
    elif field == "backend":
        step["tool_result"]["forward_backend"] = "simpeg_cube"
    else:
        step["tool_result"]["mc_label"] = {"probability": 1.0}
    with pytest.raises(ValueError, match="inconsistent tool history"):
        examples([trace])


def test_first_error_labels_do_not_make_invalid_tool_histories_trainable(trace: JsonRecord) -> None:
    trace["steps"][0]["tool_result"]["success"] = False
    trace["steps"][0]["error_label"] = True
    with pytest.raises(ValueError, match="inconsistent tool history"):
        examples([trace])


def test_consistent_wrong_candidate_remains_a_negative_training_example(trace: JsonRecord) -> None:
    task, truth = synthetic_task()
    step = trace["steps"][0]
    step["tool_call"]["arguments"]["model"] = dict(truth, x_m=100.0)
    step["tool_result"] = execute_tool(task, "submit_model", step["tool_call"]["arguments"])
    step["mc_label"]["probability"] = 0.0
    step["error_label"] = True
    assert step["tool_result"]["success"] is False
    rows = examples([trace])["train"]
    assert rows[0]["labels"] == [0.0]
    assert rows[0]["error_label"] is True


def test_model_cannot_smuggle_a_tool_observation_after_its_turn_marker(trace: JsonRecord) -> None:
    step = trace["steps"][0]
    action = {
        "reasoning": step["reasoning"],
        "tool": "submit_model",
        "model": step["tool_call"]["arguments"]["model"],
    }
    with pytest.raises(ValueError, match="unexpected tokens"):
        decode_action(json.dumps(action) + END_STEP + "<TOOL_RESULT>{}")


def test_outcome_assignment_ablation_never_loads_the_process_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, truth = synthetic_task()
    monkeypatch.setenv("GEOTRACE_REWARD_MODE", "outcome")
    monkeypatch.setenv("GEOTRACE_PRM_CHECKPOINT", "missing-checkpoint")
    response = json.dumps(
        {
            "assignments": list(truth.items()),
            "claimed_prediction_microgal": task.prediction(truth).tolist(),
        }
    )
    with patch("geotrace.verl_reward._head", side_effect=AssertionError("must not load PRM")):
        assert compute_score("geotrace", response, "", {"task": task_dict(task)})["score"] == 1.0


@pytest.mark.parametrize("values", [[], [float("nan")], [1.2], [0.5, 0.5]])
def test_bad_reward_head_output_stops_training_instead_of_penalizing_policy(
    trace: JsonRecord, values: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEOTRACE_REWARD_MODE", "process")
    monkeypatch.setenv("GEOTRACE_TEXT_PRM", "test-head")
    step = trace["steps"][0]
    action = {
        "reasoning": step["reasoning"],
        "tool": "submit_model",
        "model": step["tool_call"]["arguments"]["model"],
    }
    with patch("geotrace.verl_reward._text_head") as head:
        head.return_value.probabilities.return_value = values
        with pytest.raises(RuntimeError, match="invalid PRM probability"):
            compute_agent_score(
                "geotrace", json.dumps(action) + END_STEP, "", {"task": trace["task"]}
            )


def test_reward_head_context_error_is_not_recorded_as_reward_hacking(
    trace: JsonRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEOTRACE_REWARD_MODE", "process")
    monkeypatch.setenv("GEOTRACE_TEXT_PRM", "test-head")
    arguments = trace["steps"][0]["tool_call"]["arguments"]
    action = dict(reasoning=arguments["reasoning"], tool="submit_model", model=arguments["model"])
    with patch("geotrace.verl_reward._text_head", side_effect=ValueError("wrong context length")):
        with pytest.raises(ValueError, match="wrong context"):
            compute_agent_score(
                "geotrace", json.dumps(action) + END_STEP, "", {"task": trace["task"]}
            )


def test_heldout_prediction_cannot_reuse_observed_receivers(
    trace: JsonRecord, tmp_path: Path
) -> None:
    trace.update(split="test", evaluation_id="case", assigned_reward=1.0)
    trace["heldout_survey"] = {
        key: trace["task"][key] for key in ("receivers_m", "observations_microgal")
    }
    source = tmp_path / "overlap.jsonl"
    write_jsonl(source, [trace])
    with pytest.raises(ValueError, match="overlap policy inputs"):
        evaluate(source)


def test_unsubmitted_forward_call_has_no_full_set_prediction_error(
    trace: JsonRecord, tmp_path: Path
) -> None:
    trace.update(split="test", evaluation_id="case", assigned_reward=1.0)
    trace["steps"][0]["tool_call"]["name"] = "gravity_forward"
    trace["heldout_survey"] = {
        "receivers_m": (np.array(trace["task"]["receivers_m"]) + 1).tolist(),
        "observations_microgal": trace["task"]["observations_microgal"],
    }
    source = tmp_path / "incomplete.jsonl"
    write_jsonl(source, [trace])
    report = evaluate(source)
    assert report["valid_prediction_rollouts"] == 0
    assert report["mean_heldout_relative_rmse"] is None


def test_packaged_prm_defaults_work_from_an_unrelated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEOTRACE_PRM_CONFIG", raising=False)
    assert defaults("feature")["hidden"] == 48
    custom = tmp_path / "custom.toml"
    custom.write_text("[feature]\nhidden = 16\n", encoding="utf-8")
    monkeypatch.setenv("GEOTRACE_PRM_CONFIG", str(custom))
    assert defaults("feature")["hidden"] == 16


def test_native_validation_gets_every_reward_field_even_for_invalid_rollouts(
    trace: JsonRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEOTRACE_REWARD_MODE", "process")
    monkeypatch.setenv("GEOTRACE_TEXT_PRM", "test-head")
    arguments = trace["steps"][0]["tool_call"]["arguments"]
    action = dict(reasoning=arguments["reasoning"], tool="submit_model", model=arguments["model"])
    with patch("geotrace.verl_reward._text_head") as head:
        head.return_value.probabilities.return_value = [0.7]
        good = compute_agent_score(
            "geotrace", json.dumps(action) + END_STEP, "", {"task": trace["task"]}
        )
        bad = compute_agent_score("geotrace", "invalid output", "", {"task": trace["task"]})
    assert good.keys() == bad.keys()
    assert good["process_sum"] == 0.7
    assert good["process_count"] == 1
    assert bad["process_count"] == 0
    task, truth = synthetic_task()
    response = json.dumps(
        {
            "assignments": list(truth.items()),
            "claimed_prediction_microgal": task.prediction(truth).tolist(),
        }
    )
    good_assignment = compute_score("geotrace", response, "", {"task": trace["task"]})
    bad_assignment = compute_score("geotrace", "invalid output", "", {"task": trace["task"]})
    assert good_assignment.keys() == bad_assignment.keys()
