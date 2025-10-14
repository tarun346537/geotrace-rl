"""Reject cross-file training leakage even after survey IDs are changed."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geotrace.data import generate, label_records, parse_task, write_jsonl
from geotrace.prm_provenance import survey_provenance, validate_heldout
from geotrace.rollouts import execute_tool
from geotrace.types import JsonRecord


def test_renaming_and_reordering_a_training_survey_does_not_make_it_heldout() -> None:
    records = generate(10)
    saved = survey_provenance(records)
    reused = copy.deepcopy(next(row for row in records if row["split"] == "train"))
    reused.update(survey_id="renamed", split="test")
    for key in ("receivers_m", "observations_microgal"):
        reused["task"][key].reverse()
    with pytest.raises(ValueError, match="overlaps another checkpoint split"):
        validate_heldout(survey_provenance([reused]), saved)


def test_original_test_and_new_surveys_remain_eligible() -> None:
    records = generate(10)
    saved = survey_provenance(records)
    validate_heldout(survey_provenance([row for row in records if row["split"] == "test"]), saved)
    fresh = generate(10, seed=93)
    validate_heldout(survey_provenance([row for row in fresh if row["split"] == "test"]), saved)


def test_calibration_surveys_cannot_be_relabelled_as_test_data() -> None:
    records = generate(10)
    saved = survey_provenance(records)
    reused = copy.deepcopy(next(row for row in records if row["split"] == "validation"))
    reused.update(survey_id="new-test", split="test")
    with pytest.raises(ValueError, match="overlaps"):
        validate_heldout(survey_provenance([reused]), saved)


def test_legacy_exports_cannot_claim_an_overlap_checked_evaluation() -> None:
    with pytest.raises(ValueError, match="lacks survey provenance"):
        validate_heldout(survey_provenance(generate(10)), None)


def test_feature_export_rejects_a_renamed_training_survey_in_a_separate_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from geotrace.evaluate_prm import evaluate_checkpoint
    from geotrace.train_prm import train

    records = label_records(generate(10), samples=2)
    source = tmp_path / "original.jsonl"
    write_jsonl(source, records)
    output = tmp_path / "head"
    train(
        SimpleNamespace(
            input=source,
            output=output,
            epochs=1,
            hidden=8,
            batch_size=16,
            lr=0.003,
            seed=17,
            max_steps=0,
            resume=None,
            wandb_project=None,
            run_id=None,
            synthetic=True,
        )
    )
    assert evaluate_checkpoint(source, output / "best_model.pt")["survey_overlap_checked"]
    changed = copy.deepcopy(records)
    first_train = next(row["survey_id"] for row in changed if row["split"] == "train")
    first_test = next(row["survey_id"] for row in changed if row["split"] == "test")
    for row in changed:
        if row["survey_id"] == first_train:
            row["split"] = "test"
        elif row["survey_id"] == first_test:
            row["split"] = "train"
        row["survey_id"] = "renamed-" + row["survey_id"]
        row["trace_id"] = "renamed-" + row["trace_id"]
    bad = tmp_path / "renamed.jsonl"
    write_jsonl(bad, changed)
    with pytest.raises(ValueError, match="overlaps another checkpoint split"):
        evaluate_checkpoint(bad, output / "best_model.pt")


def _text_fixture(tmp_path: Path) -> tuple[Path, Path, JsonRecord]:
    records = generate(10)
    row = next(item for item in records if item["split"] == "test")
    task = parse_task(row["task"])
    keys = list(task.grid)
    model = next(
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(task.grid[key] for key in keys))
        if not task.successful(dict(zip(keys, values, strict=True)))
    )
    steps = []
    for index, name in enumerate(("gravity_forward", "submit_model")):
        args = {"reasoning": "test candidate" if index == 0 else "submit candidate", "model": model}
        steps.append(
            {
                "reasoning": args["reasoning"],
                "tool_call": {"name": name, "arguments": args},
                "tool_result": execute_tool(task, name, args),
                "error_label": index == 1,
            }
        )
    trace = {
        "trace_id": "test-trace",
        "survey_id": row["survey_id"],
        "split": "test",
        "task": row["task"],
        "steps": steps,
    }
    source = tmp_path / "test.jsonl"
    write_jsonl(source, [trace])
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "max_length": 128,
        "truncation_side": "left",
        "dataset_sha256": "controlled-fixture",
        "survey_provenance": survey_provenance(records),
        "first_error_threshold": 0.5,
        "first_error_threshold_selected_on": "validation",
    }
    (model_dir / "prm_config.json").write_text(json.dumps(config), encoding="utf-8")
    return source, model_dir, config


def test_text_evaluator_uses_saved_threshold_and_independent_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geotrace import evaluate_text_prm

    class Scorer:
        """Controlled scores isolate threshold and held-out contracts."""

        def __init__(self, directory: object, device: str = "cpu") -> None:
            self.device = device

        def probabilities(self, task: object, steps: object) -> list[float]:
            return [0.9, 0.1]

    source, directory, _ = _text_fixture(tmp_path)
    monkeypatch.setattr(evaluate_text_prm, "TextRewardScorer", Scorer)
    report = evaluate_text_prm.evaluate_saved(source, directory)
    assert report["test"]["first_error_f1"] == 1
    assert report["threshold"] == 0.5
    assert report["survey_overlap_checked"]


def test_text_evaluator_cannot_invent_a_threshold_for_an_unannotated_export(tmp_path: Path) -> None:
    from geotrace.evaluate_text_prm import evaluate_saved

    source, directory, config = _text_fixture(tmp_path)
    config["first_error_threshold"] = None
    (directory / "prm_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-selected"):
        evaluate_saved(source, directory)
