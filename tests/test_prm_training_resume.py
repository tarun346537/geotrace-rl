"""Checkpoint regressions for the feature reward head."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from geotrace.data import generate, label_records, write_jsonl

torch = pytest.importorskip("torch")

train = pytest.importorskip("geotrace.train_prm").train


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    source = tmp_path / "labels.jsonl"
    write_jsonl(source, label_records(generate(10), samples=4))
    return source


def _args(source: Path, output: Path, **overrides: object) -> SimpleNamespace:
    fields = dict(
        input=source,
        output=output,
        epochs=1,
        hidden=12,
        batch_size=16,
        lr=0.003,
        seed=17,
        max_steps=0,
        resume=None,
        wandb_project=None,
        run_id="checkpoint-test",
        synthetic=True,
        patience=8,
        min_delta=1e-6,
    )
    return SimpleNamespace(**(fields | overrides))


def test_finished_checkpoint_can_export_to_another_directory(labels: Path, tmp_path: Path) -> None:
    original, exported = tmp_path / "original", tmp_path / "exported"
    first = train(_args(labels, original))
    second = train(_args(labels, exported, resume=original / "checkpoint.pt"))
    assert second["steps"] == first["steps"]
    assert second["test"] == first["test"]
    before = torch.load(original / "checkpoint.pt", weights_only=True)
    after = torch.load(exported / "checkpoint.pt", weights_only=True)
    for key, value in before["model"].items():
        torch.testing.assert_close(value, after["model"][key], rtol=0, atol=0)
    assert (exported / "best_model.pt").is_file()


def test_resume_rejects_changed_label_file_before_updating_weights(
    labels: Path, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    train(_args(labels, output, max_steps=1))
    saved = (output / "checkpoint.pt").read_bytes()
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resume settings differ"):
        train(_args(labels, output, resume=output / "checkpoint.pt"))
    assert (output / "checkpoint.pt").read_bytes() == saved


def test_early_stop_preserves_last_optimizer_and_best_inference_weights(
    labels: Path, tmp_path: Path
) -> None:
    output = tmp_path / "stopped"
    result = train(_args(labels, output, epochs=9, patience=1, min_delta=100.0))
    assert result["early_stopped"] is True
    assert result["completed_epochs"] == 2
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    best = torch.load(output / "best_model.pt", weights_only=True)
    assert checkpoint["stale_epochs"] == 1
    assert checkpoint["optimizer"]["state"]
    assert best["selection"] == "validation_soft_target_mse"
    for key, value in best["model"].items():
        torch.testing.assert_close(value, checkpoint["best_model"][key], rtol=0, atol=0)


def test_resume_refuses_changed_gradient_clipping_config(
    labels: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geotrace.prm_settings import DEFAULT_CONFIG

    output = tmp_path / "run"
    train(_args(labels, output, max_steps=1))
    config = tmp_path / "prm.toml"
    config.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace("grad_clip = 1.0", "grad_clip = 0.5"),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEOTRACE_PRM_CONFIG", str(config))
    with pytest.raises(ValueError, match="resume settings differ"):
        train(_args(labels, output, resume=output / "checkpoint.pt"))


@pytest.mark.parametrize(
    "field,value", [("lr", float("nan")), ("lr", float("inf")), ("min_delta", float("nan"))]
)
def test_nonfinite_training_flags_fail_before_a_checkpoint_is_written(
    field: str,
    value: float,
    labels: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="must be finite"):
        train(_args(labels, output, **{field: value}))
    assert not output.exists()
