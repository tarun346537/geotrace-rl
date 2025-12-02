"""Native FSDP checkpoint layout and integrity contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geotrace.ops.context import WorkerContext
from geotrace.ops.verl_checkpoint import (
    bind_native_directory,
    native_inventory,
    publish_native,
    restore_native,
)
from geotrace.verl_driver import configure_resume


def write_native(root: Path, step: int, world_size: int = 2, critic: bool = False) -> Path:
    """Build native checkpoint file-layout fixtures for connector tests."""
    folder = root / f"global_step_{step}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.pt").write_bytes(b"dataloader-state")
    for role in ("actor", "critic") if critic else ("actor",):
        target = folder / role
        target.mkdir(exist_ok=True)
        (target / "fsdp_config.json").write_text(json.dumps({"world_size": world_size}))
        for rank in range(world_size):
            for name in ("model", "optim", "extra_state"):
                (target / f"{name}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"state")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return folder


def native_config(root: Path) -> SimpleNamespace:
    role = SimpleNamespace(
        strategy="fsdp", checkpoint=SimpleNamespace(save_contents=["model", "optimizer", "extra"])
    )
    return SimpleNamespace(
        trainer=SimpleNamespace(
            default_local_dir=str(root),
            save_freq=1,
            del_local_ckpt_after_load=False,
            resume_mode="auto",
            resume_from_path=None,
        ),
        actor_rollout_ref=SimpleNamespace(actor=role),
        critic=role,
    )


def test_partial_newer_native_save_never_replaces_committed_resume(tmp_path: Path) -> None:
    context = WorkerContext("run", 1, str(tmp_path / "run"))
    root = tmp_path / "native"
    bind_native_directory(root, context)
    expected = write_native(root, 3)
    publish_native(context, root, 3, 2, False)
    (root / "global_step_4").mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("4")
    config = native_config(root)
    configure_resume(config, WorkerContext("run", 2, context.directory))
    assert config.trainer.resume_mode == "resume_path"
    assert config.trainer.resume_from_path == str(expected)


def test_new_managed_run_disables_implicit_native_auto_resume(tmp_path: Path) -> None:
    context = WorkerContext("run", 1, str(tmp_path / "run"))
    config = native_config(tmp_path / "native")
    configure_resume(config, context)
    assert config.trainer.resume_mode == "disable"


def test_missing_optimizer_rank_prevents_checkpoint_commit(tmp_path: Path) -> None:
    context = WorkerContext("run", 1, str(tmp_path / "run"))
    folder = write_native(tmp_path / "native", 1, critic=True)
    (folder / "critic/optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(FileNotFoundError):
        publish_native(context, folder.parent, 1, 2, True)
    assert context.restore() is None


def test_native_resume_rejects_changed_file_sizes_and_topology(tmp_path: Path) -> None:
    context = WorkerContext("run", 1, str(tmp_path / "run"))
    folder = write_native(tmp_path / "native", 1)
    publish_native(context, folder.parent, 1, 2, False)
    with pytest.raises(ValueError, match="world size differs"):
        native_inventory(folder, 8, False)
    (folder / "data.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="file sizes changed"):
        restore_native(context)


def test_one_native_directory_cannot_mix_run_ids(tmp_path: Path) -> None:
    root = tmp_path / "native"
    bind_native_directory(root, WorkerContext("grpo", 1, str(tmp_path / "grpo")))
    with pytest.raises(ValueError, match="another run"):
        bind_native_directory(root, WorkerContext("ppo", 1, str(tmp_path / "ppo")))


def test_unmanaged_checkpoint_requires_an_explicit_migration(tmp_path: Path) -> None:
    root = tmp_path / "native"
    write_native(root, 1)
    with pytest.raises(ValueError, match="unmanaged native"):
        bind_native_directory(root, WorkerContext("run", 1, str(tmp_path / "run")))
