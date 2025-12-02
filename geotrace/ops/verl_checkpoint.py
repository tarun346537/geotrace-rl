"""Bridge native FSDP checkpoints to the durable operations ledger."""

from __future__ import annotations

import json
from pathlib import Path

from ..types import JsonRecord
from .context import WorkerContext
from .io import _atomic, _json


def native_inventory(path: Path, world_size: int, use_critic: bool) -> dict[str, int]:
    """Check every native rank's model, optimizer and RNG/scheduler state."""
    required = [path / "data.pt"]
    for role in ("actor", "critic") if use_critic else ("actor",):
        metadata = json.loads((path / role / "fsdp_config.json").read_text(encoding="utf-8"))
        if metadata["world_size"] != world_size:
            raise ValueError("native checkpoint world size differs. restore on the same topology")
        required.append(path / role / "fsdp_config.json")
        for rank in range(world_size):
            required.extend(
                path / role / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                for kind in ("model", "optim", "extra_state")
            )
    sizes = {str(item.relative_to(path)): item.stat().st_size for item in required}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("native checkpoint has an empty state file. inspect the completed save")
    return sizes


def bind_native_directory(root: Path, context: WorkerContext) -> None:
    """Bind a checkpoint directory to one logical run before any native writes."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "geotrace-run.json"
    binding = {"run_id": context.run_id, "run_directory": str(Path(context.directory).resolve())}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError(
                "native checkpoint directory belongs to another run. choose a new path"
            )
    else:
        if any(root.glob("global_step_*")):
            raise ValueError(
                "unmanaged native checkpoints found. choose an empty managed directory"
            )
        _atomic(path, (_json(binding) + "\n").encode())


def publish_native(
    context: WorkerContext, root: Path, step: int, world_size: int, use_critic: bool
) -> JsonRecord:
    """Commit a small native-state reference after verl's save barrier completes."""
    path = root.resolve() / f"global_step_{step}"
    inventory = native_inventory(path, world_size, use_critic)
    value: JsonRecord = {
        "kind": "verl_fsdp_checkpoint",
        "path": str(path),
        "step": step,
        "world_size": world_size,
        "use_critic": use_critic,
        "files": inventory,
    }
    context.save_checkpoint(step, (_json(value) + "\n").encode())
    return value


def restore_native(context: WorkerContext) -> JsonRecord | None:
    """Resolve only ledger-committed native state, ignoring partial newer saves."""
    saved = context.restore()
    if saved is None:
        return None
    step, payload = saved
    value: JsonRecord = json.loads(payload)
    path = Path(value["path"])
    if value.get("kind") != "verl_fsdp_checkpoint" or value["step"] != step:
        raise ValueError("checkpoint is not a native verl reference. inspect the run directory")
    if path.name != f"global_step_{step}" or not path.is_absolute():
        raise ValueError("invalid native checkpoint path. inspect the committed reference")
    actual = native_inventory(path, value["world_size"], value["use_critic"])
    if actual != value["files"]:
        raise ValueError("native checkpoint file sizes changed. refusing restoration")
    return value
