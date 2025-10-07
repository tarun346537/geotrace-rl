"""Model trace collection and distributed labeling command entry points."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

from .config import RuntimeConfig
from .data import read_records, write_jsonl
from .logging_config import configure_logging
from .model_client import VLLMClient
from .rollouts import _label_remote, label_trace, rollout

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Collect tool traces or label their model-policy continuations."""
    configure_logging()
    cfg = RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "label"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--endpoint", default=cfg.transport.endpoint)
        command.add_argument("--model", required=True)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--max-turns", type=int, default=8)
    collect = sub.choices["collect"]
    collect.add_argument("--target-steps", type=int, default=128)
    collect.add_argument("--prm")
    collect.add_argument("--prune-threshold", type=float, default=0.05)
    label = sub.choices["label"]
    label.add_argument("--samples", type=int, default=16)
    label.add_argument("--ray-workers", type=int, default=0)
    label.add_argument("--ray-address", default="auto")
    args = parser.parse_args()
    client = VLLMClient(args.endpoint, args.model)
    if args.command == "collect":
        if args.target_steps < 1:
            parser.error("--target-steps must be positive")
        source = read_records(args.input)
        cases = list({r["survey_id"]: r for r in source}.values())
        scorer = None
        if args.prm:
            from .text_prm import TextRewardScorer

            scorer = TextRewardScorer(args.prm)
        output, count, index, empty_invalid = [], 0, 0, 0
        while count < args.target_steps:
            if index >= max(100, args.target_steps * 4):
                raise RuntimeError(
                    "too many empty/invalid rollouts to reach the requested step count"
                )
            case = cases[index % len(cases)]
            generated = rollout(
                case["task"],
                client,
                max_turns=args.max_turns,
                seed=args.seed + index * 1009,
                scorer=scorer,
                prune_threshold=args.prune_threshold,
            )
            if not generated["steps"]:
                index += 1
                empty_invalid += 1
                continue
            output.append(
                {
                    "trace_id": f"{case['survey_id']}-rollout-{index}",
                    "survey_id": case["survey_id"],
                    "split": case["split"],
                    "task": case["task"],
                    **generated,
                    "provenance": {
                        "kind": "model_tool_rollout",
                        "model": args.model,
                        "survey": case["provenance"],
                        "seed": args.seed + index * 1009,
                    },
                }
            )
            count += len(generated["steps"])
            index += 1
    else:
        from .text_prm import load_traces

        traces = load_traces(args.input)
        if args.ray_workers:
            if args.ray_workers < 1:
                parser.error("--ray-workers must be nonnegative")
            ray = importlib.import_module("ray")
            ray.init(address=None if args.ray_address == "local" else args.ray_address)
            task = ray.remote(num_cpus=1)(_label_remote)
            output, pending = [], []
            try:
                for index, trace in enumerate(traces):
                    pending.append(
                        task.remote(
                            trace,
                            args.endpoint,
                            args.model,
                            args.samples,
                            args.seed + index * 100003,
                            args.max_turns,
                        )
                    )
                    if len(pending) >= args.ray_workers:
                        output.extend(ray.get(pending))
                        pending = []
                output.extend(ray.get(pending))
            finally:
                ray.shutdown()
        else:
            output = [
                label_trace(
                    trace,
                    client,
                    samples=args.samples,
                    seed=args.seed + index * 100003,
                    max_turns=args.max_turns,
                )
                for index, trace in enumerate(traces)
            ]
    write_jsonl(args.output, output)
    summary = {
        "traces": len(output),
        "steps": sum(len(trace["steps"]) for trace in output),
        "output": str(args.output),
    }
    if args.command == "collect":
        summary.update(attempted_rollouts=index, empty_invalid_rollouts=empty_invalid)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("rollout_records_written", extra={"metrics": summary})


if __name__ == "__main__":
    main()
