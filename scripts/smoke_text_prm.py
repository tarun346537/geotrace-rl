"""Train/export/reload a tiny local transformer PRM without model downloads."""

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast


def main() -> None:
    """Train and reload a tiny local transformer to check the PRM interfaces."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )

    from geotrace.data import generate, parse_task, write_jsonl
    from geotrace.logging_config import configure_logging
    from geotrace.rollouts import execute_tool
    from geotrace.text_prm import TextRewardScorer

    configure_logging()

    root = Path("outputs/text-prm-smoke")
    model_dir = root / "tiny-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "Gravity",
                "survey",
                "reasoning",
                "model",
                "depth",
                "mass",
                "test",
                "source",
                "submit",
                "x_m",
                "depth_m",
                "mass_kg",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_factory = cast(Callable[..., PreTrainedTokenizerBase], PreTrainedTokenizerFast)
    tokenizer = tokenizer_factory(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(model_dir)
    config_factory = cast(Callable[..., BertConfig], BertConfig)
    configuration = config_factory(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    configuration.num_labels = 1
    configuration.problem_type = "multi_label_classification"
    model_factory = cast(Callable[[BertConfig], PreTrainedModel], BertForSequenceClassification)
    model = model_factory(configuration)
    model.save_pretrained(model_dir)
    records, counts = [], {"train": 0, "validation": 0, "test": 0}
    for record in generate(10):
        split = record["split"]
        if not record["trace_id"].endswith("-0") or counts[split] >= 2:
            continue
        task = parse_task(record["task"])
        arguments = {
            "reasoning": "Test source model against observations",
            "model": dict(record["assignments"]),
        }
        steps = [
            {
                "reasoning": arguments["reasoning"],
                "tool_call": {"name": name, "arguments": arguments},
                "tool_result": execute_tool(task, name, arguments),
                "mc_label": {"probability": 1.0, "samples": 1, "successes": 1},
            }
            for name in ("gravity_forward", "submit_model")
        ]
        records.append(
            {
                "trace_id": record["trace_id"],
                "survey_id": record["survey_id"],
                "split": split,
                "task": record["task"],
                "steps": steps,
                "provenance": {"kind": "controlled_training_smoke_fixture"},
            }
        )
        counts[split] += 1
    source = root / "labeled-traces.jsonl"
    write_jsonl(source, records)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "geotrace.text_prm",
            "--input",
            str(source),
            "--model",
            str(model_dir),
            "--output",
            str(root / "trained"),
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--checkpoint-steps",
            "1",
            "--batch-size",
            "2",
            "--max-length",
            "128",
            "--cpu",
        ],
        check=True,
    )
    scorer = TextRewardScorer(root / "trained/model", max_length=128)
    if scorer.tokenizer.truncation_side != "left" or scorer.max_length != 128:
        raise RuntimeError("reloaded inference context differs from training")
    probabilities = scorer.probabilities(records[0]["task"], records[0]["steps"])
    if len(probabilities) != 2 or not all(0 <= value <= 1 for value in probabilities):
        raise RuntimeError("exported transformer did not produce per-step probabilities")
    (root / "validation.json").write_text(
        json.dumps(
            {
                "kind": "tiny_transformer_prm_cpu_smoke",
                "optimizer_steps": 2,
                "exported_model_reloaded": True,
                "per_step_probabilities": probabilities,
                "model_quality_evaluated": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.getLogger(__name__).info("transformer_prm_check_passed")


if __name__ == "__main__":
    main()
