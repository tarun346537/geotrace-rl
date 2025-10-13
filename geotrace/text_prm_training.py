"""Transformer process-reward training and validation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset
from numpy.typing import NDArray
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from .prm_provenance import survey_provenance
from .prm_settings import Namespace, defaults
from .text_prm_data import Trace, examples, first_error_metrics, load_traces

LOGGER = logging.getLogger(__name__)


class TrainingLog(TrainerCallback):
    """Write trainer events through the project's structured log formatter."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        """Record actual optimizer metrics with the global checkpoint step."""
        LOGGER.info("text_prm_step", extra={"step": state.global_step, "metrics": logs or {}})


@contextmanager
def _tracking(project: str | None, run_id: str | None) -> Iterator[None]:
    run = None
    if project:
        import wandb

        run = wandb.init(project=project, id=run_id, resume="allow")
    try:
        yield
    finally:
        if run is not None:
            run.finish()


def _probabilities(raw: object) -> NDArray[np.float64]:
    value = raw[0] if isinstance(raw, tuple) else raw
    logits = np.asarray(value, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 1 or not np.isfinite(logits).all():
        raise ValueError("invalid PRM logits. configure a single finite classification output")
    return np.asarray(1 / (1 + np.exp(-np.clip(logits[:, 0], -80.0, 80.0))), dtype=float)


def _metrics(prediction: EvalPrediction) -> dict[str, float]:
    targets = np.asarray(prediction.label_ids, dtype=float)[:, 0]
    return {
        "soft_target_mse": float(np.mean((_probabilities(prediction.predictions) - targets) ** 2))
    }


def _first_error_report(
    trainer: Trainer, datasets: dict[str, Dataset], traces: list[Trace], test_logits: object
) -> Trace:
    validation = [trace for trace in traces if trace["split"] == "validation"]
    test = [trace for trace in traces if trace["split"] == "test"]
    if any("error_label" not in step for trace in validation + test for step in trace["steps"]):
        return {}
    cfg = defaults("training")
    prediction = trainer.predict(datasets["validation"])
    scores = _probabilities(prediction.predictions)
    thresholds = [
        float(value)
        for value in np.linspace(cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_steps"])
    ]
    threshold = max(
        thresholds,
        key=lambda value: first_error_metrics(validation, scores, value)["first_error_f1"],
    )
    return {
        "first_error_threshold": threshold,
        "first_error_threshold_selected_on": "validation",
        "first_error_test": first_error_metrics(test, _probabilities(test_logits), threshold),
    }


def train(args: Namespace) -> Trace:
    """Fit MC soft targets on survey-disjoint textual prefixes.

    Checkpoints retain the trainer's optimizer, scheduler and RNG state. The
    exported classifier uses the best validation checkpoint and fixed context.
    """
    cfg = defaults("training")
    patience = int(getattr(args, "patience", cfg["early_stopping_patience"]))
    if (
        min(args.epochs, args.batch_size, args.checkpoint_steps, patience) <= 0
        or args.max_length < 32
        or args.lr <= 0
    ):
        raise ValueError(
            "invalid training flags. sizes, patience and learning rate must be positive"
        )
    input_path, output = Path(args.input), Path(args.output)
    set_seed(int(args.seed))
    traces = load_traces(input_path)
    provenance = survey_provenance(traces)
    rows = examples(traces)
    if any(not value for value in rows.values()):
        raise ValueError(
            "empty survey split. supply independent train, validation and test surveys"
        )
    tokenizer_loader = cast(Callable[..., PreTrainedTokenizerBase], AutoTokenizer.from_pretrained)
    tokenizer = tokenizer_loader(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no padding or EOS token. select a supported base model")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.init_kwargs["truncation_side"] = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.freeze_backbone:
        for parameter in model.base_model.parameters():
            parameter.requires_grad_(False)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    metadata = {
        "dataset_sha256": digest,
        "base_model": str(args.model),
        "seed": int(args.seed),
        "max_length": int(args.max_length),
        "freeze_backbone": bool(args.freeze_backbone),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "patience": patience,
    }
    if args.resume:
        previous_path = Path(args.resume).parent / "run_metadata.json"
        if (
            not previous_path.is_file()
            or json.loads(previous_path.read_text(encoding="utf-8")) != metadata
        ):
            raise ValueError(
                "resume metadata differs. use the same dataset, base model and training settings"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    datasets: dict[str, Dataset] = {}

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return dict(tokenizer(batch["text"], truncation=True, max_length=int(args.max_length)))

    for split, records in rows.items():
        datasets[split] = Dataset.from_list(
            [{key: row[key] for key in ("text", "labels")} for row in records]
        ).map(_tokenize, batched=True, remove_columns=["text"])
    project = getattr(args, "wandb_project", None)
    settings = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        eval_strategy="steps",
        eval_steps=int(args.checkpoint_steps),
        save_strategy="steps",
        save_steps=int(args.checkpoint_steps),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=int(args.seed),
        data_seed=int(args.seed),
        use_cpu=bool(args.cpu),
        bf16=not args.cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=["wandb"] if project else [],
        logging_steps=1,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    with _tracking(project, getattr(args, "run_id", None)):
        trainer = Trainer(
            model=model,
            args=settings,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), TrainingLog()],
        )
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        trainer.save_model(str(output / "model"))
        tokenizer.save_pretrained(output / "model")
        prediction = trainer.predict(datasets["test"])
        report: Trace = {
            "kind": "text_prm_training_run",
            **metadata,
            "prefixes": {key: len(value) for key, value in rows.items()},
            "optimizer_steps": trainer.state.global_step,
            "test": {key: float(value) for key, value in (prediction.metrics or {}).items()},
        }
        report.update(_first_error_report(trainer, datasets, traces, prediction.predictions))
        (output / "model/prm_config.json").write_text(
            json.dumps(
                {
                    "max_length": int(args.max_length),
                    "truncation_side": "left",
                    "input_schema_version": 1,
                    "renderer": "survey_tool_prefix_v1",
                    "survey_provenance": provenance,
                    "dataset_sha256": digest,
                    "first_error_threshold": report.get("first_error_threshold"),
                    "first_error_threshold_selected_on": report.get(
                        "first_error_threshold_selected_on"
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("text_prm_training_complete", extra={"metrics": report})
    return report
