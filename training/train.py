"""Train a LoRA adapter with masked prompts and supervised response tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from training.common import encode_example, file_hash, prompt_hash, read_jsonl


def tokenize_rows(
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    encoded = [encode_example(tokenizer, row, max_length) for row in rows]
    lengths = sorted(len(row["input_ids"]) for row in encoded)
    return encoded, {
        "examples": len(encoded),
        "min_tokens": lengths[0],
        "max_tokens": lengths[-1],
        "p50_tokens": lengths[int(0.50 * (len(lengths) - 1))],
        "p95_tokens": lengths[int(0.95 * (len(lengths) - 1))],
        "p99_tokens": lengths[int(0.99 * (len(lengths) - 1))],
        "total_tokens": sum(lengths),
        "supervised_tokens": sum(
            sum(label != -100 for label in row["labels"]) for row in encoded
        ),
    }


def check_splits(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    train_seeds = {row["metadata"]["seed"] for row in train}
    validation_seeds = {row["metadata"]["seed"] for row in validation}
    if train_seeds & validation_seeds:
        raise ValueError("Training and validation seeds overlap; rerun preparation")
    train_keys = {prompt_hash(row["prompt"]) for row in train}
    if any(prompt_hash(row["prompt"]) in train_keys for row in validation):
        raise ValueError("Training and validation prompts overlap; rerun preparation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("training/configs/qwen3-8b.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, help="Override for a short smoke run")
    parser.add_argument(
        "--checkpoint-after-seconds",
        type=float,
        default=0,
        help="Save and exit 85 after this interval; single-process jobs only (0 disables)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/tokenize data without loading model weights",
    )
    parser.add_argument(
        "--tokenization-report", type=Path, help="Save preflight token counts as JSON"
    )
    args = parser.parse_args()
    if args.checkpoint_after_seconds < 0:
        parser.error("--checkpoint-after-seconds must be nonnegative")
    config = json.loads(args.config.read_text())
    target_mode = config.get("target_mode", "action")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config.get("revision", "main")
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_file = args.data_dir / "train.jsonl"
    validation_file = args.data_dir / "validation.jsonl"
    train_rows = read_jsonl(train_file, target_mode)
    validation_rows = read_jsonl(validation_file, target_mode)
    check_splits(train_rows, validation_rows)
    train_data, train_stats = tokenize_rows(tokenizer, train_rows, config["max_length"])
    validation_data, validation_stats = tokenize_rows(
        tokenizer, validation_rows, config["max_length"]
    )
    stats = {"train": train_stats, "validation": validation_stats}
    print(json.dumps(stats, indent=2))
    if args.tokenization_report:
        args.tokenization_report.parent.mkdir(parents=True, exist_ok=True)
        args.tokenization_report.write_text(
            json.dumps(
                {
                    "target_mode": target_mode,
                    "max_length": config["max_length"],
                    "splits": stats,
                },
                indent=2,
            )
            + "\n"
        )
    if args.dry_run:
        return

    import torch
    from accelerate import PartialState
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    training_options = dict(config["training"])
    if args.output_dir is not None:
        training_options["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        training_options["max_steps"] = args.max_steps
    output_dir = Path(training_options["output_dir"])
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and args.resume_from_checkpoint is None
    ):
        raise ValueError(
            f"{output_dir} is not empty. Choose a new output directory or resume explicitly."
        )
    training_args = TrainingArguments(**training_options)
    if args.checkpoint_after_seconds and training_args.world_size != 1:
        raise ValueError("Exit-driven checkpointing requires a single-process job")
    # All ranks must finish checking the output directory before rank zero can
    # create it. Otherwise a slower rank could mistake this run for an old one.
    PartialState(cpu=training_args.use_cpu).wait_for_everyone()
    set_seed(training_args.seed)
    if training_args.bf16 and not torch.cuda.is_available():
        raise ValueError("The BF16 GPU configuration requires a CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=config.get("revision", "main"),
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            **config["lora"],
            task_type="CAUSAL_LM",
            revision=getattr(model.config, "_commit_hash", None)
            or config.get("revision", "main"),
        ),
    )
    model.print_trainable_parameters()
    callbacks = []
    if args.checkpoint_after_seconds:
        from training.checkpointing import ExitCheckpointCallback

        callbacks.append(ExitCheckpointCallback(args.checkpoint_after_seconds))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_data),
        eval_dataset=Dataset.from_list(validation_data),
        processing_class=tokenizer,
        callbacks=callbacks,
        # This collator preserves our -100 labels and masks padding too.
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
    )
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        run = {
            "config": config,
            "training_arguments": training_options,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "data_sha256": {
                "train": file_hash(train_file),
                "validation": file_hash(validation_file),
            },
            "tokenization": stats,
        }
        (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output_dir))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("validation", trainer.evaluate())
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
