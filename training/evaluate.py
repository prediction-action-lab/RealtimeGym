"""Evaluate generated actions against held-out demonstration labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from training.common import ACTIONS, encode_prompt, file_hash, prompt_hash, read_jsonl


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("No predictions to evaluate")

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(rows),
            "exact_action_accuracy": sum(
                row["prediction"] == row["target"] for row in rows
            )
            / len(rows),
            "invalid_action_rate": sum(
                row["prediction"] not in ACTIONS[row["game"]] for row in rows
            )
            / len(rows),
            "confusion": dict(
                Counter(f"{row['target']} -> {row['prediction']}" for row in rows)
            ),
        }

    by_game = {
        game: metrics([row for row in predictions if row["game"] == game])
        for game in sorted({row["game"] for row in predictions})
    }
    return {
        "overall": metrics(predictions),
        "by_game": by_game,
        "macro_game_accuracy": sum(
            result["exact_action_accuracy"] for result in by_game.values()
        )
        / len(by_game),
        "by_game_and_difficulty": {
            f"{game}/{difficulty}": metrics(
                [
                    row
                    for row in predictions
                    if (row["game"], row["difficulty"]) == (game, difficulty)
                ]
            )
            for game, difficulty in sorted(
                {(row["game"], row["difficulty"]) for row in predictions}
            )
        },
    }


def majority_predictions(train_rows: list[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        counts[row["metadata"]["game"]][row["completion"]] += 1
    return {game: counts[game].most_common(1)[0][0] for game in counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sft/test.jsonl"))
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/sft/train.jsonl"),
        help="Used only for a majority-action baseline",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--revision", default="b968826d9c46dd6066d109eabc6255188de91218"
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="Debug only; reported explicitly in metrics"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="For small-model smoke tests"
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.max_new_tokens < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Batch size, max-new-tokens, and limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    rows = read_jsonl(args.data)
    train_rows = read_jsonl(args.train_data)
    if {row["metadata"]["seed"] for row in rows} & {
        row["metadata"]["seed"] for row in train_rows
    }:
        raise ValueError("Evaluation seeds overlap training data")
    train_keys = {prompt_hash(row["prompt"]) for row in train_rows}
    if any(prompt_hash(row["prompt"]) in train_keys for row in rows):
        raise ValueError("Evaluation prompts overlap training data")
    if args.limit:
        rows = rows[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_name, revision = args.model, args.revision
    if args.adapter:
        from peft import PeftConfig

        adapter_config = PeftConfig.from_pretrained(args.adapter)
        model_name = adapter_config.base_model_name_or_path
        revision = adapter_config.revision or revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if args.adapter else model_name,
        **({} if args.adapter else {"revision": revision}),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = [encode_prompt(tokenizer, row["prompt"]) for row in rows]
    if max(map(len, inputs)) + args.max_new_tokens > args.max_length:
        raise ValueError(
            "Evaluation input exceeds max-length; increase it instead of truncating the state"
        )
    if not args.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA is required; use --cpu only for small-model smoke tests")
    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.float32 if args.cpu else torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Transformers decorates .to; use the base type for its bound signature.
    cast(torch.nn.Module, model).to(torch.device("cpu" if args.cpu else "cuda"))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            ids = inputs[start : start + args.batch_size]
            batch = tokenizer.pad(
                [
                    {"input_ids": item, "attention_mask": [1] * len(item)}
                    for item in ids
                ],
                return_tensors="pt",
            ).to(model.device)
            output = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            text = tokenizer.batch_decode(
                output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, prediction in zip(rows[start : start + args.batch_size], text):
                predictions.append(
                    {
                        **row["metadata"],
                        "target": row["completion"],
                        "prediction": prediction.strip(),
                        "raw_prediction": prediction,
                    }
                )
            print(f"Evaluated {len(predictions)}/{len(rows)}", flush=True)
    report = summarize(predictions)
    majority = majority_predictions(train_rows)
    baseline = [
        {**row, "prediction": majority.get(row["game"], "")} for row in predictions
    ]
    report["majority_action_baseline"] = summarize(baseline)
    report["run"] = {
        "model": model_name,
        "revision": revision,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "adapter": str(args.adapter) if args.adapter else None,
        "data_sha256": file_hash(args.data),
        "train_data_sha256": file_hash(args.train_data),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy, thinking disabled, no action constraint",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
