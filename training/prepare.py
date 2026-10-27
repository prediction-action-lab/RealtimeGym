"""Convert raw SALT-NLP/RealtimeGym trajectories into action-only examples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.common import (
    ACTIONS,
    file_hash,
    make_prompt,
    normalize_action,
    prompt_hash,
)

FILENAME = re.compile(
    r"^(freeway|snake|overcooked)_(easy|medium|hard)_"
    r"(\d+k)_seed(\d+)_(reactive|planning|agile)\.json$"
)


def prepare(
    input_dir: Path,
    output_dir: Path,
    validation_seeds: set[int],
    test_seeds: set[int],
    games: set[str],
    agents: set[str],
    source: dict[str, str] | None = None,
    invalid_action_policy: str = "skip",
) -> dict[str, Any]:
    if invalid_action_policy not in {"skip", "error"}:
        raise ValueError("invalid_action_policy must be skip or error")
    if not validation_seeds or not test_seeds or validation_seeds & test_seeds:
        raise ValueError("Validation and test seeds must be nonempty and disjoint")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    skipped: Counter[str] = Counter()
    rejected = []
    file_records = []
    for path in sorted(input_dir.glob("*.json")):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            continue
        game, difficulty, budget, seed_text, agent = match.groups()
        if game not in games or agent not in agents:
            continue
        seed = int(seed_text)
        split = (
            "test"
            if seed in test_seeds
            else "validation"
            if seed in validation_seeds
            else "train"
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected an array of steps in {path}")
        file_records.append(
            {"file": path.name, "sha256": file_hash(path), "split": split}
        )
        previous_step = -1
        for row in rows:
            step = row.get("step")
            if type(step) is not int or step <= previous_step:
                raise ValueError(f"Non-increasing/missing step in {path}: {step!r}")
            previous_step = step
            state = row.get("state", {})
            if state.get("game_turn") != step:
                raise ValueError(f"State/step mismatch in {path}, step {step}")
            if state.get("terminal"):
                skipped["terminal_state"] += 1
                continue
            try:
                action = normalize_action(game, row.get("action"))
            except ValueError as error:
                if invalid_action_policy == "error":
                    raise ValueError(f"{path.name}, step {step}: {error}") from error
                # Plans, explanations, and unknown actions are not reliable
                # single-action labels. Record exclusions rather than guessing.
                skipped[f"{game}_invalid_action"] += 1
                rejected.append(
                    {
                        "source_file": path.name,
                        "step": step,
                        "game": game,
                        "reason": "invalid_action",
                        "raw_action": row.get("action"),
                    }
                )
                continue
            try:
                prompt = make_prompt(game, row.get("original_state"))
            except ValueError as error:
                raise ValueError(f"{path.name}, step {step}: {error}") from error
            splits[split].append(
                {
                    "prompt": prompt,
                    "completion": action,
                    "metadata": {
                        "game": game,
                        "difficulty": difficulty,
                        "seed": seed,
                        "budget": budget,
                        "agent": agent,
                        "step": step,
                        "source_file": path.name,
                    },
                }
            )
    if not file_records:
        raise ValueError(f"No matching trajectories found in {input_dir}")

    # Holdout priority is fixed. Remove repeated prompts from earlier splits,
    # regardless of the action label. Also deduplicate equal prompt/action pairs.
    heldout_prompts: set[str] = set()
    report: dict[str, Any] = {}
    for split in ("test", "validation", "train"):
        kept = []
        seen_pairs: set[tuple[str, str]] = set()
        labels_by_prompt: dict[str, set[str]] = defaultdict(set)
        original_count = len(splits[split])
        for row in splits[split]:
            key = prompt_hash(row["prompt"])
            pair = (key, row["completion"])
            if key in heldout_prompts:
                skipped[f"{split}_heldout_overlap"] += 1
                continue
            if pair in seen_pairs:
                skipped[f"{split}_duplicate_pair"] += 1
                continue
            seen_pairs.add(pair)
            labels_by_prompt[key].add(row["completion"])
            kept.append(row)
        splits[split] = kept
        heldout_prompts.update(labels_by_prompt)
        report[split] = {
            "before_deduplication": original_count,
            "examples": len(kept),
            "conflicting_prompts": sum(
                len(labels) > 1 for labels in labels_by_prompt.values()
            ),
            "games": dict(Counter(row["metadata"]["game"] for row in kept)),
            "actions_by_game": {
                game: dict(
                    Counter(
                        row["completion"]
                        for row in kept
                        if row["metadata"]["game"] == game
                    )
                )
                for game in sorted(games)
            },
        }
        if not kept:
            raise ValueError(
                f"{split} has no examples after filtering; check seeds/data"
            )
    manifest = {
        "source": source or {"local_directory": str(input_dir.resolve())},
        "validation_seeds": sorted(validation_seeds),
        "test_seeds": sorted(test_seeds),
        "games": sorted(games),
        "agents": sorted(agents),
        "splits": report,
        "removed": dict(skipped),
        "files": file_records,
        "target": "recorded action; not a guarantee of optimal play",
        "reasoning_included": False,
        "invalid_action_policy": invalid_action_policy,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in examples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, help="Existing raw JSON files; skips download"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--dataset", default="SALT-NLP/RealtimeGym")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--invalid-action-policy", choices=["skip", "error"], default="skip"
    )
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[6])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--games", nargs="+", choices=list(ACTIONS), default=list(ACTIONS)
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["reactive", "planning", "agile"],
        default=["reactive", "planning", "agile"],
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    source: dict[str, str] | None = None
    input_dir = args.input_dir
    if input_dir is None:
        from huggingface_hub import HfApi, snapshot_download

        revision = (
            HfApi().dataset_info(args.dataset, revision=args.revision, timeout=30).sha
        )
        if not isinstance(revision, str):
            raise ValueError("Hugging Face did not return a dataset commit")
        input_dir = Path(
            snapshot_download(
                args.dataset,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    f"{game}_*_{agent}.json"
                    for game in args.games
                    for agent in args.agents
                ],
            )
        )
        source = {"dataset": args.dataset, "revision": revision}
    manifest = prepare(
        input_dir,
        args.output_dir,
        set(args.validation_seeds),
        set(args.test_seeds),
        set(args.games),
        set(args.agents),
        source,
        args.invalid_action_policy,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
