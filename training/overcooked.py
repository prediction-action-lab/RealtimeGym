"""Evaluate Overcooked using the paper's 8 game seeds x 4 sampling seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from training.common import file_hash
from training.play import ActionPolicy, enable_games, run_episode

PAPER = "https://arxiv.org/html/2511.04898v1"
DIFFICULTIES = ("easy", "medium", "hard")
GAME_SEEDS = tuple(range(8))
SAMPLING_SEEDS = tuple(range(4))
HORIZON = 100
REWARD_MAX = 56.0


def summarize(
    episodes: list[dict[str, Any]], difficulties: list[str]
) -> dict[str, Any]:
    """Require a complete, unique 32-episode grid for every requested setting."""
    expected = {
        (difficulty, seed, sampling_seed)
        for difficulty in difficulties
        for seed in GAME_SEEDS
        for sampling_seed in SAMPLING_SEEDS
    }
    seen = set()
    for row in episodes:
        key = (row["difficulty"], row["seed"], row["sampling_seed"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate episode: {key}")
        if (
            row["game"] != "overcooked"
            or row["steps"] != HORIZON
            or not row["terminated"]
            or row["truncated"]
            or not math.isfinite(row["score"])
        ):
            raise ValueError(f"Incomplete or invalid Overcooked episode: {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError(
            f"Missing {len(expected - seen)} episodes; no final score reported"
        )
    groups = {}
    for difficulty in difficulties:
        rows = [row for row in episodes if row["difficulty"] == difficulty]
        scores = [row["score"] / REWARD_MAX for row in rows]
        groups[difficulty] = {
            "episodes": len(rows),
            "mean_reward": statistics.mean(row["score"] for row in rows),
            "mean_score": statistics.mean(scores),
            "std_score": statistics.stdev(scores),
            "mean_steps": statistics.mean(row["steps"] for row in rows),
            "invalid_action_rate": sum(row["invalid_actions"] for row in rows)
            / sum(row["steps"] for row in rows),
            "missed_deadlines": sum(row["missed_deadlines"] for row in rows),
        }
    return {
        "episode_count": len(episodes),
        "normalization": {"formula": "reward / 56", "min": 0, "max": REWARD_MAX},
        "by_difficulty": groups,
        "episodes": [
            {**row, "normalized_score": row["score"] / REWARD_MAX}
            for row in sorted(
                episodes,
                key=lambda row: (row["difficulty"], row["seed"], row["sampling_seed"]),
            )
        ],
    }


def run_evaluation(
    policy: ActionPolicy,
    output: Path,
    difficulties: list[str],
    tokens_per_step: int,
    save_gif: bool,
) -> dict[str, Any]:
    episodes = []
    for difficulty in difficulties:
        for seed in GAME_SEEDS:
            for sampling_seed in SAMPLING_SEEDS:
                print(
                    f"Episode {len(episodes) + 1}/{len(difficulties) * 32}: "
                    f"{difficulty} game_seed={seed} sampling_seed={sampling_seed}",
                    flush=True,
                )
                episodes.append(
                    run_episode(
                        policy,
                        "overcooked",
                        difficulty,
                        seed,
                        output
                        / difficulty
                        / f"seed-{seed}"
                        / f"sample-{sampling_seed}",
                        HORIZON,
                        "stay",
                        save_gif,
                        sampling_seed=sampling_seed,
                        tokens_per_step=tokens_per_step,
                    )
                )
    return summarize(episodes, difficulties)


def collect(source: Path, difficulties: list[str]) -> dict[str, Any]:
    """Merge extracted CHTC results, rejecting mixed configurations or reruns."""
    reports = sorted(source.rglob("metrics.json"))
    if not reports:
        raise ValueError(f"No metrics.json files under {source}")
    configuration = None
    episodes = []
    for path in reports:
        report = json.loads(path.read_text())
        settings = report["configuration"]
        if configuration is not None and settings != configuration:
            raise ValueError(f"Evaluation configurations differ: {path}")
        configuration = settings
        episodes.extend(report["episodes"])
    return {
        "configuration": configuration,
        "source_reports": [str(path) for path in reports],
        **summarize(episodes, difficulties),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--revision", default="b968826d9c46dd6066d109eabc6255188de91218"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES)
    )
    parser.add_argument(
        "--tokens-per-step", type=int, choices=(4096, 8192, 16384, 32768), default=8192
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="For tiny-model tests only")
    parser.add_argument(
        "--collect", type=Path, help="Merge extracted results without loading a model"
    )
    args = parser.parse_args()
    if len(set(args.difficulties)) != len(args.difficulties):
        parser.error("Difficulties must be unique")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error(
            "Temperature must be positive: four sampling seeds require sampling"
        )
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require 0 < top-p <= 1 and top-k >= 0")
    if not 1 <= args.max_new_tokens <= args.tokens_per_step:
        parser.error("max-new-tokens must be positive and fit within tokens-per-step")
    if args.max_length <= args.max_new_tokens:
        parser.error("max-length must leave space for a prompt and response")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    report: dict[str, Any]
    if args.collect:
        report = collect(args.collect, args.difficulties)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        enable_games()
        policy = ActionPolicy(
            args.model,
            args.revision,
            args.adapter,
            args.cpu,
            args.max_length,
            args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        configuration = {
            "protocol": "overcooked-8x4-v1",
            "paper": PAPER,
            **policy.metadata,
            "adapter_hashes": {
                path.name: file_hash(path)
                for path in sorted(args.adapter.glob("adapter*"))
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}
            }
            if args.adapter
            else None,
            "source_hashes": {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("overcooked.py", "play.py", "common.py")
            },
            "game_seeds": GAME_SEEDS,
            "sampling_seeds": SAMPLING_SEEDS,
            "horizon": HORIZON,
            "tokens_per_step": args.tokens_per_step,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "thinking": False,
            "timing": "token-based reactive: one response per step, capped within the step budget",
            "prompt": "training.common.make_prompt (action-only SFT format)",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run.json").write_text(
            json.dumps(
                {"configuration": configuration, "difficulties": args.difficulties},
                indent=2,
            )
            + "\n"
        )
        report = {
            "configuration": configuration,
            **run_evaluation(
                policy,
                args.output_dir,
                args.difficulties,
                args.tokens_per_step,
                args.save_gif,
            ),
        }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for difficulty, values in report["by_difficulty"].items():
        print(
            f"{difficulty}: {values['episodes']} episodes, mean score={values['mean_score']:.4f}, mean reward={values['mean_reward']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
