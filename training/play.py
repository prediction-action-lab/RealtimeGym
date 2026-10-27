"""Run a base model or trained action-only LoRA adapter in actual game episodes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

from training.common import ACTIONS, encode_prompt, make_prompt


def enable_games() -> None:
    # Ship the repo's exact environment source with CHTC jobs, including assets.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def state_description(game: str, state: dict[str, Any]) -> str:
    module = importlib.import_module(f"realtimegym.prompts.{game}")
    description = module.state_to_description(state, mode="reactive")
    prefix = module.FAST_AGENT_PROMPT
    if not isinstance(description, str) or not description.startswith(prefix):
        raise ValueError("Unexpected game prompt format")
    # Retain the observed state formatting used in original_state, removing the
    # legacy reasoning/boxed-answer instructions. make_prompt supplies SFT rules.
    return description[len(prefix) :]


class ActionPolicy:
    def __init__(
        self,
        model_name: str,
        revision: str,
        adapter: Path | None,
        cpu: bool,
        max_length: int,
        max_new_tokens: int,
        temperature: float | None = None,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.last_generation_tokens = 0
        if adapter:
            from peft import PeftConfig

            config = PeftConfig.from_pretrained(str(adapter))
            model_name = config.base_model_name_or_path
            revision = config.revision or revision
        if not cpu and not torch.cuda.is_available():
            raise ValueError("CUDA is required; --cpu is for tiny-model tests")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter or model_name, **({} if adapter else {"revision": revision})
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.float32 if cpu else torch.bfloat16,
            attn_implementation="sdpa",
        )
        cast(torch.nn.Module, self.model).to(torch.device("cpu" if cpu else "cuda"))
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.model.config.use_cache = True
        self.metadata = {
            "model": model_name,
            "revision": revision,
            "adapter": str(adapter) if adapter else None,
        }

    def __call__(self, prompt: list[dict[str, str]]) -> str:
        import torch

        ids = encode_prompt(self.tokenizer, prompt)
        if len(ids) + self.max_new_tokens > self.max_length:
            raise ValueError("Game state exceeds max-length; state was not truncated")
        inputs = torch.tensor([ids], device=self.model.device)
        with torch.inference_mode():
            result = cast(
                torch.Tensor,
                self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=self.temperature is not None,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p if self.temperature is not None else None,
                    top_k=self.top_k if self.temperature is not None else None,
                    min_p=None,
                    return_dict_in_generate=False,
                ),
            )
        self.last_generation_tokens = result.shape[1] - len(ids)
        return self.tokenizer.decode(result[0, len(ids) :], skip_special_tokens=True)


def run_episode(
    policy: Callable[[list[dict[str, str]]], str],
    game: str,
    difficulty: str,
    seed: int,
    output: Path,
    max_steps: int,
    invalid_action: str,
    save_gif: bool,
    sampling_seed: int | None = None,
    tokens_per_step: int | None = None,
) -> dict[str, Any]:
    import numpy as np
    import realtimegym

    random.seed(seed)
    np.random.seed(seed)
    version = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    env, actual_seed, renderer = realtimegym.make(
        f"{game.capitalize()}-v{version}", seed=seed, render=save_gif
    )
    observation, done = env.reset()
    if sampling_seed is not None:
        import torch

        # Keep the LLM sampling stream separate from Python/NumPy game randomness.
        torch.manual_seed(sampling_seed)
    output.mkdir(parents=True, exist_ok=False)
    frames = []

    def capture() -> None:
        if renderer is not None:
            import pygame
            from PIL import Image

            pixels = pygame.surfarray.array3d(renderer.render(env)).swapaxes(0, 1)
            frames.append(Image.fromarray(pixels))

    capture()
    steps, invalid, inference_seconds = 0, 0, 0.0
    total_tokens, missed_deadlines = 0, 0
    reward_total = float(env.reward)
    initial_score = reward_total
    started = time.perf_counter()
    with (output / "trajectory.jsonl").open("w") as handle:
        while not done and steps < max_steps:
            description = state_description(game, observation["state"])
            prompt = make_prompt(game, description)
            before = time.perf_counter()
            raw = policy(prompt)
            elapsed = time.perf_counter() - before
            generated_tokens = getattr(policy, "last_generation_tokens", None)
            within_budget = True
            if tokens_per_step is not None:
                if not isinstance(generated_tokens, int) or generated_tokens < 0:
                    raise ValueError(
                        "Token-timed gameplay requires generated token counts"
                    )
                within_budget = generated_tokens <= tokens_per_step
            prediction = raw.strip()
            valid = prediction in ACTIONS[game] and within_budget
            if not valid and invalid_action == "error":
                raise ValueError(f"Invalid {game} action at turn {steps}: {raw!r}")
            action = prediction if valid else "S"
            next_observation, done, reward, reset = env.step(action)
            # All three wrappers return cumulative reward, not per-step reward.
            row = {
                "step": steps,
                "game_turn": observation["game_turn"],
                "original_state": description,
                "state_string": observation["state_string"],
                "raw_prediction": raw,
                "action": action,
                "valid_action": valid,
                "reward_delta": float(reward) - reward_total,
                "reward_total": float(reward),
                "done": bool(done),
                "reset": bool(reset),
                "inference_seconds": elapsed,
                "generated_tokens": generated_tokens,
                "within_token_budget": within_budget,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            reward_total = float(reward)
            observation = next_observation
            invalid += int(not valid)
            inference_seconds += elapsed
            total_tokens += generated_tokens or 0
            missed_deadlines += int(not within_budget)
            steps += 1
            capture()
            print(
                f"{game}/{difficulty} seed={seed} step={steps} action={action} score={reward_total}",
                flush=True,
            )
    if frames:
        frames[0].save(
            output / "episode.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
    summary = {
        "game": game,
        "difficulty": difficulty,
        "seed": seed,
        "sampling_seed": sampling_seed,
        "actual_seed": actual_seed,
        "steps": steps,
        "score": reward_total,
        "initial_score": initial_score,
        "terminated": bool(done),
        "truncated": not bool(done),
        "invalid_actions": invalid,
        "invalid_action_rate": invalid / steps if steps else 0,
        "inference_seconds": inference_seconds,
        "generated_tokens": total_tokens,
        "missed_deadlines": missed_deadlines,
        "tokens_per_step": tokens_per_step,
        "mean_inference_seconds": inference_seconds / steps if steps else 0,
        "episode_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=sorted(ACTIONS), required=True)
    parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="easy"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[6])
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--revision", default="b968826d9c46dd6066d109eabc6255188de91218"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--invalid-action", choices=("stay", "error"), default="stay")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.max_steps, args.max_length, args.max_new_tokens) < 1:
        parser.error("Step and token limits must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(
        seed not in range(8) for seed in args.seeds
    ):
        parser.error("Use unique seed IDs from 0 to 7 (6 validation, 7 test)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    enable_games()
    policy = ActionPolicy(
        args.model,
        args.revision,
        args.adapter,
        args.cpu,
        args.max_length,
        args.max_new_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {
        **vars(args),
        **policy.metadata,
        "decoding": "greedy; thinking disabled",
        "timing": "synchronous: environment advances once per action; no inference-time penalty",
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, default=str, indent=2) + "\n"
    )
    episodes = [
        run_episode(
            policy,
            args.game,
            args.difficulty,
            seed,
            args.output_dir / f"seed-{seed}",
            args.max_steps,
            args.invalid_action,
            args.save_gif,
        )
        for seed in args.seeds
    ]
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "mean_score": sum(row["score"] for row in episodes) / len(episodes),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
