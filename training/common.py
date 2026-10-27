"""Shared input format and explicit response-only label construction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ACTIONS = {
    "freeway": ("U", "D", "S"),
    "snake": ("U", "D", "L", "R", "S"),
    "overcooked": ("U", "D", "L", "R", "I", "S"),
}

# No requests for reasoning and no references to another model's plan.
RULES = {
    "freeway": (
        "Play Freeway. Reach (0, 9) from (0, 0) as quickly as possible while "
        "avoiding cars in lanes y=1 through y=8. U increases y by 1, D decreases "
        "y by 1, and S stays in place. The player remains within y=0 through 9. "
        "Cars move each turn in their stated direction at their stated speed. "
        "For a right-moving car the occupied interval is [tail, head]; for a "
        "left-moving car it is [head, tail]. A collision occurs if the car's "
        "interval contains x=0 in the player's lane after the turn's updates."
    ),
    "snake": (
        "Play Snake. Maximize food rewards while surviving. U moves (x,y+1), "
        "D moves (x,y-1), L moves (x-1,y), and R moves (x+1,y). S keeps moving "
        "in the current direction; it does not stop the snake. Immediate "
        "reversal is ignored. Snake positions are listed head first. Avoid "
        "walls, obstacles, and the body. The tail normally moves away, but "
        "eating positive food grows the body, so moving onto food on the tail "
        "can cause a collision. Positive food gives +1 and grows the snake; "
        "negative food gives -1 without growth. Food entries are "
        "(x, y, life span, value). Movement happens before food life spans "
        "decrease. Food disappears when its life span expires."
    ),
    "overcooked": (
        "Play Overcooked as Alice, cooperating with Bob to deliver soups. "
        "U moves (x,y+1), D moves (x,y-1), L moves (x-1,y), and R moves "
        "(x+1,y). Movement also changes orientation, even if a tile blocks "
        "movement. S stays in place. I interacts with the adjacent tile you "
        "face: take ingredients or plates from dispensers, place or take "
        "items on counters, add ingredients to pots, collect cooked soup "
        "using a plate, or deliver soup at a serving counter. Carry at most "
        "one item. Players cannot pass through occupied tiles or each other. "
        "Three ingredients start cooking automatically. Follow the recipes, "
        "cooking times, and rewards in the state. Action histories contain "
        "only past actions, with the most recent action last."
    ),
}

REASONING_MARKERS = re.compile(
    r"<\s*/?think\s*>|guidance from a previous thinking model|"
    r"thinking model['’]s past plan|\\boxed\s*\{",
    re.IGNORECASE,
)


def make_prompt(game: str, state_text: str) -> list[dict[str, str]]:
    """Use only the observed state, never scores, reasoning, or future states."""
    if game not in ACTIONS:
        raise ValueError(f"Unknown game: {game}")
    if not isinstance(state_text, str) or not state_text.strip():
        raise ValueError("original_state must be nonempty text")
    if REASONING_MARKERS.search(state_text):
        raise ValueError("original_state contains reasoning/answer markers")
    state_text = state_text.replace("\r\n", "\n").strip()
    return [
        {
            "role": "system",
            "content": RULES[game]
            + "\nChoose the next action. Reply with exactly one letter from "
            + ", ".join(ACTIONS[game])
            + ". Do not include an explanation or any other text.",
        },
        {"role": "user", "content": state_text},
    ]


def normalize_action(game: str, action: str) -> str:
    """Normalize documented action names; never infer a label from reasoning."""
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    aliases = {
        "UP": "U",
        "DOWN": "D",
        "LEFT": "L",
        "RIGHT": "R",
        "STAY": "S",
        "KEEP": "S",
        "INTERACT": "I",
    }
    value = action.strip().upper()
    value = aliases.get(value, value)
    if value not in ACTIONS[game]:
        raise ValueError(f"Invalid {game} action: {action!r}")
    return value


def prompt_hash(prompt: list[dict[str, str]]) -> str:
    # Ignore whitespace and t_0/t_1 notation when detecting identical inputs.
    canonical = [
        {
            "role": message["role"],
            "content": re.sub(
                r"\s+", " ", re.sub(r"t_[01]", "t", message["content"])
            ).strip(),
        }
        for message in prompt
    ]
    text = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["metadata"]
            game = metadata["game"]
            # Rebuild from a whitelist, so added metadata cannot enter the prompt.
            expected = make_prompt(game, row["prompt"][-1]["content"])
            if row["prompt"] != expected:
                raise ValueError(f"{path}:{number}: unexpected prompt format")
            action = normalize_action(game, row["completion"])
            if action != row["completion"]:
                raise ValueError(f"{path}:{number}: label is not canonical")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def encode_prompt(tokenizer: Any, prompt: list[dict[str, str]]) -> list[int]:  # noqa: ANN401
    return tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def encode_example(
    tokenizer: Any,  # noqa: ANN401
    row: dict[str, Any],
    max_length: int,
) -> dict[str, list[int]]:
    """Mask prompt tokens, supervise the action and EOS, never truncate state."""
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    action_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
    if not action_ids or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must encode actions and define EOS")
    target_ids = action_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        metadata = row.get("metadata", {})
        raise ValueError(
            f"Example {metadata} needs {len(input_ids)} tokens, exceeding "
            f"max_length={max_length}. Increase max_length; state was not truncated."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }
