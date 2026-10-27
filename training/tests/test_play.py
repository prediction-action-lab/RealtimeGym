"""Exercise live game prompts, score accounting, invalid actions, and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pygame")
pytest.importorskip("gym")

from training.common import make_prompt
from training.play import enable_games, run_episode, state_description


@pytest.mark.parametrize("game", ["freeway", "snake", "overcooked"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_game_loop(game: str, difficulty: str, tmp_path: Path) -> None:
    enable_games()
    output = tmp_path / "episode"
    prompts = []

    def policy(prompt: list[dict[str, str]]) -> str:
        prompts.append(prompt)
        return "I choose U"  # Invalid text must not be parsed into an action.

    result = run_episode(policy, game, difficulty, 6, output, 3, "stay", True)
    rows = [
        json.loads(line)
        for line in (output / "trajectory.jsonl").read_text().splitlines()
    ]
    assert result["steps"] == len(rows) == len(prompts)
    assert result["invalid_actions"] == len(rows)
    assert all(row["action"] == "S" for row in rows)
    assert sum(row["reward_delta"] for row in rows) == pytest.approx(
        result["score"] - result["initial_score"]
    )
    assert (output / "episode.gif").stat().st_size > 0
    for prompt, row in zip(prompts, rows):
        assert prompt == make_prompt(game, row["original_state"])
        assert "boxed" not in row["original_state"]
        assert "<think>" not in row["original_state"]


def test_snake_state_description_matches_training_format() -> None:
    enable_games()
    state = {
        "game_turn": 3,
        "size": 8,
        "internal_obstacles": [],
        "snake": [(3, 3)],
        "snake_dir": "L",
        "foods": [(2, 1, 10, 1)],
    }
    assert state_description("snake", state) == (
        "**Current Turn**: \\( t_0 = 3 \\)\n"
        "**Cells occupied by walls**:\n"
        "\t - Border Cells: x=0/x=7 or y=0/y=7.\n"
        "\t - Internal Obstacles: No internal obstacles\n"
        "**Snake Positions**:[(3, 3)]\n**Snake Head Direction**: L\n"
        "**Food Positions, Life Span and Value**:\n\t- (2, 1, 10, 1)\n"
    )
