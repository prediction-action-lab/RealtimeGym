"""Check complete evaluation grids, seeded live gameplay, and result collection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from training.overcooked import collect, main, run_evaluation, summarize
from training.play import ActionPolicy, enable_games, run_episode


def episode_grid(difficulty: str) -> list[dict[str, Any]]:
    return [
        {
            "game": "overcooked",
            "difficulty": difficulty,
            "seed": seed,
            "sampling_seed": sampling_seed,
            "steps": 100,
            "score": 28.0,
            "terminated": True,
            "truncated": False,
            "invalid_actions": 0,
            "missed_deadlines": 0,
        }
        for seed in range(8)
        for sampling_seed in range(4)
    ]


def test_paper_score_uses_fixed_reference_and_requires_all_32_episodes() -> None:
    rows = episode_grid("easy")
    report = summarize(rows, ["easy"])
    assert report["episode_count"] == 32
    assert report["by_difficulty"]["easy"]["mean_score"] == 0.5
    assert report["by_difficulty"]["easy"]["mean_reward"] == 28
    assert report["by_difficulty"]["easy"]["std_score"] == 0
    rows[0]["score"] = 112
    assert summarize(rows, ["easy"])["episodes"][0]["normalized_score"] == 2
    with pytest.raises(ValueError, match="Missing 1 episodes"):
        summarize(rows[:-1], ["easy"])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + [rows[0]], ["easy"])


@pytest.mark.parametrize(
    "change",
    [
        {"steps": 99},
        {"terminated": False},
        {"truncated": True},
        {"score": float("nan")},
    ],
)
def test_partial_or_failed_episode_cannot_contribute_a_final_score(
    change: dict[str, Any],
) -> None:
    rows = episode_grid("easy")
    rows[0].update(change)
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        summarize(rows, ["easy"])


def test_collection_merges_three_jobs_and_rejects_mixed_policies(
    tmp_path: Path,
) -> None:
    for difficulty in ("easy", "medium", "hard"):
        directory = tmp_path / difficulty
        directory.mkdir()
        report = {
            "configuration": {
                "model": "same-base",
                "adapter_hashes": {"weights": "same-weights"},
            },
            **summarize(episode_grid(difficulty), [difficulty]),
        }
        (directory / "metrics.json").write_text(json.dumps(report))
    report = collect(tmp_path, ["easy", "medium", "hard"])
    assert report["episode_count"] == 96
    assert all(row["episodes"] == 32 for row in report["by_difficulty"].values())
    path = tmp_path / "hard/metrics.json"
    mixed = json.loads(path.read_text())
    mixed["configuration"]["adapter_hashes"]["weights"] = "different-weights"
    path.write_text(json.dumps(mixed))
    with pytest.raises(ValueError, match="configurations differ"):
        collect(tmp_path, ["easy", "medium", "hard"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--temperature", "0"],
        ["--temperature", "nan"],
        ["--max-new-tokens", "8193"],
        ["--difficulties", "easy", "easy"],
    ],
)
def test_invalid_protocol_settings_fail_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["overcooked", "--output-dir", str(tmp_path / "out"), *arguments]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


class SampledPolicy(ActionPolicy):
    def __init__(self) -> None:
        self.last_generation_tokens = 2

    def __call__(self, prompt: list[dict[str, str]]) -> str:
        import torch

        return ("U", "D", "L", "R", "I", "S")[int(torch.randint(6, (1,)).item())]


def test_sampling_seeds_reproduce_trajectories_independently_of_game_seed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pygame")
    pytest.importorskip("gym")
    enable_games()
    sequences = []
    initial_states = []
    for index, sampling_seed in enumerate((0, 1, 0)):
        directory = tmp_path / str(index)
        run_episode(
            SampledPolicy(),
            "overcooked",
            "easy",
            6,
            directory,
            10,
            "stay",
            False,
            sampling_seed=sampling_seed,
            tokens_per_step=8192,
        )
        rows = [
            json.loads(line)
            for line in (directory / "trajectory.jsonl").read_text().splitlines()
        ]
        sequences.append(
            [(row["action"], row["state_string"], row["reward_total"]) for row in rows]
        )
        initial_states.append(rows[0]["original_state"])
    assert sequences[0] == sequences[2]
    assert sequences[0] != sequences[1]
    assert initial_states[0] == initial_states[1] == initial_states[2]


def test_full_live_overcooked_grid_and_deadline_fallback(tmp_path: Path) -> None:
    pytest.importorskip("pygame")
    pytest.importorskip("gym")
    enable_games()
    policy = SampledPolicy()
    report = run_evaluation(policy, tmp_path / "evaluation", ["easy"], 8192, False)
    assert report["episode_count"] == 32
    assert report["by_difficulty"]["easy"]["mean_steps"] == 100
    assert report["by_difficulty"]["easy"]["missed_deadlines"] == 0
    assert len(list((tmp_path / "evaluation").rglob("trajectory.jsonl"))) == 32
    policy.last_generation_tokens = 8193
    summary = run_episode(
        policy,
        "overcooked",
        "easy",
        6,
        tmp_path / "late",
        2,
        "stay",
        False,
        sampling_seed=0,
        tokens_per_step=8192,
    )
    assert summary["missed_deadlines"] == summary["invalid_actions"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "late/trajectory.jsonl").read_text().splitlines()
    ]
    assert all(row["action"] == "S" for row in rows)


def test_failed_chtc_wrapper_returns_nonzero_and_archives_partial_outputs(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "chtc/overcooked.sh"
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (tmp_path / "overcooked-results.tar.gz").is_file()
