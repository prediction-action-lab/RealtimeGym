"""Check complete evaluation grids, seeded live gameplay, and result collection."""

from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from training.overcooked import collect, main, run_evaluation, summarize
from training.common import make_prompt
from training.play import ActionPolicy, enable_games, parse_action, run_episode


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>Move toward the pot; don't choose S.</think>\n\nI", "I"),
        ("<think>\nReason\n</think> R\n", "R"),
        ("<think></think> S", "S"),
        ("<think>Still deciding whether to choose U", None),
        ("<think>U</think>", None),
        ("U", None),
        ("Reason </think> U", None),
        ("<think>Reason</think> U extra", None),
        ("<think>Reason</think> U D", None),
        ("<think><think>Reason</think></think> U", None),
        ("<think>Reason</think><think>More</think> U", None),
    ],
)
def test_cot_parsing_requires_complete_reasoning_and_one_final_action(
    raw: str, expected: str | None
) -> None:
    assert parse_action(raw, "overcooked", "cot") == expected
    assert parse_action(raw, "overcooked", "action") == ("U" if raw == "U" else None)


@pytest.mark.parametrize("trained_mode", ["action", None])
def test_cot_inference_rejects_wrong_or_unverified_adapter_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained_mode: str | None
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    if trained_mode:
        (adapter / "run.json").write_text(
            json.dumps({"config": {"target_mode": trained_mode}})
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overcooked",
            "--adapter",
            str(adapter),
            "--target-mode",
            "cot",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


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


def test_cot_live_grid_uses_training_prompts_and_logs_unfinished_reasoning(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pygame")
    pytest.importorskip("gym")
    enable_games()

    class CotPolicy:
        target_mode = "cot"
        last_generation_tokens = 8

        def __call__(self, prompt: list[dict[str, str]]) -> str:
            assert prompt == make_prompt("overcooked", prompt[-1]["content"], "cot")
            return "<think>I should interact.</think>\n\nI"

    policy = CotPolicy()
    report = run_evaluation(
        policy, tmp_path / "grid", ["easy", "medium", "hard"], 8192, False
    )
    assert report["episode_count"] == 96
    assert all(group["episodes"] == 32 for group in report["by_difficulty"].values())
    assert all(row["invalid_actions"] == 0 for row in report["episodes"])

    class UnfinishedPolicy(CotPolicy):
        last_generation_tokens = 8192

        def __call__(self, prompt: list[dict[str, str]]) -> str:
            return "<think>I am considering U but have not finished"

    result = run_episode(
        UnfinishedPolicy(),
        "overcooked",
        "easy",
        6,
        tmp_path / "unfinished",
        2,
        "stay",
        False,
        sampling_seed=0,
        tokens_per_step=8192,
    )
    assert result["invalid_actions"] == result["missed_deadlines"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "unfinished/trajectory.jsonl").read_text().splitlines()
    ]
    assert all(row["action"] == "S" and row["parsed_action"] is None for row in rows)
    assert all(row["raw_prediction"].startswith("<think>") for row in rows)


def test_failed_chtc_wrapper_returns_nonzero_and_archives_partial_outputs(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "chtc/overcooked.sh"
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (tmp_path / "overcooked-results.tar.gz").is_file()


@pytest.mark.parametrize("training_mode", ["train", "smoke"])
def test_cot_wrapper_checks_completed_training_and_passes_inference_mode(
    tmp_path: Path, training_mode: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "chtc/overcooked.sh"

    def archive_files(path: Path, files: dict[str, str]) -> None:
        with tarfile.open(path, "w:gz") as archive:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    archive_files(
        tmp_path / "gameplay-payload.tar.gz",
        {
            "training/__init__.py": "",
            "training/overcooked.py": (
                "import json, sys\nfrom pathlib import Path\n"
                "Path('outputs/invocation.json').write_text(json.dumps(sys.argv[1:]))\n"
            ),
        },
    )
    archive_files(tmp_path / "qwen3-8b-cache.tar", {})
    archive_files(
        tmp_path / "adapter.tar.gz",
        {
            "outputs/job-result.json": json.dumps(
                {
                    "mode": training_mode,
                    "target_mode": "cot",
                    "training_exit_code": 0,
                }
            ),
        },
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python").symlink_to(sys.executable)
    result = subprocess.run(
        ["bash", str(script), "easy", "adapter", "adapter.tar.gz", "8192", "cot"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        timeout=30,
    )
    assert (tmp_path / "overcooked-results.tar.gz").is_file()
    if training_mode == "train":
        assert result.returncode == 0, result.stderr
        arguments = json.loads((tmp_path / "outputs/invocation.json").read_text())
        assert arguments[arguments.index("--target-mode") + 1] == "cot"
        assert arguments[arguments.index("--adapter") + 1] == "trained/outputs/adapter"
        assert arguments[arguments.index("--tokens-per-step") + 1] == "8192"
    else:
        assert result.returncode != 0
        assert "Expected completed full CoT training" in result.stderr
        assert not (tmp_path / "outputs/invocation.json").exists()
