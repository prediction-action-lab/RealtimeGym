"""Verify packaging boundaries without a cluster or a model download."""

from pathlib import Path
import json
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from training.chtc.bundle import build_bundle


def test_bundle_has_only_training_sources_and_required_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        (data / name).write_text("{}\n")
    (data / ".env").write_text("SECRET=not-for-the-cluster")
    output = tmp_path / "bundle"
    build_bundle(Path(__file__).resolve().parents[2], data, output)
    with tarfile.open(output / "payload.tar.gz") as archive:
        names = archive.getnames()
    assert "training/train.py" in names
    assert "training/checkpointing.py" in names
    assert "training/chtc/job.py" in names
    assert "data/sft/train.jsonl" in names
    assert all(".env" not in name and ".venv" not in name for name in names)
    assert (output / "run.sh").stat().st_mode & 0o111
    with pytest.raises(ValueError, match="must be empty"):
        build_bundle(Path(__file__).resolve().parents[2], data, output)


def test_cot_bundle_selects_matching_config_and_data(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text('{"target_mode":"cot"}\n')
    (data / "manifest.json").write_text('{"target_mode":"cot"}')
    config = repo / "training/configs/qwen3-8b-cot.json"
    output = tmp_path / "cot-bundle"
    build_bundle(repo, data, output, config=config)
    with tarfile.open(output / "payload.tar.gz") as archive:
        worker_config = archive.extractfile("training/configs/qwen3-8b.json")
        assert worker_config is not None
        assert json.load(worker_config) == json.loads(config.read_text())
        assert (
            json.loads((output / "model-config.json").read_text())["target_mode"]
            == "cot"
        )
        assert "data/sft/train.jsonl" in archive.getnames()
    assert "cot-$(mode)-" in (output / "train.sub").read_text()
    (data / "manifest.json").write_text('{"target_mode":"action"}')
    with pytest.raises(ValueError, match="different target modes"):
        build_bundle(repo, data, tmp_path / "wrong-mode", config=config)


@pytest.mark.parametrize("mode", ["smoke", "train"])
def test_cot_worker_smoke_resume_and_full_training_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import torch
    from training.chtc import job

    repo = Path(__file__).resolve().parents[2]
    config = json.loads((repo / "training/configs/qwen3-8b-cot.json").read_text())
    config_dir = tmp_path / "training/configs"
    config_dir.mkdir(parents=True)
    (config_dir / "qwen3-8b.json").write_text(json.dumps(config))
    data = tmp_path / "data/sft"
    data.mkdir(parents=True)
    for split in ("train", "validation"):
        (data / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"reasoning": "r" * size}) for size in range(1, 130))
            + "\n"
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["job", mode])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "test GPU")
    calls = []
    outcomes = iter([85, 0] if mode == "smoke" else [0])

    def run(command: list[str], check: bool) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=next(outcomes))

    monkeypatch.setattr(job.subprocess, "run", run)
    with pytest.raises(SystemExit) as first_exit:
        job.main()
    assert first_exit.value.code == (85 if mode == "smoke" else 0)
    options = json.loads((tmp_path / "job-config.json").read_text())["training"]
    if mode == "smoke":
        assert options["gradient_accumulation_steps"] == 1
        longest = json.loads(
            (tmp_path / "smoke-data/train.jsonl").read_text().splitlines()[0]
        )
        assert len(longest["reasoning"]) == 129
        checkpoint = tmp_path / "outputs/adapter/checkpoint-1"
        checkpoint.mkdir(parents=True)
        with pytest.raises(SystemExit) as resumed_exit:
            job.main()
        assert resumed_exit.value.code == 0
        assert "--resume-from-checkpoint" in calls[1]
        assert calls[1][-1] == "0"
    else:
        assert options["gradient_accumulation_steps"] == 16
        assert options["num_train_epochs"] == 3
        assert options["save_steps"] == options["eval_steps"] == 250
        assert calls[0][-1] == "10800"
    result = json.loads((tmp_path / "outputs/job-result.json").read_text())
    assert result == {"mode": mode, "target_mode": "cot", "training_exit_code": 0}
    assert (tmp_path / "results.tar.gz").is_file()


def test_gameplay_bundle_runs_without_repository_or_dataset(tmp_path: Path) -> None:
    pytest.importorskip("pygame")
    pytest.importorskip("gym")
    output = tmp_path / "bundle"
    build_bundle(
        Path(__file__).resolve().parents[2],
        tmp_path / "absent-data",
        output,
        gameplay=True,
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with tarfile.open(output / "gameplay-payload.tar.gz") as archive:
        assert not any(name.startswith("data/") for name in archive.getnames())
        archive.extractall(sandbox, filter="data")
    subprocess.run(
        [sys.executable, "-m", "training.overcooked", "--help"],
        cwd=sandbox,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from training.play import enable_games, run_episode
enable_games()
for game in ('freeway', 'snake', 'overcooked'):
    result = run_episode(lambda prompt: 'S', game, 'easy', 6, Path(game), 2, 'error', True)
    assert result['steps'] == 2
    assert Path(game, 'episode.gif').is_file()
""",
        ],
        cwd=sandbox,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
