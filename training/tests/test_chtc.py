"""Verify packaging boundaries without a cluster or a model download."""

from pathlib import Path
import subprocess
import sys
import tarfile

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
