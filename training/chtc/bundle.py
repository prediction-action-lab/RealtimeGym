"""Package explicit training inputs and CHTC setup files, excluding local secrets."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def build_bundle(repo: Path, data: Path, output: Path, gameplay: bool = False) -> None:
    required = [data / f"{split}.jsonl" for split in ("train", "validation", "test")]
    required.append(data / "manifest.json")
    if gameplay:
        required = []
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Bundle output must be empty: {output}")
    chtc = repo / "training/chtc"
    for name in (
        "image.def",
        "build.sub",
        "train.sub",
        "run.sh",
        "stage_model.py",
        "README.md",
    ):
        shutil.copy2(chtc / name, output / name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(repo / "training" / name, output / name)
    shutil.copy2(repo / "training/configs/qwen3-8b.json", output / "model-config.json")
    (output / "run.sh").chmod(0o755)
    if gameplay:
        for name in (
            "play-image.def",
            "play-build.sub",
            "play.sub",
            "play.sh",
            "PLAY.md",
            "overcooked.sub",
            "overcooked.sh",
            "OVERCOOKED.md",
        ):
            shutil.copy2(chtc / name, output / name)
        (output / "play.sh").chmod(0o755)
        (output / "overcooked.sh").chmod(0o755)
    # Explicit source allowlist: never archive the whole repo or a virtualenv.
    sources = sorted((repo / "training").glob("*.py"))
    sources += sorted(chtc.glob("*.py"))
    sources.append(repo / "training/configs/qwen3-8b.json")
    if gameplay:
        allowed = {".py", ".yaml", ".json", ".png", ".ttf", ".layout"}
        sources += sorted(
            path
            for path in (repo / "src/realtimegym").rglob("*")
            if path.is_file()
            and path.suffix in allowed
            and not {"__pycache__", "testing"}.intersection(path.parts)
        )
        sources += [
            repo / "src/realtimegym/environments/overcooked_new/LICENSE",
            repo / "src/realtimegym/environments/overcooked_new/THIRD_PARTY_NOTICE.md",
        ]
    archive_name = "gameplay-payload.tar.gz" if gameplay else "payload.tar.gz"
    with tarfile.open(output / archive_name, "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(repo)))
        for path in required:
            archive.add(path, arcname=f"data/sft/{path.name}")
    print(f"Bundle ready: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gameplay", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or Path(
        "outputs/chtc-play-bundle" if args.gameplay else "outputs/chtc-bundle"
    )
    build_bundle(
        Path(__file__).resolve().parents[2], args.data_dir, output, args.gameplay
    )


if __name__ == "__main__":
    main()
