"""Download pinned weights once in a CHTC build job, preserving HF cache paths."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    config = json.loads(Path("model-config.json").read_text())
    output = Path("qwen3-8b-cache.tar")
    if output.exists():
        raise FileExistsError(output)
    snapshot_download(
        config["model"],
        revision=config["revision"],
        cache_dir="hf-cache/hub",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    # Keep blob/snapshot symlinks: they point within this same archive.
    with tarfile.open(output, "w") as archive:
        archive.add("hf-cache", arcname="hf-cache")
    print(f"Created {output}: {output.stat().st_size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
