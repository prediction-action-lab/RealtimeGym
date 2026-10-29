"""Single-GPU worker entrypoint; preserve checkpoints on 85, archive final outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    args = parser.parse_args()
    import torch
    from torch.version import cuda

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This job requires a BF16-capable CUDA GPU")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={cuda}"
    )
    config = json.loads(Path("training/configs/qwen3-8b.json").read_text())
    target_mode = config.get("target_mode", "action")
    print(
        f"Training target: {target_mode}; max_length={config['max_length']}", flush=True
    )
    options = config["training"]
    options.update(
        output_dir="outputs/adapter",
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=250,
        eval_steps=250,
    )
    data = Path("data/sft")
    if args.mode == "smoke":
        # Exercise real 8B loading, optimization, validation, and one forced restart.
        options.update(max_steps=5, save_steps=2, eval_steps=2)
        if target_mode == "cot":
            # Exercise the longest traces on the GPU without accumulating 16
            # full-context microbatches per smoke update. Full training keeps 16.
            options["gradient_accumulation_steps"] = 1
        data = Path("smoke-data")
        data.mkdir(exist_ok=True)
        for split, limit in (("train", 128), ("validation", 16)):
            lines = Path(f"data/sft/{split}.jsonl").read_text().splitlines()
            if target_mode == "cot":
                lines.sort(
                    key=lambda line: len(json.loads(line)["reasoning"]), reverse=True
                )
            (data / f"{split}.jsonl").write_text("\n".join(lines[:limit]) + "\n")
    Path("job-config.json").write_text(json.dumps(config, indent=2))
    output = Path(options["output_dir"])
    checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--config",
        "job-config.json",
        "--data-dir",
        str(data),
    ]
    if checkpoint:
        print(f"Resuming {checkpoint}", flush=True)
        command += ["--resume-from-checkpoint", checkpoint]
    # Three hours of training plus evaluation/save/transfer overhead per allocation.
    seconds = 10800 if args.mode == "train" else (0 if checkpoint else 0.001)
    command += ["--checkpoint-after-seconds", str(seconds)]
    result = subprocess.run(command, check=False)
    if result.returncode == 85:
        raise SystemExit(85)
    # Preserve diagnostics/checkpoints even on a training error. Do not call it success.
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/job-result.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "target_mode": target_mode,
                "training_exit_code": result.returncode,
            }
        )
        + "\n"
    )
    with tarfile.open("results.tar.gz", "w:gz") as archive:
        archive.add("outputs", arcname="outputs")
    raise SystemExit(result.returncode if result.returncode >= 0 else 1)


if __name__ == "__main__":
    main()
