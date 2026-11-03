#!/bin/bash
set -euo pipefail
mkdir -p outputs
# Return logs and any completed episode summaries even when a later episode fails.
finish() {
    status=$?
    trap - EXIT
    tar -czf overcooked-results.tar.gz outputs || exit 1
    exit "$status"
}
trap finish EXIT

tar -xzf gameplay-payload.tar.gz
tar -xf qwen3-8b-cache.tar
mkdir -p trained
tar -xzf "$3" -C trained
export HF_HOME="$PWD/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
adapter_args=()
target_mode="${5:-action}"
case "$target_mode" in
    action|cot) ;;
    *) echo "Unknown target mode: $target_mode" >&2; exit 2 ;;
esac
case "$2" in
    adapter) adapter_args=(--adapter trained/outputs/adapter) ;;
    base) ;;
    *) echo "Unknown policy: $2" >&2; exit 2 ;;
esac
if [[ "$target_mode" == cot && "$2" == adapter ]]; then
    # A successful smoke job also exits zero; require completed full CoT training.
    python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("trained/outputs/job-result.json").read_text())
if result != {"mode": "train", "target_mode": "cot", "training_exit_code": 0}:
    raise SystemExit(f"Expected completed full CoT training, got: {result}")
PY
fi
python -m training.overcooked --difficulties "$1" --tokens-per-step "$4" \
    --target-mode "$target_mode" \
    --output-dir outputs/overcooked --save-gif "${adapter_args[@]}"
