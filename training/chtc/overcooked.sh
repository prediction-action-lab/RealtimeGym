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
case "$2" in
    adapter) adapter_args=(--adapter trained/outputs/adapter) ;;
    base) ;;
    *) echo "Unknown policy: $2" >&2; exit 2 ;;
esac
python -m training.overcooked --difficulties "$1" --tokens-per-step "$4" \
    --output-dir outputs/overcooked --save-gif "${adapter_args[@]}"
