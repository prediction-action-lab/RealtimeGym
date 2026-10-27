#!/bin/bash
set -euo pipefail
echo "Extracting gameplay code and model cache"
tar -xzf gameplay-payload.tar.gz
tar -xf qwen3-8b-cache.tar
mkdir -p trained outputs
tar -xzf "$5" -C trained
export HF_HOME="$PWD/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
adapter_args=()
case "$4" in
    adapter) adapter_args=(--adapter trained/outputs/adapter) ;;
    base) ;;
    *) echo "Unknown policy: $4" >&2; exit 2 ;;
esac
echo "Playing $1 / $2 / seed $3 with $4"
status=0
python -m training.play --game "$1" --difficulty "$2" --seeds "$3" \
    --output-dir outputs/gameplay --save-gif "${adapter_args[@]}" || status=$?
tar -czf gameplay-results.tar.gz outputs
exit "$status"
