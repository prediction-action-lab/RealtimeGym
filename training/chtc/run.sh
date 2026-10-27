#!/bin/bash
set -euo pipefail
tar -xzf payload.tar.gz
tar -xf qwen3-8b-cache.tar
export HF_HOME="$PWD/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec python -m training.chtc.job "$@"
