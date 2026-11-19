#!/usr/bin/env bash
# Two separate SFT runs from the SALT-NLP/RealtimeGym trajectories:
#   reactive : answer-only targets, enable_thinking=False  (matches the reactive eval config)
#   planning : full <think> trace + plan, enable_thinking=True (matches the planning eval config)
# Usage: bash analysis/run_sft_split.sh [game] [outdir] [base_model]
set -euo pipefail
GAME="${1:-freeway}"; OUT="${2:-sft_out}"; BASE="${3:-Qwen/Qwen3-8B}"
mkdir -p "$OUT"

python analysis/build_sft.py --game "$GAME" --mode reactive --style short --out "$OUT/${GAME}_reactive.jsonl"
python analysis/build_sft.py --game "$GAME" --mode planning --out "$OUT/${GAME}_planning.jsonl"

# reactive: short targets, so the default 8k window is plenty
python analysis/train_sft.py --data "$OUT/${GAME}_reactive.jsonl" --base "$BASE" \
    --thinking off --out "$OUT/${GAME}_reactive_adapter"

# planning: traces are ~8-10k tokens; keep the whole <think>...plan, no action
# rebalancing (a plan is not one action), and use fewer epochs on the small set
python analysis/train_sft.py --data "$OUT/${GAME}_planning.jsonl" --base "$BASE" \
    --thinking on --max-len 16384 --balance none --accum 8 --out "$OUT/${GAME}_planning_adapter"
