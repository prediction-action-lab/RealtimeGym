#!/usr/bin/env bash
set -Eeuo pipefail

# Snake SFT using the analysis/ pipeline (build_sft.py -> train_sft.py).
# Run from a directory holding .env and RealtimeGym.tar.gz, e.g. inside a
# `condor_submit -i job_i.sub` session:
#   bash job_ft_analysis.sh
#
# Trains the two Snake adapters that job.sh evaluates, in one job:
#   reactive: answer-only targets, thinking OFF  -> outputs/snake_reactive_adapter
#   planning: full <think> trace + plan, thinking ON -> outputs/snake_planning_adapter
#
# Then (RUN_EVAL=1, default) it runs job.sh in the same job: vLLM serves both adapters and
# the Snake evals run. Results go to staging (printed at the end by job.sh).
#
# Knobs (env vars): MODE=both|reactive|planning   RANK=32  RUN_TRAIN=1  RUN_EVAL=1   LIMIT=0 (e.g. 200 = smoke test)
#   EPOCHS_REACTIVE=3  EPOCHS_PLANNING=2  MAX_LEN_PLANNING=16384

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"
: "${CHTC_USER:?CHTC_USER must be set in .env}"
: "${_CONDOR_SCRATCH_DIR:?run this inside a CHTC job}"

MODE="${MODE:-both}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RANK="${RANK:-32}"
LIMIT="${LIMIT:-0}"
EPOCHS_REACTIVE="${EPOCHS_REACTIVE:-3}"
EPOCHS_PLANNING="${EPOCHS_PLANNING:-2}"
MAX_LEN_PLANNING="${MAX_LEN_PLANNING:-16384}"

SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}"
PROJECT_DIR="${SCRATCH_DIR}/RealtimeGym"
STAGING_DIR="${FINETUNE_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/realtimegym-snake-sft}"
OUT_ROOT="${STAGING_DIR}/outputs"

export HOME="${SCRATCH_DIR}"
export HF_HOME="${SCRATCH_DIR}/hf_home"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/models"
export XDG_CACHE_HOME="${SCRATCH_DIR}/xdg_cache"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"
mkdir -p "${HF_HOME}" "${OUT_ROOT}"

if [[ ! -d "${PROJECT_DIR}/analysis" ]]; then
    tar -xzf "${SCRIPT_DIR}/RealtimeGym.tar.gz" -C "${SCRATCH_DIR}"
fi
cd "${PROJECT_DIR}"

# The container already ships torch/transformers; only add peft if missing.
python3 -c "import peft" 2>/dev/null || python3 -m pip install "peft>=0.18,<0.19"
[[ -n "${HF_TOKEN:-}" ]] && hf auth login --token "${HF_TOKEN}" || true

run_one() {
    local mode="$1"; shift
    local out="${OUT_ROOT}/snake_${mode}_adapter"
    local data="${SCRATCH_DIR}/sft_snake_${mode}.jsonl"
    local style=full; [[ "${mode}" == reactive ]] && style=short
    echo "== build Snake data (${mode}, style=${style}) -> ${data}"
    python3 analysis/build_sft.py --game snake --mode "${mode}" --style "${style}" --out "${data}"
    wc -l "${data}"
    echo "== LoRA SFT (${mode}) -> ${out}"
    python3 analysis/train_sft.py --data "${data}" --base Qwen/Qwen3-8B --out "${out}" \
        --rank "${RANK}" --limit "${LIMIT}" "$@"
    ls -la "${out}"
}

if [[ "${RUN_TRAIN}" == 1 ]]; then
    # --thinking auto: each row carries enable_thinking (reactive=false, planning=true)
    if [[ "${MODE}" == both || "${MODE}" == reactive ]]; then
        run_one reactive --thinking off --epochs "${EPOCHS_REACTIVE}" --max-len 8192
    fi
    if [[ "${MODE}" == both || "${MODE}" == planning ]]; then
        # ~10k-token traces; no action rebalancing (a plan is not one action)
        run_one planning --thinking on --epochs "${EPOCHS_PLANNING}" --max-len "${MAX_LEN_PLANNING}" \
            --balance none --accum 8
    fi
fi

echo "Adapters in ${OUT_ROOT}:"
ls "${OUT_ROOT}"

if [[ "${RUN_EVAL}" == 1 ]]; then
    # job.sh expects to run from the scratch dir (holds .env and the archive),
    # and needs the training process gone so vLLM can claim the GPU.
    cd "${SCRATCH_DIR}"
    echo "== evaluating both adapters on Snake (job.sh)"
    MAX_LORA_RANK="${RANK}" FINETUNE_STAGING_DIR="${STAGING_DIR}" bash "${SCRIPT_DIR}/job.sh" 0 0
fi
