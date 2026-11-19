#!/usr/bin/env bash
set -Eeuo pipefail

# SFT-only QLoRA training for Qwen3-8B on official RealtimeGym Snake data.
# This script intentionally contains no DPO, RL, preference generation, merge,
# or evaluation stage.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
ARCHIVE_PATH="${SCRIPT_DIR}/RealtimeGym.tar.gz"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}. Copy .env.template to .env and fill it in." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"
: "${CHTC_USER:?CHTC_USER must be set in ${ENV_FILE}}"

if [[ -z "${_CONDOR_SCRATCH_DIR:-}" ]]; then
    echo "_CONDOR_SCRATCH_DIR is not set; run this through HTCondor." >&2
    exit 1
fi
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Missing transferred archive ${ARCHIVE_PATH}." >&2
    exit 1
fi

SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}"
PROJECT_DIR="${SCRATCH_DIR}/RealtimeGym"
FT_DIR="${PROJECT_DIR}/logs/finetune_snake_qwen"
DATA_DIR="${PROJECT_DIR}/datasets/snake_sft_official_realtimegym/realtimegym"
STAGING_DIR="${FINETUNE_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/realtimegym-snake-sft}"
RUN_OUTPUT_DIR="${FT_DIR}/outputs/qwen3-8b-snake-sft-official-r1"

export HOME="${SCRATCH_DIR}"
export HF_HOME="${SCRATCH_DIR}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/datasets_cache"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/models"
export XDG_CACHE_HOME="${SCRATCH_DIR}/xdg_cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/xdg_config"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/torch_cache"
export TORCH_COMPILE_CACHE_DIR="${SCRATCH_DIR}/torch_compile_cache"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_COMPILE_CACHE_DIR}" \
    "${STAGING_DIR}"

copy_directory_contents() {
    local source="$1"
    local destination="$2"
    if [[ -d "${source}" ]]; then
        mkdir -p "${destination}"
        cp -a "${source}/." "${destination}/"
    fi
}

persist_outputs() {
    if [[ -d "${FT_DIR}/outputs" ]]; then
        echo "Persisting checkpoints to ${STAGING_DIR}/outputs..."
        copy_directory_contents "${FT_DIR}/outputs" "${STAGING_DIR}/outputs"
    fi
}

on_exit() {
    local status=$?
    trap - EXIT
    if [[ -d "${FT_DIR}" ]]; then
        persist_outputs || echo "Warning: final staging copy failed." >&2
    fi
    echo "Snake SFT job finished with exit code ${status}."
    exit "${status}"
}
trap on_exit EXIT

echo "Extracting project archive..."
tar -xzf "${ARCHIVE_PATH}" -C "${SCRATCH_DIR}"

if [[ ! -d "${FT_DIR}" ]]; then
    echo "Missing fine-tuning package ${FT_DIR}." >&2
    exit 1
fi
echo "Installing SFT dependencies..."
python3 -m pip install -r "${FT_DIR}/requirements.txt"
python3 -m pip install -e "${FT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}"
else
    echo "HF_TOKEN is unset; public model download may still work."
fi

export PYTHONPATH="${FT_DIR}:${PYTHONPATH:-}"
cd "${FT_DIR}"

echo "Downloading and preparing official RealtimeGym Snake planning traces..."
python3 scripts/prepare_official_snake_dataset.py --output-dir "${DATA_DIR}"
if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/validation.jsonl" ]]; then
    echo "Official Snake dataset preparation did not create train/validation data." >&2
    exit 1
fi
echo "Official dataset manifest:"
cat "${DATA_DIR}/manifest.json"

# Resume support: copy any previously staged Trainer checkpoints back first.
copy_directory_contents "${STAGING_DIR}/outputs" "${FT_DIR}/outputs"
mkdir -p "${RUN_OUTPUT_DIR}"
cp -f "${DATA_DIR}/manifest.json" "${RUN_OUTPUT_DIR}/dataset_manifest.json"

echo "CUDA visibility check:"
python3 - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the CHTC job")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY

echo "Starting Qwen3-8B Snake QLoRA SFT at $(date)..."
accelerate launch --num_machines 1 --num_processes 1 \
    scripts/train_sft.py --config configs/sft.yaml

persist_outputs
echo "Training completed at $(date)."
echo "Persistent adapter and checkpoints: ${STAGING_DIR}/outputs"
