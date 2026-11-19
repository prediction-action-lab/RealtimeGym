#!/usr/bin/env bash
set -Eeuo pipefail

# RealtimeGym fine-tuning pipeline for a CHTC interactive GPU allocation.
#
# Usage:
#   bash job_ft.sh                  # run every stage
#   bash job_ft.sh sft              # run one stage
#   bash job_ft.sh prepare,sft      # run selected stages in order
#
# Supported stages: prepare, sft, merge, preferences, dpo, evaluate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
ARCHIVE_PATH="${SCRIPT_DIR}/RealtimeGym.tar.gz"
REQUESTED_STAGES="${1:-all}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"
: "${CHTC_USER:?CHTC_USER must be set in ${ENV_FILE}}"

if [[ -z "${_CONDOR_SCRATCH_DIR:-}" ]]; then
    echo "_CONDOR_SCRATCH_DIR is not set. Run this inside a CHTC job." >&2
    exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Missing ${ARCHIVE_PATH}. Check transfer_input_files in job_i.sub." >&2
    exit 1
fi

SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}"
PROJECT_NAME="RealtimeGym"
PROJECT_DIR="${SCRATCH_DIR}/${PROJECT_NAME}"
FT_DIR="${PROJECT_DIR}/logs/finetune_realtimegym"
STAGING_DIR="${FINETUNE_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/realtimegym-finetune}"

export HOME="${SCRATCH_DIR}"
export HF_HOME="${SCRATCH_DIR}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/datasets"
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

stage_selected() {
    local stage="$1"
    [[ "${REQUESTED_STAGES}" == "all" || ",${REQUESTED_STAGES}," == *",${stage},"* ]]
}

validate_stages() {
    local value
    local stage
    if [[ "${REQUESTED_STAGES}" == "all" ]]; then
        return
    fi
    IFS=',' read -r -a selected <<< "${REQUESTED_STAGES}"
    for value in "${selected[@]}"; do
        case "${value}" in
            prepare|sft|merge|preferences|dpo|evaluate) ;;
            *)
                echo "Unknown stage: ${value}" >&2
                echo "Use: prepare,sft,merge,preferences,dpo,evaluate or all" >&2
                exit 2
                ;;
        esac
    done
}

copy_directory_contents() {
    local source="$1"
    local destination="$2"
    if [[ -d "${source}" ]]; then
        mkdir -p "${destination}"
        cp -a "${source}/." "${destination}/"
    fi
}

restore_artifacts() {
    echo "Restoring existing fine-tuning artifacts from ${STAGING_DIR} when present..."
    copy_directory_contents "${STAGING_DIR}/outputs" "${FT_DIR}/outputs"
    copy_directory_contents "${STAGING_DIR}/prepared" "${FT_DIR}/data/prepared"
    copy_directory_contents "${STAGING_DIR}/preferences" "${FT_DIR}/data/preferences"
}

persist_artifacts() {
    echo "Persisting fine-tuning artifacts to ${STAGING_DIR}..."
    copy_directory_contents "${FT_DIR}/outputs" "${STAGING_DIR}/outputs"
    copy_directory_contents "${FT_DIR}/data/prepared" "${STAGING_DIR}/prepared"
    copy_directory_contents "${FT_DIR}/data/preferences" "${STAGING_DIR}/preferences"
}

on_exit() {
    local status=$?
    trap - EXIT
    if [[ -d "${FT_DIR}" ]]; then
        persist_artifacts || echo "Warning: final staging copy failed." >&2
    fi
    echo "Fine-tuning job finished with exit code ${status}."
    exit "${status}"
}
trap on_exit EXIT

run_stage() {
    local name="$1"
    shift
    if ! stage_selected "${name}"; then
        return
    fi
    echo
    echo "======================================================================"
    echo "Starting stage: ${name} at $(date)"
    echo "======================================================================"
    "$@"
    persist_artifacts
    echo "Completed stage: ${name} at $(date)"
}

validate_stages

echo "Extracting ${ARCHIVE_PATH} into ${SCRATCH_DIR}..."
tar -xzf "${ARCHIVE_PATH}" -C "${SCRATCH_DIR}"

if [[ ! -d "${FT_DIR}" ]]; then
    echo "Fine-tuning package not found at ${FT_DIR}." >&2
    exit 1
fi

echo "Installing RealtimeGym and fine-tuning dependencies..."
python3 -m pip install -e "${PROJECT_DIR}"
python3 -m pip install -r "${FT_DIR}/requirements.txt"
python3 -m pip install -e "${FT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}"
else
    echo "HF_TOKEN is not set; continuing without Hugging Face authentication."
fi

export PYTHONPATH="${PROJECT_DIR}:${FT_DIR}:${PYTHONPATH:-}"
cd "${FT_DIR}"
restore_artifacts

echo "CUDA visibility check:"
python3 - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the CHTC job")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

run_stage prepare \
    python3 scripts/prepare_data.py --config configs/prepare.yaml

run_stage sft \
    accelerate launch --num_machines 1 --num_processes 1 \
    scripts/train_sft.py --config configs/sft.yaml

run_stage merge \
    python3 scripts/merge_adapter.py --config configs/merge.yaml

run_stage preferences \
    python3 scripts/generate_preferences.py --config configs/preferences.yaml

run_stage dpo \
    accelerate launch --num_machines 1 --num_processes 1 \
    scripts/train_dpo.py --config configs/dpo.yaml

run_stage evaluate \
    python3 scripts/evaluate.py --config configs/evaluate.yaml

echo
echo "All requested stages completed successfully."
echo "Persistent results: ${STAGING_DIR}"
