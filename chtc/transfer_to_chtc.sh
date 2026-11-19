#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PARENT="$(dirname -- "${PROJECT_DIR}")"
PROJECT_NAME="$(basename -- "${PROJECT_DIR}")"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}. Create it with CHTC_USER=<your CHTC username>." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"
: "${CHTC_USER:?CHTC_USER must be set in ${ENV_FILE}}"

CHTC_HOST="${CHTC_HOST:-ap2001.chtc.wisc.edu}"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

required_local_files=(
    "${SCRIPT_DIR}/job_ft.sub"
    "${SCRIPT_DIR}/job_ft_snake.sh"
    "${SCRIPT_DIR}/job.sh"
    "${SCRIPT_DIR}/job.sub"
    "${PROJECT_DIR}/configs/qwen-3-8B-snake-sft-reactive.yaml"
    "${PROJECT_DIR}/configs/qwen-3-8B-snake-sft-planning.yaml"
    "${PROJECT_DIR}/logs/finetune_snake_qwen/scripts/train_sft.py"
    "${PROJECT_DIR}/logs/finetune_snake_qwen/scripts/prepare_official_snake_dataset.py"
    "${PROJECT_DIR}/logs/finetune_snake_qwen/configs/sft.yaml"
)
for required_file in "${required_local_files[@]}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing required Snake SFT file: ${required_file}" >&2
        exit 1
    fi
done

TEMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TEMP_DIR}/${PROJECT_NAME}.tar.gz"

cleanup() {
    rm -f -- "${ARCHIVE_PATH}"
    rmdir -- "${TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "============================================"
echo "1. Creating transfer archive..."
echo "   Including: ${PROJECT_NAME}/logs/ and official Snake data preparation code"
echo "   Excluding: ${PROJECT_NAME}/logs0/ and ${PROJECT_NAME}/logs1/"
echo "============================================"

tar -C "${PROJECT_PARENT}" \
    --exclude="${PROJECT_NAME}/.git" \
    --exclude="${PROJECT_NAME}/.idea" \
    --exclude="${PROJECT_NAME}/.venv" \
    --exclude="${PROJECT_NAME}/logs0" \
    --exclude="${PROJECT_NAME}/logs1" \
    --exclude="${PROJECT_NAME}/snake_logs" \
    --exclude="${PROJECT_NAME}/snake_logs_1" \
    --exclude="${PROJECT_NAME}/logs/rendered_episodes" \
    --exclude="${PROJECT_NAME}/logs/rendered_episodes_test2" \
    --exclude="${PROJECT_NAME}/logs/rendered_episodes_test3" \
    --exclude="${PROJECT_NAME}/datasets/snake_sft_r1_style_synthetic" \
    --exclude="${PROJECT_NAME}/datasets/snake_sft_r1_style_synthetic.tar.gz" \
    --exclude="${PROJECT_NAME}/datasets/snake_sft_synthetic" \
    --exclude="${PROJECT_NAME}/datasets/snake_sft_synthetic.tar.gz" \
    --exclude="${PROJECT_NAME}/realtimegym_eval_0.log" \
    --exclude='*/__pycache__' \
    --exclude='*/.DS_Store' \
    --exclude='*.pyc' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='*.log' \
    -czf "${ARCHIVE_PATH}" "${PROJECT_NAME}"

archive_has_path() {
    tar -tzf "${ARCHIVE_PATH}" | awk -v prefix="$1" \
        'index($0, prefix) == 1 { found = 1; exit } END { exit !found }'
}

if archive_has_path "${PROJECT_NAME}/logs0/" || archive_has_path "${PROJECT_NAME}/logs1/"; then
    echo "Archive validation failed: an old logs directory was included." >&2
    exit 1
fi

if ! archive_has_path "${PROJECT_NAME}/logs/finetune_realtimegym/"; then
    echo "Archive validation failed: fine-tuning code is missing." >&2
    exit 1
fi
if ! archive_has_path "${PROJECT_NAME}/logs/finetune_snake_qwen/scripts/train_sft.py"; then
    echo "Archive validation failed: Snake SFT training script is missing." >&2
    exit 1
fi
if ! archive_has_path "${PROJECT_NAME}/logs/finetune_snake_qwen/scripts/prepare_official_snake_dataset.py"; then
    echo "Archive validation failed: official Snake data preparation script is missing." >&2
    exit 1
fi
if ! archive_has_path "${PROJECT_NAME}/logs/finetune_snake_qwen/configs/sft.yaml"; then
    echo "Archive validation failed: Snake SFT configuration is missing." >&2
    exit 1
fi
if ! archive_has_path "${PROJECT_NAME}/configs/qwen-3-8B-snake-sft-reactive.yaml"; then
    echo "Archive validation failed: fine-tuned reactive config is missing." >&2
    exit 1
fi
if ! archive_has_path "${PROJECT_NAME}/configs/qwen-3-8B-snake-sft-planning.yaml"; then
    echo "Archive validation failed: fine-tuned planning config is missing." >&2
    exit 1
fi

echo "Archive validated: $(du -h "${ARCHIVE_PATH}" | awk '{print $1}') compressed."

if [[ "${DRY_RUN}" == true ]]; then
    echo "Dry run complete; no network transfer was attempted."
    exit 0
fi

echo "============================================"
echo "2. Transferring archive to CHTC..."
echo "============================================"
ssh "${CHTC_USER}@${CHTC_HOST}" "mkdir -p ~/${PROJECT_NAME}"
scp "${ARCHIVE_PATH}" "${CHTC_USER}@${CHTC_HOST}:~/${PROJECT_NAME}/"

echo "============================================"
echo "3. Syncing the chtc folder to CHTC..."
echo "============================================"
rsync -avz \
    --exclude='logs*' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.err' \
    "${SCRIPT_DIR}/" "${CHTC_USER}@${CHTC_HOST}:~/${PROJECT_NAME}/chtc/"

echo "Done! The archive contains the Snake SFT package, official-data preparation code, and evaluation configs."
echo "The chtc sync includes both fine-tuning and evaluation submit files."
echo "For fine-tuning: cd ~/${PROJECT_NAME}/chtc && condor_submit job_ft.sub"
echo "For evaluation: create results/log directories, then submit job.sub."
