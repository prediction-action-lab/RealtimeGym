#!/bin/bash
set -o pipefail

# RealtimeGym CHTC Interactive Job Script
# Adapted from verl template for real-time reasoning evaluation
# Supports time pressure control: token budget or wall-clock time

source .env
: "${CHTC_USER:?CHTC_USER must be set in .env}"

pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1

# ==================== ENVIRONMENT SETUP ====================
# Set all paths to use local scratch directory for performance

export HOME=$_CONDOR_SCRATCH_DIR
export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics
export HF_HOME=$_CONDOR_SCRATCH_DIR/hf_home

# vLLM 0.12 expects numeric GPU indices.
export CUDA_VISIBLE_DEVICES=0

export TORCHINDUCTOR_CACHE_DIR=$_CONDOR_SCRATCH_DIR/torch_cache
export TORCH_COMPILE_CACHE=$_CONDOR_SCRATCH_DIR/torch_compile_cache

# Cache directories
export XDG_CACHE_HOME=$_CONDOR_SCRATCH_DIR/xdg_cache
export XDG_CONFIG_HOME=$_CONDOR_SCRATCH_DIR/xdg_config

# Ray configuration (for distributed evaluation if needed)
export USER=condor
export LOGNAME=condor
export USERNAME=condor
export RAY_TMPDIR=/tmp/ray_$USER

# Python configuration
export PYTHONUNBUFFERED=1

# Fine-tuned Snake adapter produced by job_ft.sub. vLLM serves this PEFT
# adapter on top of the original Qwen3-8B weights, so no merge step is needed.
# Two adapters, one per agent, both served by the same vLLM on top of the base
# Qwen3-8B weights (train them with analysis/run_sft_split.sh):
#   reactive adapter (thinking OFF) -> served as snake-reactive
#   planning adapter (thinking ON)  -> served as snake-planning
# The model names must match `model:` in the two configs/qwen-3-8B-snake-sft-*.yaml.
REACTIVE_ALIAS="snake-reactive"
PLANNING_ALIAS="snake-planning"
FINETUNE_STAGING_DIR="${FINETUNE_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/realtimegym-snake-sft}"
REACTIVE_SOURCE="${REACTIVE_SOURCE:-${FINETUNE_STAGING_DIR}/outputs/snake_reactive_adapter}"
PLANNING_SOURCE="${PLANNING_SOURCE:-${FINETUNE_STAGING_DIR}/outputs/snake_planning_adapter}"
REACTIVE_DIR="${_CONDOR_SCRATCH_DIR}/snake_reactive_adapter"
PLANNING_DIR="${_CONDOR_SCRATCH_DIR}/snake_planning_adapter"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
EVAL_STAGING_DIR="${EVAL_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/logs}"
RESULTS_STAGING_DIR="${RESULTS_STAGING_DIR:-/staging/${CHTC_USER:0:1}/${CHTC_USER}/realtimegym-snake-eval}"

# CHTC sets HTTP(S) proxy variables; local vLLM calls must bypass Squid.
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

# ==================== CODE FETCH & SETUP ====================

CODENAME=RealtimeGym
ARCHIVE_PATH="${PWD}/${CODENAME}.tar.gz"
echo "Extracting transferred code archive ${ARCHIVE_PATH}"
if [[ ! -s "${ARCHIVE_PATH}" ]]; then
  echo "Missing transferred project archive: ${ARCHIVE_PATH}" >&2
  exit 1
fi
tar -xzf "${ARCHIVE_PATH}"
rm "${ARCHIVE_PATH}"
cd "${CODENAME}"

# Ensure python looks in the current folder
export PYTHONPATH=$(pwd):$PYTHONPATH

# ==================== AUTHENTICATION ====================

# Weights & Biases authentication
wandb login ${WANDB_API_KEY}

# HuggingFace authentication
hf auth login --token ${HF_TOKEN}

# ==================== DEPENDENCY INSTALLATION ====================

echo "Installing RealtimeGym and dependencies..."

# Install RealtimeGym in editable mode
pip install -e . --quiet

# Install development dependencies
pip install -e ".[dev]" --quiet

# Install numpy compatibility fix for Overcooked environment
pip install "numpy<2.0" --quiet

echo "✓ Dependencies installed"

# ==================== REALTIMEGYM EVALUATION ====================

# You can modify these parameters for different experiments
# ========================================================

# Output configuration. Keep the complete evaluation output in CHTC staging so
# it survives job completion/eviction even if Condor's output transfer fails.
LOG_DIR="${EVAL_STAGING_DIR}/snake_sft_eval_${pid}"
VLLM_LOG="${LOG_DIR}/vllm.log"
SAVE_GIFS=false                # set to true to save trajectory GIFs
mkdir -p "${LOG_DIR}"
echo "Evaluation logs will be saved persistently in ${LOG_DIR}"

# ========================================================

export USER=condor
export LOGNAME=condor
export USERNAME=condor
export HOME=$_CONDOR_SCRATCH_DIR

echo "========== Starting reactive vLLM =========="

for pair in "${REACTIVE_SOURCE}:${REACTIVE_DIR}" "${PLANNING_SOURCE}:${PLANNING_DIR}"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [[ ! -s "${src}/adapter_config.json" || ! -s "${src}/adapter_model.safetensors" ]]; then
    echo "Adapter missing or incomplete: ${src}" >&2
    echo "Expected adapter_config.json and adapter_model.safetensors." >&2
    exit 1
  fi
  echo "Copying adapter ${src}..."
  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
done

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enable-lora \
  --lora-modules "${REACTIVE_ALIAS}=${REACTIVE_DIR}" "${PLANNING_ALIAS}=${PLANNING_DIR}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-loras 2 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
until curl --noproxy '*' -sf http://127.0.0.1:8000/ping >/dev/null; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  echo "Waiting for vLLM..."
  tail -20 "${VLLM_LOG}"
  sleep 5
done

echo "vLLM models:"
curl --noproxy '*' -sf http://127.0.0.1:8000/v1/models
echo

# echo "========== Starting planning vLLM =========="

# CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
#   --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#   --host 127.0.0.1 \
#   --port 8001 \
#   --gpu-memory-utilization 0.9 \
#   > vllm.log 2>&1 &
# VLLM_PID=$!
# until curl --noproxy '*' -sf http://127.0.0.1:8001/ping >/dev/null; do
#   echo "Waiting for vLLM..."
#   tail -20 vllm.log
#   sleep 5
# done

mkdir -p "${RESULTS_STAGING_DIR}"

# Experiment configuration
GAME="snake"                   # freeway, snake, or overcooked
COGNITIVE_LOADS=("E" "M" "H")
MODES=("reactive" "planning")
TIME_UNIT="token"
TIME_PRESSURES=(8192)

# Reactive agent's internal reasoning budget
INTERNAL_BUDGET=8192

# Evaluation scale
SEED_NUM=8
REPEAT_TIMES=4

echo "========== vLLM is ready =========="

echo "RealtimeGym Cognitive-Load Evaluation Configuration:"
echo "  Game: $GAME"
echo "  Time Unit: $TIME_UNIT"
echo "  Time Pressures: ${TIME_PRESSURES[*]}"
echo "  Internal Budget: $INTERNAL_BUDGET"
echo "  Seeds: $SEED_NUM"
echo "  Repeats per seed: $REPEAT_TIMES"
echo ""
echo "Starting evaluation at $(date)"
echo "======================================================================"
for TIME_PRESSURE in "${TIME_PRESSURES[@]}"; do
  for MODE in "${MODES[@]}"; do
    for COGNITIVE_LOAD in "${COGNITIVE_LOADS[@]}"; do
      echo "======================================================================"
      echo "  Cognitive Load: $COGNITIVE_LOAD"
      echo "  Modes: $MODE"
      echo "======================================================================"
      echo "======================================================================"
      echo "Time Pressure: $TIME_PRESSURE"
      echo "======================================================================"
      EVAL_LOG="${LOG_DIR}/realtimegym_eval_load_${pid}_${MODE}_${TIME_PRESSURE}_${COGNITIVE_LOAD}.log"
      : > "$EVAL_LOG"

      if [ "$MODE" = "planning" ]; then
        RUN_INTERNAL_BUDGET=0
      else
        RUN_INTERNAL_BUDGET="$INTERNAL_BUDGET"
      fi

      PYTHONUNBUFFERED=1 python3 -m realtimegym.agile_eval \
      --game "$GAME" \
      --cognitive_load "$COGNITIVE_LOAD" \
      --mode "$MODE" \
      --time_unit "$TIME_UNIT" \
      --time_pressure "$TIME_PRESSURE" \
      --internal_budget "$RUN_INTERNAL_BUDGET" \
      --seed_num "$SEED_NUM" \
      --repeat_times "$REPEAT_TIMES" \
      --log_dir "$LOG_DIR" \
      --reactive-model-config "configs/qwen-3-8B-snake-sft-reactive.yaml" \
      --planning-model-config "configs/qwen-3-8B-snake-sft-planning.yaml" \
      2>&1 | tee "$EVAL_LOG"
    done
  done
done

# Experiment configuration
GAME="snake"                   # freeway, snake, or overcooked
COGNITIVE_LOADS=("M")
MODES=("reactive" "planning")
TIME_UNIT="token"
TIME_PRESSURES=(4096 8192 16384 32768)

# Fixed reactive-agent internal reasoning budget
INTERNAL_BUDGET=4096

# Evaluation scale
SEED_NUM=8
REPEAT_TIMES=4

echo "RealtimeGym Time-Pressure Evaluation Configuration:"
echo "  Game: $GAME"
echo "  Time Unit: $TIME_UNIT"
echo "  Time Pressures: ${TIME_PRESSURES[*]}"
echo "  Internal Budget: $INTERNAL_BUDGET"
echo "  Seeds: $SEED_NUM"
echo "  Repeats per seed: $REPEAT_TIMES"
echo ""
echo "Starting evaluation at $(date)"
echo "======================================================================"
for TIME_PRESSURE in "${TIME_PRESSURES[@]}"; do
  for MODE in "${MODES[@]}"; do
    for COGNITIVE_LOAD in "${COGNITIVE_LOADS[@]}"; do
      echo "======================================================================"
      echo "  Cognitive Load: $COGNITIVE_LOAD"
      echo "  Modes: $MODE"
      echo "======================================================================"
      echo "======================================================================"
      echo "Time Pressure: $TIME_PRESSURE"
      echo "======================================================================"
      EVAL_LOG="${LOG_DIR}/realtimegym_eval_time_${pid}_${MODE}_${TIME_PRESSURE}_${COGNITIVE_LOAD}.log"
      : > "$EVAL_LOG"

      if [ "$MODE" = "planning" ]; then
        RUN_INTERNAL_BUDGET=0
      else
        RUN_INTERNAL_BUDGET="$INTERNAL_BUDGET"
      fi

      PYTHONUNBUFFERED=1 python3 -m realtimegym.agile_eval \
      --game "$GAME" \
      --cognitive_load "$COGNITIVE_LOAD" \
      --mode "$MODE" \
      --time_unit "$TIME_UNIT" \
      --time_pressure "$TIME_PRESSURE" \
      --internal_budget "$RUN_INTERNAL_BUDGET" \
      --seed_num "$SEED_NUM" \
      --repeat_times "$REPEAT_TIMES" \
      --log_dir "$LOG_DIR" \
      --reactive-model-config "configs/qwen-3-8B-snake-sft-reactive.yaml" \
      --planning-model-config "configs/qwen-3-8B-snake-sft-planning.yaml" \
      2>&1 | tee "$EVAL_LOG"
    done
  done
done

EVAL_EXIT_CODE=$?
echo "Eval exit code: $EVAL_EXIT_CODE"

if [ "$EVAL_EXIT_CODE" -ne 0 ]; then
    echo "========== Eval failed: likely error lines =========="
    grep -nE "Traceback|Exception|Error|404|Connection|ModuleNotFound|SyntaxError|ImportError|ValueError|RuntimeError" "$EVAL_LOG" | tail -80 || true
    echo "========== Last 120 eval log lines =========="
    tail -120 "$EVAL_LOG"
fi

if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
    wait "$VLLM_PID" 2>/dev/null || true
fi
# ==================== RESULTS PACKAGING ====================

echo ""
echo "======================================================================"
echo "Evaluation completed with exit code: $EVAL_EXIT_CODE"
echo "Time: $(date)"
echo ""

# Package results for transfer back
if [ -d "$LOG_DIR" ]; then
    echo "Packaging results..."
    tar -czf "results_${pid}.tar.gz" -C "${LOG_DIR}" .
    cp -f "results_${pid}.tar.gz" "${RESULTS_STAGING_DIR}/results_${pid}.tar.gz"
    echo "✓ Persistent logs saved in ${LOG_DIR}"
    echo "✓ Results archive saved to results_${pid}.tar.gz"
    echo "✓ Staging backup saved to ${RESULTS_STAGING_DIR}/results_${pid}.tar.gz"
fi

exit $EVAL_EXIT_CODE
