#!/usr/bin/env bash
# Start a vLLM server hosting UI-TARS-1.5-7B (xLangAI's GUI agent
# trained on the MOBILE_USE_DOUBAO / COMPUTER_USE_DOUBAO prompt format).
#
# UI-TARS is a Qwen2-VL-7B fine-tune. vLLM supports it natively via the
# Qwen2-VL backbone; no trust_remote_code needed.
#
# Usage:
#   scripts/start_ui_tars_vllm.sh                      # GPU 0, port 4246
#   GPU=2 PORT=4246 scripts/start_ui_tars_vllm.sh
#
# Default ports across the repo (adjust if collisions):
#   OpenCUA  -> 4243
#   GUI-Owl  -> 4244
#   Jedi     -> 4245
#   UI-TARS  -> 4246
set -euo pipefail

MODEL="${MODEL:-./models/UI-TARS-1.5-7B}"
SERVED="${SERVED:-ui-tars-1.5-7b}"
PORT="${PORT:-4246}"
HOST="${HOST:-127.0.0.1}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"

IMAGE_LIMIT="${IMAGE_LIMIT:-}"
[[ -z "$IMAGE_LIMIT" ]] && IMAGE_LIMIT='{"image": 2}'

VLLM_BIN="vllm"
if [[ -x ".venv-vllm/bin/vllm" ]]; then
    VLLM_BIN=".venv-vllm/bin/vllm"
fi

echo "[ui-tars-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[ui-tars-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
