#!/usr/bin/env bash
# Start a vLLM server hosting xlangai/Jedi-7B-1080p.
#
# Jedi-7B is a Qwen2.5-VL-7B-Instruct fine-tune for high-resolution
# GUI grounding (1080p resolution profile). vLLM supports it natively
# via the Qwen2.5-VL backbone — no trust_remote_code needed.
#
# Usage:
#   scripts/start_jedi_vllm.sh                   # GPU 0, port 4245
#   GPU=2 PORT=4245 scripts/start_jedi_vllm.sh
#
# Default ports across the repo (adjust if collisions):
#   OpenCUA  -> 4243
#   GUI-Owl  -> 4244
#   Jedi     -> 4245
set -euo pipefail

MODEL="${MODEL:-./models/Jedi-7B-1080p}"
SERVED="${SERVED:-jedi-7b-1080p}"
PORT="${PORT:-4245}"
HOST="${HOST:-127.0.0.1}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 1080p screenshots can be large — bump max_pixels to fit a single
# 1080p frame (~2 MP) without aggressive downscale.
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"

IMAGE_LIMIT="${IMAGE_LIMIT:-}"
[[ -z "$IMAGE_LIMIT" ]] && IMAGE_LIMIT='{"image": 2}'

VLLM_BIN="vllm"
if [[ -x ".venv-vllm/bin/vllm" ]]; then
    VLLM_BIN=".venv-vllm/bin/vllm"
fi

echo "[jedi-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[jedi-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
