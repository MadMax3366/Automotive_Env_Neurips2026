#!/usr/bin/env bash
# Start a vLLM server hosting mPLUG/GUI-Owl-7B.
#
# Per the model card's recommended deployment recipe (validated on
# A100/96GB):
#   PIXEL_ARGS='{"min_pixels":3136,"max_pixels":10035200}'
#   IMAGE_LIMIT_ARGS='image=2'
#   vllm serve $CKPT --max-model-len 32768 ${MM_KWARGS[@]} \
#       --tensor-parallel-size 1 --port 4243
#
# We default --max_pixels lower (~2 MP) to fit the multi-image context
# of M3A within max_model_len; bump back to 10 MP if you need full
# resolution and don't need history.
#
# Usage:
#   scripts/start_gui_owl_vllm.sh                      # GPU 0, port 4244
#   GPU=2 PORT=4244 scripts/start_gui_owl_vllm.sh
#
# Note: pick a different --port from the OpenCUA vLLM server if both
# run concurrently. Defaults: OpenCUA -> 4243, GUI-Owl -> 4244.
set -euo pipefail

MODEL="${MODEL:-./models/GUI-Owl-7B}"
SERVED="${SERVED:-gui-owl-7b}"
PORT="${PORT:-4244}"
HOST="${HOST:-127.0.0.1}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-2007040}"
# vLLM >=0.20 expects JSON for --limit-mm-per-prompt. Override via env
# var: IMAGE_LIMIT='{"image": 5}' if you need more screenshots/prompt.
IMAGE_LIMIT="${IMAGE_LIMIT:-}"
[[ -z "$IMAGE_LIMIT" ]] && IMAGE_LIMIT='{"image": 2}'

VLLM_BIN="vllm"
if [[ -x ".venv-vllm/bin/vllm" ]]; then
    VLLM_BIN=".venv-vllm/bin/vllm"
fi

echo "[gui-owl-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[gui-owl-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
