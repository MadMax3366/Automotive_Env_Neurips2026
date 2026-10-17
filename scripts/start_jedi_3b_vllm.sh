#!/usr/bin/env bash
# Start a vLLM server hosting xlangai/Jedi-3B-1080p.
#
# Same recipe as start_jedi_vllm.sh (which hosts the 7B variant) but
# targeting the 3B weights and port 4247 so both Jedi sizes can run
# concurrently:
#   Jedi-7B  -> port 4245
#   Jedi-3B  -> port 4247
#
# Usage:
#   scripts/start_jedi_3b_vllm.sh                 # GPU 0, port 4247
#   GPU=2 PORT=4247 scripts/start_jedi_3b_vllm.sh
set -euo pipefail

MODEL="${MODEL:-./models/Jedi-3B-1080p}"
SERVED="${SERVED:-jedi-3b-1080p}"
PORT="${PORT:-4247}"
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

echo "[jedi-3b-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[jedi-3b-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
