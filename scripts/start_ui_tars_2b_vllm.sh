#!/usr/bin/env bash
# Start a vLLM server hosting ByteDance-Seed/UI-TARS-2B-SFT.
#
# Same recipe as the 7B start scripts but targets the 2B-SFT weights
# and port 4250 so all UI-TARS variants can run concurrently:
#   UI-TARS-1.5-7B  -> 4246
#   UI-TARS-7B-DPO  -> 4248
#   UI-TARS-7B-SFT  -> 4249
#   UI-TARS-2B-SFT  -> 4250
#
# Usage:
#   scripts/start_ui_tars_2b_vllm.sh                  # GPU 0, port 4250
#   GPU=2 PORT=4250 scripts/start_ui_tars_2b_vllm.sh
set -euo pipefail

MODEL="${MODEL:-./models/UI-TARS-2B-SFT}"
SERVED="${SERVED:-ui-tars-2b-sft}"
PORT="${PORT:-4250}"
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

echo "[ui-tars-2b-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[ui-tars-2b-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
