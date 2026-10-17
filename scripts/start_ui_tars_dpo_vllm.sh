#!/usr/bin/env bash
# Start a vLLM server hosting ByteDance-Seed/UI-TARS-7B-DPO.
#
# Same recipe as start_ui_tars_vllm.sh (which hosts 1.5-7B) but
# targets the DPO variant and port 4248 so both UI-TARS sizes can
# run concurrently:
#   UI-TARS-1.5-7B  -> 4246
#   UI-TARS-7B-DPO  -> 4248
#   UI-TARS-7B-SFT  -> 4249
#
# Usage:
#   scripts/start_ui_tars_dpo_vllm.sh                 # GPU 0, port 4248
#   GPU=2 PORT=4248 scripts/start_ui_tars_dpo_vllm.sh
set -euo pipefail

MODEL="${MODEL:-./models/UI-TARS-7B-DPO}"
SERVED="${SERVED:-ui-tars-7b-dpo}"
PORT="${PORT:-4248}"
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

echo "[ui-tars-dpo-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[ui-tars-dpo-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt "$IMAGE_LIMIT"
