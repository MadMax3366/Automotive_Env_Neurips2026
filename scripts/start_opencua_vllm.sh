#!/usr/bin/env bash
# Start a vLLM server hosting OpenCUA-7B (or 32B/72B).
#
# Why vLLM and not HF transformers: OpenCUA's bundled
# modeling_opencua.py keeps drifting against newer transformers
# (tie_weights signature, mm_token_type_ids, tuple indexing). vLLM
# (>=0.12.0) has its own native OpenCUA implementation, so the buggy
# custom Python code is never imported.
#
# Usage:
#   scripts/start_opencua_vllm.sh                    # 7B, port 4243, GPU 0
#   GPU=2 PORT=4243 scripts/start_opencua_vllm.sh    # explicit GPU/port
#   MODEL=./models/OpenCUA-32B SERVED=opencua-32b scripts/start_opencua_vllm.sh
set -euo pipefail

MODEL="${MODEL:-./models/OpenCUA-7B}"
SERVED="${SERVED:-opencua-7b}"
PORT="${PORT:-4243}"
HOST="${HOST:-127.0.0.1}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Per the model card's recommended pixel range for OpenCUA inference.
MIN_PIXELS="${MIN_PIXELS:-3136}"        # 56*56
MAX_PIXELS="${MAX_PIXELS:-2007040}"     # ~1280*28*28; below the card's
                                        # 10035200 cap to fit a 5-image
                                        # context within max_model_len.

# Use a separate venv for vLLM if .venv-vllm exists, otherwise fall
# back to whatever vllm is on PATH.
VLLM_BIN="vllm"
if [[ -x ".venv-vllm/bin/vllm" ]]; then
    VLLM_BIN=".venv-vllm/bin/vllm"
fi

echo "[opencua-vllm] model=$MODEL  served=$SERVED  port=$PORT  gpu=$GPU"
echo "[opencua-vllm] using $VLLM_BIN"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --trust-remote-code \
    --served-model-name "$SERVED" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --mm-processor-kwargs "{\"min_pixels\":$MIN_PIXELS,\"max_pixels\":$MAX_PIXELS}" \
    --limit-mm-per-prompt '{"image": 2}'
