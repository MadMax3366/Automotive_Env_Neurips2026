#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for MAI-UI-8B (default) or 2B.
#
# Usage:
#   bash scripts/start_mai_ui_vllm.sh                     # 8B on GPU 4 port 8000, foreground
#   bash scripts/start_mai_ui_vllm.sh --gpu 5             # different GPU
#   bash scripts/start_mai_ui_vllm.sh --2b                # serve 2B instead
#   bash scripts/start_mai_ui_vllm.sh --port 8001         # different port
#   bash scripts/start_mai_ui_vllm.sh --bg                # backgrounded; PID -> /tmp/vllm_<name>.pid
#
# Stop with `scripts/stop_mai_ui_vllm.sh [MAI-UI-8B|MAI-UI-2B]`.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="MAI-UI-8B"
MODEL_PATH="$PROJECT_ROOT/models/MAI-UI-8B"
PORT=8000
GPU=4
BG=0

while [ $# -gt 0 ]; do
    case "$1" in
        --2b)
            MODEL_NAME="MAI-UI-2B"
            MODEL_PATH="$PROJECT_ROOT/models/MAI-UI-2B"
            ;;
        --port)
            PORT="$2"; shift ;;
        --gpu)
            GPU="$2"; shift ;;
        --bg)
            BG=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model path not found: $MODEL_PATH" >&2
    echo "Download first: python3 scripts/download_mai_ui.py [--8b]" >&2
    exit 2
fi

# Avoid clashing with an existing server on the same port.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "ERROR: port $PORT already in use. Pick another with --port, or stop the running server." >&2
    exit 3
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL_NAME --host 0.0.0.0 --port $PORT --tensor-parallel-size 1 --trust-remote-code --gpu-memory-utilization 0.85"

echo "[start_mai_ui_vllm] $MODEL_NAME  GPU=$GPU  port=$PORT"
echo "[start_mai_ui_vllm] $MODEL_PATH"

if [ $BG -eq 1 ]; then
    LOGFILE="/tmp/vllm_${MODEL_NAME}.log"
    PIDFILE="/tmp/vllm_${MODEL_NAME}.pid"
    nohup bash -c "$CMD" > "$LOGFILE" 2>&1 &
    PID=$!
    echo $PID > "$PIDFILE"
    echo "[start_mai_ui_vllm] backgrounded  PID=$PID  log=$LOGFILE"
    echo "[start_mai_ui_vllm] tail:  tail -f $LOGFILE"
    echo "[start_mai_ui_vllm] stop:  bash scripts/stop_mai_ui_vllm.sh $MODEL_NAME"
else
    eval "$CMD"
fi
