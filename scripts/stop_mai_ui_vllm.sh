#!/usr/bin/env bash
# Stop a vLLM server previously started with `start_mai_ui_vllm.sh --bg`.
#
# Usage:
#   bash scripts/stop_mai_ui_vllm.sh                # stops MAI-UI-8B (default)
#   bash scripts/stop_mai_ui_vllm.sh MAI-UI-2B      # stops 2B
MODEL_NAME="${1:-MAI-UI-8B}"
PIDFILE="/tmp/vllm_${MODEL_NAME}.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[stop_mai_ui_vllm] killing PID $PID ($MODEL_NAME)"
        kill "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
    else
        echo "[stop_mai_ui_vllm] PID $PID already gone"
    fi
    rm -f "$PIDFILE"
else
    echo "[stop_mai_ui_vllm] no $PIDFILE; falling back to pkill on served-model-name"
    pkill -f "vllm.entrypoints.openai.api_server.*served-model-name $MODEL_NAME" || true
fi
