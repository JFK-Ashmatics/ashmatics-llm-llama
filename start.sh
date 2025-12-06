#!/usr/bin/env bash
set -euo pipefail

# Trap signals to ensure clean shutdown of background processes
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

MODEL_PATH="${MODEL_PATH:-/app/models/model.gguf}"
LLAMA_PORT="${LLAMA_PORT:-11434}"
CONTEXT="${CONTEXT:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
THREADS="${THREADS:-0}"           # 0 means auto (nproc)
MODEL_ALIAS="${MODEL_ALIAS:-kb-llm}"

# Use env var for binary path if set, otherwise default
LLAMA_BIN="${LLAMA_BIN_PATH:-/app/bin/llama-server}"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "Error: llama-server binary not found at $LLAMA_BIN"
    exit 1
fi

echo "Starting llama-server..."
# Start llama-server in the background
$LLAMA_BIN \
  -m "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --port "$LLAMA_PORT" \
  --ctx-size "$CONTEXT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --threads "$THREADS" \
  --host 0.0.0.0 &

# Wait until llama-server is ready
echo "Waiting for llama-server on :$LLAMA_PORT ..."
# Loop with a timeout
MAX_RETRIES=30
count=0
until curl -s "http://127.0.0.1:${LLAMA_PORT}/v1/models" >/dev/null; do
  sleep 1
  count=$((count+1))
  if [ $count -ge $MAX_RETRIES ]; then
      echo "Timeout waiting for llama-server"
      exit 1
  fi
done
echo "llama-server is up."

# Start FastAPI
# We don't use exec here so that the trap can catch signals and kill the background process
uvicorn main:app --host 0.0.0.0 --port 9000

# Wait for background processes
wait

