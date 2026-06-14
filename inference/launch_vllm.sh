#!/bin/bash
set -e

MODEL_PATH=${1:-"./models/vaelon-7b"}
PORT=${2:-8000}

echo "Starting Vaelon inference server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

if command -v docker &> /dev/null; then
    docker run --gpus all -p $PORT:8000 \
        -v $(pwd)/$MODEL_PATH:/model \
        qythera-vaelon:latest \
        python3 -m inference.server --model /model --port 8000
else
    python3 -m inference.server --model "$MODEL_PATH" --port "$PORT"
fi
