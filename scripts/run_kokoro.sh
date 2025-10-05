#!/bin/bash

# Get project root directory
# cd ../Kokoro-FastAPI
# PROJECT_ROOT=$(pwd)

# # Set environment variables
# export USE_GPU=true
# export USE_ONNX=false
# export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
# export MODEL_DIR=src/models
# export VOICES_DIR=src/voices/v1_0
# export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# # Run FastAPI with GPU extras using uv run
# # Note: espeak may still require manual installation,
# uv pip install -e ".[gpu]"
# uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
# uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880

# This is an outdated version but I have to use it as my Driver/CUDA is old
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:v0.1.3