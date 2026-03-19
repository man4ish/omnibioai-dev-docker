#!/bin/bash
# ==========================================================
# Script Name: run_ai_dev.sh (Hardened Version)
# ==========================================================

IMAGE_NAME="omnibioai-dev-env"
TAG="latest"

echo "----------------------------------------------------------"
echo "OmniBioAI Foundry: Launching AI Dev Environment"
echo "----------------------------------------------------------"

# 1. Check if NVIDIA Container Toolkit is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "CRITICAL ERROR: nvidia-smi not found. GPU acceleration will fail."
    echo "Please install NVIDIA drivers and nvidia-container-toolkit."
    exit 1
fi

# 2. Ensure the image exists (Build if missing)
if [[ "$(docker images -q ${IMAGE_NAME}:${TAG} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_NAME}:${TAG} not found. Building now..."
    docker build -t ${IMAGE_NAME}:${TAG} .
fi

# 3. Clean up any existing dead containers with the same name
docker rm -f omnibio_dev_foundry &> /dev/null

# 4. Launch with full integrations
# --ipc=host is critical for PyTorch DataLoader (shared memory)
# --ulimit memlock=-1 is often needed for high-performance CUDA
docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name omnibio_dev_foundry \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.ollama:/root/.ollama \
  -v "$(pwd)":/workspace \
  -p 8888:8888 \
  -p 11434:11434 \
  -e HF_HOME=/root/.cache/huggingface \
  -e OLLAMA_HOST=0.0.0.0 \
  -it ${IMAGE_NAME}:${TAG} bash

echo "----------------------------------------------------------"
echo "Container exited. Work saved in $(pwd)"
echo "----------------------------------------------------------"