#!/usr/bin/env bash
# Quick shell alternative to download the model
# Requires: huggingface_hub (pip install huggingface_hub)

set -euo pipefail

REPO_ID="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODELS_DIR="$(dirname "$0")/../models"

mkdir -p "$MODELS_DIR"

echo "Downloading $FILENAME from $REPO_ID ..."
echo "Target directory: $MODELS_DIR"

python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

MODELS_DIR = Path('$MODELS_DIR')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id='$REPO_ID',
    filename='$FILENAME',
    local_dir=MODELS_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f'Downloaded to: {path}')
"
