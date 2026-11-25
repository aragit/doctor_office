#!/bin/bash
# scripts/tune_ollama.sh
# Zero-risk Ollama runtime tuning for CPU inference.
# Run as your normal user (not root).
# Usage: bash scripts/tune_ollama.sh

set -euo pipefail

echo "=== Ollama Runtime Tuning ==="

# 1. Kill existing ollama serve
echo "[1/4] Stopping existing ollama processes..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# 2. Detect physical CPU cores (not logical/HT)
# lscpu -p prints "CPU,Core,Socket,Node,,L1d,L1i,L2,L3" - we want unique Core IDs
PHYSICAL_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,2 | wc -l)
if [ "$PHYSICAL_CORES" -eq 0 ]; then
    # Fallback: nproc shows logical cores; halve as rough estimate
    PHYSICAL_CORES=$(( $(nproc) / 2 ))
    echo "  lscpu parse failed, fallback to nproc/2 = $PHYSICAL_CORES"
else
    echo "  Detected physical cores: $PHYSICAL_CORES"
fi

# 3. Export tuned environment
export OLLAMA_NUM_THREADS="$PHYSICAL_CORES"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

echo "[2/4] Environment configured:"
echo "  OLLAMA_NUM_THREADS=$OLLAMA_NUM_THREADS"
echo "  OLLAMA_FLASH_ATTENTION=$OLLAMA_FLASH_ATTENTION"
echo "  OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
echo "  OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"

# 4. Start ollama serve in background
echo "[3/4] Starting ollama serve..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for readiness
echo "[4/4] Waiting for ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  ✅ Ollama is ready (pid=$OLLAMA_PID)"
        echo "  Log: /tmp/ollama.log"
        exit 0
    fi
    sleep 1
done

echo "  ❌ Ollama did not become ready in time"
echo "  Check log: /tmp/ollama.log"
exit 1