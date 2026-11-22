#!/usr/bin/env python3
"""
Download the required LLM model for doctor_office.
Place the model into models/ directory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# ============================================================
# CONFIGURE THIS SECTION
# ============================================================
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"   # ← change to your actual model
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"    # ← change to the exact .gguf file
# Or use snapshot_download for whole repos

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================

def download_gguf():
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    print(f"Target directory: {MODELS_DIR}")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n✅ Model downloaded successfully to:\n{path}")
        return path
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_gguf()
