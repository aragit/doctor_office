"""Central model configuration for the Doctor Office MVP."""

import os
from typing import Optional

# Model presets - logical mode labels, not specific model files
MODEL_PRESETS = {
    "fast": "qwen2.5-1.5b-instruct",
    "quality": "qwen2.5-1.5b-instruct",  # Same weights unless user loads larger GGUF
}

DEFAULT_MODE = "fast"

# LLM connection settings (OpenAI-compatible API)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "local")

# Generation limits for CPU efficiency
FAST_MAX_TOKENS = int(os.environ.get("LLM_FAST_MAX_TOKENS", "384"))
QUALITY_MAX_TOKENS = int(os.environ.get("LLM_QUALITY_MAX_TOKENS", "384"))

# RAG configuration
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "2"))

# Synthetic-data guard configuration
ALLOW_REAL_PHI = os.environ.get("ALLOW_REAL_PHI", "false").lower() == "true"


def get_model_name(mode: str = DEFAULT_MODE) -> str:
    """Get the model label for a given mode.

    Args:
        mode: Either "fast" or "quality". Defaults to "fast".

    Returns:
        The model label string.
    """
    return MODEL_PRESETS.get(mode, MODEL_PRESETS[DEFAULT_MODE])


def get_available_modes() -> list[str]:
    """Get list of available mode names."""
    return list(MODEL_PRESETS.keys())


def get_max_tokens(mode: str = DEFAULT_MODE) -> int:
    """Get the maximum tokens for generation based on mode.

    Args:
        mode: Either "fast" or "quality". Defaults to "fast".

    Returns:
        Maximum tokens for generation.
    """
    if mode == "quality":
        return QUALITY_MAX_TOKENS
    return FAST_MAX_TOKENS


# Re-export synthetic data utilities from the single source of truth
from utils.synthetic import (  # noqa: F401
    is_known_synthetic_patient,
    enforce_synthetic_guard,
    SyntheticGuardError,
)