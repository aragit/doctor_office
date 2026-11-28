import logging
import os
from typing import Optional

from langchain_openai import ChatOpenAI

from .config import (
    get_model_name,
    get_max_tokens,
    LLM_BASE_URL,
    LLM_API_KEY,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300


def _get_timeout_seconds() -> int:
    """Resolve per-request timeout in seconds from env, with a safe CPU default."""
    raw = os.getenv("LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        logger.warning("Invalid LLM_TIMEOUT_SECONDS=%r; using %ds", raw, DEFAULT_TIMEOUT_SECONDS)
        return DEFAULT_TIMEOUT_SECONDS


def get_llm(
    mode: str = "fast",
    temperature: float = 0.1,
    num_predict: Optional[int] = None,
    model_name: Optional[str] = None,
) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance pointed at llama-server.

    Args:
        mode: Model mode - "fast" or "quality". Defaults to "fast".
            Ignored when `model_name` is supplied.
        temperature: LLM temperature. Defaults to 0.1.
        num_predict: Maximum number of tokens to generate.
            If None, uses mode-based defaults from config.
            Defaults to 384 for fast mode, 384 for quality mode.
        model_name: Explicit model to use instead of the mode preset. This is
            how callers pin an exact model while keeping the mode
            for anything else that depends on it. `None` keeps the mode-based default.
            Note: The llama-server often ignores the model name when one model is loaded,
            but we keep it for compatibility.

    Returns:
        Configured ChatOpenAI instance.
    """
    # Priority: explicit model_name > LLM_MODEL_NAME env var (via get_model_name) > mode preset
    resolved_model = (model_name or "").strip() or get_model_name(mode)

    # Use provided num_predict or fall back to mode-based default
    if num_predict is None:
        num_predict = get_max_tokens(mode)

    timeout_seconds = _get_timeout_seconds()
    logger.info(
        "Using LLM model: %s (mode=%s, explicit_model=%s, timeout=%ss)",
        resolved_model,
        mode,
        model_name,
        timeout_seconds,
    )

    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        max_tokens=num_predict,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=timeout_seconds,
        max_retries=1,  # Fail fast, don't retry on timeout
    )


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("What are the first-line treatments for Type 2 Diabetes?")
    print(response.content)