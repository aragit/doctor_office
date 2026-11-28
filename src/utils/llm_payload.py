"""
Shared helpers for building Pydantic models from imperfect LLM JSON.

Small local models (qwen2.5:1.5b, tinyllama) routinely omit keys - most often
the list-valued ones such as ``icd10_suggestions`` or ``cpt_suggestions``. Because
``JsonOutputParser`` returns a plain dict, ``Model(**result)`` then raises a
``ValidationError`` and aborts the entire visit.

Every agent therefore parses through :func:`build_model`, which fills safe defaults
based on the model's own field annotations, logs exactly what it had to invent, and
never raises for a payload that is merely incomplete.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, get_origin

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_SEQUENCE_ORIGINS = (list, set, tuple, frozenset)
_SOAP_NARRATIVE_FIELDS = ("subjective", "objective", "assessment", "plan")


def coerce_str(value: Any) -> str:
    """Render any value as a plain string (never raises)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (coerce_str(v) for v in value) if part)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def format_code_entry(item: Any) -> str:
    """Render one clinical code as ``"CODE - Description"`` where possible.

    Handles the ``{"code": "E11.9", "description": "..."}`` shape that models
    sometimes emit instead of the required string form.
    """
    if isinstance(item, dict):
        code = coerce_str(item.get("code")).strip()
        description = coerce_str(item.get("description")).strip()
        if code and description:
            return f"{code} - {description}"
        if code or description:
            return code or description
    return coerce_str(item).strip()


def coerce_code_list(value: Any) -> List[str]:
    """Normalise any value into a list of non-empty ``"CODE - Description"`` strings."""
    if value is None:
        return []
    if isinstance(value, dict):
        entry = format_code_entry(value)
        return [entry] if entry else []
    if isinstance(value, (list, tuple, set)):
        return [entry for entry in (format_code_entry(item) for item in value) if entry]
    text = coerce_str(value).strip()
    return [text] if text else []


def default_for(annotation: Any) -> Any:
    """Safe empty value for a field annotation: ``[]`` for sequences, ``""`` otherwise."""
    origin = get_origin(annotation)
    if origin in _SEQUENCE_ORIGINS or annotation in _SEQUENCE_ORIGINS:
        return []
    if origin is dict or annotation is dict:
        return {}
    return ""


def _coerce(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin in _SEQUENCE_ORIGINS or annotation in _SEQUENCE_ORIGINS:
        return coerce_code_list(value)
    if origin is dict or annotation is dict:
        return value if isinstance(value, dict) else coerce_str(value)
    return coerce_str(value)


def normalize_payload(model_cls: Type[ModelT], payload: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Fill a raw LLM payload with safe defaults for every field of *model_cls*.

    Args:
        model_cls: Target Pydantic model; its annotations decide the defaults.
        payload: Raw value returned by the model (usually a dict, never trusted).

    Returns:
        ``(data, applied_defaults)`` - *data* always has every field of
        *model_cls*, and *applied_defaults* holds one message per field that was
        missing, ``None`` or unusable.
    """
    applied: List[str] = []
    source: Dict[str, Any] = payload if isinstance(payload, dict) else {}

    if not isinstance(payload, dict):
        logger.warning(
            "LLM payload was %s, not a dict - defaulting every field", type(payload).__name__
        )

    data: Dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        raw = source.get(name)
        if raw is None:
            applied.append(f"{name} missing - defaulting to {default_for(field.annotation)!r}")
        data[name] = _coerce(field.annotation, raw)
    return data, applied


def build_model(model_cls: Type[ModelT], payload: Any, source: str = "llm output") -> ModelT:
    """Build *model_cls* from *payload*, defaulting anything the model omitted.

    Missing strings become ``""`` and missing lists become ``[]`` instead of raising.
    If pydantic still rejects the payload, an empty-but-valid note is returned, so a
    malformed generation degrades to a flagged empty note rather than killing the run.
    """
    data, applied = normalize_payload(model_cls, payload)
    if applied:
        logger.warning("%s incomplete - %s", source, "; ".join(applied))

    try:
        return model_cls(**data)
    except ValidationError as exc:
        logger.error(
            "%s still failed validation after defaulting (%s) - returning empty %s",
            source,
            exc,
            model_cls.__name__,
        )
        # model_construct skips validation, so even this last resort cannot raise.
        empty = {name: default_for(field.annotation) for name, field in model_cls.model_fields.items()}
        return model_cls.model_construct(**empty)


def has_content(model: BaseModel) -> bool:
    """True when a model carries at least one non-empty field."""
    for name in type(model).model_fields:
        value = getattr(model, name, None)
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
        elif coerce_str(value).strip():
            return True
    return False


def has_soap_sections(model: BaseModel) -> bool:
    """True when at least one core SOAP narrative field is non-empty."""
    return any(
        coerce_str(getattr(model, field, None)).strip()
        for field in _SOAP_NARRATIVE_FIELDS
        if field in type(model).model_fields
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from raw model text.

    Handles fenced blocks, JSON wrapped in prose, and output truncated by the
    token budget.
    """
    if not text:
        return None

    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        for parse in (json.loads, _parse_partial_json):
            try:
                parsed = parse(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _parse_partial_json(text: str) -> Any:
    """Thin wrapper so a missing langchain helper can never break extraction."""
    try:
        from langchain_core.utils.json import parse_partial_json
    except Exception:  # pragma: no cover - defensive
        return None
    return parse_partial_json(text)


__all__ = [
    "build_model",
    "coerce_code_list",
    "coerce_str",
    "default_for",
    "extract_json_object",
    "format_code_entry",
    "has_content",
    "has_soap_sections",
    "normalize_payload",
]