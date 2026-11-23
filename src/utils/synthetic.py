"""
Synthetic patient guard — single source of truth for A1.
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

_SYNTHETIC_DATA = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "synthetic" / "patients.json"
)


class SyntheticGuardError(RuntimeError):
    """Raised when a record lacks a verifiable synthetic marker and ALLOW_REAL_PHI is false."""


@lru_cache(maxsize=1)
def _synthetic_index() -> frozenset:
    if not _SYNTHETIC_DATA.exists():
        return frozenset()
    with open(_SYNTHETIC_DATA, encoding="utf-8") as f:
        data = json.load(f)
    return frozenset(str(p["patient_id"]) for p in data if p.get("is_synthetic") is True)


def is_known_synthetic_patient(patient_id: Optional[str]) -> bool:
    return patient_id is not None and patient_id in _synthetic_index()


def _allow_real_phi() -> bool:
    return os.environ.get("ALLOW_REAL_PHI", "false").strip().lower() == "true"


def enforce_synthetic_guard(patient_id: Optional[str], is_synthetic: bool) -> None:
    """Refuse unless ALLOW_REAL_PHI=true OR the caller asserts synthetic AND ID verifies."""
    if _allow_real_phi():
        return
    if not is_synthetic:
        raise SyntheticGuardError(
            f"Refusing: patient '{patient_id or '<anonymous>'}' is not marked synthetic "
            f"and ALLOW_REAL_PHI is false."
        )
    if not patient_id:
        raise SyntheticGuardError(
            "Refusing: is_synthetic=True requires an explicit patient_id."
        )
    if not is_known_synthetic_patient(patient_id):
        raise SyntheticGuardError(
            f"Refusing: patient '{patient_id}' claimed synthetic but is not in "
            f"the synthetic index ({sorted(_synthetic_index())})."
        )