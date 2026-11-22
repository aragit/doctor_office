"""
Coding and grounding tools.

These are the safety-relevant tools. They are deterministic, LLM-free
functions so the grounding decision never depends on model compliance:

- `validate_icd_against_patient` - is this code supported by the chart?
- `simple_symptom_to_icd_suggestion` - map a presenting symptom to a
  candidate ICD-10 code so the agents stop dropping acute symptoms.
"""

import re
from typing import Any, Dict, List, Optional

from memory.patient_profile import parse_patient_context
from utils.symptom_codes import (
    SYMPTOM_CODES as _CANONICAL_SYMPTOM_CODES,
    SYMPTOM_PHRASES as _CANONICAL_SYMPTOM_PHRASES,
    find_symptom_candidates,
    symptom_phrase_present,
)
from utils.validation import CONDITION_ICD10_MAP, ICD10_CONDITION_MAP

DESCRIPTION_VALIDATE = (
    "Check whether an ICD-10 code is grounded in the patient's documented "
    "conditions or chief complaint. Returns a verdict dict; never raises."
)
DESCRIPTION_SYMPTOM = (
    "Map a presenting symptom phrase to candidate ICD-10-CM codes, e.g. "
    "'chest pain' -> R07.9. Use to make sure acute symptoms are not omitted."
)

# ICD-10 prefix -> phrases that justify it, and prefix -> canonical code.
#
# Both tables now live in `utils.symptom_codes`, the single source of truth
# shared with validation and the Supervisor (Phase A3). They are re-exported
# here because the tool surface is the public name for this behaviour, and
# several callers - and the registry's tool docs - import them from here.
SYMPTOM_PHRASES: Dict[str, List[str]] = {
    prefix: list(phrases)
    for prefix, phrases in _CANONICAL_SYMPTOM_PHRASES.items()
}

# Canonical code + description for each supported prefix.
SYMPTOM_CODES: Dict[str, str] = dict(_CANONICAL_SYMPTOM_CODES)

# Prefix -> canonical code, derived from SYMPTOM_CODES so the two tables
# cannot drift apart.
PREFIX_TO_SYMPTOM_CODE: Dict[str, str] = {
    code[:3]: code for code in SYMPTOM_CODES
}

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")


def _extract_code(code_or_text: str) -> Optional[str]:
    """Pull the ICD-10 code out of a bare code or a 'CODE - Description' string."""
    if not code_or_text:
        return None
    match = _CODE_RE.search(str(code_or_text).strip().upper())
    return match.group(1) if match else None


def validate_icd_against_patient(
    code: str,
    patient_context: str,
    chief_complaint: str = "",
) -> Dict[str, Any]:
    """Check whether an ICD-10 code is grounded in the patient chart.

    A code passes when it maps to a documented condition, or when it is a
    symptom code whose phrase appears in the chief complaint.

    Args:
        code: ICD-10 code, either bare ("J45.909") or "CODE - Description".
        patient_context: Formatted patient context block.
        chief_complaint: Reason for the visit (optional, enables symptom codes).

    Returns:
        Dict with `valid` (bool), `code`, `reason`, `matched_against`,
        `suggestion` and `quality_flag`.
    """
    extracted = _extract_code(code)
    if not extracted:
        return {
            "valid": False,
            "code": code,
            "reason": "No ICD-10 code found in the input",
            "matched_against": None,
            "quality_flag": f"Could not parse an ICD-10 code from {code!r}",
        }

    prefix = extracted[:3]
    profile = parse_patient_context(patient_context)

    # 1. Charted condition?
    condition_name = ICD10_CONDITION_MAP.get(prefix)
    if condition_name:
        for documented in profile.conditions:
            documented_lower = documented.lower()
            if documented_lower in condition_name.lower() or condition_name.lower() in documented_lower:
                return {
                    "valid": True,
                    "code": extracted,
                    "reason": f"Maps to documented condition '{documented}'",
                    "matched_against": documented,
                    "suggestion": None,
                    "quality_flag": None,
                }

    # 2. Known condition synonyms that the chart may phrase differently.
    for keyword, codes in CONDITION_ICD10_MAP.items():
        if any(code.startswith(extracted[:3]) for code in codes) and keyword in profile.raw_context.lower():
            return {
                "valid": True,
                "code": extracted,
                "reason": f"Patient context mentions '{keyword}'",
                "matched_against": keyword,
                "suggestion": None,
                "quality_flag": None,
            }

    # 3. Symptom justified by the chief complaint? Any phrase for the prefix
    #    counts, via the central map - so "dyspnea" grounds R06 just as
    #    "shortness of breath" does.
    if symptom_phrase_present(prefix, chief_complaint or ""):
        return {
            "valid": True,
            "code": extracted,
            "reason": "Symptom present in chief complaint",
            "matched_against": "chief_complaint",
            "suggestion": None,
            "quality_flag": None,
        }

    return {
        "valid": False,
        "code": extracted,
        "reason": (
            f"Code {extracted} does not match any documented condition "
            f"({', '.join(profile.conditions) or 'none listed'}) or complaint symptom"
        ),
        "matched_against": None,
        "suggestion": None,
        "quality_flag": f"Ungrounded ICD-10 code {extracted} - not in patient context or chief complaint",
    }


def simple_symptom_to_icd_suggestion(symptom: str) -> Dict[str, Any]:
    """Suggest ICD-10 candidates for a presenting symptom.

    Args:
        symptom: Free-text symptom, e.g. "chest pain" or "diarrhea".

    Returns:
        Dict with `symptom`, `codes` (list of "CODE - Description"),
        `matched_phrase` and `found` (bool).
    """
    text = (symptom or "").strip().lower()
    if not text:
        return {"symptom": symptom, "codes": [], "matched_phrase": None, "found": False}

    matches: List[str] = []
    matched_phrase = None
    for prefix, phrases in SYMPTOM_PHRASES.items():
        for phrase in phrases:
            if phrase in text:
                code = PREFIX_TO_SYMPTOM_CODE.get(prefix)
                if code and code not in matches:
                    matches.append(code)
                if matched_phrase is None or len(phrase) > len(matched_phrase):
                    matched_phrase = phrase

    return {
        "symptom": symptom,
        "codes": matches,
        "matched_phrase": matched_phrase,
        "found": bool(matches),
    }


def suggest_codes_for_complaint(
    chief_complaint: str,
    patient_context: str = "",
    max_codes: int = 4,
    visit_context: Any = None,
    already_coded: Optional[List[str]] = None,
) -> List[str]:
    """Candidate symptom codes for this encounter.

    Used to brief the Coding agent on acute symptoms it routinely drops. This is
    *guidance*, not an answer: the validation layer still checks the final codes
    and the Supervisor can still complete a true gap.

    Delegates to `utils.symptom_codes.find_symptom_candidates`, so the tool, the
    validation gap flag and the Supervisor all resolve phrases the same way
    (Phase A3). That includes two behaviours this function did not have: visit
    data (HPI, exam) is searched as well as the complaint, and a symptom is
    suppressed when a more specific diagnosis for it is already charted.

    Only sign/symptom (R) and screening (Z) codes are returned - a disease code
    is never proposed from a phrase.
    """
    if not chief_complaint and not visit_context:
        return []

    return [
        candidate.code
        for candidate in find_symptom_candidates(
            chief_complaint=chief_complaint,
            visit_context=visit_context,
            patient_context=patient_context,
            already_coded=already_coded,
            max_codes=max_codes,
        )
    ]


def validate_codes_against_patient(
    codes: List[str],
    patient_context: str,
    chief_complaint: str = "",
) -> Dict[str, List[str]]:
    """Validate a list of codes at once.

    Returns:
        Dict with `valid_codes`, `invalid_codes`, `flags` and `results`.
    """
    results = []
    valid_codes: List[str] = []
    invalid_codes: List[str] = []
    flags: List[str] = []

    for code in codes or []:
        verdict = validate_icd_against_patient(code, patient_context, chief_complaint)
        results.append(verdict)
        if verdict["valid"]:
            valid_codes.append(code)
        else:
            invalid_codes.append(code)
            if verdict.get("quality_flag"):
                flags.append(verdict["quality_flag"])

    return {
        "valid_codes": valid_codes,
        "invalid_codes": invalid_codes,
        "flags": flags,
        "results": results,
    }


__all__ = [
    "SYMPTOM_CODES",
    "SYMPTOM_PHRASES",
    "simple_symptom_to_icd_suggestion",
    "suggest_codes_for_complaint",
    "validate_codes_against_patient",
    "validate_icd_against_patient",
    "DESCRIPTION_VALIDATE",
    "DESCRIPTION_SYMPTOM",
]
