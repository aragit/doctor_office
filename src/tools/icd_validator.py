"""
ICD-10-CM existence / plausibility validation (pure Python, no LLM).

Provides a lightweight, local check that an ICD-10 code is:
1. well-formed (letter + digits, optional decimal),
2. known / plausible (present in a static reference table of common codes
   merged with the maps the rest of the pipeline already uses).

It deliberately does NOT try to hold the entire ICD-10-CM catalog - it uses a
curated set of common codes. Codes that are well-formed but not in the table
are flagged with an advisory "verify manually" quality flag rather than
hard-rejected, so a legitimate but uncatalogued code can still pass with a
paper trail.

Grounding (is this code *supported by this patient*) remains the job of
`tools.coding_tools.validate_icd_against_patient`. This module only answers
"is this a real code shape and is it known to us".
"""

import re
from typing import Any, Dict, List

# Seed with the symptom/condition codes the rest of the pipeline already knows,
# so the existence table can never drift from grounding logic.
from tools.coding_tools import SYMPTOM_CODES  # noqa: E402
from utils.validation import CONDITION_ICD10_MAP  # noqa: E402

DESCRIPTION_VALIDATE_ICD = (
    "Check that an ICD-10 code is well-formed and exists in the local ICD-10 "
    "reference table. Returns a verdict dict; never raises."
)

# Common codes beyond the symptom/condition set used by grounding.
_COMMON_ICD10 = {
    # Endocrine / metabolic
    "E03.9": "Hypothyroidism, unspecified",
    "E66.9": "Obesity, unspecified",
    "E78.5": "Hyperlipidemia, unspecified",
    "E78.00": "Pure hypercholesterolemia, unspecified",
    "E11.21": "Type 2 diabetes with diabetic nephropathy",
    "E11.22": "Type 2 diabetes with diabetic chronic kidney disease",
    "E11.29": "Type 2 diabetes with other diabetic kidney complication",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "I10": "Essential (primary) hypertension",
    "I20.9": "Angina pectoris, unspecified",
    # Respiratory
    "J30.9": "Allergic rhinitis, unspecified",
    "J45.901": "Unspecified asthma with (acute) exacerbation",
    "J45.909": "Unspecified asthma",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    # GI
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "K30": "Functional dyspepsia",
    # MSK / neuro
    "M54.5": "Low back pain, unspecified",
    "M54.2": "Cervicalgia",
    "M25.50": "Pain in unspecified joint",
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    "G89.29": "Other chronic pain",
    # Mental / behavioral
    "F41.9": "Anxiety disorder, unspecified",
    "F32.9": "Major depressive disorder, single episode, unspecified",
    # Renal
    "N18.1": "Chronic kidney disease, stage 1",
    "N18.9": "Chronic kidney disease, unspecified",
    "N39.0": "Urinary tract infection, site not specified",
    # Z-codes (encounters / status)
    "Z00.00": "Encounter for general adult medical examination, no abnormal findings",
    "Z00.129": "Encounter for routine child health examination without abnormal findings",
    "Z01.81": "Encounter for pre-procedural examination",
    "Z13.9": "Encounter for screening, unspecified",
    "Z23": "Encounter for immunization",
    "Z71.3": "Dietary counseling and surveillance",
    "Z71.89": "Other specified counseling",
    "Z79.01": "Long term (current) use of anticoagulants",
    "Z79.899": "Other long term (current) use of systemic anti-infectives",
    # Existing symptom codes (from coding_tools, kept authoritative)
    "R07.9": "R07.9 - Chest pain, unspecified",
    "R06.02": "R06.02 - Shortness of breath",
    "R05.9": "R05.9 - Cough, unspecified",
    "R10.9": "R10.9 - Unspecified abdominal pain",
    "R19.7": "R19.7 - Diarrhea, unspecified",
    "R50.9": "R50.9 - Fever, unspecified",
    "R51.9": "R51.9 - Headache, unspecified",
    "R42": "R42 - Dizziness and giddiness",
    "R53.83": "R53.83 - Other fatigue",
    "R11.2": "R11.2 - Nausea with vomiting, unspecified",
    "R21": "R21 - Rash and other non-specific skin eruption",
    "R31.9": "R31.9 - Hematuria, unspecified",
    "R45.9": "R45.9 - Anxiety, unspecified",
}

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")


def _normalize(text: str) -> str:
    """Pull a bare ICD-10 code out of 'CODE - Description' or 'CODEs' text."""
    if not text or not str(text).strip():
        return ""
    match = _CODE_RE.search(str(text).strip().upper())
    return match.group(1) if match else ""


def _well_formed(code: str) -> bool:
    """Loose structural check: letter + digits, optional decimal section."""
    if len(code) < 3 or not code[0].isalpha():
        return False
    body = code[1:]
    if "." in body:
        int_part, dec_part = body.split(".", 1)
        if not int_part.isdigit() or not dec_part.isdigit():
            return False
        # ICD-10-CM allows 1-3 digits after the decimal.
        return 1 <= len(dec_part) <= 3 and len(int_part) == 2
    # Bare category codes are exactly letter + 2 digits (e.g. Z23, I10).
    return body.isdigit() and len(body) == 2


# Build the exact-code + prefix tables from every source so they stay in sync.
KNOWN_CODES: Dict[str, str] = {}
for table in (SYMPTOM_CODES, _COMMON_ICD10, CONDITION_ICD10_MAP):
    for key, value in table.items():
        for candidate in (key, value):
            normalized = _normalize(candidate)
            if normalized and normalized not in KNOWN_CODES:
                KNOWN_CODES[normalized] = str(value)

KNOWN_EXACT = set(KNOWN_CODES)
KNOWN_PREFIXES = {code[:3] for code in KNOWN_EXACT}


def validate_icd10_code(text: str) -> Dict[str, Any]:
    """Validate an ICD-10 code for shape and local existence.

    Args:
        text: A code ('E11.9') or 'CODE - Description' string.

    Returns:
        Dict with `valid` (well-formed), `exists` (in the local table),
        `plausible` (well-formed AND known prefix), `code`, `reason` and an
        optional advisory `quality_flag`. Never raises.
    """
    code = _normalize(text)
    if not code:
        return {
            "valid": False,
            "exists": False,
            "plausible": False,
            "code": str(text),
            "reason": "No ICD-10 code found in the input",
            "quality_flag": f"Could not parse an ICD-10 code from {text!r}",
        }

    well_formed = _well_formed(code)
    exists = code in KNOWN_EXACT
    plausible = well_formed and code[:3] in KNOWN_PREFIXES

    if not well_formed:
        return {
            "valid": False,
            "exists": exists,
            "plausible": False,
            "code": code,
            "reason": f"Code {code!r} is not a well-formed ICD-10 code",
            "quality_flag": f"Malformed ICD-10 code {code} - reject or verify manually",
        }

    if not plausible:
        return {
            "valid": True,
            "exists": exists,
            "plausible": False,
            "code": code,
            "reason": f"Code {code} is well-formed but its prefix is not in the local reference table",
            "quality_flag": f"ICD-10 code {code} not in the local reference table - verify manually",
        }

    if not exists:
        return {
            "valid": True,
            "exists": False,
            "plausible": True,
            "code": code,
            "reason": f"Code {code} has a known prefix but is not in the local table",
            "quality_flag": None,
        }

    return {
        "valid": True,
        "exists": True,
        "plausible": True,
        "code": code,
        "reason": f"{code} found in local ICD-10 reference table",
        "quality_flag": None,
    }


def validate_icd10_list(codes: List[str]) -> Dict[str, Any]:
    """Validate a list of ICD-10 codes at once.

    Returns:
        Dict with `valid`, `well_formed`, `invalid`, `unknown` (advisory),
        `flags` and `results`.
    """
    results = [validate_icd10_code(code) for code in codes or []]
    invalid = [r["code"] for r in results if not r["valid"]]
    unknown = [r["code"] for r in results if r["valid"] and not r["plausible"]]
    flags = [r["quality_flag"] for r in results if r.get("quality_flag")]
    return {
        "valid": not invalid,
        "well_formed": [r["code"] for r in results if r["valid"]],
        "invalid": invalid,
        "unknown": unknown,
        "flags": flags,
        "results": results,
    }


__all__ = [
    "KNOWN_CODES",
    "KNOWN_EXACT",
    "KNOWN_PREFIXES",
    "validate_icd10_code",
    "validate_icd10_list",
    "DESCRIPTION_VALIDATE_ICD",
]