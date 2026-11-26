"""
Fast Path / Slow Path decision gate for the Coding Agent.

This module provides a pure, deterministic function to decide whether a visit
can use the Fast Path (deterministic tools only) or must use the Slow Path
(full LLM Coding Agent call).

The decision is 100% deterministic and auditable - no LLM involved.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.symptom_codes import (
    AUTO_ADDABLE_PREFIXES,
    SYMPTOM_CODES,
    find_symptom_candidates,
    canonical_code_for_prefix,
)
from tools.em_calculator import derive_mdm_domains, calculate_em_code
from agents.visit_context import visit_context_to_grounding_text

logger = logging.getLogger(__name__)

# Chronic conditions that map cleanly to ICD-10 and are common in follow-up visits
CHRONIC_CONDITION_ICD10 = {
    "type 2 diabetes": "E11.9 - Type 2 diabetes mellitus without complications",
    "diabetes mellitus type 2": "E11.9 - Type 2 diabetes mellitus without complications",
    "diabetes": "E11.9 - Type 2 diabetes mellitus without complications",
    "hypertension": "I10 - Essential (primary) hypertension",
    "essential hypertension": "I10 - Essential (primary) hypertension",
    "htn": "I10 - Essential (primary) hypertension",
}

# Strong uncertainty phrases that MUST force Slow Path (per decision rules)
# These indicate genuine diagnostic uncertainty, not routine chronic management
STRONG_UNCERTAINTY_PHRASES = (
    # Rule-out / differential language
    "rule out",
    "r/o",
    "r/o ",
    "possible ",
    "possibly",
    "suspect",
    "suspected",
    "suspicious for",
    # Differential diagnosis language
    "differential",
    "ddx",
    "working diagnosis",
    # New onset / first presentation language
    "new onset",
    "new-onset",
    "first presentation",
    "first episode",
    # Unclear etiology language
    "undifferentiated",
    "unclear etiology",
    "etiology unclear",
    # Explicit workup needed language
    "cannot rule out",
    "needs further workup",
    "further evaluation needed",
)

# Words that should NOT force Slow Path by themselves (common in chronic follow-up)
# These are explicitly allowed in chronic management context
ALLOWED_IN_CHRONIC_CONTEXT = (
    "acute",
    "chronic",
    "worsening",
    "improving",
    "stable",
    "uncontrolled",
    "referral",
    "follow-up",
    "followup",
    "monitor",
    "continue",
    "adjust",
    "increase",
    "decrease",
    "mild",
    "moderate",
    "severe",
    "consult",
    "workup",
    "probable",
    "possible",  # Only "possible " (with space) is in STRONG_UNCERTAINTY_PHRASES
)

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")

# Maximum symptom codes we can confidently handle in Fast Path
MAX_FAST_PATH_SYMPTOM_CODES = 3


def _extract_chronic_conditions(patient_context: str) -> Dict[str, str]:
    """Extract known chronic conditions from patient_context with their ICD-10 mappings."""
    context_lower = patient_context.lower()
    found = {}
    for condition, icd10 in CHRONIC_CONDITION_ICD10.items():
        if condition in context_lower:
            found[condition] = icd10
    return found


def _has_uncertainty_signals(
    chief_complaint: str,
    soap_note: Dict[str, Any],
    patient_context: str,
    visit_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Detect signals that suggest diagnostic uncertainty (force Slow Path).

    Strong uncertainty phrases are checked ONLY on original clinical inputs
    (chief_complaint, patient_context, visit_context HPI) to avoid false
    positives from Scribe-generated soft uncertainty language in the SOAP note.
    """
    reasons = []

    # Build text from ORIGINAL clinical inputs only (not generated SOAP)
    original_text_parts = [
        chief_complaint or "",
        patient_context or "",
    ]
    if visit_context:
        hpi = visit_context.get("hpi", "") or ""
        if hpi:
            original_text_parts.append(hpi)
    original_text = "\n".join(original_text_parts).lower()

    # Check if patient has recognized chronic conditions
    chronic_conditions = _extract_chronic_conditions(patient_context)
    has_chronic = len(chronic_conditions) > 0

    # Check if chief complaint aligns with chronic condition follow-up
    complaint_lower = chief_complaint.lower()
    chronic_keywords = [
        "diabetes", "dm", "hba1c", "glucose", "hyperglycemia", "diabetic",
        "hypertension", "htn", "blood pressure", "bp", "hypertensive",
        "hyperlipidemia", "hld", "cholesterol", "lipid",
        "hypothyroidism", "thyroid", "tsh",
        "ckd", "chronic kidney", "renal",
        "heart failure", "chf", "hfref", "hfpef",
        "follow-up", "followup", "management", "monitoring", "control",
        "medication", "refill", "adherence", "compliance",
        "routine", "chronic", "stable", "uncontrolled",
        "fatigue", "dizziness", "weakness", "tired", "exhaustion",
        "polyuria", "polydipsia", "nocturia",
        "adjust", "increase", "decrease", "intensif", "optimiz",
    ]
    chief_aligns_with_chronic = any(kw in complaint_lower for kw in chronic_keywords)

    # Check for STRONG uncertainty phrases in ORIGINAL clinical inputs only
    # (chief complaint, patient context, visit context HPI)
    # This prevents false positives from Scribe-generated soft uncertainty language
    for phrase in STRONG_UNCERTAINTY_PHRASES:
        if phrase in original_text:
            reasons.append(f"Strong uncertainty phrase in original input: '{phrase}'")

    # Acute symptoms that suggest NEW problem (not chronic management)
    # These force Slow Path even if patient has chronic conditions,
    # UNLESS the chief complaint clearly aligns with chronic follow-up
    new_acute_symptoms = (
        "chest pain",
        "shortness of breath",
        "dyspnea",
        "abdominal pain",
        "diarrhea",
        "fever",
        "headache",
        "vomiting",
        "nausea",
        "cough",
        "rash",
        "blood in urine",
        "hematuria",
        "palpitation",
        "syncope",
        "fall",
        "confusion",
        "altered mental status",
        "sudden onset",
        "acute onset",
    )

    # Check for new acute symptoms in chief complaint
    has_new_acute = any(symptom in complaint_lower for symptom in new_acute_symptoms)

    # Force Slow Path if:
    # - No chronic conditions and new acute symptoms present, OR
    # - Chronic conditions exist but chief complaint does NOT align with chronic management
    #   AND new acute symptoms are present
    if has_new_acute and (not has_chronic or not chief_aligns_with_chronic):
        reasons.append(
            "Chief complaint contains new acute symptoms not aligned with chronic condition management"
        )

    # Check for incomplete documentation (empty assessment/plan in generated SOAP)
    # This is still important - we need the SOAP to be usable
    if not soap_note.get("assessment", "").strip():
        reasons.append("SOAP assessment section is empty")
    if not soap_note.get("plan", "").strip():
        reasons.append("SOAP plan section is empty")

    return reasons


def _can_derive_primary_icd(
    patient_context: str,
    chief_complaint: str,
    soap_note: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """
    Determine if primary ICD-10 can be deterministically derived.
    Returns (primary_icd, missing_items).
    """
    chronic = _extract_chronic_conditions(patient_context)
    missing = []

    if not chronic:
        missing.append("No recognized chronic conditions in patient context")
        return None, missing

    # For follow-up visits, the primary is typically the main chronic condition
    # Check if chief complaint aligns with a chronic condition
    complaint_lower = chief_complaint.lower()

    # Priority: diabetes follow-up, hypertension follow-up, or both
    primary_candidates = []
    if any(c in complaint_lower for c in ["diabetes", "dm", "hba1c", "glucose", "hyperglycemia"]):
        if "type 2 diabetes" in chronic or "diabetes" in chronic:
            primary_candidates.append(chronic.get("type 2 diabetes") or chronic.get("diabetes"))
    if any(c in complaint_lower for c in ["hypertension", "htn", "blood pressure", "bp"]):
        if "hypertension" in chronic or "essential hypertension" in chronic or "htn" in chronic:
            primary_candidates.append(chronic.get("hypertension") or chronic.get("essential hypertension") or chronic.get("htn"))

    # If chief complaint mentions both, pick the first mentioned or diabetes as default
    if len(primary_candidates) == 1:
        return primary_candidates[0], missing
    elif len(primary_candidates) > 1:
        # Default to diabetes for diabetes+HTN follow-up (most common)
        for cand in primary_candidates:
            if "E11" in cand:
                return cand, missing
        return primary_candidates[0], missing

    # Check SOAP assessment for clues
    assessment = (soap_note.get("assessment") or "").lower()
    for condition, icd in chronic.items():
        if condition in assessment:
            return icd, missing

    missing.append("Primary diagnosis cannot be clearly mapped to a chronic condition")
    return None, missing


def _resolve_symptom_codes_deterministic(
    chief_complaint: str,
    patient_context: str,
    visit_context: Optional[Dict[str, Any]] = None,
    already_coded: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Resolve symptom codes using deterministic tools.
    Returns (symptom_codes, missing_items, flags).
    """
    symptom_codes = []
    missing = []
    flags = []

    try:
        candidates = find_symptom_candidates(
            chief_complaint=chief_complaint,
            visit_context=visit_context,
            patient_context=patient_context,
            already_coded=already_coded or [],
            max_codes=MAX_FAST_PATH_SYMPTOM_CODES,
        )
    except Exception as exc:
        logger.warning("Fast Path: symptom candidate resolution failed: %s", exc)
        missing.append(f"Symptom resolution failed: {exc}")
        return [], missing, flags

    for candidate in candidates:
        if candidate.prefix in AUTO_ADDABLE_PREFIXES:
            symptom_codes.append(candidate.description)
            flags.append(
                f"Fast Path: added symptom code {candidate.description} "
                f"('{candidate.phrase}' documented in {candidate.source.replace('_', ' ')})"
            )

    return symptom_codes, missing, flags


def _compute_cpt_deterministic(
    soap_note: Dict[str, Any],
    patient_context: str,
) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """
    Compute CPT via deterministic MDM calculator.
    Returns (cpt_result, missing_items, flags).
    """
    missing = []
    flags = []

    try:
        domains = derive_mdm_domains(soap_note, patient_context)
        em = calculate_em_code(
            domains["problems_level"],
            domains["data_level"],
            domains["risk_level"],
        )
    except Exception as exc:
        logger.warning("Fast Path: MDM calculation failed: %s", exc)
        missing.append(f"MDM calculation failed: {exc}")
        return None, missing, flags

    if not em.get("valid"):
        missing.append(f"Invalid MDM levels: {em.get('error')}")
        return None, missing, flags

    cpt_result = {
        "cpt_description": em.get("cpt_description", "99213 - Office visit, established patient, low medical decision making"),
        "cpt_code": em.get("cpt_code", "99213"),
        "overall_mdm_level": em.get("overall_mdm_level", "low"),
        "justification": em.get("justification", ""),
        "domains": domains,
    }
    flags.append(f"Fast Path: deterministic CPT {cpt_result['cpt_code']} (MDM {cpt_result['overall_mdm_level']})")
    return cpt_result, missing, flags


def should_use_fast_path(
    cleaned_soap_note: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    visit_context: Optional[Dict[str, Any]] = None,
    grounding_facts: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if the Fast Path (deterministic tools only) can be used.

    Returns:
        (use_fast_path: bool, decision_metadata: dict)

    decision_metadata contains:
        - reason: short human-readable explanation
        - confidence: "high" | "medium" | "low"
        - missing_items: list of things that prevented fast path (if any)
        - deterministic_primary_icd: str | None
        - deterministic_secondary_icds: list[str]
        - deterministic_cpt: str | None
    """
    missing_items: List[str] = []
    flags: List[str] = []
    secondary_icds: List[str] = []
    cpt_result: Optional[Dict[str, Any]] = None

    # 1. Check for uncertainty signals (hard blockers)
    uncertainty_reasons = _has_uncertainty_signals(chief_complaint, cleaned_soap_note, patient_context, visit_context)
    if uncertainty_reasons:
        missing_items.extend(uncertainty_reasons)

    # 2. Try to derive primary ICD-10 deterministically
    primary_icd, primary_missing = _can_derive_primary_icd(
        patient_context, chief_complaint, cleaned_soap_note
    )
    missing_items.extend(primary_missing)

    # 3. Resolve symptom codes deterministically
    scribe_codes = list(cleaned_soap_note.get("icd10_suggestions") or [])
    symptom_codes, symptom_missing, symptom_flags = _resolve_symptom_codes_deterministic(
        chief_complaint, patient_context, visit_context, already_coded=scribe_codes
    )
    missing_items.extend(symptom_missing)
    flags.extend(symptom_flags)
    secondary_icds.extend(symptom_codes)

    # 4. Compute CPT deterministically
    cpt_result, cpt_missing, cpt_flags = _compute_cpt_deterministic(
        cleaned_soap_note, patient_context
    )
    missing_items.extend(cpt_missing)
    flags.extend(cpt_flags)

    # 5. Check if we have enough for Fast Path
    can_use_fast_path = (
        primary_icd is not None
        and cpt_result is not None
        and len(missing_items) == 0
    )

    # Build decision metadata
    if can_use_fast_path:
        reason = "All coding elements derivable deterministically from chart and encounter"
        confidence = "high"
    else:
        reason = "Fast Path ineligible: " + "; ".join(missing_items) if missing_items else "Unknown reason"
        confidence = "low"

    decision_metadata = {
        "reason": reason,
        "confidence": confidence,
        "missing_items": missing_items,
        "deterministic_primary_icd": primary_icd,
        "deterministic_secondary_icds": secondary_icds,
        "deterministic_cpt": cpt_result["cpt_description"] if cpt_result else None,
        "decision_flags": flags,
    }

    logger.info(
        "Coding path decision: %s – reason: %s (confidence: %s)",
        "FAST" if can_use_fast_path else "SLOW",
        reason,
        confidence,
    )

    return can_use_fast_path, decision_metadata


def execute_fast_path(
    cleaned_soap_note: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    visit_context: Optional[Dict[str, Any]] = None,
    decision_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Execute the Fast Path: build complete coding_result using deterministic tools only.

    Returns:
        (coding_result, tool_trace, quality_flags)
    """
    if decision_metadata is None:
        raise ValueError("decision_metadata required for Fast Path execution")

    primary_icd = decision_metadata.get("deterministic_primary_icd")
    secondary_icds = list(decision_metadata.get("deterministic_secondary_icds") or [])
    cpt_result = decision_metadata.get("deterministic_cpt")
    cpt_code = cpt_result.split(" - ")[0] if cpt_result else "99213"
    cpt_domains = None

    # Re-compute CPT domains for trace
    if cpt_result and "domains" in decision_metadata:
        cpt_domains = decision_metadata.get("domains")
    else:
        # Fallback: compute fresh
        domains = derive_mdm_domains(cleaned_soap_note, patient_context)
        cpt_domains = domains

    # Build coding result
    coding_result = {
        "primary_icd10": primary_icd or "",
        "secondary_icd10": secondary_icds,
        "cpt_suggestions": [cpt_result] if cpt_result else ["99213 - Office visit, established patient, low medical decision making"],
        "medical_necessity_note": (
            f"CPT {cpt_code} supported by {primary_icd or 'chronic condition management'} "
            f"and symptom management"
        ),
    }

    # Build tool trace (same structure as Slow Path)
    scribe_codes = list(cleaned_soap_note.get("icd10_suggestions") or [])
    candidates = find_symptom_candidates(
        chief_complaint=chief_complaint,
        visit_context=visit_context,
        patient_context=patient_context,
        already_coded=scribe_codes,
        max_codes=MAX_FAST_PATH_SYMPTOM_CODES,
    )

    tool_trace = {
        "simple_symptom_to_icd_suggestion": {
            "chief_complaint": chief_complaint,
            "suggested_codes": [c.code for c in candidates[:MAX_FAST_PATH_SYMPTOM_CODES]],
            "candidates": [
                {
                    "code": c.code,
                    "description": c.description,
                    "prefix": c.prefix,
                    "phrase": c.phrase,
                    "source": c.source,
                }
                for c in candidates[:MAX_FAST_PATH_SYMPTOM_CODES]
            ],
            "added_to_result": secondary_icds,
        },
        "validate_icd10_code": {
            "codes_checked": 1 + len(secondary_icds),
            "malformed_dropped": [],
        },
        "derive_mdm_domains": {
            "overall_mdm_level": cpt_domains.get("overall_mdm_level", "low") if cpt_domains else "low",
            "domains": {
                "problems": cpt_domains.get("problems_level", "low") if cpt_domains else "low",
                "data": cpt_domains.get("data_level", "low") if cpt_domains else "low",
                "risk": cpt_domains.get("risk_level", "low") if cpt_domains else "low",
            },
        },
        "calculate_em_code": {
            "cpt_code": cpt_code,
            "justification": cpt_domains.get("justification", "Fast Path deterministic calculation") if cpt_domains else "Fast Path deterministic calculation",
        },
        "validate_icd_against_patient": {
            "codes_checked": 1 + len(secondary_icds),
            "flags": [],
        },
        "fast_path": {
            "used": True,
            "reason": decision_metadata.get("reason", ""),
            "confidence": decision_metadata.get("confidence", "high"),
        },
    }

    # Quality flags from decision
    quality_flags = list(decision_metadata.get("decision_flags") or [])

    return coding_result, tool_trace, quality_flags


__all__ = [
    "should_use_fast_path",
    "execute_fast_path",
    "MAX_FAST_PATH_SYMPTOM_CODES",
]