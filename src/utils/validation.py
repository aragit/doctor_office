"""
Validation utilities for clinical coding accuracy.
Provides lightweight guardrails to detect hallucinated diagnoses and incorrect codes.
"""

import re
from typing import List, Dict, Any, Tuple

from utils.cardiac_claims import (
    has_charted_cardiac_diagnosis,
    scrub_numbered_lines,
    scrub_unsupported_cardiac_claims,
)
from utils.symptom_codes import symptom_hint_rows


# Known condition to ICD-10 mappings for validation
CONDITION_ICD10_MAP = {
    "type 2 diabetes": ["E11.9", "E11"],
    "diabetes": ["E11.9", "E11"],
    "hypertension": ["I10"],
    "essential hypertension": ["I10"],
    "htn": ["I10"],
    "asthma": ["J45.909", "J45"],
    "seasonal allergies": ["J30.9"],
    "allergic rhinitis": ["J30.9"],
    "angina": ["I20"],
    "chest pain": ["R07.9", "R07"],
}

# ICD-10 codes that indicate specific conditions we can check for
ICD10_CONDITION_MAP = {
    "E11": "Type 2 Diabetes",
    "I10": "Essential Hypertension",
    "J45": "Asthma",
    "J30": "Allergic Rhinitis/Seasonal Allergies",
    "I20": "Angina/Ischemic Heart Disease",
    "R07": "Chest Pain",
}

# Hallucinated terms to watch for (when not in patient context)
HALLUCINATED_TERMS = {
    "asthma": ["asthma", "albuterol", "salbutamol", "prednisone", "peak flow", "wheezing", "j45"],
    "copd": ["copd", "tiotropium", "spivia", "ipratropium", "fev1", "j44"],
    "heart failure": ["heart failure", "furosemide", "lasix", "bnp", "ejection fraction", "chf", "i50"],
    "ckd": ["ckd", "chronic kidney", "creatinine", "gfr", "dialysis", "n18"],
}

# Upper bound on symptom/sign codes a single visit may carry, enforced only on
# the post-fix re-validation. Matches the candidate cap the Coding node uses, so
# an auto-fix cannot push the result past what the Coding path is allowed to
# propose.
MAX_POST_FIX_SYMPTOM_CODES = 4


def extract_conditions_from_context(patient_context: str) -> List[str]:
    """Extract known conditions from patient context string."""
    conditions = []
    context_lower = patient_context.lower()
    
    # Look for conditions in the context
    condition_keywords = [
        "type 2 diabetes", "diabetes", "hypertension", "htn", 
        "asthma", "seasonal allergies", "allergic rhinitis"
    ]
    
    for cond in condition_keywords:
        if cond in context_lower:
            conditions.append(cond)
    
    return conditions


def extract_icd10_codes(text: str) -> List[str]:
    """Extract ICD-10 codes from text (format: CODE - Description)."""
    pattern = r'([A-Z]\d{2}(?:\.\d+)?)'
    codes = re.findall(pattern, text)
    return codes


def extract_icd10_from_list(icd10_list: List[str]) -> List[str]:
    """Extract just the code portion from ICD-10 strings."""
    codes = []
    for item in icd10_list:
        match = re.match(r'([A-Z]\d{2}(?:\.\d+)?)', item)
        if match:
            codes.append(match.group(1))
    return codes


def validate_icd10_against_context(
    icd10_codes: List[str],
    patient_context: str,
    chief_complaint: str = "",
    extra_context: str = ""
) -> Dict[str, Any]:
    """
    Validate that ICD-10 codes only reference conditions present in patient_context or chief_complaint.

    Args:
        extra_context: Optional additional clinician-provided text for this visit
            (HPI, exam, tests). Treated as authoritative grounding, the same as
            the chart, so a condition documented today is not flagged.

    Returns:
        dict with keys:
        - valid: bool
        - flagged_codes: list of codes that may be hallucinated
        - warnings: list of warning messages
        - quality_flags: list of quality flags for output
    """
    # Extract conditions from patient context (+ any visit data supplied)
    grounding_text = _with_extra_context(patient_context, extra_context)
    known_conditions = extract_conditions_from_context(grounding_text)
    
    # Also check chief complaint for acute symptoms
    complaint_lower = chief_complaint.lower()
    if "chest pain" in complaint_lower:
        known_conditions.append("chest pain")
    if "shortness of breath" in complaint_lower or "dyspnea" in complaint_lower:
        known_conditions.append("shortness of breath")
    
    flagged_codes = []
    warnings = []
    quality_flags = []
    
    for code in icd10_codes:
        code_prefix = code[:3]  # e.g., "E11", "I10", "I20"
        
        # Check if this ICD-10 maps to a known condition
        condition_name = ICD10_CONDITION_MAP.get(code_prefix)
        
        if condition_name:
            # Check if this condition is in known conditions
            condition_found = False
            for known in known_conditions:
                if known in condition_name.lower() or condition_name.lower() in known:
                    condition_found = True
                    break
            
            if not condition_found:
                # Special case: I20 (angina) requires explicit cardiac diagnosis
                # Consider extra_context (visit HPI/exam/tests) the same as cleanup does.
                grounding = _with_extra_context(patient_context, extra_context)
                has_angina_diagnosis = "angina" in grounding.lower() or "ischemic" in grounding.lower()
                has_cardiac_chest_pain = "cardiac" in complaint_lower and "chest pain" in complaint_lower
                if code_prefix == "I20":
                    if not has_angina_diagnosis and not has_cardiac_chest_pain:
                        flagged_codes.append(code)
                        warnings.append(f"Code {code} ({condition_name}) used without documented angina/cardiac ischemia diagnosis")
                        quality_flags.append(f"Possible hallucinated diagnosis: {condition_name} (code {code}) removed")
                # Special case: J45 (asthma) requires explicit asthma diagnosis
                elif code_prefix == "J45":
                    has_asthma = "asthma" in patient_context.lower()
                    if not has_asthma:
                        flagged_codes.append(code)
                        warnings.append(f"Code {code} ({condition_name}) used without asthma in patient context")
                        quality_flags.append(f"Possible hallucinated diagnosis: {condition_name} (code {code}) removed")
                # General case for other conditions
                else:
                    flagged_codes.append(code)
                    warnings.append(f"Code {code} ({condition_name}) references condition not in patient context")
                    quality_flags.append(f"Possible hallucinated diagnosis: {condition_name} (code {code}) needs review")
    
    return {
        "valid": len(flagged_codes) == 0,
        "flagged_codes": flagged_codes,
        "warnings": warnings,
        "quality_flags": quality_flags
    }


def validate_cpt_for_patient_status(
    cpt_codes: List[str],
    patient_context: str
) -> Dict[str, Any]:
    """
    Validate CPT codes are appropriate for patient status (new vs established).
    
    Returns:
        dict with quality_flags for any issues
    """
    quality_flags = []
    context_lower = patient_context.lower()
    
    # Check if patient appears established (has prior visit/conditions)
    is_established = (
        "last visit" in context_lower or 
        "conditions:" in context_lower or
        "patient_id" in context_lower
    )
    
    for cpt in cpt_codes:
        # Extract CPT code number
        match = re.match(r'(\d{5})', cpt)
        if match:
            cpt_num = int(match.group(1))
            
            # New patient codes (99202-99205) for established patients
            if 99202 <= cpt_num <= 99205 and is_established:
                quality_flags.append(
                    f"CPT {cpt_num} is for new patients; patient appears established. Consider 99213-99215."
                )
            
            # Established patient codes for potentially new patients (less critical)
            if 99211 <= cpt_num <= 99215 and not is_established:
                quality_flags.append(
                    f"CPT {cpt_num} is for established patients; verify patient status."
                )
    
    return {"quality_flags": quality_flags}


# Chief complaint phrases that should normally carry a symptom ICD-10 code.
# Built from the central map (Phase A3) so this gap flag, the Coding candidates
# and the Supervisor's auto-completion can never disagree about which phrase
# implies which code.
#
# Every phrase in the central map, one gap-flag row each. The I10 row is
# deliberately NOT part of that map: "high blood pressure" is a hypothesis about
# a chronic diagnosis, not a symptom. It stays here as a review prompt for the
# human coder only - nothing adds I10 automatically, because a chronic diagnosis
# has to come from the chart.
CHIEF_COMPLAINT_CODE_HINTS: List[Tuple[str, str, str]] = symptom_hint_rows() + [
    ("high blood pressure", "I10", "high blood pressure")
]


def validate_chief_complaint_coverage(
    icd10_codes: List[str],
    chief_complaint: str
) -> List[str]:
    """
    Flag chief complaint symptoms that have no corresponding ICD-10 code.

    This is a review prompt, not a hallucination guard: the symptom legitimately
    comes from the patient, so it is the *absence* of a code that is suspicious
    (the agents focus on chronic problems and can drop acute presenting
    symptoms). Nothing is removed - the coder reviews and decides.

    Only symptom codes from `utils.symptom_codes` are ever flagged as missing,
    so this cannot ask for a diagnosis code that the deterministic layer would
    then invent. A symptom that is already coded - by the Scribe or by Coding -
    raises nothing, which is what makes the Coding path the primary source of
    symptom codes and the Supervisor the fallback.

    Returns:
        List of advisory quality flags.
    """
    if not chief_complaint:
        return []

    complaint_lower = chief_complaint.lower()
    prefixes = {code[:3].upper() for code in icd10_codes if code}

    flags = []
    seen_prefixes = set()
    for phrase, prefix, label in CHIEF_COMPLAINT_CODE_HINTS:
        if phrase not in complaint_lower or prefix in prefixes or prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        flags.append(
            f"Chief complaint mentions '{label}' but no {prefix} code was generated - "
            f"review and add a symptom code if clinically appropriate"
        )

    return flags


def run_full_validation(
    soap_icd10: List[str],
    coding_result: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    extra_context: str = ""
) -> Dict[str, Any]:
    """
    Run complete validation on pipeline output.

    Args:
        extra_context: Optional clinician-provided visit text (HPI, exam, tests)
            that widens the grounding set.

    Returns combined quality_flags and validation results.
    """
    all_quality_flags = []
    
    # Validate Scribe ICD-10 suggestions
    scribe_codes = extract_icd10_from_list(soap_icd10)
    scribe_validation = validate_icd10_against_context(
        scribe_codes, patient_context, chief_complaint, extra_context
    )
    all_quality_flags.extend(scribe_validation["quality_flags"])
    
    # Validate Coding Agent primary and secondary ICD-10
    coding_codes = [coding_result.get("primary_icd10", "")] + coding_result.get("secondary_icd10", [])
    coding_codes_extracted = extract_icd10_from_list(coding_codes)
    coding_validation = validate_icd10_against_context(
        coding_codes_extracted, patient_context, chief_complaint, extra_context
    )
    all_quality_flags.extend(coding_validation["quality_flags"])
    
    # Validate CPT codes
    cpt_codes = coding_result.get("cpt_suggestions", [])
    cpt_validation = validate_cpt_for_patient_status(cpt_codes, patient_context)
    all_quality_flags.extend(cpt_validation["quality_flags"])
    
    # Flag chief complaint symptoms that never made it into an ICD-10 code.
    all_codes = scribe_codes + coding_codes_extracted
    all_quality_flags.extend(
        validate_chief_complaint_coverage(all_codes, chief_complaint)
    )
    
    return {
        "quality_flags": all_quality_flags,
        "has_issues": len(all_quality_flags) > 0
    }


def validate_post_fix_integrity(
    coding_result: Dict[str, Any],
    auto_added_codes: List[str],
) -> List[str]:
    """Structural re-check of a code list the Supervisor itself changed.

    `run_full_validation` grounds codes and checks symptom coverage, but the
    Supervisor's deterministic symptom completion never passes through the Coding
    node's structural existence pre-check or the `validate_icd10_code` tool - it
    appends to `secondary_icd10` directly. This function closes that gap. It runs
    only on the post-fix re-validation, so the ordinary path pays nothing and the
    normal flag vocabulary is unchanged.

    Checks, in order:
    1. every entry still parses as a real ICD-10 code (structural validity);
    2. no exact duplicate code - the failure mode the Coding node's pre-check
       exists to prevent, guarded here because re-applying a state update can
       otherwise append the same completion twice;
    3. no runaway symptom-code list (more than `MAX_POST_FIX_SYMPTOM_CODES`);
    4. each code the Supervisor claims to have added is actually present, so a
       silent write failure cannot read as a successful fix.

    Grounding and symptom-gap coverage are not re-implemented here: the caller
    runs `run_full_validation` on the same mutated result, which covers both.

    Args:
        coding_result: The coding result *after* the fix was applied.
        auto_added_codes: The codes the Supervisor reported adding.

    Returns:
        Quality flags. Structural and duplication problems are worded for the
        Coding bucket ("Post-fix re-validation: ...") so a real defect routes to
        a coding retry instead of being filed as an advisory.
    """
    flags: List[str] = []
    primary = str(coding_result.get("primary_icd10", "") or "")
    secondary = [str(code or "") for code in (coding_result.get("secondary_icd10") or [])]

    seen: Dict[str, int] = {}
    for entry in [primary] + secondary:
        match = re.match(r"^\s*([A-Z]\d{2}(?:\.\d{1,4})?)\b", entry)
        if not match:
            if entry.strip():
                flags.append(
                    f"Post-fix re-validation: malformed ICD-10 entry {entry!r} in the "
                    f"coding result; it cannot be validated or billed"
                )
            continue
        code = match.group(1)
        seen[code] = seen.get(code, 0) + 1

    duplicates = sorted(code for code, count in seen.items() if count > 1)
    if duplicates:
        flags.append(
            "Post-fix re-validation: duplicate ICD-10 code(s) in the coding result: "
            + ", ".join(duplicates)
            + "; remove the repeats"
        )

    symptom_count = sum(1 for code in seen if code[:1] in "RZ")
    if symptom_count > MAX_POST_FIX_SYMPTOM_CODES:
        flags.append(
            f"Post-fix re-validation: {symptom_count} symptom/sign code(s) present "
            f"(more than {MAX_POST_FIX_SYMPTOM_CODES}); confirm the list reflects "
            f"this visit"
        )

    for code in auto_added_codes or []:
        token = re.match(r"^\s*([A-Z]\d{2}(?:\.\d{1,4})?)\b", str(code))
        wanted = token.group(1) if token else str(code)
        if wanted not in seen:
            flags.append(
                f"Post-fix re-validation: the Supervisor reported adding {wanted} "
                f"but it is absent from the coding result"
            )

    return flags


def _with_extra_context(patient_context: str, extra_context: str = "") -> str:
    """Join the chart with any clinician-provided visit data for grounding."""
    if not extra_context or not str(extra_context).strip():
        return patient_context or ""
    return f"{patient_context or ''}\n{extra_context}"


def _get_known_conditions_set(patient_context: str, chief_complaint: str) -> set:
    """Get set of normalized condition names that are allowed."""
    known = extract_conditions_from_context(patient_context)
    known_lower = {c.lower() for c in known}
    
    # Add acute symptoms from chief complaint
    complaint_lower = chief_complaint.lower()
    if "chest pain" in complaint_lower:
        known_lower.add("chest pain")
    if "shortness of breath" in complaint_lower or "dyspnea" in complaint_lower:
        known_lower.add("shortness of breath")
    if "diarrhea" in complaint_lower:
        known_lower.add("diarrhea")
    
    return known_lower


def clean_soap_note(
    soap_note: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    extra_context: str = ""
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Remove hallucinated content from SOAP note.

    Scans plan and icd10_suggestions for references to conditions not in patient_context.

    Args:
        extra_context: Optional clinician-provided text for this visit (HPI,
            exam, tests). Widens the grounding set so findings documented today
            are not scrubbed as hallucinations.

    Returns:
        Tuple of (cleaned_soap_note, list_of_removed_items)
    """
    import copy
    cleaned = copy.deepcopy(soap_note)
    removed_items = []
    grounding_text = _with_extra_context(patient_context, extra_context)
    known_conditions = _get_known_conditions_set(grounding_text, chief_complaint)
    context_lower = grounding_text.lower()
    
    # Check which hallucinated condition categories are NOT in patient context
    forbidden_categories = []
    for category, terms in HALLUCINATED_TERMS.items():
        # If none of the known conditions match this category, it's forbidden
        category_allowed = any(
            any(term in known for term in terms) 
            for known in known_conditions
        )
        if not category_allowed:
            forbidden_categories.append((category, terms))
    
    # Clean ICD-10 suggestions
    if "icd10_suggestions" in cleaned:
        original_icd10 = cleaned["icd10_suggestions"]
        cleaned_icd10 = []
        for code_desc in original_icd10:
            code_match = re.match(r'([A-Z]\d{2}(?:\.\d+)?)', code_desc)
            if code_match:
                code = code_match.group(1)
                code_prefix = code[:3]
                condition_name = ICD10_CONDITION_MAP.get(code_prefix)
                
                # Check if this code is allowed
                allowed = False
                if condition_name:
                    for known in known_conditions:
                        if known in condition_name.lower() or condition_name.lower() in known:
                            allowed = True
                            break
                
                # Special handling for I20 (angina) - needs explicit cardiac diagnosis
                if not allowed and code_prefix == "I20":
                    has_angina = "angina" in context_lower or "ischemic" in context_lower
                    has_cardiac_cp = "cardiac" in chief_complaint.lower() and "chest pain" in chief_complaint.lower()
                    allowed = has_angina or has_cardiac_cp
                
                if allowed:
                    cleaned_icd10.append(code_desc)
                else:
                    removed_items.append(f"Removed ICD-10: {code_desc}")
            else:
                cleaned_icd10.append(code_desc)
        cleaned["icd10_suggestions"] = cleaned_icd10
    
    # Clean Plan section - remove lines referencing forbidden conditions
    if "plan" in cleaned and cleaned["plan"]:
        plan_lines = cleaned["plan"].split("\n")
        cleaned_lines = []
        for line in plan_lines:
            line_lower = line.lower()
            is_forbidden = False
            for category, terms in forbidden_categories:
                if any(term in line_lower for term in terms):
                    is_forbidden = True
                    removed_items.append(f"Removed plan line: {line.strip()}")
                    break
            if not is_forbidden:
                cleaned_lines.append(line)
        cleaned["plan"] = "\n".join(cleaned_lines)
    
    # Clean Assessment section - remove references to forbidden conditions
    if "assessment" in cleaned and cleaned["assessment"]:
        assessment = cleaned["assessment"]
        for category, terms in forbidden_categories:
            for term in terms:
                # Replace mentions with [REMOVED] marker
                pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
                if pattern.search(assessment):
                    assessment = pattern.sub('[REMOVED - hallucinated]', assessment)
                    removed_items.append(f"Sanitized assessment: '{term}'")
        cleaned["assessment"] = assessment
    
    # Remove cardiac diagnoses the chart does not support.
    #
    # A real Cardiology run produced "most consistent with ischemic etiology" and
    # "further evaluation for ischemic heart disease" for a chart holding only
    # diabetes and hypertension. The codes were right; the narrative was not, and
    # nothing in the graph would have caught it. Narrative sections are scrubbed
    # here, sentence by sentence, and the removal is reported like any other
    # cleanup item - no retry, the physician sees exactly what was taken out.
    cardiac_diagnosis = has_charted_cardiac_diagnosis(patient_context)
    if not cardiac_diagnosis:
        for section in ("subjective", "objective", "assessment"):
            if cleaned.get(section):
                scrubbed, claims = scrub_unsupported_cardiac_claims(
                    cleaned[section], has_diagnosis=False
                )
                cleaned[section] = scrubbed
                for claim in claims:
                    removed_items.append(
                        f"Removed unsupported cardiac claim from {section}: {claim}"
                    )
        if cleaned.get("plan"):
            scrubbed, claims = scrub_numbered_lines(cleaned["plan"], has_diagnosis=False)
            cleaned["plan"] = scrubbed
            for claim in claims:
                removed_items.append(
                    f"Removed plan line for unsupported cardiac claim: {claim}"
                )

    return cleaned, removed_items


if __name__ == "__main__":
    # Quick test
    patient_context = """
    Patient: John Doe, 58yo male
    Conditions: Type 2 Diabetes, Hypertension
    Medications: Metformin 500mg BID, Lisinopril 10mg daily
    Allergies: Penicillin
    Last Visit: Blood pressure 148/92. HbA1c 7.8%.
    """
    
    # Test with hallucinated asthma
    test_icd10 = ["E11.9 - Type 2 diabetes mellitus without complications", 
                  "I10 - Essential (primary) hypertension",
                  "J45.909 - Unspecified asthma"]
    
    result = validate_icd10_against_context(test_icd10, patient_context, "routine follow-up")
    print("Validation result:", result)