"""
Validation utilities for clinical coding accuracy.
Provides lightweight guardrails to detect hallucinated diagnoses and incorrect codes.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

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
    "hyperlipidemia": ["E78.5", "E78"],
    "hld": ["E78.5", "E78"],
    "dyslipidemia": ["E78.5", "E78"],
    "ckd": ["N18.9", "N18"],
    "chronic kidney disease": ["N18.9", "N18"],
    "chronic kidney": ["N18.9", "N18"],
    "asthma": ["J45.909", "J45"],
    "seasonal allergies": ["J30.9"],
    "allergic rhinitis": ["J30.9"],
    "angina": ["I20"],
    "chest pain": ["R07.9", "R07"],
    "hypothyroidism": ["E03.9", "E03"],
    "heart failure": ["I50.9", "I50"],
    "hfpef": ["I50.9", "I50"],
    "hfref": ["I50.9", "I50"],
    "atrial fibrillation": ["I48.91", "I48"],
    "afib": ["I48.91", "I48"],
    "depression": ["F32.9", "F33.9", "F32", "F33"],
    "major depressive disorder": ["F32.9", "F33.9", "F32", "F33"],
    "copd": ["J44.9", "J44"],
    "chronic obstructive pulmonary disease": ["J44.9", "J44"],
    "migraine": ["G43.909", "G43"],
}

# ICD-10 codes that indicate specific conditions we can check for
ICD10_CONDITION_MAP = {
    "E11": "Type 2 Diabetes",
    "I10": "Essential Hypertension",
    "E78": "Hyperlipidemia/Dyslipidemia",
    "N18": "Chronic Kidney Disease",
    "J45": "Asthma",
    "J30": "Allergic Rhinitis/Seasonal Allergies",
    "I20": "Angina/Ischemic Heart Disease",
    "R07": "Chest Pain",
    "E03": "Hypothyroidism",
    "I50": "Heart Failure",
    "I48": "Atrial Fibrillation",
    "F32": "Depression",
    "F33": "Depression",
    "J44": "COPD",
    "G43": "Migraine",
}

# Hallucinated terms to watch for (when not in patient context)
HALLUCINATED_TERMS = {
    "asthma": ["asthma", "albuterol", "salbutamol", "prednisone", "peak flow", "wheezing", "j45"],
    "copd": ["copd", "tiotropium", "spivia", "ipratropium", "fev1", "j44"],
    "heart failure": ["heart failure", "furosemide", "lasix", "bnp", "ejection fraction", "chf", "i50"],
    "ckd": ["ckd", "chronic kidney", "creatinine", "gfr", "dialysis", "n18"],
}

# Common medication / drug-class keywords for grounding & allergy checks
COMMON_DRUG_TERMS = (
    "metformin", "insulin", "glargine", "lispro", "aspart", "glipizide", "empagliflozin",
    "lisinopril", "losartan", "valsartan", "amlodipine", "hydrochlorothiazide", "hctz",
    "metoprolol", "carvedilol", "atenolol", "furosemide", "spironolactone",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    "aspirin", "clopidogrel", "apixaban", "rivaroxaban", "warfarin", "heparin",
    "albuterol", "fluticasone", "montelukast", "levothyroxine", "omeprazole", "pantoprazole",
    "ibuprofen", "naproxen", "acetaminophen", "amoxicillin", "penicillin", "azithromycin",
    "sumatriptan", "sertraline", "escitalopram",
)

ALLERGY_CLASS_MAP = {
    "penicillin": ("penicillin", "amoxicillin", "ampicillin", "piperacillin"),
    "sulfa": ("sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim", "furosemide", "hydrochlorothiazide"),
    "nsaid": ("ibuprofen", "naproxen", "meloxicam", "ketorolac", "celecoxib"),
    "aspirin": ("aspirin", "salicylate"),
    "statin": ("atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"),
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

    condition_keywords = [
        "type 2 diabetes", "diabetes", "hypertension", "htn",
        "hyperlipidemia", "hld", "dyslipidemia",
        "ckd", "chronic kidney disease", "chronic kidney",
        "asthma", "seasonal allergies", "allergic rhinitis",
        "hypothyroidism",
        "heart failure", "hfpef", "hfref",
        "atrial fibrillation", "afib",
        "depression", "major depressive disorder",
        "copd", "chronic obstructive pulmonary disease",
        "migraine",
    ]

    for cond in condition_keywords:
        if cond in context_lower:
            conditions.append(cond)

    return conditions


def extract_icd10_codes(text: str) -> List[str]:
    """Extract ICD-10 codes from text (format: CODE - Description)."""
    pattern = r'([A-Z]\d{2}(?:\.\d+)?)'
    return re.findall(pattern, text)


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
    grounding_text = _with_extra_context(patient_context, extra_context)
    known_conditions = extract_conditions_from_context(grounding_text)

    complaint_lower = chief_complaint.lower()
    if "chest pain" in complaint_lower:
        known_conditions.append("chest pain")
    if "shortness of breath" in complaint_lower or "dyspnea" in complaint_lower:
        known_conditions.append("shortness of breath")

    flagged_codes = []
    warnings = []
    quality_flags = []

    for code in icd10_codes:
        code_prefix = code[:3]
        condition_name = ICD10_CONDITION_MAP.get(code_prefix)

        if condition_name:
            condition_found = any(
                known in condition_name.lower() or condition_name.lower() in known
                for known in known_conditions
            )

            if not condition_found:
                grounding = _with_extra_context(patient_context, extra_context)
                has_angina_diagnosis = "angina" in grounding.lower() or "ischemic" in grounding.lower()
                has_cardiac_chest_pain = "cardiac" in complaint_lower and "chest pain" in complaint_lower
                if code_prefix == "I20":
                    if not has_angina_diagnosis and not has_cardiac_chest_pain:
                        flagged_codes.append(code)
                        warnings.append(f"Code {code} ({condition_name}) used without documented angina/cardiac ischemia diagnosis")
                        quality_flags.append(f"Possible hallucinated diagnosis: {condition_name} (code {code}) removed")
                elif code_prefix == "J45":
                    has_asthma = "asthma" in patient_context.lower()
                    if not has_asthma:
                        flagged_codes.append(code)
                        warnings.append(f"Code {code} ({condition_name}) used without asthma in patient context")
                        quality_flags.append(f"Possible hallucinated diagnosis: {condition_name} (code {code}) removed")
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

    is_established = (
        "last visit" in context_lower or
        "conditions:" in context_lower or
        "patient_id" in context_lower
    )

    for cpt in cpt_codes:
        match = re.match(r'(\d{5})', cpt)
        if match:
            cpt_num = int(match.group(1))

            if 99202 <= cpt_num <= 99205 and is_established:
                quality_flags.append(
                    f"CPT {cpt_num} is for new patients; patient appears established. Consider 99213-99215."
                )

            if 99211 <= cpt_num <= 99215 and not is_established:
                quality_flags.append(
                    f"CPT {cpt_num} is for established patients; verify patient status."
                )

    return {"quality_flags": quality_flags}


CHIEF_COMPLAINT_CODE_HINTS: List[Tuple[str, str, str]] = symptom_hint_rows() + [
    ("high blood pressure", "I10", "high blood pressure")
]


def validate_chief_complaint_coverage(
    icd10_codes: List[str],
    chief_complaint: str
) -> List[str]:
    """
    Flag chief complaint symptoms that have no corresponding ICD-10 code.

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


def validate_plan_and_medication_grounding(
    soap_note: Optional[Dict[str, Any]],
    patient_context: str,
    chief_complaint: str = "",
    visit_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Advisory neuro-symbolic checks on Plan medication mentions/actions,
    documented allergy conflicts, and acute-acuity vs. follow-up timing.

    Returns non-blocking quality flags for clinician review.
    """
    if not soap_note:
        return []

    plan_text = str(soap_note.get("plan", "") or "").strip()
    if not plan_text:
        return []

    flags: List[str] = []
    plan_lower = plan_text.lower()

    visit_text = _extract_visit_field_text(
        visit_context,
        (
            "hpi", "physical_exam", "additional_findings", "vitals_note",
            "blood_pressure", "heart_rate", "temperature", "respiratory_rate",
            "spo2", "weight_kg", "bmi",
        ),
    )
    chart_and_visit = f"{patient_context or ''}\n{visit_text}\n{chief_complaint or ''}".lower()

    # 1. Medication grounding check: flag drug names mentioned in Plan that do not
    # appear anywhere in patient_context, visit_context, or chief_complaint.
    mentioned_drugs = [
        drug for drug in COMMON_DRUG_TERMS
        if re.search(rf"\b{re.escape(drug)}\b", plan_lower)
    ]
    ungrounded_drugs = [
        drug for drug in mentioned_drugs
        if not re.search(rf"\b{re.escape(drug)}\b", chart_and_visit)
    ]
    if ungrounded_drugs:
        flags.append(
            "Medication grounding review: Plan references medication(s) not found in "
            f"chart/visit context ({', '.join(sorted(set(ungrounded_drugs)))}) - verify clinical basis"
        )

    # 2. Medication action specificity check: if Plan mentions adjusting/starting/
    # stopping/continuing a medication regimen without naming a specific drug or dose,
    # flag for clinician clarification.
    action_verb_present = bool(
        re.search(
            r"\b(?:start|initiate|stop|discontinue|hold|titrate|adjust|increase|decrease|continue|prescribe)\b",
            plan_lower,
        )
    )
    med_noun_present = bool(re.search(r"\b(?:medication|medications|meds|regimen|therapy|dose|dosage)\b", plan_lower))
    has_explicit_dose = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml)\b", plan_lower))
    if action_verb_present and med_noun_present and not mentioned_drugs and not has_explicit_dose:
        flags.append(
            "Medication action specificity: Plan references medication actions/regimen "
            "without naming specific medication(s) or dose(s) - review before sign-off"
        )

    # 3. Allergy cross-check against patient_context allergies
    patient_lower = (patient_context or "").lower()
    allergies_match = re.search(r"allergies\s*:\s*([^\n]+)", patient_lower)
    allergies_text = allergies_match.group(1) if allergies_match else patient_lower
    if allergies_text and "nkda" not in allergies_text and "none" not in allergies_text:
        for allergy_key, related_drugs in ALLERGY_CLASS_MAP.items():
            if allergy_key in allergies_text:
                conflicts = [
                    d for d in mentioned_drugs
                    if d == allergy_key or d in related_drugs
                ]
                if conflicts:
                    flags.append(
                        f"Allergy check: Plan mentions {', '.join(sorted(set(conflicts)))} "
                        f"with documented '{allergy_key}' allergy in chart - verify safety"
                    )

    # 4. Acuity vs. follow-up timing check for high-acuity presentations
    high_acuity_signals = (
        "chest pain", "troponin", "st depression", "st elevation",
        "acute coronary", "acs", "myocardial infarction",
        "shortness of breath", "dyspnea", "spo2 9", "hypoxia",
    )
    is_high_acuity = any(sig in f"{chief_complaint or ''} {visit_text}".lower() for sig in high_acuity_signals)
    delayed_followup = bool(
        re.search(
            r"\b(?:follow[- ]?up\s+(?:in\s+)?(?:2|3|4|6)\s+weeks?|routine\s+(?:outpatient\s+)?follow[- ]?up)\b",
            plan_lower,
        )
    )
    has_urgent_escalation = bool(
        re.search(
            r"\b(?:emergency\s+(?:department|room)|ed\b|er\b|911|immediate\s+(?:transfer|evaluation|cardiology)|urgent)\b",
            plan_lower,
        )
    )
    if is_high_acuity and delayed_followup and not has_urgent_escalation:
        flags.append(
            "Acuity vs. follow-up mismatch: high-acuity presentation paired with routine/delayed "
            "follow-up wording without explicit urgent/ED escalation - review carefully"
        )

    return flags


def run_full_validation(
    soap_icd10: List[str],
    coding_result: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    extra_context: str = "",
    soap_note: Optional[Dict[str, Any]] = None,
    visit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run complete validation on pipeline output.

    Args:
        extra_context: Optional clinician-provided visit text (HPI, exam, tests)
            that widens the grounding set.
        soap_note: Optional full SOAP note to check for empty sections and plan guards.
        visit_context: Optional structured visit dictionary for acuity/medication checks.

    Returns combined quality_flags and validation results.
    """
    all_quality_flags = []

    if soap_note:
        subjective = str(soap_note.get("subjective", "") or "").strip()
        objective = str(soap_note.get("objective", "") or "").strip()
        assessment = str(soap_note.get("assessment", "") or "").strip()
        plan = str(soap_note.get("plan", "") or "").strip()
        if not subjective and not objective and not assessment and not plan:
            all_quality_flags.append("soap_empty: All SOAP sections (subjective, objective, assessment, plan) are empty")

    scribe_codes = extract_icd10_from_list(soap_icd10)
    scribe_validation = validate_icd10_against_context(
        scribe_codes, patient_context, chief_complaint, extra_context
    )
    all_quality_flags.extend(scribe_validation["quality_flags"])

    coding_codes = [coding_result.get("primary_icd10", "")] + coding_result.get("secondary_icd10", [])
    coding_codes_extracted = extract_icd10_from_list(coding_codes)
    coding_validation = validate_icd10_against_context(
        coding_codes_extracted, patient_context, chief_complaint, extra_context
    )
    all_quality_flags.extend(coding_validation["quality_flags"])

    cpt_codes = coding_result.get("cpt_suggestions", [])
    cpt_validation = validate_cpt_for_patient_status(cpt_codes, patient_context)
    all_quality_flags.extend(cpt_validation["quality_flags"])

    all_codes = scribe_codes + coding_codes_extracted
    all_quality_flags.extend(
        validate_chief_complaint_coverage(all_codes, chief_complaint)
    )

    all_quality_flags.extend(
        validate_plan_and_medication_grounding(
            soap_note=soap_note,
            patient_context=patient_context,
            chief_complaint=chief_complaint,
            visit_context=visit_context,
        )
    )

    return {
        "quality_flags": all_quality_flags,
        "has_issues": len(all_quality_flags) > 0
    }


def validate_post_fix_integrity(
    coding_result: Dict[str, Any],
    auto_added_codes: List[str],
) -> List[str]:
    """Structural re-check of a code list the Supervisor itself changed."""
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


# ---------------------------------------------------------------------------
# Numeric & clinical-qualifier grounding check
# ---------------------------------------------------------------------------
def _extract_visit_field_text(visit_context: Optional[Dict[str, Any]], keys: Tuple[str, ...]) -> str:
    """Extract formatted `key: value` lines for selected visit_context fields."""
    if not visit_context:
        return ""
    lines = []
    for key in keys:
        val = visit_context.get(key)
        if val is not None and str(val).strip():
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def validate_numeric_grounding(
    soap_note: Dict[str, Any],
    patient_context: str,
    visit_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Compare key numeric values and high-risk clinical qualifiers in SOAP note
    against visit_context and patient_context.

    Today's encounter values in `visit_context` take priority over historical
    values in `patient_context` so prior-visit numbers (e.g. last visit BP) do
    not trigger false mismatch alerts when today's encounter reading matches.

    Returns list of quality flags for discrepancies (non-blocking, advisory).
    """
    flags: List[str] = []

    visit_all_keys = (
        "hpi", "physical_exam", "additional_findings", "vitals_note",
        "blood_pressure", "heart_rate", "temperature", "respiratory_rate",
        "spo2", "weight_kg", "bmi",
    )
    visit_text = _extract_visit_field_text(visit_context, visit_all_keys)
    patient_text = patient_context or ""
    combined_grounding_text = "\n".join(part for part in (visit_text, patient_text) if part)

    soap_text = " ".join([
        str(soap_note.get("subjective", "") or ""),
        str(soap_note.get("objective", "") or ""),
        str(soap_note.get("assessment", "") or ""),
        str(soap_note.get("plan", "") or ""),
    ])

    numeric_patterns = [
        ("blood pressure", r"(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3}/\d{2,3})", True),
        ("heart rate", r"(?:hr|heart rate)\s*[:=]?\s*(\d{2,3})", True),
        ("temperature", r"(?:temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)", True),
        ("respiratory rate", r"(?:rr|respiratory rate)\s*[:=]?\s*(\d{1,2})", True),
        ("spo2", r"(?:spo2|spo₂|oxygen saturation)\s*[:=]?\s*(\d{2,3})", True),
        ("weight", r"(?:weight)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)", True),
        ("bmi", r"(?:bmi)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)", True),
        ("hba1c", r"(?:hba1c|hb a1c|a1c)\s*[:=]?\s*(\d(?:\.\d)?)%?", False),
        ("glucose", r"(?:glucose|blood sugar)\s*[:=]?\s*(\d{2,3})", False),
        ("ldl", r"(?:ldl)\s*[:=]?\s*(\d{2,3})", False),
        ("egfr", r"(?:egfr|gfr)\s*[:=]?\s*(\d{2,3})", False),
        ("creatinine", r"(?:creatinine|cr)\s*[:=]?\s*(\d(?:\.\d)?)", False),
        ("microalbumin", r"(?:microalbumin|albumin)\s*[:=]?\s*(\d{1,3})", False),
    ]

    for label, pattern, prefer_visit_first in numeric_patterns:
        soap_matches = re.findall(pattern, soap_text, re.IGNORECASE)
        if not soap_matches:
            continue

        if prefer_visit_first:
            visit_matches = re.findall(pattern, visit_text, re.IGNORECASE)
            if visit_matches:
                ground_val = visit_matches[0]
                source_label = "visit context"
            else:
                patient_matches = re.findall(pattern, patient_text, re.IGNORECASE)
                if not patient_matches:
                    continue
                ground_val = patient_matches[0]
                source_label = "patient context"
        else:
            ground_matches = re.findall(pattern, combined_grounding_text, re.IGNORECASE)
            if not ground_matches:
                continue
            ground_val = ground_matches[0]
            source_label = "visit/patient context"

        soap_val = soap_matches[0]
        if ground_val != soap_val:
            flags.append(
                f"Numeric mismatch: {label} in SOAP ({soap_val}) differs from {source_label} ({ground_val})"
            )

    high_risk_qualifiers = [
        (r"\bischemic\b", "ischemic claim", True),
        (r"\bischemia\b", "ischemia claim", True),
        (r"\bangina\b", "angina claim", True),
        (r"\bcoronary\b", "coronary claim", True),
        (r"\bmyocardial infarction\b", "MI claim", True),
        (r"\bacs\b", "ACS claim", True),
        (r"\bheart failure\b", "heart failure claim", True),
        (r"\bchf\b", "CHF claim", True),
        (r"\bejection fraction\b", "ejection fraction claim", True),
        (r"\bbnp\b", "BNP claim", True),
        (r"\bstroke\b", "stroke claim", False),
        (r"\btia\b", "TIA claim", False),
        (r"\btransient ischemic\b", "transient ischemic claim", False),
        (r"\bpulmonary embolism\b", "pulmonary embolism claim", False),
        (r"\bpe\b", "PE claim", False),
        (r"\bsepsis\b", "sepsis claim", False),
        (r"\bseptic\b", "septic claim", False),
        (r"\bcancer\b", "cancer claim", False),
        (r"\bmalignan\w*\b", "malignancy claim", False),
        (r"\bmetastatic\b", "metastatic claim", False),
        (r"\bacute abdomen\b", "acute abdomen claim", False),
        (r"\bperitonitis\b", "peritonitis claim", False),
        (r"\bmost consistent with\b", "definitive diagnostic phrasing", False),
        (r"\bconsistent with\b", "diagnostic phrasing", False),
        (r"\bsuggestive of\b", "diagnostic phrasing", False),
        (r"\bindicative of\b", "diagnostic phrasing", False),
        (r"\bdiagnostic of\b", "diagnostic phrasing", False),
        (r"\bconfirmed\b", "confirmed diagnosis phrasing", False),
    ]

    cardiac_terms = [
        r"\bangina\b", r"\bischemic\b", r"\bcoronary\b", r"\bmyocardial infarction\b",
        r"\bmi\b", r"\bacs\b", r"\bheart failure\b", r"\bchf\b", r"\bcardiomyopathy\b",
        r"\batrial fibrillation\b", r"\bafib\b",
    ]
    patient_lower = patient_text.lower()
    cardiac_dx_in_chart = any(
        re.search(term, patient_lower) for term in cardiac_terms
    )

    for pattern, label, is_cardiac in high_risk_qualifiers:
        matches = re.findall(pattern, soap_text, re.IGNORECASE)
        if not matches:
            continue

        matched_snippet = matches[0]
        if is_cardiac:
            if not cardiac_dx_in_chart:
                flags.append(
                    f"Unsupported clinical qualifier ('{matched_snippet}' / {label}): "
                    f"appears in SOAP but no cardiac diagnosis is documented in patient context"
                )
        else:
            flags.append(
                f"High-risk clinical qualifier needs review ('{matched_snippet}' / {label}): "
                f"verify grounding in visit and patient data"
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

    forbidden_categories = []
    for category, terms in HALLUCINATED_TERMS.items():
        category_allowed = any(
            any(term in known for term in terms)
            for known in known_conditions
        )
        if not category_allowed:
            forbidden_categories.append((category, terms))

    if "icd10_suggestions" in cleaned:
        original_icd10 = cleaned["icd10_suggestions"]
        cleaned_icd10 = []
        for code_desc in original_icd10:
            code_match = re.match(r'([A-Z]\d{2}(?:\.\d+)?)', code_desc)
            if code_match:
                code = code_match.group(1)
                code_prefix = code[:3]
                condition_name = ICD10_CONDITION_MAP.get(code_prefix)

                allowed = False
                if condition_name:
                    for known in known_conditions:
                        if known in condition_name.lower() or condition_name.lower() in known:
                            allowed = True
                            break

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

    if "assessment" in cleaned and cleaned["assessment"]:
        assessment = cleaned["assessment"]
        for category, terms in forbidden_categories:
            for term in terms:
                pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
                if pattern.search(assessment):
                    assessment = pattern.sub('[REMOVED - hallucinated]', assessment)
                    removed_items.append(f"Sanitized assessment: '{term}'")
        cleaned["assessment"] = assessment

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
    patient_context = """
    Patient: John Doe, 58yo male
    Conditions: Type 2 Diabetes, Hypertension
    Medications: Metformin 500mg BID, Lisinopril 10mg daily
    Allergies: Penicillin
    Last Visit: Blood pressure 148/92. HbA1c 7.8%.
    """

    test_icd10 = [
        "E11.9 - Type 2 diabetes mellitus without complications",
        "I10 - Essential (primary) hypertension",
        "J45.909 - Unspecified asthma",
    ]

    result = validate_icd10_against_context(test_icd10, patient_context, "routine follow-up")
    print("Validation result:", result)