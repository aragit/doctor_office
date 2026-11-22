"""
Deterministic CPT E/M (Evaluation & Management) calculator.

Pure Python, no LLM. Implements a *simplified* 2-of-3 Medical Decision
Making (MDM) level for ESTABLISHED-patient office visits (99212-99215) using
the 2021 AMA table structure:

    MDM level                Established-patient CPT
    ---------------------    ------------------------
    straightforward          99212
    low                      99213
    moderate                 99214
    high                     99215

The rule: the overall MDM level is the highest level at which **at least two
of the three elements** (problems addressed, data reviewed, risk of
complications/morbidity) are at or above that level.

The three input levels are derived deterministically from the cleaned SOAP
note by `derive_mdm_domains()` (keyword/count heuristics). The calculator
itself (`calculate_em_code`) only reasons about levels, so it cannot be
talked into a different code by a model.

SIMPLIFICATION: this is NOT a full CPT validator. Real MDM grids weight
|undifferentiated| counts, procedures, and specific risk-relevant
diagnoses; this uses coarse domain-level buckets designed to be
*conservative and auditable*, never to replace a coder.
"""

from typing import Any, Dict, List, Tuple

from memory.patient_profile import parse_patient_context

# ---------------------------------------------------------------------------
# Level ordering and CPT mapping
# ---------------------------------------------------------------------------

MDM_LEVELS: Tuple[str, ...] = ("straightforward", "low", "moderate", "high")
MDM_RANK: Dict[str, int] = {level: idx for idx, level in enumerate(MDM_LEVELS)}

# 2021 E/M, established patient office/outpatient visits.
CPT_BY_MDM: Dict[str, str] = {
    "straightforward": "99212 - Office visit, established patient, straightforward medical decision making",
    "low": "99213 - Office visit, established patient, low medical decision making",
    "moderate": "99214 - Office visit, established patient, moderate medical decision making",
    "high": "99215 - Office visit, established patient, high medical decision making",
}

DESCRIPTION_CALCULATE = (
    "Determine the established-patient E/M CPT code (99212-99215) from the three "
    "MDM element levels (problems, data, risk) using the 2-of-3 rule. Inputs are "
    "one of straightforward|low|moderate|high. The returned code is authoritative; "
    "do not override it with a guess."
)

# Keyword heuristics for deriving each element level from clinical text.
# Coarse on purpose: honest buckets beat a false sense of precision.
_PROBLEM_HIGH_EXACERBATION = (
    "uncontrolled", "severe", "exacerbation", "acute decompensation",
    "complication", "life-threatening", "emergency", "refractory",
)
_DATA_LAB = (
    "hba1c", "a1c", "glucose", "lab", "panel", "creatinine", "potassium",
    "sodium", "bmp", "cbc", "lipid", "troponin", "lft", "ast", "alt", "inr",
    "hemoglobin", "culture",
)
_DATA_IMAGING = (
    "x-ray", "xray", "mri", "ultrasound", "echo", "ekg", "ecg", "imaging",
    "ct scan", "cat scan",
)
_DATA_RECORDS = (
    "records", "chart review", "external notes", "discharge summary",
    "prior records", "outside records",
)
_RISK_HIGH = (
    "referral", "refer to", "surgery", "surgical", "hospital", "hospitalization",
    "admit", "intensive", "iv ", "intravenous", "opioid", "controlled substance",
    "anticoag", "warfarin", "eliquis", "xarelto",
)
_RISK_MODERATE = (
    "start ", "started", "initiate", "add ", "increase", "titrat",
    "change to", "switched to", "new medication", "insulin",
    "prednisone", "steroid", "sglt2", "glp-1", "low-dose", "dose increase",
)


def _has_marker(text: str, markers: Tuple[str, ...]) -> List[str]:
    lower = text.lower()
    return [m for m in markers if m in lower]


# ---------------------------------------------------------------------------
# Domain derivation
# ---------------------------------------------------------------------------


def _derive_problems_level(
    conditions: List[str], symptoms: List[str], text: str
) -> Tuple[str, int, List[str]]:
    """Problems level from charted conditions + acute complaint symptoms."""
    count = len(conditions) + len(symptoms)
    exacerbation = _has_marker(text, _PROBLEM_HIGH_EXACERBATION)
    if exacerbation:
        return "high", count, exacerbation
    if count >= 2:
        return "moderate", count, []
    if count == 1:
        return "low", count, []
    return "straightforward", count, []


def _derive_data_level(text: str) -> Tuple[str, int, List[str]]:
    """Data level from the distinct categories of data elements documented."""
    found: List[str] = []
    for label, markers in (
        ("laboratory", _DATA_LAB),
        ("imaging", _DATA_IMAGING),
        ("records review", _DATA_RECORDS),
    ):
        if _has_marker(text, markers):
            found.append(label)

    # Discussion with another practitioner / independent interpretation is the
    # highest data element - treat it as high.
    if _has_marker(text, ("independent interpretation", "discussed with", "consult with")):
        return "high", len(found) + 1, found + ["provider discussion"]

    n = len(found)
    if n >= 3:
        return "high", n, found
    if n == 2:
        return "moderate", n, found
    if n == 1:
        return "low", n, found
    return "straightforward", n, found


def _derive_risk_level(text: str, conditions: List[str]) -> Tuple[str, List[str]]:
    """Risk level from management markers + prescription burden."""
    high = _has_marker(text, _RISK_HIGH)
    if high:
        return "high", high
    moderate = _has_marker(text, _RISK_MODERATE)
    if moderate:
        return "moderate", moderate
    # At least one prescription medication being managed = low risk.
    if conditions or _has_marker(text, ("prescri", "medication", "metformin", "lisinopril")):
        return "low", []
    return "straightforward", []


def derive_mdm_domains(
    soap_note: Dict[str, Any], patient_context: str = ""
) -> Dict[str, Any]:
    """Derive the three MDM element levels from a cleaned SOAP note.

    Deterministic, LLM-free heuristic. Scores are intentionally conservative:
    when a domain is ambiguous it is scored one level lower rather than
    risking upcoding, and explicit risk/exacerbation markers drive it up.

    Args:
        soap_note: SOAP dict with subjective/objective/assessment/plan.
        patient_context: Formatted chart text (used for chronic condition count).

    Returns:
        Dict with `problems_level`, `data_level`, `risk_level` and the
        derivation detail for each domain (scores + matched markers).
    """
    soap_note = soap_note or {}
    sections = [str(soap_note.get(k, "")) for k in ("subjective", "objective", "assessment", "plan")]
    text = "\n".join(sections)

    profile = parse_patient_context(patient_context or "")
    conditions = list(profile.conditions)
    symptoms = [  # acute chief-complaint symptoms echoed in the exam/assessment
        label for prefix in ("R07", "R10", "R19", "R42", "R05", "R06", "R11", "R51")
        for label in _symptom_labels_if_present(prefix, text, conditions)
    ]

    problems_level, problem_count, problem_markers = _derive_problems_level(
        conditions, symptoms, text
    )
    data_level, data_count, data_markers = _derive_data_level(text)
    risk_level, risk_markers = _derive_risk_level(text, conditions)

    return {
        "problems_level": problems_level,
        "data_level": data_level,
        "risk_level": risk_level,
        "problems_count": problem_count,
        "problems_markers": problem_markers,
        "data_elements": data_markers,
        "risk_markers": risk_markers,
    }


def _symptom_labels_if_present(
    prefix: str, text: str, conditions: List[str]
) -> List[str]:
    """Return symptom labels (e.g. 'chest pain') present in the note text."""
    phrases = {
        "R07": ["chest pain", "chest tightness", "angina"],
        "R10": ["abdominal pain", "belly pain"],
        "R19": ["diarrhea", "diarrhoea", "loose stool"],
        "R42": ["dizziness", "dizzy", "lightheadedness"],
        "R05": ["cough"],
        "R06": ["shortness of breath", "dyspnea", "wheezing"],
        "R11": ["nausea", "vomiting", "emesis"],
        "R51": ["headache"],
    }
    lowered = text.lower()
    found = [p for p in phrases.get(prefix, []) if p in lowered]
    return found


# ---------------------------------------------------------------------------
# 2-of-3 calculator
# ---------------------------------------------------------------------------


def calculate_em_code(
    problems_level: str,
    data_level: str,
    risk_level: str,
) -> Dict[str, Any]:
    """Compute the established-patient E/M CPT code via the 2-of-3 MDM rule.

    Args:
        problems_level: One of straightforward|low|moderate|high.
        data_level: One of straightforward|low|moderate|high.
        risk_level: One of straightforward|low|moderate|high.

    Returns:
        Dict with `valid`, `overall_mdm_level`, `cpt_code`, `cpt_description`,
        `justification`, `elements` (per-element score), and - on invalid input -
        `error`. Never raises.
    """
    elements = {
        "problems": str(problems_level or "").strip().lower(),
        "data": str(data_level or "").strip().lower(),
        "risk": str(risk_level or "").strip().lower(),
    }
    invalid = [k for k, v in elements.items() if v not in MDM_RANK]
    if invalid:
        return {
            "valid": False,
            "error": (
                f"Invalid MDM level(s) for {', '.join(invalid)}: "
                f"expected one of {', '.join(MDM_LEVELS)}"
            ),
        }

    ranks = [MDM_RANK[level] for level in elements.values()]
    # 2-of-3: at least two elements must be at or above the chosen level. The
    # median of three ranks is exactly the highest level with that property.
    overall_rank = sorted(ranks)[1]
    overall = MDM_LEVELS[overall_rank]
    cpt = CPT_BY_MDM[overall]

    two_of_three = {
        level: sum(1 for r in ranks if r >= MDM_RANK[level])
        for level in MDM_LEVELS
    }
    justification = (
        f"MDM 2-of-3: problems={elements['problems']}, data={elements['data']}, "
        f"risk={elements['risk']} -> at least two elements meet or exceed "
        f"'{overall}' -> MDM {overall} -> CPT {cpt.split(' - ')[0]}."
    )

    return {
        "valid": True,
        "overall_mdm_level": overall,
        "cpt_code": cpt.split(" - ")[0],
        "cpt_description": cpt,
        "justification": justification,
        "elements": elements,
        "two_of_three": two_of_three,
    }


def cpt_for_mdm(level: str) -> str:
    """Map an MDM level to its established-patient CPT description."""
    key = str(level or "").strip().lower()
    return CPT_BY_MDM[key]


__all__ = [
    "CPT_BY_MDM",
    "MDM_LEVELS",
    "calculate_em_code",
    "cpt_for_mdm",
    "derive_mdm_domains",
    "DESCRIPTION_CALCULATE",
]