"""
Central symptom -> ICD-10-CM map and the pure functions built on it.

This is the single source of truth for "which symptom code does this phrase
justify". Phase A3 replaced three drifting copies of the same idea:

- `tools.coding_tools.SYMPTOM_PHRASES` / `SYMPTOM_CODES` (what the tool matches)
- `utils.validation.CHIef_COMPLAINT_CODE_HINTS` (what validation flags as missing)
- `agents.supervisor_policy._symptom_phrase_present` (what the Supervisor may complete)

with one table plus one resolver, so a phrase added here reaches all three
consumers. Nothing in this module imports an LLM, the graph, or the tools layer -
it is pure text matching, which is what makes it safe to trust.

Three rules hold everywhere:

1. **A symptom code is a symptom, never a diagnosis.** `AUTO_ADDABLE_PREFIXES`
   is the allow-list the Supervisor's deterministic completion and the Coding
   candidates are restricted to. A disease code (I10, I20, J45, ...) never comes
   from here: a phrase such as "high blood pressure" is a *hypothesis*, and
   guessing a chronic diagnosis from a chief complaint is exactly the failure
   mode A2.1 exists to prevent. Diagnosis codes stay in the hands of the chart.
2. **Never an upgrade.** `find_symptom_candidates` can only return a code from
   this table, all of which are R-codes (signs and symptoms) or the Z13
   screening code. There is no code path from "chest pain" to I20/I21.
3. **Grounded or nothing.** A candidate is returned only when the phrase is
   actually present in the supplied text, and it is suppressed when a more
   specific diagnosis for the same problem is already grounded in the chart, so
   a charted cardiac diagnosis does not also collect R07.9.
"""

import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "AUTO_ADDABLE_PREFIXES",
    "NEGATION_CUES",
    "NEGATION_WINDOW",
    "SYMPTOM_CODES",
    "SYMPTOM_PHRASES",
    "SYMPTOM_PREFIXES",
    "SUPPRESSED_BY_GROUNDED_DIAGNOSIS",
    "SymptomCandidate",
    "canonical_code_for_prefix",
    "describe_prefix",
    "find_symptom_candidates",
    "is_symptom_prefix",
    "phrase_present_not_negated",
    "symptom_gap_labels",
    "symptom_hint_rows",
    "symptom_phrase_for_prefix",
    "symptom_phrase_present",
]


# ---------------------------------------------------------------------------
# Negation-aware phrase matching
# ---------------------------------------------------------------------------

# Cues that turn a symptom phrase into a denial. Matched
# case-insensitively as whole words (except "(-)", which is the literal
# chart shorthand for "negative").
NEGATION_CUES: Tuple[str, ...] = (
    "denies", "denied", "deny", "no", "without", "not",
    "negative for", "(-)",
)

# How far before the phrase a negation cue still counts, in characters.
# A cue in the same sentence (or within this window) negates the phrase;
# a cue further back belongs to an earlier sentence/clause and does not.
NEGATION_WINDOW: int = 120

# Sentence terminators: a cue never crosses one of these to reach the
# phrase, even when the character distance is inside the window.
_SENTENCE_END_RE = re.compile(r"[.!?]")

# Word-boundary form of each cue, so "no" does not match "notable" and
# "deny" does not match "denying". "(-)" has no word characters, so it
# is matched literally.
_NEGATION_CUE_RES: Tuple[re.Pattern, ...] = tuple(
    re.compile(re.escape(cue) if not cue.isalnum() else rf"\b{re.escape(cue)}\b",
               re.IGNORECASE)
    for cue in NEGATION_CUES
)


def phrase_present_not_negated(phrase: str, text: str) -> bool:
    """True when `phrase` appears in `text` and is not negated.

    Rule (documented for auditors and tests):

    1. Find every case-insensitive occurrence of `phrase`.
    2. For each occurrence, look at the text immediately before it,
       bounded by BOTH of:
         - the nearest preceding sentence terminator (`.`, `!`, `?`),
           and
         - a `NEGATION_WINDOW`-character window (120 chars).
       The effective search span is the text between the later of those
       two boundaries and the phrase start.
    3. If any negation cue (`NEGATION_CUES`) appears in that span, the
       occurrence is a denial and does not count.
    4. The phrase is "present" when at least one occurrence survives.

    Examples:
        "Denies chest pain"            -> False (cue in same sentence)
        "no chest pain or SOB"         -> False
        "chest pain for 1 day"         -> True
        "chest pain and diarrhea"      -> True (both phrases)
        "denies chest pain. Has fever" -> chest pain: False, fever: True
        "no chest pain, but fever"     -> chest pain: False, fever: True
    """
    if not phrase or not text:
        return False
    haystack = str(text).lower()
    needle = str(phrase).lower()
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        # Span start: the later of the sentence boundary and the window.
        sentence_start = 0
        for match in _SENTENCE_END_RE.finditer(haystack, 0, idx):
            sentence_start = match.end()
        window_start = max(0, idx - NEGATION_WINDOW)
        span_start = max(sentence_start, window_start)
        span = haystack[span_start:idx]
        if not any(cue_re.search(span) for cue_re in _NEGATION_CUE_RES):
            return True
        start = idx + len(needle)
    return False


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

# ICD-10 prefix -> phrases that justify it. Phrases are lowercase substrings and
# are matched against the chief complaint and the clinician's visit data.
SYMPTOM_PHRASES: Dict[str, Tuple[str, ...]] = {
    # "substernal" / "precordial" are included because they are specific chest
    # descriptors: a clinician writing "Substernal pressure, radiates to left
    # arm" has documented chest pain without using the words "chest pain". A
    # bare "chest" is deliberately NOT a phrase - "chest exam", "chest x-ray"
    # and "chest clear" are not symptoms.
    "R07": ("chest pain", "chest tightness", "chest discomfort", "chest pressure",
            "substernal", "retrosternal", "precordial", "angina"),
    "R06": ("shortness of breath", "dyspnea", "breathless", "breathlessness",
            "difficulty breathing", "wheezing"),
    "R05": ("cough", "coughing"),
    "R10": ("abdominal pain", "stomach pain", "belly pain", "abdominal cramping",
            "abdominal tenderness"),
    "R19": ("diarrhea", "diarrhoea", "loose stool", "bloody stool"),
    "R50": ("fever", "febrile", "chills"),
    "R51": ("headache", "head ache", "migraine"),
    "R42": ("dizziness", "dizzy", "lightheadedness", "vertigo"),
    "R53": ("weakness", "fatigue", "lethargy", "exhaustion", "tiredness"),
    "R11": ("vomiting", "nausea", "emesis"),
    "R21": ("rash", "skin eruption", "hives"),
    "R31": ("blood in urine", "hematuria", "haematuria"),
    "R45": ("anxiety", "depression", "depressed"),
    # NOTE: "follow-up" is deliberately NOT mapped to Z13. A chronic-disease
    # follow-up is coded with the chronic diagnoses (E11/I10) and the E/M level
    # in CPT; pushing a screening Z-code here would dilute the diagnosis list.
    "Z13": ("screening", "health screening"),
}

# Canonical code + description for each prefix. Keys and prefixes cannot drift:
# `find_symptom_candidates` only emits codes listed here.
SYMPTOM_CODES: Dict[str, str] = {
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
    "Z13.9": "Z13.9 - Encounter for screening, unspecified",
}

# Every prefix the map can produce.
SYMPTOM_PREFIXES: Tuple[str, ...] = tuple(
    sorted({code[:3] for code in SYMPTOM_CODES})
)

# Prefixes that may be *added automatically* (Coding candidates and the
# Supervisor's deterministic completion). Signs and symptoms only. A disease
# code can never enter the result through this module, so no phrase here can
# turn into I10, I20, I21, J45 or any other diagnosis.
#
# The check is a prefix-shape test rather than a table: every canonical code in
# this module is verified to start with R (signs and symptoms) or Z (screening),
# and anything else is dropped by the tests rather than trusted here.
AUTO_ADDABLE_PREFIXES: frozenset = frozenset(
    code[:3]
    for code in SYMPTOM_CODES
    if code[:3].startswith(("R", "Z"))
)

# Phrase -> code for every prefix, longest phrase first so "shortness of breath"
# is reported rather than "breath".
_PREFIX_TO_CODE: Dict[str, str] = {
    prefix: code for code, prefix in
    ((code, code[:3]) for code in SYMPTOM_CODES)
}

# Human label used in the "no <PREFIX> code was generated" gap flag.
_PREFIX_LABELS: Dict[str, str] = {
    "R07": "chest pain",
    "R06": "shortness of breath",
    "R05": "cough",
    "R10": "abdominal pain",
    "R19": "diarrhea",
    "R50": "fever",
    "R51": "headache",
    "R42": "dizziness",
    "R53": "fatigue",
    "R11": "nausea",
    "R21": "rash",
    "R31": "blood in urine",
    "R45": "anxiety",
    "Z13": "screening",
}

# A specific diagnosis for the same problem makes the symptom code redundant.
# Checked against the *chart*, never against the chief complaint, because a
# phrase in the complaint is a hypothesis and a condition on the chart is fact.
# R07 is additionally suppressed by any charted cardiac diagnosis (see
# `_cardiac_diagnosis_charted`), because a charted coronary disease already
# explains the chest pain and coding both invites double-billing.
SUPPRESSED_BY_GROUNDED_DIAGNOSIS: Dict[str, Tuple[str, ...]] = {
    "R07": ("I20", "I21", "I25"),
    "R06": ("I50", "J44", "J45"),
    "R05": ("J30", "J45", "J44"),
    "R10": ("K35", "K80", "K21"),
    "R19": ("A09", "K59"),
    "R51": ("G43", "G44"),
    "R42": ("I77", "H81"),
    "R53": ("D50", "D51"),
    "R45": ("F41", "F32", "F33"),
}

# Chart conditions that mean the same as their ICD prefix for suppression
# purposes. Substring matching, lower-cased.
_DIAGNOSIS_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "I20": ("angina",),
    "I21": ("myocardial infarction", "heart attack", "mi "),
    "I25": ("coronary artery disease", "coronary heart disease", "ischemic heart disease",
            "ischaemic heart disease", "coronary atherosclerosis"),
    "I50": ("heart failure", "chf", "congestive heart failure"),
    "J44": ("copd",),
    "J45": ("asthma",),
    "J30": ("allergic rhinitis", "seasonal allergies", "hay fever"),
    "A09": ("infectious gastroenteritis", "gastroenteritis"),
    "G43": ("migraine",),
    "F41": ("anxiety disorder", "generalized anxiety", "generalised anxiety"),
    "F32": ("major depression", "depressive disorder"),
    "D50": ("iron deficiency anemia", "iron deficiency anaemia"),
}

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")


class SymptomCandidate(NamedTuple):
    """One grounded symptom-code candidate.

    Attributes:
        prefix: ICD-10 prefix, e.g. "R07".
        code: Canonical code, e.g. "R07.9".
        description: "CODE - Description" string for prompts and code lists.
        phrase: The phrase that matched, so the coder can see the justification.
        source: Which text justified it ("chief_complaint" / "visit_context").
    """

    prefix: str
    code: str
    description: str
    phrase: str
    source: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def is_symptom_prefix(prefix: str) -> bool:
    """True when `prefix` is one this module can emit."""
    return str(prefix or "").strip().upper()[:3] in _PREFIX_TO_CODE


def canonical_code_for_prefix(prefix: str) -> Optional[str]:
    """The canonical code for a prefix, or None when the prefix is not a symptom."""
    return _PREFIX_TO_CODE.get(str(prefix or "").strip().upper()[:3])


def describe_prefix(prefix: str) -> Optional[str]:
    """The "CODE - Description" string for a prefix."""
    code = canonical_code_for_prefix(prefix)
    return SYMPTOM_CODES.get(code) if code else None


def symptom_phrase_for_prefix(prefix: str) -> Optional[str]:
    """A representative phrase for a prefix, for flag text and feedback."""
    normalized = str(prefix or "").strip().upper()[:3]
    if normalized not in SYMPTOM_PHRASES:
        return None
    # The longest phrase reads best as a label ("shortness of breath").
    return max(SYMPTOM_PHRASES[normalized], key=len)


def symptom_phrase_present(prefix: str, text: str) -> bool:
    """True when any phrase for `prefix` appears in `text` un-negated.

    This is the check the Supervisor's auto-completion uses, and the check
    that used to miss "dyspnea" because it only ever looked for one
    hard-coded phrase per prefix. Matching is negation-aware: a phrase the
    clinician explicitly denied ("Denies chest pain") is not present.
    """
    normalized = str(prefix or "").strip().upper()[:3]
    phrases = SYMPTOM_PHRASES.get(normalized)
    if not phrases or not text:
        return False
    return any(phrase_present_not_negated(phrase, text) for phrase in phrases)


def symptom_hint_rows() -> List[Tuple[str, str, str]]:
    """Every `(phrase, prefix, label)` row validation needs for gap flags.

    One row per phrase (not one per prefix), so "dyspnea" and "shortness of
    breath" both raise the R06 gap flag, and the label stays the readable
    symptom name rather than whichever phrase happened to be longest.
    """
    return [
        (phrase, prefix, _PREFIX_LABELS.get(prefix, phrase))
        for prefix in sorted(_PREFIX_TO_CODE)
        for phrase in SYMPTOM_PHRASES.get(prefix, ())
    ]


def symptom_gap_labels(text: str) -> List[Tuple[str, str, str]]:
    """`(prefix, label, phrase)` for every symptom prefix present in `text`.

    Validation uses this to raise a gap flag; the label is what a clinician reads,
    the phrase is what the code is justified by. Matching is negation-aware:
    a denied symptom ("Denies chest pain") is not a gap to flag.
    """
    if not text:
        return []
    found: List[Tuple[str, str, str]] = []
    for prefix in _PREFIX_TO_CODE:
        phrases = SYMPTOM_PHRASES.get(prefix, ())
        match = next((phrase for phrase in sorted(phrases, key=len, reverse=True)
                      if phrase_present_not_negated(phrase, text)), None)
        if match:
            found.append((prefix, _PREFIX_LABELS.get(prefix, match), match))
    return found


def _cardiac_diagnosis_charted(patient_context: str) -> bool:
    """True when the chart itself documents a cardiac diagnosis.

    Deliberately scoped to the *chart*: A2.1 established that the words
    "ischemic" in a visit ECG note must not be read as a cardiac diagnosis, so
    this never looks at visit text.
    """
    if not patient_context:
        return False
    # Local import keeps this module free of agent dependencies at import time.
    from utils.cardiac_claims import has_charted_cardiac_diagnosis

    return has_charted_cardiac_diagnosis(patient_context)


def _chart_prefixes(patient_context: str) -> set:
    """Prefixes whose condition is documented in the chart text."""
    text = str(patient_context or "").lower()
    return {
        prefix
        for prefix, keywords in _DIAGNOSIS_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }


def _suppressed(prefix: str, patient_context: str) -> bool:
    """True when a more specific grounded diagnosis covers this symptom."""
    if prefix == "R07" and _cardiac_diagnosis_charted(patient_context):
        return True
    charted = _chart_prefixes(patient_context)
    return any(blocked in charted
               for blocked in SUPPRESSED_BY_GROUNDED_DIAGNOSIS.get(prefix, ()))


def _codes_already_present(codes: Iterable[str]) -> set:
    """ICD prefixes already present in a code list, bare or "CODE - Text"."""
    present = set()
    for entry in codes or ():
        if not entry:
            continue
        match = _CODE_RE.search(str(entry).upper())
        if match:
            present.add(match.group(1)[:3])
    return present


def find_symptom_candidates(
    chief_complaint: str = "",
    visit_context: Any = None,
    patient_context: str = "",
    already_coded: Optional[Sequence[str]] = None,
    max_codes: int = 4,
    allow_disease_prefixes: bool = False,
) -> List[SymptomCandidate]:
    """Ground symptom-code candidates for this encounter.

    Scans the chief complaint first, then the clinician's visit data (HPI, exam,
    documented findings), so a symptom the clinician wrote down is coded even
    when the patient did not open with it.

    Args:
        chief_complaint: Reason for the visit.
        visit_context: Normalized visit context dict, or pre-extracted text.
            Anything with a `visit_context_to_grounding_text`-shaped mapping is
            accepted; a plain string is treated as the visit text.
        patient_context: Chart text, used to suppress a symptom that a more
            specific documented diagnosis already covers.
        already_coded: Codes the note already carries, so a candidate is never
            proposed twice.
        max_codes: Upper bound on returned candidates, highest-signal first.
        allow_disease_prefixes: Off by default. Nothing in this module emits a
            disease code; the parameter exists so a caller that deliberately
            wants diagnosis hints has to say so out loud.

    Returns:
        Grounded `SymptomCandidate`s, chief-complaint matches first. Every code
        is a sign/symptom (R) or screening (Z) code from `SYMPTOM_CODES`.
    """
    candidates: List[SymptomCandidate] = []
    seen_prefixes: set = set()
    coded = _codes_already_present(already_coded or ())

    sources = [("chief_complaint", chief_complaint), ("visit_context", _visit_text(visit_context))]
    for source, text in sources:
        if not text:
            continue
        for prefix, _label, phrase in symptom_gap_labels(text):
            if prefix in seen_prefixes or prefix in coded:
                continue
            if prefix not in AUTO_ADDABLE_PREFIXES and not allow_disease_prefixes:
                continue
            code = canonical_code_for_prefix(prefix)
            if not code or _suppressed(prefix, patient_context):
                continue
            seen_prefixes.add(prefix)
            candidates.append(
                SymptomCandidate(
                    prefix=prefix,
                    code=code,
                    description=SYMPTOM_CODES[code],
                    phrase=phrase,
                    source=source,
                )
            )
            if len(candidates) >= max_codes:
                return candidates
    return candidates


def _visit_text(visit_context: Any) -> str:
    """Flatten a visit context into searchable text, without importing agents."""
    if not visit_context:
        return ""
    if isinstance(visit_context, str):
        return visit_context
    if not isinstance(visit_context, dict):
        return ""
    # Already-normalized visit context: the specialty fields and the plain
    # encounter fields are both worth searching.
    values = []
    for key, value in visit_context.items():
        if key == "specialty_fields" and isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
    return "\n".join(values)
