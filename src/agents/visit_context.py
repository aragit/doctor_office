"""
Visit context - the clinician-provided data for a single encounter.

This module defines the canonical structured contract for visit context.
All downstream nodes (Scribe, Cleanup, Coding, Validation, Supervisor, UI)
depend on this contract. Future Phase A quality work and FHIR mapping build
on this stable shape instead of growing more string parsing.

Two rules shape this module:

1. **Everything is optional.** An empty visit context produces an empty prompt
   block, so a chief-complaint-only visit behaves exactly as it did before.

2. **Clinician-provided data is authoritative grounding**, no different from the
   chart itself. The grounding text is what the cleanup and validation gates
   treat as "known conditions", which stops a condition the clinician documented
   today (e.g. in the exam) from being scrubbed as a hallucination.

Anti-hallucination matters as much as enrichment: a model that invents an
exam finding is worse than one that admits it has none, so
:func:`find_unprovided_vitals` reports numeric vitals in the note that appear
nowhere in the supplied context.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Re-exported so callers do not need to reach into `specialty_profiles` for the
# list of taggable sections.
from agents.specialty_profiles import PROMPT_SECTIONS

# Canonical vitals fields: state key -> (UI label, expected unit).
# `weight_kg` and `bmi` are optional extras; the rest are the core set.
VITALS_FIELDS = (
    ("blood_pressure", "Blood Pressure", "mmHg"),
    ("heart_rate", "Heart Rate", "bpm"),
    ("temperature", "Temperature", "deg F"),
    ("respiratory_rate", "Respiratory Rate", "/min"),
    ("spo2", "SpO2", "%"),
    ("weight_kg", "Weight (kg)", "kg"),
    ("bmi", "BMI", "kg/m2"),
)

# Free-text companion to the structured vitals.
VITALS_NOTE_KEY = "vitals_note"

# Free-text sections, in prompt order.
TEXT_SECTIONS = (
    ("hpi", "History of Present Illness"),
    ("physical_exam", "Physical Examination"),
    ("additional_findings", "Additional Findings / Tests"),
)


class VisitVitals(BaseModel):
    """Structured vital signs for a visit."""
    blood_pressure: Optional[str] = Field(default=None, description="Blood pressure in mmHg, e.g. '148/92'")
    heart_rate: Optional[str] = Field(default=None, description="Heart rate in bpm, e.g. '78'")
    temperature: Optional[str] = Field(default=None, description="Temperature in deg F, e.g. '98.6'")
    respiratory_rate: Optional[str] = Field(default=None, description="Respiratory rate in /min, e.g. '16'")
    spo2: Optional[str] = Field(default=None, description="SpO2 in %, e.g. '98'")
    weight_kg: Optional[str] = Field(default=None, description="Weight in kg, e.g. '72.5'")
    bmi: Optional[str] = Field(default=None, description="BMI in kg/m2, e.g. '26.4'")
    vitals_note: Optional[str] = Field(default=None, description="Free-text vitals observations")

    def model_dump_nonempty(self) -> Dict[str, Any]:
        """Return dict with only non-None/non-empty values, for prompt rendering."""
        result: Dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            result[key] = value
        return result


class VisitContext(BaseModel):
    """
    Canonical structured visit context for a single encounter.

    All fields are optional. An empty VisitContext() produces an empty prompt
    block, preserving chief-complaint-only behaviour.

    Fields:
        chief_complaint: The patient's stated reason for visit.
        hpi: History of Present Illness - patient-reported narrative.
        vitals: Structured vital signs.
        physical_exam: Clinician-observed examination findings.
        additional_findings: Labs, imaging, test results documented today.
        specialty: Medical specialty ID (e.g., "cardiology", "neurology").
        specialty_fields: Specialty-specific extra fields (chest pain characteristics,
            ECG notes, NYHA class, etc.) as declared in specialty_profiles.
    """
    chief_complaint: Optional[str] = Field(default=None, description="Patient's stated reason for visit")
    hpi: Optional[str] = Field(default=None, description="History of Present Illness - patient-reported narrative")
    vitals: Optional[VisitVitals] = Field(default=None, description="Structured vital signs")
    physical_exam: Optional[str] = Field(default=None, description="Physical examination findings")
    additional_findings: Optional[str] = Field(default=None, description="Additional findings / test results")
    specialty: Optional[str] = Field(default=None, description="Medical specialty ID")
    specialty_fields: Dict[str, str] = Field(default_factory=dict, description="Specialty-specific extra fields")

    def model_dump_nonempty(self) -> Dict[str, Any]:
        """Return dict with only non-None/non-empty values, for prompt rendering.

        Uses exclude_none=True to recursively drop None values, then filters
        empty strings and empty dicts.
        """
        # First pass: exclude None values recursively
        dumped = self.model_dump(exclude_none=True)
        result: Dict[str, Any] = {}
        for key, value in dumped.items():
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, dict) and not value:
                continue
            result[key] = value
        return result


def visit_vitals_from_dict(raw: Dict[str, Any]) -> Optional[VisitVitals]:
    """Construct VisitVitals from a flat dict (UI/state shape)."""
    vitals_keys = {k for k, _, _ in VITALS_FIELDS} | {VITALS_NOTE_KEY}
    vitals_data = {k: v for k, v in raw.items() if k in vitals_keys and v}
    if not vitals_data:
        return None
    return VisitVitals(**vitals_data)


def specialty_field_specs() -> Dict[str, Any]:
    """Map every specialty-specific field key to its `ExtraInputField` definition.

    Read from `specialty_profiles.SPECIALTY_PROFILES` rather than hard-coded, so
    a new specialty (or a new field on an existing one) is picked up
    automatically and the Scribe prompt stays configuration-compatible. The
    import is deferred and the failure is swallowed: visit context must keep
    working even if the specialty configuration cannot be loaded.
    """
    specs: Dict[str, Any] = {}
    try:
        from agents.specialty_profiles import SPECIALTY_PROFILES

        for profile in SPECIALTY_PROFILES.values():
            for extra in profile.extra_input_fields:
                specs.setdefault(extra.key, extra)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read specialty field specs: %s", exc)
    return specs


def specialty_field_catalog() -> Dict[str, str]:
    """Map every specialty-specific field key to its human label.

    Derived from `specialty_field_specs`, so this helper keeps its original
    contract (key -> label) while the richer spec stays available for the
    section tagging below.
    """
    return {key: extra.label for key, extra in specialty_field_specs().items()}


def specialty_field_sections() -> Dict[str, str]:
    """Map section-tagged field keys to the SOAP section they belong in.

    Fields without a `prompt_section` are absent, which is how an untagged
    specialty (neurology, dermatology, oncology) keeps its current rendering.
    """
    return {
        key: extra.prompt_section
        for key, extra in specialty_field_specs().items()
        if extra.prompt_section
    }


def _is_tagged_section(section: str) -> bool:
    """True when a declared section is one this pipeline knows how to tag."""
    return section.lower() in PROMPT_SECTIONS

# Cap the injected block so a long HPI cannot dominate the prompt, but leave
# enough room for a full encounter (vitals, exam, labs and HPI) to survive.
# The budget covers data AND rules: rules are never truncated, the data is.
MAX_BLOCK_CHARS = 1600

BLOCK_HEADER = "CLINICAL DATA PROVIDED BY THE CLINICIAN FOR THIS VISIT (authoritative):"
BLOCK_RULES = (
    "RULES FOR THIS BLOCK: use only what is written above. Never invent an "
    "examination finding, vital sign, or lab result that is missing here. "
    "Vitals and exam findings belong in Objective; patient-reported history "
    "belongs in Subjective. If a field was not provided, simply do not "
    "mention it."
)
TAGGED_RULES = (
    "RULES FOR TAGGED SPECIALTY FINDINGS: every [SECTION] item above was "
    "recorded for this visit - put it in that section and do not leave it "
    "out of the note."
)
TRUNCATION_MARKER = "[visit context truncated]"

_UNIT_SUFFIX_RE = re.compile(r"[a-zA-Z°%/]+$")


def _clean(value: Any) -> str:
    """Normalise any UI value to a trimmed single-line-ish string."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    # Collapse newlines so a pasted value cannot break the prompt layout.
    return re.sub(r"\s*\n\s*", " ", text).strip()


def normalize_visit_context(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce a UI/state visit context into the canonical shape.

    Unknown keys are dropped and every value is cleaned, so downstream code can
    rely on the key set and on strings. The exception is specialty-specific
    fields: keys declared by `specialty_profiles` are kept under a separate
    ``specialty_fields`` sub-dict, because Phase 2 lets the clinician record
    specialty data (chest-pain characteristics, ECG notes, NYHA class) that the
    Scribe prompt needs. Arbitrary unknown keys are still discarded.

    Args:
        raw: Partial dict straight from the UI or a caller.

    Returns:
        Dict with only the keys that carry data:
        ``hpi``, ``physical_exam``, ``additional_findings``, ``vitals_note``,
        any of the ``VITALS_FIELDS`` keys, and ``specialty_fields``.
    """
    if not isinstance(raw, dict):
        return {}

    clean: Dict[str, Any] = {}
    for key, _label, _unit in VITALS_FIELDS:
        value = _clean(raw.get(key))
        if value:
            clean[key] = value
    for key in (VITALS_NOTE_KEY,) + tuple(k for k, _label in TEXT_SECTIONS):
        value = _clean(raw.get(key))
        if value:
            clean[key] = value

    specialty_fields = normalize_specialty_fields(raw)
    if specialty_fields:
        clean["specialty_fields"] = specialty_fields
    return clean


def normalize_specialty_fields(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Pull the config-declared specialty fields out of a raw visit context.

    Accepts both shapes so that normalising is idempotent: specialty data
    arriving flat from the UI (``{"ecg_rhythm_notes": "..."}``) and an already
    normalised context (``{"specialty_fields": {"ecg_rhythm_notes": "..."}}``)
    produce the same result.

    Only keys present in `specialty_field_catalog()` survive, so a stray field
    name is never treated as clinical data.
    """
    if not isinstance(raw, dict):
        return {}

    catalog = specialty_field_catalog()
    if not catalog:
        return {}

    # Flat UI shape plus any nested block from a previous normalisation pass.
    sources: List[Dict[str, Any]] = [raw]
    nested = raw.get("specialty_fields")
    if isinstance(nested, dict):
        sources.append(nested)

    collected: Dict[str, str] = {}
    for source in sources:
        for key in catalog:
            value = _clean(source.get(key))
            if value:
                collected[key] = value
    return collected


def get_specialty_fields(visit_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Read the specialty fields back out of an already-normalised context."""
    if not isinstance(visit_context, dict):
        return {}
    fields = visit_context.get("specialty_fields")
    return fields if isinstance(fields, dict) else {}


def has_visit_context(visit_context: Optional[Dict[str, Any]]) -> bool:
    """True when the clinician supplied anything beyond the chief complaint."""
    return bool(normalize_visit_context(visit_context))


def format_vitals(vitals: Dict[str, Any]) -> str:
    """Render the structured vitals as a compact one-line summary."""
    parts: List[str] = []
    for key, label, unit in VITALS_FIELDS:
        value = _clean(vitals.get(key))
        if not value:
            continue
        # Don't double up a unit the clinician already typed ("98.6 F").
        if not _UNIT_SUFFIX_RE.search(value):
            separator = "" if unit == "%" else " "
            value = f"{value}{separator}{unit}"
        parts.append(f"{label}: {value}")

    note = _clean(vitals.get(VITALS_NOTE_KEY))
    if note:
        parts.append(f"Vitals note: {note}")
    return "; ".join(parts)


def specialty_fields_text(visit_context: Optional[Dict[str, Any]]) -> str:
    """Render the specialty-specific findings as plain labelled lines.

    Used to widen the grounding set (a condition the clinician documents in a
    specialty field today must not be scrubbed as a hallucination) and by the
    Physician Review summary.
    """
    fields = get_specialty_fields(visit_context)
    if not fields:
        return ""

    catalog = specialty_field_catalog()
    lines: List[str] = []
    for key, value in fields.items():
        label = catalog.get(key, key.replace("_", " ").title())
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def visit_context_text(visit_context: Optional[Dict[str, Any]]) -> str:
    """Plain text of everything the clinician provided today.

    Used to widen the grounding set for the cleanup and validation gates.
    """
    clean = normalize_visit_context(visit_context)
    if not clean:
        return ""

    blocks: List[str] = []
    for key, label in TEXT_SECTIONS:
        value = _clean(clean.get(key))
        if value:
            blocks.append(f"{label}: {value}")

    specialty_text = specialty_fields_text(clean)
    if specialty_text:
        blocks.insert(0, f"Specialty-Specific Findings:\n{specialty_text}")

    vitals = format_vitals(clean)
    if vitals:
        blocks.insert(0, f"Vitals: {vitals}")

    return "\n".join(blocks)


def build_visit_context_block(
    visit_context: Optional[Dict[str, Any]],
    char_limit: int = MAX_BLOCK_CHARS,
) -> str:
    """Render the visit context as a labelled, explicitly-grounded prompt block.

    Specialty fields that declare a `prompt_section` are tagged inline
    (`[SUBJECTIVE] Chest Pain Characteristics: ...`) so the Scribe knows which
    section the documented finding belongs in, instead of having to infer it.
    Untagged specialties render exactly as before.

    Returns an empty string when nothing was provided, which keeps the
    chief-complaint-only prompt byte-for-byte what it was before Phase 1.
    """
    clean = normalize_visit_context(visit_context)
    if not clean:
        return ""

    data_lines: List[str] = [BLOCK_HEADER]

    vitals = format_vitals(clean)
    if vitals:
        data_lines.append(f"- Vitals: {vitals}")

    specialty_fields = get_specialty_fields(clean)
    tagged_keys: List[str] = []
    if specialty_fields:
        catalog = specialty_field_catalog()
        sections = specialty_field_sections()
        tagged_keys = [
            key for key in specialty_fields
            if _is_tagged_section(sections.get(key, ""))
        ]
        header = "- Specialty-specific findings (documented today)"
        if tagged_keys:
            data_lines.append(f"{header} - [SECTION] marks the SOAP section each belongs in:")
        else:
            data_lines.append(f"{header}:")
        for key, value in specialty_fields.items():
            label = catalog.get(key, key.replace("_", " ").title())
            section = sections.get(key, "")
            prefix = f"[{section.upper()}] " if _is_tagged_section(section) else ""
            data_lines.append(f"  * {prefix}{label}: {value}")

    for key, label in TEXT_SECTIONS:
        value = _clean(clean.get(key))
        if value:
            data_lines.append(f"- {label}: {value}")

    # The rules govern everything above them, so they get first claim on the
    # budget: a long HPI loses data, never the instruction that constrains it.
    rules = [BLOCK_RULES]
    if tagged_keys:
        rules.append(TAGGED_RULES)
    rules_text = "\n".join(rules)

    data_text = "\n".join(data_lines)
    data_budget = char_limit - len(rules_text) - 1
    if data_budget > 0 and len(data_text) > data_budget:
        clipped = data_text[:data_budget].rsplit("\n", 1)[0].rstrip()
        data_text = f"{clipped}\n{TRUNCATION_MARKER}" if clipped else data_text[:data_budget]
    return f"{data_text}\n{rules_text}"


# --- Anti-hallucination: numeric vitals in the note that were never supplied ---

# Each entry: (label, pattern, aliases that must appear in the provided text
# for a number to count as supplied). The aliases stop an unrelated number that
# happens to match from being treated as evidence that a vital was documented.
_VITAL_PATTERNS = (
    ("blood pressure", re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b"), ("blood pressure", "bp", "mmhg")),
    ("heart rate", re.compile(r"\b(?:hr|heart\s+rate|pulse)\s*(?:is|of|[:=])?\s*(\d{2,3})\b", re.I),
     ("heart rate", "hr", "pulse", "bpm")),
    ("heart rate", re.compile(r"\b(\d{2,3})\s*(?:bpm|beats)\b", re.I), ("heart rate", "hr", "pulse", "bpm")),
    ("temperature", re.compile(r"\b(?:temp|temperature|t)\s*(?:is|of|[:=])?\s*(\d{2,3}(?:\.\d)?)\b", re.I),
     ("temperature", "temp", "deg f", "°f")),
    ("temperature", re.compile(r"\b(\d{2,3}(?:\.\d)?)\s*(?:°\s*)?(?:deg(?:rees)?\s*)?[FC]\b"), ("temperature", "temp", "deg f", "°f")),
    ("SpO2", re.compile(r"\b(?:spo2|sao2|o2\s?sat\w*)\s*(?:is|of|[:=])?\s*(\d{2,3})\s*%?", re.I),
     ("spo2", "sao2", "o2 sat", "o2sat", "%")),
    ("respiratory rate", re.compile(r"\b(?:rr|respiratory\s+rate|resp)\s*(?:is|of|[:=])?\s*(\d{1,2})\b", re.I),
     ("respiratory rate", "rr", "/min")),
)


def _normalise_number(token: str) -> str:
    """Normalise ``"78.0"``/``"078"`` to ``"78"`` for comparison."""
    token = token.strip().rstrip(".")
    if "." in token:
        token = token.split(".")[0]
    try:
        return str(int(float(token)))
    except ValueError:
        return token


def find_unprovided_vitals(text: str, provided_text: str) -> List[str]:
    """Report numeric vitals in *text* that do not appear in *provided_text*.

    Deliberately narrow: it only fires when the note states a specific
    measurable value (e.g. "BP 160/100") that appears nowhere in the supplied
    context. It says nothing about qualitative findings, so it stays quiet on
    ordinary prose.

    Args:
        text: Generated note text (typically the Objective section).
        provided_text: Everything the clinician supplied - the visit context
            plus the patient chart, so values legitimately repeated from the
            chart are not reported.

    Returns:
        Human-readable findings, e.g. ``["blood pressure 160/100"]``.
    """
    if not text or not provided_text:
        return []

    provided = provided_text.lower()
    provided_numbers = {
        _normalise_number(t) for t in re.findall(r"\d+(?:\.\d+)?", provided)
    }
    provided_pairs = {
        f"{_normalise_number(a)}/{_normalise_number(b)}"
        for a, b in re.findall(r"(\d{2,3})\s*/\s*(\d{2,3})", provided)
    }

    findings: List[str] = []
    for label, pattern, aliases in _VITAL_PATTERNS:
        label_documented = any(alias in provided for alias in aliases)
        for match in pattern.finditer(text):
            groups = match.groups()
            if label == "blood pressure":
                value = f"{_normalise_number(groups[0])}/{_normalise_number(groups[1])}"
                known = value in provided_pairs or (
                    label_documented
                    and _normalise_number(groups[0]) in provided_numbers
                    and _normalise_number(groups[1]) in provided_numbers
                )
            else:
                value = _normalise_number(groups[0])
                known = label_documented and value in provided_numbers
            if not known and f"{label.lower()} {value}" not in findings:
                findings.append(f"{label.lower()} {value}")
    return findings


__all__ = [
    "BLOCK_HEADER",
    "BLOCK_RULES",
    "MAX_BLOCK_CHARS",
    "PROMPT_SECTIONS",
    "TAGGED_RULES",
    "TEXT_SECTIONS",
    "TRUNCATION_MARKER",
    "VITALS_FIELDS",
    "VITALS_NOTE_KEY",
    "VisitContext",
    "VisitVitals",
    "build_visit_context_block",
    "find_unprovided_vitals",
    "format_vitals",
    "get_specialty_fields",
    "has_visit_context",
    "normalize_specialty_fields",
    "normalize_visit_context",
    "specialty_field_catalog",
    "specialty_field_sections",
    "specialty_field_specs",
    "specialty_fields_text",
    "visit_context_text",
    "visit_context_to_grounding_text",
    "visit_context_to_prompt_block",
    "visit_vitals_from_dict",
]


# --- Explicit helper aliases for the single rendering / prompt contract ---
# These are the canonical names downstream nodes should use.

def visit_context_to_prompt_block(
    visit_context: Optional[Dict[str, Any]],
    char_limit: int = MAX_BLOCK_CHARS,
) -> str:
    """
    Render the visit context as a labelled, explicitly-grounded prompt block.

    This is the SINGLE prompt rendering function for the visit context.
    All nodes (Scribe, Coding) must use this instead of ad-hoc string building.

    Args:
        visit_context: Normalized or raw visit context dict.
        char_limit: Maximum characters for the rendered block (default 1200).

    Returns:
        Formatted prompt block, or empty string if no visit context provided.
    """
    return build_visit_context_block(visit_context, char_limit)


def visit_context_to_grounding_text(visit_context: Optional[Dict[str, Any]]) -> str:
    """
    Render the visit context as plain text for grounding/validation gates.

    This is the SINGLE grounding text function for the visit context.
    Cleanup and Validation nodes must use this to widen the grounding set.

    Args:
        visit_context: Normalized or raw visit context dict.

    Returns:
        Plain text of all clinician-provided data, or empty string if none.
    """
    return visit_context_text(visit_context)