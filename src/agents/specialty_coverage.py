"""
Specialty coverage guard - a deterministic backstop for documented specialty data.

The Scribe prompt is what makes a specialty visit *use* the fields the clinician
filled in, but a small model can silently drop them. A real Cardiology run
documented chest-pain characteristics, an ECG and an NYHA class, and the note
came back with a one-line Subjective ("chest pain x 1 day") and an Objective that
recorded vitals only. The physician has to catch that manually.

This module is the deterministic counterpart to that risk, in the same spirit as
`visit_context.find_unprovided_vitals`:

* It only ever looks at fields the clinician actually supplied, and skips fields
  they explicitly declined ("Not assessed").
* As of Phase A2.1 the check is **section-scoped**: a documented field must be
  reflected in the section it is tagged for - chest-pain descriptors in
  SUBJECTIVE, ECG findings in OBJECTIVE, functional class in ASSESSMENT. Putting
  everything in the Assessment no longer satisfies the check, because that is
  how the fidelity loss showed up in practice.
* Reflection means either the section's clinical vocabulary appears (the
  ``SECTION_EVIDENCE`` patterns) or enough of the clinician's own words from the
  supplied value survive (a 5-character-prefix token overlap, so "radiates" and
  "radiating" match). It never checks that values were reproduced faithfully.
* It also reports a cardiac claim that survived `utils.validation.clean_soap_note`,
  which is the safety-critical case: a diagnosis the chart does not support.

Every flag is advisory, so a visit with one finalises and is presented for review
rather than looping the graph.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.cardiac_claims import (
    build_cardiac_claim_flag,
    has_charted_cardiac_diagnosis,
)

logger = logging.getLogger(__name__)

# Evidence that a documented finding made it into the section, per SOAP section
# tag. Deliberately narrow, and *excluding* the bare symptom name: naming "chest
# pain" is not the same as carrying the documented chest-pain descriptors, and
# treating it as use is exactly how a thin Subjective slipped through in Phase A2.
# These match the vocabulary of the field itself, so this asks "does this section
# talk like the documented finding", not a fidelity check on the values.
SECTION_EVIDENCE: Dict[str, Tuple[re.Pattern, ...]] = {
    "subjective": (
        re.compile(
            r"\b(chest pressure|chest tightness|chest discomfort|substernal"
            r"|retrosternal|precordial|exertional|exertion|radiat\w*"
            r"|precipitat\w*|relieved|alleviat\w*|th pressure)",
            re.I,
        ),
    ),
    "objective": (
        re.compile(
            r"\b(ecg|ekg|electrocardiogram|rhythm|sinus|afib|a-fib"
            r"|st-t|st segment|st-segment|t-wave|t wave|qrs|pr interval|qtc)",
            re.I,
        ),
    ),
    "assessment": (
        re.compile(
            r"(nyha|functional class|functional status|cardiac function"
            r"|\bclass (i{1,3}|iv)\b)",
            re.I,
        ),
    ),
}

# Sections where the field's own vocabulary is enough to count as use, because a
# clinician who wrote an ECG note has to use ECG words to write it. The SUBJECTIVE
# is excluded on purpose: the weak markers there ("exertional", "chest pressure")
# are exactly the fragments a thin Subjective keeps when it drops every recorded
# descriptor, so Subjective always has to reproduce the clinician's own words.
_VOCABULARY_IS_ENOUGH = frozenset({"objective", "assessment"})

NOTE_SECTIONS: Tuple[str, ...] = ("subjective", "objective", "assessment", "plan")

# A clinician can explicitly record "nothing to report" - a select dropdown's
# "Not assessed" default, or a blank-ish free-text answer. There is nothing for
# the note to reflect in that case, so the field must not raise a flag.
#
# Matched on the WHOLE value only, never as a prefix: "no acute ischemic
# changes" is a perfectly good ECG finding, while "no" on its own is a decline.
NO_DATA_VALUES: Tuple[str, ...] = (
    "not assessed", "none", "n/a", "na", "-", "unknown", "not applicable",
    "not done", "not documented", "no", "nil", "not relevant",
)
# These are unambiguous even when the clinician explains themselves afterwards,
# e.g. "Not assessed (patient asymptomatic)".
NO_DATA_PREFIXES: Tuple[str, ...] = (
    "not assessed", "not applicable", "not done", "not documented", "not relevant",
)
_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")

COVERAGE_FLAG_TEMPLATE = (
    "Specialty data documented this visit is not reflected in the {section_title} "
    "section: {label} - carry the documented findings into that section"
)

# Words that carry no clinical content, dropped before token comparison.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "has",
    "have", "had", "not", "but", "you", "are", "his", "her", "its", "per",
    "she", "him", "day", "days", "ago", "now", "also", "any", "all", "can",
    "will", "would", "there", "their", "than", "when", "what", "which", "who",
    "how", "out", "off", "about", "into", "over", "one", "two", "some",
})
_TOKEN_PREFIX = 5

# Token overlap below this many stems counts as "not used". The share scales with
# the length of the recorded value, but is capped so a clinician who dictates
# fifteen descriptors does not make fidelity impossible, and a very short value
# ("Radiates to left arm") is not held to an impossible bar - see
# `_overlap_threshold`.
_MIN_OVERLAP = 4
_OVERLAP_SHARE = 0.5
_MAX_OVERLAP = 6
# A value this short is reproduced nearly in full rather than by share.
_SHORT_VALUE_TERMS = 4


def _token_keys(text: str) -> Set[str]:
    """Distinctive word stems, compared on a prefix so plurals and simple
    inflections ("radiates" / "radiating") still match."""
    words = re.findall(r"[a-z][a-z0-9-]{2,}", str(text or "").lower())
    return {
        word[:_TOKEN_PREFIX]
        for word in words
        if word not in _STOPWORDS and len(word) >= 3
    }


def _overlap_threshold(supplied_terms: Set[str]) -> int:
    """How many of the recorded stems a section must reproduce."""
    count = len(supplied_terms)
    if not count:
        return _MIN_OVERLAP
    if count <= _SHORT_VALUE_TERMS:
        # "Class II - Slight limitation" has three stems; demanding all three
        # would punish a note that says the same thing in other words.
        return max(1, count - 1)
    return min(max(_MIN_OVERLAP, round(_OVERLAP_SHARE * count)), _MAX_OVERLAP)


def _has_recorded_value(value: Any) -> bool:
    """True unless the clinician explicitly recorded nothing to report."""
    text = str(value or "").strip()
    # Drop a trailing parenthetical so "Not assessed (asymptomatic)" still counts
    # as a decline rather than as recorded data.
    text = _TRAILING_PARENTHETICAL.sub("", text).strip().rstrip(".").lower()
    if not text or text in NO_DATA_VALUES:
        return False
    return not text.startswith(NO_DATA_PREFIXES)


def _supplied_fields(visit_context: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Field keys the clinician actually filled in, with their labels."""
    if not visit_context or not isinstance(visit_context, dict):
        return []

    from agents.visit_context import (
        get_specialty_fields,
        specialty_field_catalog,
        specialty_field_sections,
    )

    supplied = get_specialty_fields(visit_context)
    if not supplied:
        return []

    catalog = specialty_field_catalog()
    sections = specialty_field_sections()
    return [
        (key, catalog.get(key, key.replace("_", " ").title()))
        for key, value in supplied.items()
        if str(sections.get(key, "")).lower() in SECTION_EVIDENCE
        and _has_recorded_value(value)
    ]


def _missing_fields(
    note: Any,
    visit_context: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """`(key, label, section)` for every supplied field the note dropped."""
    fields = _supplied_fields(visit_context)
    if not fields:
        return []

    sections_text = _section_text(note)
    values = _supplied_values(visit_context)

    missing: List[Tuple[str, str, str]] = []
    for key, label in fields:
        section = _section_for(key)
        section_text = sections_text.get(section, "")
        if not section_text.strip():
            missing.append((key, label, section))
            continue
        # Section vocabulary counts as use ("ECG: normal sinus rhythm"), except in
        # the Subjective, which must carry the documented descriptors themselves.
        if (section in _VOCABULARY_IS_ENOUGH
                and any(pattern.search(section_text)
                        for pattern in SECTION_EVIDENCE.get(section, ()))):
            continue
        # Otherwise the clinician's own words must survive.
        terms = _token_keys(values.get(key, ""))
        overlap = len(terms & _token_keys(section_text))
        if terms and overlap >= _overlap_threshold(terms):
            continue
        missing.append((key, label, section))
    return missing


def find_unreflected_specialty_fields(
    note: Any,
    visit_context: Optional[Dict[str, Any]],
) -> List[str]:
    """Labels of documented specialty fields missing from their tagged section.

    Args:
        note: Generated SOAP note (dict or pydantic model).
        visit_context: The visit context the note was generated from.

    Returns:
        Human-readable field labels, e.g. ``["ECG / Rhythm Notes"]``. Empty when
        every documented field is reflected in the section it is tagged for, or
        when no tagged field was supplied.
    """
    return [label for _key, label, _section in _missing_fields(note, visit_context)]


def build_specialty_coverage_flag(
    note: Any,
    visit_context: Optional[Dict[str, Any]],
    patient_context: str = "",
) -> List[str]:
    """Advisory flags for documented specialty data the note did not carry.

    Args:
        note: Generated SOAP note (dict or pydantic model).
        visit_context: The visit context the note was generated from.
        patient_context: The chart text, used for the cardiac-claim backstop.
            Empty means "not evaluated" and the claim flag is skipped.

    Returns:
        One flag per unreflected field, plus one if an unsupported cardiac claim
        survived cleanup. Advisory by design: the visit finalises for physician
        review instead of retrying, so a note that ignored a field cannot start
        a repair loop.
    """
    flags: List[str] = [
        COVERAGE_FLAG_TEMPLATE.format(
            section_title=section.upper() or "NOTE",
            label=label,
        )
        for _key, label, section in _missing_fields(note, visit_context)
    ]

    if patient_context:
        # Per section, not one joined blob: the claim detector suppresses a match
        # when a negation sits nearby ("no acute ischemic changes"), and joining
        # sections would let an Objective negation excuse an Assessment claim.
        has_diagnosis = has_charted_cardiac_diagnosis(patient_context)
        for section_text in _section_text(note).values():
            flags.extend(build_cardiac_claim_flag(section_text, has_diagnosis=has_diagnosis))

    for flag in flags:
        logger.info("specialty coverage guard: %s", flag)
    return flags


def _section_for(key: str) -> str:
    """The SOAP section a field is tagged for ("" when untagged)."""
    from agents.visit_context import specialty_field_sections

    return str(specialty_field_sections().get(key, "")).lower()


def _supplied_values(visit_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """The clinician's own text for each supplied specialty field."""
    if not visit_context or not isinstance(visit_context, dict):
        return {}
    from agents.visit_context import get_specialty_fields

    return get_specialty_fields(visit_context)


def _section_text(note: Any) -> Dict[str, str]:
    """Flatten a SOAP note into per-section text."""
    if note is None:
        return {}
    if hasattr(note, "model_dump"):
        note = note.model_dump()
    if not isinstance(note, dict):
        return {"": str(note)}
    return {
        section: str(note.get(section, "") or "")
        for section in NOTE_SECTIONS
        if section in note
    }


__all__ = [
    "COVERAGE_FLAG_TEMPLATE",
    "NO_DATA_VALUES",
    "NOTE_SECTIONS",
    "SECTION_EVIDENCE",
    "build_specialty_coverage_flag",
    "find_unreflected_specialty_fields",
]