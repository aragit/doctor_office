"""
Presentation helpers for the physician review surface.

Pure functions, no Streamlit import: the UI renders what this module decides,
and the tests assert the decisions without a browser. It exists so that
"what the physician is told" is unit-testable and so the same wording is used
in the Coding tab and in Physician Review.

Two responsibilities:

1. **Symptom-coding provenance.** `symptom_coding_rows` turns the pipeline's
   symptom metadata into one row per candidate code with an explicit origin:
   already present, added by the Coding node, or added by the Supervisor
   fallback. The physician can see *where a code came from*, which is the point
   of A4 - a code appearing in the list with no explanation is not reviewable.

2. **Flag grouping.** `categorize_quality_flags` sorts the raw quality-flag
   strings into three readable buckets with short human labels:

   - `auto_fixed` - something was completed automatically (a symptom code added
     by Coding or by the Supervisor). Informational: nothing was lost.
   - `review_advised` - something is incomplete or under-reflected and only the
     physician can judge it (a dropped specialty field, a symptom that was not
     coded, an incomplete note).
   - `safety_scrub` - something was *removed* because it was unsafe (a cardiac
     claim the chart does not support, a hallucinated diagnosis, an ungrounded
     or malformed code).

   Raw flag text is never dropped: every returned row carries it in `detail`,
   and the Raw JSON tab still shows the untouched list. Nothing here changes
   what the pipeline produced - it only reads the metadata the pipeline already
   recorded.
"""

import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

__all__ = [
    "FLAG_CATEGORIES",
    "ReviewFlag",
    "SymptomCodingRow",
    "SymptomCodingSummary",
    "category_meta",
    "categorize_quality_flags",
    "flag_groups",
    "origin_label",
    "source_label",
    "specialty_fidelity_notes",
    "symptom_coding_rows",
    "symptom_coding_summary",
]


class ReviewFlag(NamedTuple):
    """One quality flag, grouped for display.

    Attributes:
        category: One of the keys in `FLAG_CATEGORIES`.
        label: Short human sentence, e.g. "Removed a cardiac claim the chart
            does not support".
        detail: The raw flag string, preserved verbatim for debugging.
    """

    category: str
    label: str
    detail: str


class SymptomCodingRow(NamedTuple):
    """One symptom-code candidate and what became of it.

    Attributes:
        code: Canonical code, e.g. "R07.9".
        description: "CODE - Description" as shown to the physician.
        phrase: The documented text that justified the candidate.
        source: Where that text came from ("chief_complaint" / "visit_context").
        origin: "already_present" | "added_by_coding" | "added_by_supervisor"
            | "considered_only".
        present_in_result: Whether the code is in the final result.
    """

    code: str
    description: str
    phrase: str
    source: str
    origin: str
    present_in_result: bool


class SymptomCodingSummary(NamedTuple):
    """Aggregated symptom-coding outcome for the UI.

    Attributes:
        rows: One `SymptomCodingRow` per candidate, considered order preserved.
        considered: How many candidates the map produced.
        already_present: How many were already coded by the Scribe or Coding.
        coding_added: How many the Coding node completed deterministically.
        supervisor_added: How many the Supervisor fallback completed.
        missing: How many candidates are not in the final result.
    """

    rows: List[SymptomCodingRow]
    considered: int
    already_present: int
    coding_added: int
    supervisor_added: int
    missing: int


# ---------------------------------------------------------------------------
# Flag categories
# ---------------------------------------------------------------------------

# Ordered so the physician reads the informational bucket first and the safety
# removals last, when both are present.
FLAG_CATEGORIES: Tuple[Dict[str, str], ...] = (
    {
        "key": "auto_fixed",
        "title": "Completed automatically",
        "caption": "Added for you - nothing was lost, but check the code list.",
        "severity": "info",
    },
    {
        "key": "review_advised",
        "title": "Review advised",
        "caption": "Incomplete or under-reflected. Only you can judge these.",
        "severity": "warning",
    },
    {
        "key": "safety_scrub",
        "title": "Removed by a safety check",
        "caption": "Taken out of the note or the code list. See the detail.",
        "severity": "error",
    },
)

_CATEGORY_BY_KEY: Dict[str, Dict[str, str]] = {
    entry["key"]: entry for entry in FLAG_CATEGORIES
}


def category_meta(key: str) -> Dict[str, str]:
    """Presentation metadata for a category key, with a safe default."""
    return _CATEGORY_BY_KEY.get(
        key,
        {"key": key, "title": key, "caption": "", "severity": "warning"},
    )


# --- flag matchers ----------------------------------------------------------
# Ordered most specific first: the first pattern that matches decides the
# bucket, so an ambiguous flag can never be double-counted.

_FIDELITY_RE = re.compile(
    r"Specialty data documented this visit is not reflected in the "
    r"(?P<section>[A-Z]+) section: (?P<field>[^-]+)",
    re.IGNORECASE,
)
_SYMPTOM_GAP_RE = re.compile(
    r"Chief complaint mentions '(?P<label>[^']+)' but no (?P<prefix>[A-Z]\d{2}) code was generated",
    re.IGNORECASE,
)
_CARDIAC_CLAIM_RE = re.compile(
    r"Cardiac claim not supported by the chart: (?P<claims>[^(\-]+)",
    re.IGNORECASE,
)
_SCRIBER_INCOMPLETE_RE = re.compile(
    r"Scribe content incomplete", re.IGNORECASE
)

# (pattern, category, label template). `{claims}`-style groups are formatted in.
_FLAG_RULES: Tuple[Tuple[re.Pattern, str, str], ...] = (
    # --- auto-completed symptom codes -------------------------------------
    (
        re.compile(
            r"Coding: added symptom code (?P<code>[A-Z]\d{2}(?:\.\d+)?)\b[^(]*"
            r"\((?P<phrase>[^)]*)\)",
            re.IGNORECASE,
        ),
        "auto_fixed",
        "Added symptom code {code} - {phrase}",
    ),
    (
        re.compile(r"Added symptom code (?P<code>[A-Z]\d{2}(?:\.\d+)?)", re.IGNORECASE),
        "auto_fixed",
        "Added symptom code {code} (validation fallback)",
    ),
    (
        re.compile(r"Added missing symptom code", re.IGNORECASE),
        "auto_fixed",
        "Added the missing symptom code",
    ),
    # --- safety scrubs ----------------------------------------------------
    (
        _CARDIAC_CLAIM_RE,
        "safety_scrub",
        "Cardiac wording not supported by the chart: {claims}",
    ),
    (
        re.compile(r"removed unsupported cardiac claim", re.IGNORECASE),
        "safety_scrub",
        "Removed a cardiac diagnosis from the medical necessity text",
    ),
    # Cleanup phrases this either way round: "Removed unsupported cardiac claim
    # from assessment" / "Removed plan line for unsupported cardiac claim".
    (
        re.compile(r"unsupported cardiac claim", re.IGNORECASE),
        "safety_scrub",
        "Removed a cardiac diagnosis the chart does not support",
    ),
    (
        re.compile(r"Possible hallucinated diagnosis", re.IGNORECASE),
        "safety_scrub",
        "Removed a diagnosis that is not in the chart",
    ),
    (
        re.compile(r"Sanitized assessment", re.IGNORECASE),
        "safety_scrub",
        "Removed a term that is not in the chart",
    ),
    (
        re.compile(r"dropped malformed ICD-10 code", re.IGNORECASE),
        "safety_scrub",
        "Dropped a malformed ICD-10 code",
    ),
    (
        re.compile(r"Ungrounded ICD-10 code", re.IGNORECASE),
        "safety_scrub",
        "Flagged a code the chart does not support",
    ),
    # --- review advised ---------------------------------------------------
    (
        _FIDELITY_RE,
        "review_advised",
        "{field} not reflected in the {section} section",
    ),
    (
        _SYMPTOM_GAP_RE,
        "review_advised",
        "Symptom '{label}' has no {prefix} code - decide whether to add one",
    ),
    (
        _SCRIBER_INCOMPLETE_RE,
        "review_advised",
        "Note was incomplete and some fields were defaulted",
    ),
    # Catch-all for any cleanup removal, including shapes added later: a
    # "Cleanup: Removed ..." flag is by definition something taken out, so it
    # belongs with the safety bucket rather than the review bucket.
    (
        re.compile(r"^Cleanup: Removed", re.IGNORECASE),
        "safety_scrub",
        "Removed from the note by the grounding check",
    ),
    (
        re.compile(r"Max retries reached", re.IGNORECASE),
        "review_advised",
        "Retries were exhausted and this visit needs human review",
    ),
    (
        re.compile(
            r"CPT .*(?:new|established) patients|consider 992|verify patient status",
            re.IGNORECASE,
        ),
        "review_advised",
        "CPT level may not match the patient's established/new status",
    ),
)


def _label_for(flag: str) -> Tuple[str, str]:
    """Classify one raw flag into `(category, human label)`."""
    for pattern, category, template in _FLAG_RULES:
        match = pattern.search(flag)
        if not match:
            continue
        fields = {
            key: (value or "").strip()
            for key, value in match.groupdict().items()
        }
        if "section" in fields:
            fields["section"] = fields["section"].title()
        label = template
        for key, value in fields.items():
            label = label.replace("{" + key + "}", value)
        return category, label
    # Unknown flags stay visible in the review bucket rather than disappearing.
    return "review_advised", _shorten(flag)


def _shorten(flag: str, limit: int = 110) -> str:
    """Collapse a long internal flag to a readable first sentence."""
    text = str(flag or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def categorize_quality_flags(flags: Optional[Iterable[str]]) -> List[ReviewFlag]:
    """Group raw quality flags into the three display buckets.

    Args:
        flags: The pipeline's raw `quality_flags` list.

    Returns:
        One `ReviewFlag` per input flag, each carrying its category, a short
        human label and the untouched raw string.
    """
    grouped: List[ReviewFlag] = []
    for flag in flags or []:
        if not flag:
            continue
        category, label = _label_for(str(flag))
        grouped.append(ReviewFlag(category, label, str(flag)))
    return grouped


def flag_groups(
    flags: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    """Buckets ready to render, in `FLAG_CATEGORIES` order.

    Returns:
        One dict per category that has at least one entry, each with the
        category metadata plus its `flags`. Empty categories are omitted so the
        UI shows no empty headings.
    """
    grouped = categorize_quality_flags(flags)
    buckets: Dict[str, List[ReviewFlag]] = {key: [] for key in _CATEGORY_BY_KEY}
    for item in grouped:
        buckets.setdefault(item.category, []).append(item)

    sections: List[Dict[str, Any]] = []
    for entry in FLAG_CATEGORIES:
        items = buckets.get(entry["key"]) or []
        if not items:
            continue
        sections.append({**entry, "flags": items})
    return sections


def specialty_fidelity_notes(
    flags: Optional[Iterable[str]],
) -> List[str]:
    """One plain line per under-reflected specialty field.

    Returns:
        Lines like "ECG / Rhythm Notes - not reflected in the Objective
        section", empty when no fidelity flag fired.
    """
    notes: List[str] = []
    for flag in flags or []:
        match = _FIDELITY_RE.search(str(flag or ""))
        if not match:
            continue
        field = (match.group("field") or "").strip()
        section = (match.group("section") or "").strip().title()
        notes.append(f"{field} - not reflected in the {section} section")
    return notes


# ---------------------------------------------------------------------------
# Symptom-coding provenance
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")

_SOURCE_LABELS = {
    "chief_complaint": "chief complaint",
    "visit_context": "visit data (HPI / exam / results)",
}

_ORIGIN_LABELS = {
    "already_present": "Already coded",
    "added_by_coding": "Added by Coding",
    "added_by_supervisor": "Added by Supervisor",
    "considered_only": "Not in the final result",
}


def _codes_in(coding_result: Optional[Dict[str, Any]]) -> List[str]:
    entries = list((coding_result or {}).get("secondary_icd10") or [])
    primary = (coding_result or {}).get("primary_icd10")
    if primary:
        entries = [primary] + entries
    codes: List[str] = []
    for entry in entries:
        match = _CODE_RE.search(str(entry).upper())
        if match:
            codes.append(match.group(1))
    return codes


def _codes_from_text(values: Optional[Iterable[str]]) -> List[str]:
    found: List[str] = []
    for value in values or []:
        match = _CODE_RE.search(str(value).upper())
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def symptom_coding_rows(
    metadata: Optional[Dict[str, Any]],
    coding_result: Optional[Dict[str, Any]] = None,
) -> List[SymptomCodingRow]:
    """One row per symptom-code candidate, with its origin.

    Reads `metadata["symptom_coding"]` when present, and falls back to the tool
    trace and Supervisor fixes when an older result predates that block, so the
    panel never silently goes empty.

    Args:
        metadata: The pipeline result's `metadata` dict.
        coding_result: The result's `coding_result`, used to decide whether a
            candidate actually made it into the final code list.

    Returns:
        Rows in candidate order.
    """
    metadata = metadata or {}
    block = dict(metadata.get("symptom_coding") or {})
    if not block:
        block = _symptom_block_from_legacy(metadata)
    if not block:
        return []

    candidates = list(block.get("candidates") or [])
    coding_added = _codes_from_text(block.get("coding_added"))
    supervisor_added = _codes_from_text(block.get("supervisor_added"))
    final_codes = _codes_in(coding_result)

    rows: List[SymptomCodingRow] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = {"code": _CODE_RE.search(candidate.upper()).group(1)
                         if _CODE_RE.search(candidate.upper()) else candidate,
                         "description": candidate}
        code = str(candidate.get("code") or "").upper()
        if not code:
            continue
        present = code in final_codes
        if code in coding_added:
            origin = "added_by_coding"
        elif code in supervisor_added:
            origin = "added_by_supervisor"
        elif present:
            origin = "already_present"
        else:
            origin = "considered_only"
        rows.append(
            SymptomCodingRow(
                code=code,
                description=str(candidate.get("description") or candidate.get("code") or code),
                phrase=str(candidate.get("phrase") or ""),
                source=str(candidate.get("source") or ""),
                origin=origin,
                present_in_result=present,
            )
        )
    return rows


def _symptom_block_from_legacy(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the symptom block for a result that predates it.

    Keeps the panel working on stored results and on any caller that assembles a
    result dict by hand (tests, scripts) without the newer metadata.
    """
    tool_results = (metadata.get("tools") or {}).get("tool_results") or {}
    trace = (tool_results.get("coding_node") or {}).get(
        "simple_symptom_to_icd_suggestion"
    ) or {}
    supervisor_trace = (tool_results.get("supervisor") or {}).get(
        "simple_symptom_to_icd_suggestion"
    ) or {}
    candidates = trace.get("candidates") or []
    if not candidates:
        candidates = [
            {"code": code} for code in trace.get("suggested_codes") or []
        ]
    coding_added = trace.get("added_to_result") or []
    supervisor_added = supervisor_trace.get("added") or []
    if not candidates and not coding_added and not supervisor_added:
        return {}
    return {
        "candidates": candidates,
        "coding_added": list(coding_added),
        "supervisor_added": list(supervisor_added),
    }


def symptom_coding_summary(
    metadata: Optional[Dict[str, Any]],
    coding_result: Optional[Dict[str, Any]] = None,
) -> SymptomCodingSummary:
    """Aggregate counts plus the rows, for the section header."""
    rows = symptom_coding_rows(metadata, coding_result)
    return SymptomCodingSummary(
        rows=rows,
        considered=len(rows),
        already_present=sum(1 for row in rows if row.origin == "already_present"),
        coding_added=sum(1 for row in rows if row.origin == "added_by_coding"),
        supervisor_added=sum(1 for row in rows if row.origin == "added_by_supervisor"),
        missing=sum(1 for row in rows if not row.present_in_result),
    )


def source_label(source: str) -> str:
    """Readable label for a candidate's source."""
    return _SOURCE_LABELS.get(source, source.replace("_", " ") or "visit")


def origin_label(origin: str) -> str:
    """Readable label for a candidate's origin."""
    return _ORIGIN_LABELS.get(origin, origin.replace("_", " "))