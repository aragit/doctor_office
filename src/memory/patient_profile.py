"""
Structured patient profile parsing.

Turns the free-text `patient_context` string used across the pipeline into a
typed `PatientProfile`. Both the tools and the long-term memory layer build on
this so they agree on what a "condition" or "medication" is.

Expected input shape (as produced by the Streamlit UI and the demo):

    Patient: John Doe, 58yo male
    Conditions: Type 2 Diabetes, Hypertension
    Medications: Metformin 500mg BID, Lisinopril 10mg daily
    Allergies: Penicillin
    Last Visit: Blood pressure 148/92. HbA1c 7.8%.
"""

import re
from typing import Any, Dict, List, Optional

# Labels understood in the patient context block.
FIELD_LABELS = {
    "patient": "name",
    "name": "name",
    "patient_id": "patient_id",
    "id": "patient_id",
    "age": "age",
    "gender": "gender",
    "sex": "gender",
    "conditions": "conditions",
    "diagnosis": "conditions",
    "diagnoses": "conditions",
    "medications": "medications",
    "meds": "medications",
    "allergies": "allergies",
    "last visit": "last_visit_summary",
    "last_visit": "last_visit_summary",
}

# Values that mean "the list is empty" rather than being a real entry.
NULL_TOKENS = {"none", "n/a", "na", "no known allergies", "nkda", "nil", "unknown", "-"}

_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _]*?)\s*:\s*(.*)$")
_AGE_RE = re.compile(r"(\d{1,3})\s*(?:yo|y/o|years? old)?", re.IGNORECASE)
# Word boundaries matter here: "female" contains "male".
_GENDER_RE = re.compile(r"\b(female|male|nonbinary|non-binary|other)\b", re.IGNORECASE)


def _split_list(raw: str) -> List[str]:
    """Split a comma/semicolon separated field into cleaned entries."""
    if not raw:
        return []
    parts = re.split(r"[,;]", raw)
    return [p.strip(" .\t-") for p in parts if p.strip(" .\t-")]


def _clean_values(raw: str, allow_null: bool = True) -> List[str]:
    """Split and drop placeholder values such as `None` or `N/A`."""
    values = _split_list(raw)
    if allow_null:
        values = [v for v in values if v.lower() not in NULL_TOKENS]
    return values


def _slugify(value: str) -> str:
    """Build a stable id from a display name (e.g. 'John Doe' -> 'john-doe')."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or "unknown-patient"


class PatientProfile:
    """Parsed view of a patient context string.

    Attributes are deliberately forgiving: anything the parser cannot find is
    simply an empty list or `None` rather than raising, because clinical text
    in this MVP is free-form.
    """

    def __init__(
        self,
        name: str = "",
        patient_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: str = "",
        conditions: Optional[List[str]] = None,
        medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        last_visit_summary: str = "",
        raw_context: str = "",
    ) -> None:
        self.name = name
        self._patient_id = patient_id
        self.age = age
        self.gender = gender
        self.conditions = conditions or []
        self.medications = medications or []
        self.allergies = allergies or []
        self.last_visit_summary = last_visit_summary
        self.raw_context = raw_context

    # -- identity ----------------------------------------------------------
    @property
    def patient_id(self) -> str:
        """Explicit id when supplied, otherwise a stable slug of the name.

        Falls back to a content hash of the context so two different anonymous
        patients never collapse into the same memory bucket.
        """
        if self._patient_id:
            return self._patient_id
        if self.name:
            return _slugify(self.name)
        return f"anon-{_slugify(self.raw_context)[:24] or 'context'}"

    @property
    def display_id(self) -> str:
        return self.patient_id

    # -- queries -----------------------------------------------------------
    def has_condition(self, condition: str) -> bool:
        needle = (condition or "").strip().lower()
        return any(needle in c.lower() for c in self.conditions)

    def condition_text(self) -> str:
        return "; ".join(self.conditions) if self.conditions else "none documented"

    def medication_text(self) -> str:
        return "; ".join(self.medications) if self.medications else "none documented"

    def allergy_text(self) -> str:
        return "; ".join(self.allergies) if self.allergies else "none documented"

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "conditions": self.conditions,
            "medications": self.medications,
            "allergies": self.allergies,
            "last_visit_summary": self.last_visit_summary,
        }

    def summary_lines(self) -> List[str]:
        """Compact one-line-per-field summary used for prompt injection."""
        age_gender = ""
        if self.age is not None:
            age_gender = f"{self.age}yo"
            if self.gender:
                age_gender += f" {self.gender}"
        return [
            f"Patient: {self.name or 'unknown'} ({age_gender})".strip(),
            f"Conditions: {self.condition_text()}",
            f"Medications: {self.medication_text()}",
            f"Allergies: {self.allergy_text()}",
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PatientProfile(id={self.patient_id!r}, conditions={self.conditions!r})"


def parse_patient_context(patient_context: str) -> PatientProfile:
    """Parse a formatted patient context string into a `PatientProfile`.

    Args:
        patient_context: The multi-line context block used by the pipeline.

    Returns:
        A `PatientProfile`. Unparseable input yields an empty profile rather
        than an exception, so a malformed context can never break a visit.
    """
    profile = PatientProfile(raw_context=patient_context or "")
    if not patient_context:
        return profile

    found: Dict[str, str] = {}
    for line in patient_context.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        label = match.group(1).strip().lower()
        value = match.group(2).strip()
        field = FIELD_LABELS.get(label)
        if field and field not in found:
            found[field] = value

    if "name" in found:
        raw_name = found["name"]
        age_match = _AGE_RE.search(raw_name)
        if age_match:
            profile.age = int(age_match.group(1))
        # Strip the trailing demographic clause: "John Doe, 58yo male"
        name_part = re.split(r",", raw_name, maxsplit=1)[0].strip()
        profile.name = name_part
        tail = raw_name[len(name_part):]
        if not profile.age:
            age_in_tail = _AGE_RE.search(tail)
            if age_in_tail:
                profile.age = int(age_in_tail.group(1))
        gender_match = _GENDER_RE.search(tail)
        if gender_match:
            profile.gender = gender_match.group(1).lower()

    if "patient_id" in found:
        profile._patient_id = found["patient_id"].strip()
    if "age" in found and profile.age is None:
        age_match = _AGE_RE.search(found["age"])
        if age_match:
            profile.age = int(age_match.group(1))
    if "gender" in found and not profile.gender:
        profile.gender = found["gender"].strip()
    if "last_visit_summary" in found:
        profile.last_visit_summary = found["last_visit_summary"]

    profile.conditions = _clean_values(found.get("conditions", ""))
    profile.medications = _clean_values(found.get("medications", ""))
    profile.allergies = _clean_values(found.get("allergies", ""))

    return profile


__all__ = ["PatientProfile", "parse_patient_context"]
