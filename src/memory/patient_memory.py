"""
Long-term patient memory (lightweight, file-backed).

Scope for this stage - deliberately small and reliable:

* One JSON file per project (`data/memory/patients.json`) holding a summary
  record and a bounded visit history per patient.
* An in-process cache with a write-through file, so repeated runs are cheap.
* Thread-safe writes (Streamlit reruns share a process) and atomic file
  replacement so a crash cannot corrupt the store.

**Draft vs approved.** Every visit row carries a status:

* `draft` - pipeline output. The visit exists in the store for audit and
  continuity, but no physician has signed it.
* `approved` - a physician pressed Approve in Physician Review. Only these rows
  are longitudinal history: only they are injected into the Scribe prompt, and
  only they are counted in `visit_count`.

The pipeline records a **draft**; `approve_visit` is the only path that produces
an approved row. Rejecting leaves the draft in place, unapproved and unread by
the Scribe - there is no un-store operation, by design.

Legacy rows written before this distinction existed carry no status and are
treated as **approved**: they predate the split, they were the only write path,
and downgrading them would silently empty every existing patient's context.
`visit_status()` is the single place that rule lives.

This is NOT a clinical record system. It stores only what the pipeline
produces plus the facts already present in the patient context, and every
consumer treats it as *advisory* context that the grounding rules must still
respect.
"""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.patient_profile import PatientProfile, parse_patient_context

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory" / "patients.json"

# Keep history bounded - this is a lightweight summary store, not an EHR.
MAX_VISITS_PER_PATIENT = 10
SCHEMA_VERSION = 1

# Visit statuses. Only these two exist; anything else on a stored row is read
# through `visit_status`, which normalises legacy rows to approved.
VISIT_STATUS_DRAFT = "draft"
VISIT_STATUS_APPROVED = "approved"
VISIT_STATUSES = (VISIT_STATUS_DRAFT, VISIT_STATUS_APPROVED)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def visit_status(visit: Optional[Dict[str, Any]]) -> str:
    """Status of a stored visit row.

    A row with no `status` predates the draft/approved split and is reported as
    approved, so existing stores keep working (see the module docstring).
    Unrecognised values are also read as approved rather than silently hiding a
    row from history: showing a visit as unsigned when the store is ambiguous is
    the safer error for a *history* read, and the UI shows the raw row anyway.

    Args:
        visit: One entry from a patient's `visits` list.

    Returns:
        `VISIT_STATUS_DRAFT` or `VISIT_STATUS_APPROVED`.
    """
    raw = str((visit or {}).get("status") or "").strip().lower()
    return raw if raw in VISIT_STATUSES else VISIT_STATUS_APPROVED


def _is_approved(visit: Optional[Dict[str, Any]]) -> bool:
    return visit_status(visit) == VISIT_STATUS_APPROVED


def _has_visit_content(
    chief_complaint: str,
    coding_result: Optional[Dict[str, Any]] = None,
    soap_note: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether there is a visit here worth recording at all.

    Used only by the approve fallback path: writing an approved row with no
    complaint, no codes and no assessment would put a meaningless visit into a
    patient's history, so an approval with nothing to promote and nothing to say
    is refused instead.
    """
    if str(chief_complaint or "").strip():
        return True
    coding = coding_result or {}
    if str(coding.get("primary_icd10") or "").strip():
        return True
    if list(coding.get("secondary_icd10") or []):
        return True
    if list(coding.get("cpt_suggestions") or []):
        return True
    note = soap_note or {}
    return any(str(note.get(field) or "").strip() for field in ("assessment", "plan"))


def _merge_unique(existing: List[str], incoming: List[str], limit: int = 50) -> List[str]:
    """Case-insensitive union that preserves order and caps length."""
    seen = {item.strip().lower() for item in existing if item and item.strip()}
    merged = list(existing or [])
    for item in incoming or []:
        if item and item.strip() and item.strip().lower() not in seen:
            seen.add(item.strip().lower())
            merged.append(item.strip())
    return merged[-limit:]


class JsonPatientMemoryRepository:
    """File-backed store of patient summaries and visit history (JSON implementation).

    This is the default repository implementation. It implements the
    `PatientMemoryRepository` protocol and can be swapped for a Postgres
    backend without changing agent code.

    Usage:
        repo = JsonPatientMemoryRepository()
        summary = repo.get_summary("P001")
        # Pipeline output is a draft, never history:
        repo.record_visit("P001", patient_context=ctx, coding_result=coding)
        # Only a physician Approve promotes it into longitudinal history:
        repo.approve_visit("P001", visit_id="visit-1")
    """

    def __init__(self, path: Optional[Path] = None, autosave: bool = True) -> None:
        self.path = Path(path) if path else DEFAULT_MEMORY_PATH
        self.autosave = autosave
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Any]] = None

    # -- persistence -------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        """Return the whole store, reading from disk once per process."""
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            self._cache = {"schema_version": SCHEMA_VERSION, "patients": {}}
            return self._cache

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Patient memory unreadable at %s (%s); starting empty", self.path, exc)
            data = {}

        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("patients", {})
        self._cache = data
        return data

    def save(self) -> None:
        """Atomically write the store to disk."""
        with self._lock:
            data = self._load()
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp file in the same directory, then replace, so a
            # crash mid-write cannot leave a truncated JSON file behind.
            fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, default=str)
                os.replace(tmp_path, self.path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise

    def clear_cache(self) -> None:
        """Drop the in-process cache (used by tests and after external edits)."""
        with self._lock:
            self._cache = None

    def reset(self) -> None:
        """Delete all stored patients. Intended for tests and demos."""
        with self._lock:
            self._cache = {"schema_version": SCHEMA_VERSION, "patients": {}}
            self.save()

    # -- reads -------------------------------------------------------------
    def list_patient_ids(self) -> List[str]:
        return sorted(self._load()["patients"].keys())

    def get_summary(self, patient_id: str) -> Dict[str, Any]:
        """Return the stored summary for a patient (empty dict if unknown)."""
        with self._lock:
            record = self._load()["patients"].get(patient_id)
            if not record:
                return {}
            # Deep-ish copy so callers cannot mutate the cache by accident.
            summary = json.loads(json.dumps(record))
            # Counters are derived from the rows on read, not only on write, so a
            # store written before the draft/approved split still reports honest
            # approved/draft totals instead of reading as zero approved visits.
            self._refresh_counts(summary)
            return summary

    def get_visit_history(
        self,
        patient_id: str,
        limit: int = 3,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent visits for a patient, newest first.

        Args:
            patient_id: Stable patient identifier.
            limit: Maximum rows to return.
            status: `VISIT_STATUS_DRAFT` or `VISIT_STATUS_APPROVED` to filter;
                None (default) returns both, which is what an audit view wants.
                Anything the Scribe reads must pass a status explicitly.
        """
        with self._lock:
            record = self._load()["patients"].get(patient_id)
            if not record:
                return []
            visits = list(record.get("visits", []))
            if status:
                visits = [v for v in visits if visit_status(v) == status]
            visits.reverse()
            return visits[:limit]

    def get_approved_visit_history(
        self, patient_id: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Most recent physician-approved visits, newest first."""
        return self.get_visit_history(patient_id, limit=limit, status=VISIT_STATUS_APPROVED)

    def get_longitudinal_facts(
        self,
        patient_id: str,
        limit: int = 3,
        status: str = VISIT_STATUS_APPROVED,
    ) -> List[str]:
        """Human-readable fact strings from recent visits, newest first.

        These are injected into the Scribe prompt as *historical context*.
        They must never be treated as new diagnoses on their own.

        Defaults to approved visits only: unsigned pipeline output must not be
        presented to the model as this patient's signed history.
        """
        facts: List[str] = []
        for visit in self.get_visit_history(patient_id, limit=limit, status=status):
            complaint = visit.get("chief_complaint", "").strip()
            if not complaint:
                continue
            codes = ", ".join(visit.get("icd10", []) or []) or "no codes recorded"
            when = visit.get("timestamp", "")[:10]
            facts.append(f"[{when}] Visit: {complaint} | Codes: {codes}")
        return facts

    def build_context_block(self, patient_id: str, visit_limit: int = 2, char_limit: int = 600) -> str:
        """Render prior-visit context as a compact prompt block.

        Only physician-approved visits contribute. A pipeline draft is invisible
        here by construction: it has not been signed, so it is not history yet.

        Returns an empty string when there is no approved history, so the caller
        can skip the section entirely.
        """
        summary = self.get_summary(patient_id)
        if not summary:
            return ""

        parts: List[str] = []
        prior_conditions = summary.get("history", {}).get("conditions_ever", [])
        if prior_conditions:
            # Anything in conditions_ever that is NOT in the current chart is
            # explicitly excluded - it is history, not a current diagnosis.
            current = {c.lower() for c in summary.get("current", {}).get("conditions", [])}
            historical_only = [c for c in prior_conditions if c.lower() not in current]
            if historical_only:
                parts.append(
                    "Previously recorded (historical only, NOT current diagnoses): "
                    + "; ".join(historical_only[:8])
                )

        longitudinal = self.get_longitudinal_facts(
            patient_id, limit=visit_limit, status=VISIT_STATUS_APPROVED
        )
        if longitudinal:
            parts.append("Approved past visits:")
            parts.extend(f"  - {fact}" for fact in longitudinal)

        if not parts:
            return ""

        block = (
            "PATIENT HISTORY (physician-approved visits only, reference only):\n"
            + "\n".join(parts)
        )
        return block[:char_limit]
    # -- writes ------------------------------------------------------------
    def _ensure_patient_live(self, patient_id: str, patient_context: str) -> Dict[str, Any]:
        """Create/refresh a patient and return the LIVE cache reference.

        Internal: callers must treat the returned dict as borrowed and either
        mutate it immediately (followed by `save()`) or deep-copy it first.
        Use `ensure_patient` for a safe copy, or `get_summary` to read.
        """
        profile = parse_patient_context(patient_context)
        store = self._load()
        record = store["patients"].get(patient_id)

        if record is None:
            record = {
                "patient_id": patient_id,
                "name": profile.name,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "current": {},
                "history": {},
                "visits": [],
            }
            store["patients"][patient_id] = record

        record["name"] = profile.name or record.get("name", "")
        record["updated_at"] = _now_iso()
        record["current"] = {
            "age": profile.age,
            "gender": profile.gender,
            "conditions": profile.conditions,
            "medications": profile.medications,
            "allergies": profile.allergies,
            "last_visit_summary": profile.last_visit_summary,
        }

        history = record.setdefault("history", {})
        history["conditions_ever"] = _merge_unique(
            history.get("conditions_ever", []), profile.conditions
        )
        history["medications_ever"] = _merge_unique(
            history.get("medications_ever", []), profile.medications
        )
        history["allergies_ever"] = _merge_unique(
            history.get("allergies_ever", []), profile.allergies
        )
        record.setdefault("visit_count", 0)
        record.setdefault("visits", [])
        return record

    def ensure_patient(self, patient_id: str, patient_context: str) -> Dict[str, Any]:
        """Create or refresh a patient record from the current chart.

        Conditions/medications/allergies are replaced with the *current* chart
        values; `*_ever` history lists only ever grow.

        Returns:
            A copy of the stored record, safe for the caller to mutate.
        """
        with self._lock:
            record = self._ensure_patient_live(patient_id, patient_context)
            if self.autosave:
                self.save()
            return json.loads(json.dumps(record))

    def record_visit(
        self,
        patient_id: str,
        patient_context: str,
        chief_complaint: str = "",
        coding_result: Optional[Dict[str, Any]] = None,
        soap_note: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
        status: str = VISIT_STATUS_DRAFT,
    ) -> Dict[str, Any]:
        """Persist one visit for a patient.

        Defaults to `VISIT_STATUS_DRAFT`, so the safe status is also the default:
        producing approved history requires an explicit, deliberate argument.
        The pipeline's `memory_save_node` relies on that - it records a draft and
        nothing else. `approve_visit` is what turns one into history.

        Args:
            patient_id: Stable patient identifier.
            patient_context: The chart context used for this visit.
            chief_complaint: Reason for the visit.
            coding_result: Coding Agent output dict.
            soap_note: Cleaned SOAP note dict (stored truncated).
            quality_flags: Flags raised during the visit.
            status: `VISIT_STATUS_DRAFT` or `VISIT_STATUS_APPROVED`.

        Returns:
            The updated patient record.
        """
        if status not in VISIT_STATUSES:
            raise ValueError(
                f"unknown visit status {status!r}; expected one of {VISIT_STATUSES}"
            )

        coding_result = coding_result or {}
        with self._lock:
            # Mutate the live cache record, then persist - mutating the copy
            # returned by ensure_patient() would silently drop the visit.
            record = self._ensure_patient_live(patient_id, patient_context)

            visit = self._build_visit(
                chief_complaint=chief_complaint,
                coding_result=coding_result,
                soap_note=soap_note,
                quality_flags=quality_flags,
                status=status,
            )

            record["visits"] = self._append_visit(record, visit)
            self._refresh_counts(record)

            if self.autosave:
                self.save()
            return json.loads(json.dumps(record))

    def approve_visit(
        self,
        patient_id: str,
        visit_id: Optional[str] = None,
        patient_context: str = "",
        chief_complaint: str = "",
        coding_result: Optional[Dict[str, Any]] = None,
        soap_note: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Mark a recorded visit physician-approved, in place when possible.

        This is the only path that creates approved longitudinal history, and it is
        idempotent: approving a row that is already approved changes nothing and
        writes no second row, so a double Approve cannot inflate the history. When
        there is no draft to promote - an escalated visit, a run with memory off, or
        a draft that was trimmed out of the rolling window - the supplied visit
        content is written as an approved row instead, and if no content was supplied
        the call raises rather than inventing an empty visit.

        Args:
            patient_id: Stable patient identifier.
            visit_id: The `id` of the draft row to promote, when the caller knows
                it (the pipeline echoes it back in result metadata).
            patient_context: Chart text, used only when a new row has to be
                written because the draft is gone.
            chief_complaint: Reason for the visit - same, fallback only.
            coding_result: Coding output - fallback only.
            soap_note: Cleaned SOAP note - fallback only.
            quality_flags: Visit flags - fallback only.

        Returns:
            The updated patient record.

        Raises:
            KeyError: The patient is unknown and no chart was supplied.
            ValueError: Nothing to promote and no content to write.
        """
        with self._lock:
            record = self._load()["patients"].get(patient_id)
            if not record:
                if not patient_context:
                    raise KeyError(f"unknown patient {patient_id!r} and no chart to create one")
                # Nothing on record: the only honest way to approve is to write
                # the visit, and to say so through the row itself.
                return self.record_visit(
                    patient_id=patient_id,
                    patient_context=patient_context,
                    chief_complaint=chief_complaint,
                    coding_result=coding_result,
                    soap_note=soap_note,
                    quality_flags=quality_flags,
                    status=VISIT_STATUS_APPROVED,
                )

            target = self._find_approvable(record, visit_id)
            if target is not None:
                visit = record["visits"][target]
                if not _is_approved(visit):
                    visit["status"] = VISIT_STATUS_APPROVED
                    visit["approved_at"] = _now_iso()
                    record["updated_at"] = _now_iso()
                    logger.info(
                        "approve_visit: promoted %s visit %s to approved",
                        patient_id, visit.get("id") or "(legacy row)",
                    )
                # Already approved: nothing to change.
            elif visit_id and any(
                v.get("id") == visit_id for v in (record.get("visits") or [])
            ):
                # The addressed row is already approved. Approving it again is a
                # double click, and must not append an empty second visit.
                logger.info(
                    "approve_visit: %s visit %s was already approved; no write",
                    patient_id, visit_id,
                )
            else:
                # The draft was trimmed away, memory was off during the run, or
                # the visit was escalated before memory_save_node. Write the
                # approved visit directly rather than approving a different one -
                # but only if there is anything to write. Inventing an empty
                # approved row would put a visit with no complaint and no codes
                # into a patient's history, which is worse than refusing.
                if not _has_visit_content(chief_complaint, coding_result, soap_note):
                    raise ValueError(
                        f"approve_visit: {patient_id} has no draft visit to approve "
                        f"(visit_id={visit_id!r}) and no visit content was supplied"
                    )
                visit = self._build_visit(
                    chief_complaint=chief_complaint,
                    coding_result=coding_result or {},
                    soap_note=soap_note or {},
                    quality_flags=quality_flags,
                    status=VISIT_STATUS_APPROVED,
                )
                record["visits"] = self._append_visit(record, visit)
                logger.info(
                    "approve_visit: %s had no draft row to promote; wrote a new approved visit",
                    patient_id,
                )

            self._refresh_counts(record)
            if self.autosave:
                self.save()
            return json.loads(json.dumps(record))

    # -- visit-row helpers -------------------------------------------------
    def _build_visit(
        self,
        chief_complaint: str,
        coding_result: Dict[str, Any],
        soap_note: Dict[str, Any],
        quality_flags: Optional[List[str]],
        status: str,
    ) -> Dict[str, Any]:
        """Assemble one visit row. Internal."""
        icd10 = [coding_result.get("primary_icd10", "")] + list(
            coding_result.get("secondary_icd10", []) or []
        )
        icd10 = [code for code in icd10 if code]
        visit = {
            "timestamp": _now_iso(),
            "status": status,
            "chief_complaint": (chief_complaint or "").strip(),
            "icd10": icd10,
            "cpt": list(coding_result.get("cpt_suggestions", []) or []),
            "assessment": str(soap_note.get("assessment", ""))[:400],
            "plan": str(soap_note.get("plan", ""))[:400],
            "quality_flags": list(quality_flags or []),
        }
        if status == VISIT_STATUS_APPROVED:
            visit["approved_at"] = visit["timestamp"]
        return visit

    def _append_visit(self, record: Dict[str, Any], visit: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append a row, stamp its id, and trim to the window. Internal.

        The trim drops the oldest **draft** first, so unsigned work can never
        push a signed visit out of the store. Only when drafts alone exceed the
        window does the oldest row of any status go.
        """
        seq = int(record.get("visit_seq", 0) or 0) + 1
        record["visit_seq"] = seq
        visit["id"] = f"visit-{seq}"

        visits = list(record.get("visits", []))
        visits.append(visit)
        while len(visits) > MAX_VISITS_PER_PATIENT:
            victim = next(
                (i for i, v in enumerate(visits) if not _is_approved(v)),
                0,  # nothing unsigned left: fall back to the oldest row
            )
            visits.pop(victim)
        return visits

    def _find_approvable(self, record: Dict[str, Any], visit_id: Optional[str]) -> Optional[int]:
        """Index of the row an Approve may promote, or None. Internal.

        An explicit `visit_id` is honoured only when that row is still a draft:
        approving a row that is already approved, or promoting a row id we do
        not recognise, must never silently sign a *different* visit.
        """
        visits = list(record.get("visits", []))
        if visit_id:
            for index, visit in enumerate(visits):
                if visit.get("id") == visit_id and not _is_approved(visit):
                    return index
            return None
        for index in range(len(visits) - 1, -1, -1):
            if not _is_approved(visits[index]):
                return index
        return None

    def _refresh_counts(self, record: Dict[str, Any]) -> None:
        """Recompute the approved/draft counters. Internal.

        `visit_count` means *approved visits on record*, which is what every
        consumer shows ("N visit(s) on record"). It never decreases: a legacy
        store may hold a count higher than the rows still inside the bounded
        window, and losing that number on the first write would be a regression.
        """
        visits = list(record.get("visits", []))
        approved = sum(1 for visit in visits if _is_approved(visit))
        record["visit_count"] = max(int(record.get("visit_count", 0) or 0), approved)
        record["approved_visit_count"] = approved
        record["draft_visit_count"] = len(visits) - approved
        approved_visits = [v for v in visits if _is_approved(v)]
        if approved_visits:
            last = approved_visits[-1]
            record["last_visit"] = {
                "timestamp": last.get("timestamp", ""),
                "chief_complaint": last.get("chief_complaint", ""),
            }
        elif approved == 0:
            # No approved history: clear a stale pointer rather than leave one
            # that points at an unsigned visit.
            record["last_visit"] = {}


_DEFAULT_REPO: Optional[JsonPatientMemoryRepository] = None
_DEFAULT_REPO_LOCK = threading.Lock()


def get_memory_repo() -> JsonPatientMemoryRepository:
    """Return the process-wide memory repository singleton (JSON backend)."""
    global _DEFAULT_REPO
    if _DEFAULT_REPO is None:
        with _DEFAULT_REPO_LOCK:
            if _DEFAULT_REPO is None:
                _DEFAULT_REPO = JsonPatientMemoryRepository()
    return _DEFAULT_REPO


# Backward-compatibility aliases
PatientMemoryStore = JsonPatientMemoryRepository
get_memory_store = get_memory_repo


def resolve_patient_id(patient_id: Optional[str], patient_context: str) -> str:
    """Pick the best available patient id.

    An explicit `patient_id` always wins; otherwise fall back to the id
    derived from the parsed chart.
    """
    if patient_id:
        return str(patient_id).strip()
    profile: PatientProfile = parse_patient_context(patient_context)
    return profile.patient_id


__all__ = [
    "DEFAULT_MEMORY_PATH",
    "MAX_VISITS_PER_PATIENT",
    "VISIT_STATUS_APPROVED",
    "VISIT_STATUS_DRAFT",
    "VISIT_STATUSES",
    "JsonPatientMemoryRepository",
    "PatientMemoryStore",  # backward compatibility
    "get_memory_repo",
    "get_memory_store",  # backward compatibility
    "resolve_patient_id",
    "visit_status",
]
