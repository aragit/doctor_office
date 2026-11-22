"""
Patient Memory Repository Interface - abstraction for long-term patient storage.

This module defines a repository protocol so the JSON file-backed store can
later be swapped for Postgres (or any other backend) without changing the
memory nodes or any agent logic.

The interface mirrors the current `PatientMemoryStore` public API, so
`JsonPatientMemoryRepository` is a drop-in replacement, extended with the
draft/approved split: `record_visit` writes a draft by default and
`approve_visit` is the only operation that creates approved history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class PatientMemoryRepository(Protocol):
    """Protocol for patient memory persistence.

    Implementations must be thread-safe for concurrent Streamlit reruns
    and support atomic writes so a crash cannot corrupt stored data.

    Implementations must also honour the draft/approved split: any read used as
    patient history defaults to approved visits, and only `approve_visit` may
    produce an approved row.
    """

    def list_patient_ids(self) -> List[str]:
        """Return all known patient IDs, sorted."""
        ...

    def get_summary(self, patient_id: str) -> Dict[str, Any]:
        """Return the stored summary for a patient (empty dict if unknown).

        Includes `visit_count` (approved visits on record), plus
        `approved_visit_count` and `draft_visit_count`.
        """
        ...

    def get_visit_history(
        self, patient_id: str, limit: int = 3, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return the most recent visits for a patient, newest first.

        `status=None` returns both draft and approved rows (audit view);
        pass a status to filter.
        """
        ...

    def get_approved_visit_history(
        self, patient_id: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the most recent physician-approved visits, newest first."""
        ...

    def get_longitudinal_facts(
        self, patient_id: str, limit: int = 3, status: str = "approved"
    ) -> List[str]:
        """Human-readable fact strings from recent visits, newest first.

        These are injected into the Scribe prompt as *historical context*.
        They must never be treated as new diagnoses on their own, and they
        default to approved visits only.
        """
        ...

    def build_context_block(
        self, patient_id: str, visit_limit: int = 2, char_limit: int = 600
    ) -> str:
        """Render prior-visit context as a compact prompt block.

        Approved visits only. Returns an empty string when there is no approved
        history, so the caller can skip the section entirely.
        """
        ...

    def ensure_patient(self, patient_id: str, patient_context: str) -> Dict[str, Any]:
        """Create or refresh a patient record from the current chart.

        Conditions/medications/allergies are replaced with the *current* chart
        values; `*_ever` history lists only ever grow.

        Returns:
            A copy of the stored record, safe for the caller to mutate.
        """
        ...

    def record_visit(
        self,
        patient_id: str,
        patient_context: str,
        chief_complaint: str = "",
        coding_result: Optional[Dict[str, Any]] = None,
        soap_note: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
        status: str = "draft",
    ) -> Dict[str, Any]:
        """Persist one visit for a patient.

        Defaults to a **draft**: pipeline output is recorded for audit but is not
        history until a physician approves it.

        Args:
            patient_id: Stable patient identifier.
            patient_context: The chart context used for this visit.
            chief_complaint: Reason for the visit.
            coding_result: Coding Agent output dict.
            soap_note: Cleaned SOAP note dict (stored truncated).
            quality_flags: Flags raised during the visit.
            status: "draft" (default) or "approved".

        Returns:
            The updated patient record.
        """
        ...

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
        """Mark a recorded visit physician-approved - the only path to history.

        Promotes the addressed draft row in place, or writes an approved row when
        no draft remains to promote. Idempotent.

        Returns:
            The updated patient record.
        """
        ...

    def save(self) -> None:
        """Atomically write the store to disk."""
        ...

    def clear_cache(self) -> None:
        """Drop the in-process cache (used by tests and after external edits)."""
        ...

    def reset(self) -> None:
        """Delete all stored patients. Intended for tests and demos."""
        ...


# Re-export the JSON implementation and factory
from memory.patient_memory import (  # noqa: E402
    DEFAULT_MEMORY_PATH,
    MAX_VISITS_PER_PATIENT,
    VISIT_STATUS_APPROVED,
    VISIT_STATUS_DRAFT,
    VISIT_STATUSES,
    JsonPatientMemoryRepository,
    get_memory_repo,
    resolve_patient_id,
    visit_status,
)

__all__ = [
    "DEFAULT_MEMORY_PATH",
    "MAX_VISITS_PER_PATIENT",
    "VISIT_STATUS_APPROVED",
    "VISIT_STATUS_DRAFT",
    "VISIT_STATUSES",
    "PatientMemoryRepository",
    "JsonPatientMemoryRepository",
    "get_memory_repo",
    "resolve_patient_id",
    "visit_status",
]