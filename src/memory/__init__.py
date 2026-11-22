"""Memory layer for the Doctor Office MVP.

- `patient_profile` - structured parsing of the free-text patient context.
- `patient_memory` - lightweight, file-backed long-term patient memory, where
  every visit row is either an unsigned `draft` (what a pipeline run writes) or
  `approved` history (what only a physician's Approve creates).

The two status constants and `visit_status` are re-exported here because "is
this row history?" is a question callers outside the store have to be able to
ask without reaching into a raw dict.
"""

from memory.patient_profile import PatientProfile, parse_patient_context
from memory.patient_memory import (
    DEFAULT_MEMORY_PATH,
    MAX_VISITS_PER_PATIENT,
    VISIT_STATUS_APPROVED,
    VISIT_STATUS_DRAFT,
    VISIT_STATUSES,
    JsonPatientMemoryRepository,
    PatientMemoryStore,
    get_memory_store,
    resolve_patient_id,
    visit_status,
)

__all__ = [
    "PatientProfile",
    "parse_patient_context",
    "PatientMemoryStore",
    "JsonPatientMemoryRepository",
    "get_memory_store",
    "resolve_patient_id",
    "visit_status",
    "DEFAULT_MEMORY_PATH",
    "MAX_VISITS_PER_PATIENT",
    "VISIT_STATUS_APPROVED",
    "VISIT_STATUS_DRAFT",
    "VISIT_STATUSES",
]