"""
Patient inspection tools.

Each function is a plain Python function (directly testable) paired with a
LangChain `StructuredTool` wrapper in `tools.registry`, so the same
implementation can later be bound to a model that supports tool calling.

These tools are read-only and never call an LLM, so they are safe to run
before, during or after generation.
"""

from typing import Any, Dict, List

from memory.patient_profile import parse_patient_context

DESCRIPTION_CONDITIONS = (
    "Return the patient's documented chronic conditions as a clean list. "
    "Use this to ground any statement about diagnoses; the list is authoritative."
)
DESCRIPTION_MEDICATIONS = (
    "Return the patient's documented medications as a clean list. "
    "Use this to ground medication statements in the plan."
)
DESCRIPTION_ALLERGIES = (
    "Return the patient's documented allergies as a clean list. "
    "Returns an empty list when no allergies are recorded."
)
DESCRIPTION_PROFILE = (
    "Return the full structured patient profile: id, name, age, gender, "
    "conditions, medications, allergies and last visit summary."
)


def get_patient_conditions(patient_context: str) -> List[str]:
    """Get the patient's documented conditions.

    Args:
        patient_context: Formatted patient context block.

    Returns:
        Clean list of conditions, e.g. ["Type 2 Diabetes", "Hypertension"].
        Empty when the context lists none.
    """
    return parse_patient_context(patient_context).conditions


def get_patient_medications(patient_context: str) -> List[str]:
    """Get the patient's documented medications.

    Args:
        patient_context: Formatted patient context block.

    Returns:
        Clean list of medications, e.g. ["Metformin 500mg BID"].
    """
    return parse_patient_context(patient_context).medications


def get_patient_allergies(patient_context: str) -> List[str]:
    """Get the patient's documented allergies.

    Args:
        patient_context: Formatted patient context block.

    Returns:
        Clean list of allergies. Placeholders such as "None" / "N/A" are
        treated as an empty list rather than a real allergy.
    """
    return parse_patient_context(patient_context).allergies


def get_patient_profile(patient_context: str) -> Dict[str, Any]:
    """Get the full structured patient profile.

    Args:
        patient_context: Formatted patient context block.

    Returns:
        Dict with patient_id, name, age, gender, conditions, medications,
        allergies and last_visit_summary.
    """
    return parse_patient_context(patient_context).to_dict()


__all__ = [
    "get_patient_conditions",
    "get_patient_medications",
    "get_patient_allergies",
    "get_patient_profile",
    "DESCRIPTION_CONDITIONS",
    "DESCRIPTION_MEDICATIONS",
    "DESCRIPTION_ALLERGIES",
    "DESCRIPTION_PROFILE",
]
