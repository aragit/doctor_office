"""
validation_node - LangGraph node running the final clinical quality gate.

Wraps `utils.validation.run_full_validation`, which checks that every
ICD-10 code is grounded in the patient context, the chief complaint or the
clinician-provided visit data (HPI, exam, tests), and that CPT codes match
the patient's established/new status.

The node has a second job: it is the *only* thing that can clear
`pending_revalidation`, so the Supervisor can never finalize a coding result
that this node has not seen. When it runs against a result the Supervisor
mutated, it additionally re-checks the code list's structure, which the
deterministic symptom completion bypasses (see
`utils.validation.validate_post_fix_integrity`).
"""

import logging
import time

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState
from agents.visit_context import visit_context_to_grounding_text
from utils.validation import run_full_validation, validate_post_fix_integrity

logger = logging.getLogger(__name__)


def validation_node(state: AgentState) -> dict:
    """Validate ICD-10 grounding and CPT appropriateness.

    Args:
        state: Current graph state, must contain `cleaned_soap_note` and
            `coding_result`.

    Returns:
        Partial state update with accumulated quality flags and timing. Always
        clears `pending_revalidation`, because having executed is itself the proof
        that the current coding result has been validated.
    """
    started = time.time()

    soap_note = state.get("cleaned_soap_note") or state.get("soap_note") or {}
    coding_result = state.get("coding_result") or {}

    validation_result = run_full_validation(
        soap_icd10=soap_note.get("icd10_suggestions", []),
        coding_result=coding_result,
        patient_context=state["patient_context"],
        chief_complaint=state["chief_complaint"],
        extra_context=visit_context_to_grounding_text(state.get("visit_context")),
        soap_note=soap_note,
    )

    quality_flags = list(validation_result.get("quality_flags", []))

    # Re-validate the Supervisor's own mutation. Only on this path: the ordinary
    # pass already had these codes checked structurally by the Coding node, so
    # repeating the check would only add noise to the normal flag vocabulary.
    revalidating = bool(state.get("pending_revalidation", False))
    if revalidating:
        added = list(state.get("supervisor_symptom_codes") or [])
        integrity_flags = validate_post_fix_integrity(coding_result, added)
        quality_flags.extend(integrity_flags)
        logger.info(
            "validation_node re-validated an auto-fixed coding result: %d added "
            "code(s), %d integrity flag(s)",
            len(added), len(integrity_flags),
        )

    elapsed = round(time.time() - started, 1)
    logger.info(
        "validation_node finished in %ss with %d quality flag(s)%s",
        elapsed, len(quality_flags), " (post-fix re-validation)" if revalidating else "",
    )

    if quality_flags:
        summary = f"Validation raised {len(quality_flags)} quality flag(s)."
    else:
        summary = "Validation passed - all codes grounded in patient context."

    update = {
        "quality_flags": quality_flags,
        "cycle_flags": {"validation": list(quality_flags)},
        "current_step": "validation",
        "timing": {"validation_seconds": elapsed},
        "messages": [AIMessage(content=summary, name="validation_node")],
    }

    if revalidating:
        update["pending_revalidation"] = False
        update["revalidation_passes"] = (
            int(state.get("revalidation_passes", 0) or 0) + 1
        )

    return update


__all__ = ["validation_node"]
