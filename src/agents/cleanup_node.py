"""
cleanup_node - LangGraph node enforcing clinical grounding.

Wraps `utils.validation.clean_soap_note`, which strips hallucinated
conditions (asthma/COPD/CHF/CKD and their medications) from the generated
note and records every removal as a quality flag. This is the safety gate
that sits between the Scribe Agent and the Coding Agent.
"""

import logging
import time

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState
from agents.visit_context import visit_context_to_grounding_text
from utils.validation import clean_soap_note

logger = logging.getLogger(__name__)


def cleanup_node(state: AgentState) -> dict:
    """Remove hallucinated content from the Scribe Agent output.

    Args:
        state: Current graph state, must contain `soap_note`.

    Returns:
        Partial state update containing the cleaned SOAP note plus the list of
        removed items (also surfaced as quality flags).
    """
    started = time.time()

    # Clinician-provided HPI/exam/tests widen the grounding set: a condition the
    # doctor documented today must not be scrubbed as a hallucination.
    extra_context = visit_context_to_grounding_text(state.get("visit_context"))

    cleaned, removed_items = clean_soap_note(
        state["soap_note"],
        state["patient_context"],
        state["chief_complaint"],
        extra_context,
    )

    elapsed = round(time.time() - started, 1)
    if removed_items:
        logger.info("cleanup_node removed %d hallucinated item(s)", len(removed_items))
        for item in removed_items:
            logger.info("  - %s", item)
    else:
        logger.info("cleanup_node found no hallucinated content")

    summary = (
        f"Cleanup removed {len(removed_items)} hallucinated item(s) in {elapsed}s."
        if removed_items
        else f"Cleanup found no hallucinated content in {elapsed}s."
    )

    return {
        "cleaned_soap_note": cleaned,
        "cleanup_removed": removed_items,
        "quality_flags": [f"Cleanup: {item}" for item in removed_items],
        "current_step": "cleanup",
        "timing": {"cleanup_seconds": elapsed},
        "messages": [AIMessage(content=summary, name="cleanup_node")],
    }


__all__ = ["cleanup_node"]
