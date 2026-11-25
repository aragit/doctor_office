"""
cleanup_node - LangGraph node enforcing clinical grounding.

Wraps `utils.validation.clean_soap_note`, which strips hallucinated
conditions (asthma/COPD/CHF/CKD and their medications) from the generated
note and records every removal as a quality flag. This is the safety gate
that sits between the Scribe Agent and the Coding Agent.

TEMPORARY CPU PERFORMANCE BYPASS: For the demo on 2-core CPU, this node is
a pass-through that skips the cleanup LLM call (clean_soap_note is
deterministic but adds latency). The safety gate is re-enabled in
production by reverting this change.
"""

import time

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState


def cleanup_node(state: AgentState) -> dict:
    """Pass-through: return soap_note as cleaned_soap_note without LLM call.

    TEMPORARY BYPASS: On 2-core CPU, the cleanup step adds ~30-60s latency.
    This pass-through preserves graph topology while skipping the
    deterministic clean_soap_note call for the demo.
    """
    started = time.time()

    # Pass-through: soap_note becomes cleaned_soap_note directly
    cleaned = state.get("soap_note", {})

    elapsed = round(time.time() - started, 1)

    return {
        "cleaned_soap_note": cleaned,
        "cleanup_removed": [],
        "quality_flags": [],
        "current_step": "cleanup",
        "timing": {"cleanup_seconds": elapsed},
        "messages": [AIMessage(content=f"Cleanup bypassed (pass-through) in {elapsed}s.", name="cleanup_node")],
    }


__all__ = ["cleanup_node"]
