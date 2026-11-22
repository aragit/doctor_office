"""
Memory nodes - LangGraph nodes for long-term patient memory.

Two supervisor-controlled nodes, both pure Python (no LLM, microseconds):

- `memory_load_node`  runs before the Scribe Agent and pulls prior-visit
  context into working state. Approved visits only: a pipeline draft has not
  been signed, so it is not history and never reaches the prompt.
- `memory_save_node`  runs after validation and records the finished visit as a
  **draft**. It cannot create approved history; only `approve_visit`, reached
  from the physician's Approve button, does that.

Keeping memory as explicit graph nodes (rather than hidden helper calls inside
the scribe) is what preserves the Central Executor pattern: the graph decides
when memory is read and written, and the agents only ever see the results in
state.

Both nodes depend on the `PatientMemoryRepository` interface (defaulting to
the JSON implementation) so the store can be swapped for Postgres without
changing the nodes.
"""

import logging
import time
from typing import Any, Dict

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState
from memory.repository import get_memory_repo, resolve_patient_id
from memory.patient_memory import VISIT_STATUS_DRAFT, visit_status
from memory.patient_profile import parse_patient_context
from tools.registry import MEMORY_AGENT, call_tool

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 600


def memory_load_node(state: AgentState) -> dict:
    """Load long-term memory for this patient into working state.

    Never fails a visit: a missing patient or an unreadable store simply
    yields empty memory fields. Honours `use_memory=False` by doing nothing.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with `patient_id`, `patient_summary` and
        `memory_context`.
    """
    started = time.time()

    patient_context = state["patient_context"]

    if not state.get("use_memory", True):
        return {
            "patient_id": resolve_patient_id(state.get("patient_id"), patient_context),
            "patient_summary": {},
            "memory_context": "",
            "memory_saved": False,
            "current_step": "memory_load",
            "timing": {"memory_load_seconds": round(time.time() - started, 1)},
            "messages": [
                AIMessage(content="Memory disabled for this run.", name="memory_load_node")
            ],
        }

    patient_id = resolve_patient_id(state.get("patient_id"), patient_context)
    store = get_memory_repo()

    summary: Dict[str, Any] = {}
    memory_context = ""
    try:
        # Refresh the "current" chart section from the context we were handed,
        # so memory never contradicts the live chart.
        store.ensure_patient(patient_id, patient_context)
        summary = store.get_summary(patient_id)
        memory_context = store.build_context_block(patient_id, char_limit=MAX_HISTORY_CHARS)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("memory_load_node failed for %s: %s", patient_id, exc)

    profile = parse_patient_context(patient_context)
    visit_count = int(summary.get("visit_count", 0))

    elapsed = round(time.time() - started, 1)
    logger.info(
        "memory_load_node: patient=%s prior_visits=%s chars=%s",
        patient_id, visit_count, len(memory_context),
    )

    return {
        "patient_id": patient_id,
        "patient_summary": summary,
        "memory_context": memory_context,
        "current_step": "memory_load",
        "timing": {"memory_load_seconds": elapsed},
        "messages": [
            AIMessage(
                content=(
                    f"Memory loaded for {profile.name or patient_id} "
                    f"({visit_count} prior visit(s) on record)."
                ),
                name="memory_load_node",
            )
        ],
    }


def memory_save_node(state: AgentState) -> dict:
    """Record the completed visit as a **draft** in long-term memory.

    Runs only after validation, so a failed visit is not written at all. What it
    writes is a draft: the note and codes exist in the store for audit and
    continuity, but nothing in this node can create approved longitudinal
    history. `approve_visit` - reached only from the physician's Approve button -
    is what promotes the row. Never raises.

    Args:
        state: Current graph state (should contain cleaned note + coding).

    Returns:
        Partial state update recording what was stored: `memory_saved`,
        `memory_visit_id` and `memory_visit_status`.
    """
    started = time.time()

    if not state.get("use_memory", True):
        return {
            "memory_saved": False,
            "memory_visit_id": "",
            "memory_visit_status": "",
            "current_step": "memory_save",
            "timing": {"memory_save_seconds": round(time.time() - started, 1)},
            "messages": [
                AIMessage(content="Memory disabled; visit not recorded.", name="memory_save_node")
            ],
        }

    patient_id = state.get("patient_id") or resolve_patient_id(
        state.get("patient_id"), state["patient_context"]
    )
    soap_note = state.get("cleaned_soap_note") or state.get("soap_note") or {}

    saved = False
    visit_id = ""
    summary = None
    error = None
    try:
        record = get_memory_repo().record_visit(
            patient_id=patient_id,
            patient_context=state["patient_context"],
            chief_complaint=state["chief_complaint"],
            coding_result=state.get("coding_result") or {},
            soap_note=soap_note,
            quality_flags=list(state.get("quality_flags") or []),
            status=VISIT_STATUS_DRAFT,
        )
        saved = True
        # Echoed so Physician Review can approve *this* row. Read from the
        # newest draft rather than assuming one write: a revision run appends a
        # second row and must not promote the draft it replaced.
        drafts = [
            visit for visit in (record.get("visits") or [])
            if visit_status(visit) == VISIT_STATUS_DRAFT
        ]
        if drafts:
            visit_id = str(drafts[-1].get("id") or "")
        # The summary in state was loaded before this run wrote anything, so its
        # counters are stale by one visit. Refresh it, or the UI would report the
        # store as it looked before the draft existed.
        summary = record
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("memory_save_node failed for %s: %s", patient_id, exc)
        error = f"Memory save failed: {exc}"

    elapsed = round(time.time() - started, 1)
    logger.info(
        "memory_save_node: patient=%s draft=%s id=%s in %ss",
        patient_id, saved, visit_id or "-", elapsed,
    )

    update: Dict[str, Any] = {
        "current_step": "memory_save",
        "timing": {"memory_save_seconds": elapsed},
        "memory_saved": saved,
        "memory_visit_id": visit_id,
        "memory_visit_status": VISIT_STATUS_DRAFT if saved else "",
        "messages": [
            AIMessage(
                content=(
                    f"Draft visit recorded to long-term memory for {patient_id} "
                    f"(not history until a physician approves it)."
                    if saved
                    else "Visit not recorded to long-term memory."
                ),
                name="memory_save_node",
            )
        ],
    }
    if summary is not None:
        update["patient_summary"] = summary
    if error:
        update["errors"] = [error]
    return update


__all__ = ["memory_load_node", "memory_save_node"]
