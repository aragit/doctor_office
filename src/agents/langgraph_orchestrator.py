"""
LangGraph orchestrator - the multi-agent visit documentation graph.

Graph topology (supervisor-routed):

    START -> memory_load_node -> scribe_node -> cleanup_node -> coding_node
          -> validation_node -> supervisor_node -+-> memory_save_node -> END
                                                  +-> scribe_node       (retry 1/2)
                                                  +-> coding_node       (retry 1/2)
                                                  +-> validation_node   (symptom-code re-check)
                                                  +-> human_review_node -> END

Central Executor / Supervisor pattern:
- The graph owns the order of work. Agents do not call each other and do not
  decide when to run; the supervisor node does.
- Each agent node runs a *scoped, deterministic* toolset (see
  `tools.registry.AGENT_TOOLS`) and returns results through `AgentState`.
- Memory reads and writes are explicit nodes, so persistence is visible in the
  graph and auditable, not hidden inside an agent.
- The Supervisor is a zero-LLM node: it reads validation quality_flags, routes
  retries with targeted feedback, auto-fixes symptom-code gaps deterministically,
  and escalates to human_review_node when retries are exhausted. A visit sent
  to human review is never written to long-term memory.

Two control-flow invariants hold across the graph (P1):

- A retry starts from an empty per-cycle flag view. The Supervisor writes every
  `cycle_flags` bucket empty before routing back, so no flag from the previous
  pass can consume the retry budget or escalate a visit that has been corrected.
- The `supervisor -> validation_node -> supervisor` loop is the post-fix re-check.
  When the Supervisor mutates `coding_result` it sets `pending_revalidation`,
  which the Supervisor will not finalize on and only `validation_node` clears -
  so an auto-added code is always validated in its final form before the visit is
  stored. Bounded by `MAX_REVALIDATIONS`; `MAX_RETRIES` still bounds the LLM
  passes.

The Scribe and Coding agent logic, the retrieved-note filtering and the
cleanup/validation guardrails are reused unchanged from earlier stages.
"""

import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.cleanup_node import cleanup_node
from agents.coding_node import coding_node
from agents.graph_state import (
    PIPELINE_STEPS,
    PIPELINE_VERSION,
    AgentState,
    build_initial_state,
)
from agents.memory_node import memory_load_node, memory_save_node
from agents.scribe_node import scribe_node
from agents.supervisor_node import (
    human_review_node,
    route_after_validation,
    supervisor_node,
)
from agents.validation_node import validation_node
from utils.config import get_model_name, enforce_synthetic_guard

logger = logging.getLogger(__name__)


def build_graph() -> CompiledStateGraph:
    """Construct and compile the visit documentation StateGraph.

    Returns:
        A compiled LangGraph graph ready to invoke or stream.
    """
    graph = StateGraph(AgentState)

    graph.add_node("memory_load_node", memory_load_node)
    graph.add_node("scribe_node", scribe_node)
    graph.add_node("cleanup_node", cleanup_node)
    graph.add_node("coding_node", coding_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("memory_save_node", memory_save_node)
    graph.add_node("human_review_node", human_review_node)

    graph.add_edge(START, "memory_load_node")
    graph.add_edge("memory_load_node", "scribe_node")
    graph.add_edge("scribe_node", "cleanup_node")
    graph.add_edge("cleanup_node", "coding_node")
    graph.add_edge("coding_node", "validation_node")
    graph.add_edge("validation_node", "supervisor_node")

    # Supervisor decides the next step from validation outcome.
    graph.add_conditional_edges(
        "supervisor_node",
        route_after_validation,
        {
            "scribe_node": "scribe_node",
            "coding_node": "coding_node",
            "validation_node": "validation_node",
            "memory_save_node": "memory_save_node",
            "human_review_node": "human_review_node",
        },
    )

    graph.add_edge("memory_save_node", END)
    graph.add_edge("human_review_node", END)

    return graph.compile()


_GRAPH: Optional[CompiledStateGraph] = None
_GRAPH_LOCK = threading.Lock()


def get_graph() -> CompiledStateGraph:
    """Return the compiled graph, building it once per process.

    Compiling is cheap but the graph object is reused across Streamlit
    reruns and across consecutive test cases.
    """
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = build_graph()
    return _GRAPH


def _steps_completed(state: AgentState) -> List[str]:
    """Order the steps that actually ran, from per-node timing.

    Retries mean steps can execute more than once: a scribe retry is still
    one "scribe" entry (we report first-pass order, mirroring the canonical
    PIPELINE_STEPS list), but `retry_count` in metadata carries the extra
    passes. `human_review` is appended when a visit was escalated.
    """
    timing = state.get("timing") or {}
    step_order = {
        "memory_load": "memory_load_seconds",
        "scribe": "scribe_seconds",
        "cleanup": "cleanup_seconds",
        "coding": "coding_seconds",
        "validation": "validation_seconds",
        "supervisor": "supervisor_seconds",
        "memory_save": "memory_save_seconds",
    }
    completed = [step for step in PIPELINE_STEPS if step_order.get(step) in timing]
    if "human_review_seconds" in timing:
        completed.append("human_review")
    return completed


def _symptom_coding_metadata(state: AgentState) -> Dict[str, Any]:
    """Provenance for every symptom code the pipeline considered or added.

    Reads what already happened - the Coding node's tool trace and the
    Supervisor's deterministic completions - and returns it as data. Nothing
    here decides anything, so adding it cannot alter a clinical outcome.

    Returns:
        Dict with `candidates` (what the symptom map proposed, and why),
        `coding_added` and `supervisor_added` (what each layer added).
    """
    trace = (state.get("tool_results") or {}).get("coding_node") or {}
    symptom_trace = trace.get("simple_symptom_to_icd_suggestion") or {}

    candidates = list(symptom_trace.get("candidates") or [])
    if not candidates:
        # Older traces carried only the bare code list.
        candidates = [
            {"code": code} for code in symptom_trace.get("suggested_codes") or []
        ]

    return {
        "candidates": candidates,
        "coding_added": list(symptom_trace.get("added_to_result") or []),
        "supervisor_added": list(state.get("supervisor_symptom_codes") or []),
    }


def _control_flow_metadata(state: AgentState) -> Dict[str, Any]:
    """The Supervisor's control-flow record for this run.

    Reports the three facts a reviewer needs in order to trust an auto-fix: was
    one applied, was the mutated result re-validated before anything was
    decided, and where the run ended. Reads only state the graph already
    produced, so it cannot change a clinical outcome.

    Args:
        state: Final graph state.

    Returns:
        Dict with the auto-fix, re-validation and routing record. The
        `pending_*` fields can only be True if a run ended mid-re-validation,
        which the graph has no path for - reported so an unexpected state is
        visible rather than hidden.
    """
    auto_fix_rounds = int(state.get("auto_fix_rounds", 0) or 0)
    return {
        "auto_fix_applied": bool(auto_fix_rounds),
        "auto_fix_rounds": auto_fix_rounds,
        "auto_fix_codes": list(state.get("supervisor_symptom_codes") or []),
        "revalidations": int(state.get("revalidation_passes", 0) or 0),
        "revalidation_stalls": int(state.get("revalidation_stalls", 0) or 0),
        "pending_revalidation": bool(state.get("pending_revalidation", False)),
        "cycle_flag_resets": int(state.get("cycle_flag_resets", 0) or 0),
        "cleared_cycle_flags": list(state.get("cleared_cycle_flags") or []),
        "final_route": state.get("route_to") or "",
        "final_decision": state.get("supervisor_decision") or "",
        "escalation_reason": state.get("escalation_reason") or "",
        "empty_soap_retries": int(state.get("empty_soap_retries", 0) or 0),
    }


def _state_to_result(state: AgentState, mode: str, model_name: Optional[str]) -> Dict[str, Any]:
    """Convert final graph state into the legacy result dictionary.

    The shape matches the pre-LangGraph orchestrator exactly so the Streamlit
    UI and any existing scripts keep working without modification. Memory,
    tool and supervisor details are additive keys under `metadata`.
    """
    timing = dict(state.get("timing") or {})
    timing["total_seconds"] = round(state.get("total_seconds") or 0.0, 1)

    patient_summary = state.get("patient_summary") or {}
    tool_results = state.get("tool_results") or {}

    supervisor_applied = list(state.get("supervisor_applied") or [])

    return {
        "soap_note": state.get("cleaned_soap_note") or state.get("soap_note") or {},
        "coding_result": state.get("coding_result") or {},
        "quality_flags": list(state.get("quality_flags") or []),
        "metadata": {
            "model_used": model_name or get_model_name(mode),
            "mode": mode,
            "pipeline_version": PIPELINE_VERSION,
            "orchestrator": "langgraph",
            "steps_completed": _steps_completed(state),
            "graph": state.get("current_step", "memory_save"),
            "timing": timing,
            "memory": {
                "patient_id": state.get("patient_id"),
                "prior_visits": int(patient_summary.get("visit_count", 0) or 0),
                "memory_context_used": bool(state.get("memory_context")),
                "saved": bool(state.get("memory_saved")),
                # P2: what this run wrote, and what it is worth. A recorded
                # visit is a draft until a physician approves it, so `saved`
                # alone must never be read as "this is now history".
                "visit_id": state.get("memory_visit_id") or "",
                "visit_status": state.get("memory_visit_status") or "",
                "approved": False,
                "approved_visit_count": int(
                    patient_summary.get("approved_visit_count", 0) or 0
                ),
                "draft_visit_count": int(
                    patient_summary.get("draft_visit_count", 0) or 0
                ),
                "conditions_ever": list(
                    (patient_summary.get("history") or {}).get("conditions_ever", []) or []
                ),
            },
            "supervisor": {
                "retry_count": int(state.get("retry_count", 0) or 0),
                "needs_human_review": bool(state.get("needs_human_review", False)),
                "applied": supervisor_applied,
                "feedback": list(state.get("feedback") or []),
            },
            # P1 control flow: whether a deterministic fix was applied, whether
            # the fixed codes were re-validated, and how the run ended. Pure
            # metadata read back from Supervisor/validation state, so a run can
            # be audited for "was the auto-fix actually checked?" without
            # replaying it.
            "control_flow": _control_flow_metadata(state),
            # Phase A4: where every symptom code came from. Pure metadata, read
            # back from the Coding tool trace and the Supervisor state - it adds
            # no decision and changes nothing the pipeline produced.
            "symptom_coding": _symptom_coding_metadata(state),
            # Echoed back so the Physician Review tab can show the clinician
            # exactly which encounter data produced this note.
            "visit_context": dict(state.get("visit_context") or {}),
            "specialty": state.get("specialty"),
            # Coding path metadata (Fast Path / Slow Path)
            "coding_path": state.get("coding_path", "slow"),
            "coding_path_decision": state.get("coding_path_decision", {}),
            "tools": {
                "tools_used": sorted(
                    {name for trace in tool_results.values() if isinstance(trace, dict)
                     for name in trace}
                ),
                "tool_results": tool_results,
            },
        },
    }


def run_visit_pipeline(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str,
    mode: str = "fast",
    model_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    use_memory: bool = True,
    physician_feedback: Optional[str] = None,
    visit_context: Optional[Dict[str, Any]] = None,
    specialty: Optional[str] = None,
    is_synthetic: bool = False,
    allow_real_phi: bool = False,
) -> Dict[str, Any]:
    """Execute the complete visit documentation pipeline via LangGraph.

    Nodes executed: memory load -> Scribe Agent -> grounding cleanup ->
    Coding Agent -> validation -> memory save.

    Args:
        patient_context: Formatted patient demographics, conditions, medications,
            allergies, last visit.
        chief_complaint: Patient's current complaint / reason for visit.
        visit_context: Optional clinician-provided encounter data - HPI, vitals,
            physical examination, additional findings. Omit for a
            chief-complaint-only visit.
        retrieved_notes: Relevant clinical reference notes from RAG.
        mode: Model mode - "fast" (qwen2.5:1.5b) or "quality" (gemma3:4b).
            Defaults to "fast".
        model_name: Optional specific model name (overrides `mode`).
        patient_id: Optional stable patient id for long-term memory. Defaults
            to an id derived from the patient context.
        use_memory: Set False to skip reading and writing long-term memory.
        physician_feedback: Optional free-text feedback from a physician
            requesting changes to the generated note (for revision runs).
        specialty: Optional medical specialty ID for specialty-specific guidance.
        is_synthetic: Whether the patient record is known synthetic data.
            Defaults to False. When True, the synthetic-data guard is bypassed.
        allow_real_phi: Whether to allow real PHI. Defaults to False.
            Only set True in isolated dev environments with synthetic data you control.

    Returns:
        Dictionary with `soap_note`, `coding_result`, `quality_flags` and `metadata`.
    """
    enforce_synthetic_guard(patient_id, is_synthetic)

    started_at = time.time()
    initial_state = build_initial_state(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        model_name=model_name,
        patient_id=patient_id,
        use_memory=use_memory,
        started_at=started_at,
        physician_feedback=physician_feedback,
        visit_context=visit_context,
        specialty=specialty,
        is_synthetic=is_synthetic,
        allow_real_phi=allow_real_phi,
    )

    logger.info("Starting LangGraph visit pipeline (mode=%s)", mode)
    final_state = get_graph().invoke(initial_state)

    final_state["total_seconds"] = round(time.time() - started_at, 1)
    logger.info("LangGraph pipeline finished in %ss", final_state["total_seconds"])

    return _state_to_result(final_state, mode, model_name)


def stream_visit_pipeline(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str,
    mode: str = "fast",
    model_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    use_memory: bool = True,
    physician_feedback: Optional[str] = None,
    visit_context: Optional[Dict[str, Any]] = None,
    specialty: Optional[str] = None,
    is_synthetic: bool = False,
    allow_real_phi: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Run the pipeline, yielding a legacy-shaped result after each node.

    Useful for progressive UI updates: the caller sees a partial (but
    schema-identical) result as soon as each node finishes, in graph order.
    Reducers are applied by LangGraph itself, so each yielded snapshot is a
    fully merged state.

    Yields:
        One dict per completed node with the same keys as `run_visit_pipeline`.
    """
    enforce_synthetic_guard(patient_id, is_synthetic)

    started_at = time.time()
    initial_state = build_initial_state(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        model_name=model_name,
        patient_id=patient_id,
        use_memory=use_memory,
        started_at=started_at,
        physician_feedback=physician_feedback,
        visit_context=visit_context,
        specialty=specialty,
        is_synthetic=is_synthetic,
        allow_real_phi=allow_real_phi,
    )

    for snapshot in get_graph().stream(initial_state, stream_mode="values"):
        # The first "values" snapshot is the untouched input state.
        if snapshot.get("current_step") == "start" and not snapshot.get("soap_note"):
            continue

        snapshot = {
            **snapshot,
            "total_seconds": round(time.time() - started_at, 1),
        }
        yield _state_to_result(snapshot, mode, model_name)


def graph_mermaid() -> str:
    """Return the Mermaid diagram of the compiled graph (for docs/debugging)."""
    return get_graph().get_graph().draw_mermaid()


def run_demo(mode: str = "fast") -> Dict[str, Any]:
    """Run the built-in end-to-end demo for John Doe and print the results."""
    import json

    from rag.simple_rag import get_vectorstore

    patient = {
        "name": "John Doe",
        "age": 58,
        "gender": "male",
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
        "allergies": ["Penicillin"],
        "last_visit_summary": "Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence.",
    }

    patient_context = f"""
Patient: {patient['name']}, {patient['age']}yo {patient['gender']}
Conditions: {', '.join(patient['conditions'])}
Medications: {', '.join(patient['medications'])}
Allergies: {', '.join(patient['allergies'])}
Last Visit: {patient['last_visit_summary']}
"""

    vs = get_vectorstore()
    from utils.config import RAG_TOP_K
    retrieved_docs = vs.similarity_search("diabetes hypertension management", k=RAG_TOP_K)
    retrieved_notes = "\n\n".join([doc.page_content for doc in retrieved_docs])

    chief_complaint = (
        "Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
    )

    print("=" * 60)
    print(f"LANGGRAPH VISIT PIPELINE DEMO - John Doe ({mode.title()} Mode)")
    print("=" * 60)
    print("\nGraph topology:\n")
    print(graph_mermaid())
    print()

    enforce_synthetic_guard("P001", is_synthetic=True)

    result = run_visit_pipeline(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        patient_id="P001",
        is_synthetic=True,
    )

    print("\n--- SOAP NOTE ---")
    print(result["soap_note"]["subjective"][:200] + "...")
    print("\n--- CODING RESULT ---")
    print(f"Primary ICD-10: {result['coding_result']['primary_icd10']}")
    print(f"Secondary ICD-10: {result['coding_result']['secondary_icd10']}")
    print(f"CPT Suggestions: {result['coding_result']['cpt_suggestions']}")
    print(f"Medical Necessity: {result['coding_result']['medical_necessity_note'][:200]}...")
    print(f"Quality Flags: {result['quality_flags']}")

    print("\n--- MEMORY & TOOLS ---")
    print(f"Memory: {result['metadata']['memory']}")
    print(f"Tools used: {result['metadata']['tools']['tools_used']}")

    print("\n" + "=" * 60)
    print("FULL PIPELINE OUTPUT (JSON)")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_demo()
