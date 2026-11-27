"""
Shared LangGraph state for the Doctor Office visit documentation pipeline.

The `AgentState` TypedDict is the single source of truth that flows between
graph nodes. Nodes never pass data to each other directly - they receive the
accumulated state and return a partial update, which LangGraph merges using the
reducers declared here.
"""

import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


PIPELINE_VERSION = "3.1-supervisor-revalidate"

# Canonical happy-path node order. Used for metadata reporting, progress
# rendering and the streaming snapshot order.
PIPELINE_STEPS = (
    "memory_load",
    "scribe",
    "cleanup",
    "coding",
    "validation",
    "supervisor",
    "memory_save",
)

# Nodes that run no LLM (fast, deterministic).
DETERMINISTIC_STEPS = (
    "memory_load",
    "cleanup",
    "validation",
    "supervisor",
    "human_review",
    "memory_save",
)

# Target names the Supervisor may route to after validation.
ROUTE_FINALIZE = "memory_save_node"
ROUTE_RETRY_CODING = "coding_node"
ROUTE_RETRY_SCRIBE = "scribe_node"
ROUTE_REVALIDATE = "validation_node"
ROUTE_HUMAN_REVIEW = "human_review_node"

# Stages that write into the per-cycle flag buckets. A retry re-runs the stage
# the Supervisor routed to, so the buckets of every stage in the cycle are
# cleared explicitly before the next pass - see `reset_cycle_flags`.
CYCLE_FLAG_BUCKETS = ("scribe", "cleanup", "coding", "validation")


def append_unique(existing: List[Any], new: List[Any]) -> List[Any]:
    """Reducer that appends items while dropping duplicates.

    LangGraph re-applies reducers on every node execution, so a plain
    `operator.add` would duplicate flags whenever a node is revisited
    (retries, conditional loops). Keeping this idempotent makes the state
    safe to replay.
    """
    result = list(existing or [])
    for item in new or []:
        if item not in result:
            result.append(item)
    return result


def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that shallow-merges dict updates (used for per-node timing)."""
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


def merge_tool_results(
    existing: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer that merges per-node tool traces without clobbering other nodes.

    A node returns `{node_name: {tool_name: result}}`; nested dicts are merged
    one level down so re-running one node does not erase another's trace.
    """
    merged: Dict[str, Any] = dict(existing or {})
    for node, node_tools in (new or {}).items():
        if isinstance(node_tools, dict):
            existing_node = dict(merged.get(node) or {})
            existing_node.update(node_tools)
            merged[node] = existing_node
        else:
            merged[node] = node_tools
    return merged


def reset_cycle_flags() -> Dict[str, List[str]]:
    """Explicit empty write for every per-cycle flag bucket.

    `cycle_flags` is merged per bucket (`merge_dicts`), so a bucket survives
    until the node that owns it writes again. That is the wrong default for a
    retry: if the re-run node returns no flags at all - or the pipeline takes a
    path that skips a stage - the previous pass's flag stays in state and the
    Supervisor can be re-triggered by a pass it already corrected.

    Writing all buckets empty in one update is the explicit overwrite that makes
    "this pass raised nothing" a state the Supervisor can see, instead of an
    absence it has to infer. Returned as a plain dict so the caller can merge it
    into its own update alongside the rest of the routing decision.

    Returns:
        Dict with every `CYCLE_FLAG_BUCKETS` entry mapped to an empty list.
    """
    return {bucket: [] for bucket in CYCLE_FLAG_BUCKETS}


class AgentState(TypedDict, total=False):
    """State shared by every node in the visit documentation graph.

    Attributes:
        patient_context: Demographics, conditions, medications, allergies, last visit.
        chief_complaint: Reason for the current visit.
        specialty: Selected medical specialty (general, cardiology, neurology, etc.).
        visit_context: Clinician-provided data for this encounter - HPI, vitals,
            physical examination, additional findings. Optional; an empty dict
            keeps the chief-complaint-only behaviour of earlier versions. See
            `agents.visit_context` for the canonical shape and rendering.
        retrieved_notes: Raw RAG notes (filtered inside the Scribe Agent).
        soap_note: SOAP note as produced by the Scribe Agent (pre-cleanup).
        cleaned_soap_note: SOAP note after grounding cleanup - this is what the
            Coding Agent consumes and what callers should read.
        coding_result: Coding Agent output (primary ICD-10, secondary, CPT, necessity).
        quality_flags: Accumulated warnings from cleanup and validation. This is
            an **audit log**, not a routing input: it accumulates across retries
            with `append_unique` so the review UI can show what every pass
            reported. The Supervisor decides only on `cycle_flags`.
        cleanup_removed: Individual items removed by the grounding cleanup.
        messages: Interleaved agent trace (LangChain messages).
        mode: Model mode, "fast" (default) or "quality".
        model_name: Optional explicit model name overriding `mode`.
        current_step: Name of the most recently executed node.
        timing: Per-node wall-clock seconds.
        total_seconds: Total pipeline wall-clock seconds.
        started_at: Unix timestamp recorded when the pipeline starts.
        errors: Errors captured by recoverable nodes.

        patient_id: Stable patient identifier for long-term memory.
        patient_summary: Stored long-term memory record for this patient.
        memory_context: Prior-visit text injected into the Scribe prompt.
            Approved visits only - a pipeline draft is not history (P2).
        memory_saved: Whether a visit row was persisted on completion. The row is
            a **draft**; only a physician Approve turns it into history.
        memory_visit_id: Id of the draft row the pipeline recorded, echoed into
            result metadata so Physician Review approves the row this run wrote
            rather than whichever draft happens to be newest.
        memory_visit_status: Status of the row this run recorded - always
            `VISIT_STATUS_DRAFT`, or empty when nothing was recorded. Present so
            a reviewer can tell "pipeline output, unsigned" apart from "approved
            and in history" without reaching into the store.
        tool_results: Tool calls executed by nodes, keyed by node name.
        grounding_facts: Verified facts the tools derived from the chart.

        retry_count: How many LLM passes have been re-run by the Supervisor.
        feedback: Short, actionable messages the Supervisor sends to the
            next Scribe or Coding pass when routing back.
        physician_feedback: Free-text feedback from the reviewing physician
            requesting changes to the generated note. Carried forward on revision runs.
        needs_human_review: Set by the Supervisor when retries are exhausted.
        supervisor_applied: Deterministic fixes the Supervisor applied itself
            (e.g. re-inserted a symptom code the model dropped).
        supervisor_symptom_codes: The structured form of the symptom codes the
            Supervisor completed, so the review UI can attribute a code to the
            Supervisor without parsing `supervisor_applied` prose (Phase A4).
        route_to: Internal Supervisor routing signal consumed by the
            conditional edge after `supervisor_node`.
        supervisor_decision: Class of the last Supervisor decision
            ("finalize" | "revalidate" | "retry_scribe" | "retry_coding" |
            "human_review"), reported as metadata so a run can be audited
            without replaying it.
        auto_fix_rounds: How many times the Supervisor mutated `coding_result`
            itself. Bounds deterministic auto-fixes independently of
            `retry_count`, which counts only LLM passes (P1 control flow).
        revalidation_passes: How many validation passes ran against a mutated
            `coding_result`, i.e. how many times the post-fix re-check
            actually executed rather than being skipped.
        revalidation_stalls: How many times a re-validation was demanded while
            a fix was still pending - that is, how often a validation pass
            failed to consume the pending marker. It is the bound that keeps the
            fix/re-validate loop finite even if the validating node is replaced
            by one that does not clear `pending_revalidation`.
        pending_revalidation: True from the moment the Supervisor changes codes
            until `validation_node` runs again. The Supervisor refuses to
            finalize while it is set, so an auto-fix can never be signed off on
            the validation that predates it.
        cycle_flag_resets: How many times the Supervisor explicitly cleared the
            per-cycle flag buckets, i.e. how many passes were protected from
            stale-flag re-triggering.
        cleared_cycle_flags: The flags removed from the routing view by the last
            reset, kept as data so "what the previous pass said" stays auditable
            while being unable to route anything.
    """

    # --- Inputs -------------------------------------------------------------
    patient_context: str
    chief_complaint: str
    specialty: str
    visit_context: Dict[str, Any]
    retrieved_notes: str
    mode: str
    model_name: Optional[str]
    patient_id: Optional[str]
    use_memory: bool
    physician_feedback: Optional[str]
    is_synthetic: bool
    allow_real_phi: bool

    # --- Intermediate / final artifacts -------------------------------------
    soap_note: Dict[str, Any]
    cleaned_soap_note: Dict[str, Any]
    coding_result: Dict[str, Any]
    quality_flags: Annotated[List[str], append_unique]
    cleanup_removed: Annotated[List[str], append_unique]
    errors: Annotated[List[str], append_unique]

    # Current-cycle quality flags, per producing stage. Overwritten every pass,
    # so the Supervisor judges only the latest scribe/cleanup/coding/validation
    # cycle and cannot be re-triggered by a stale flag from an earlier retry.
    # `reset_cycle_flags` makes that overwrite explicit before every retry.
    cycle_flags: Annotated[Dict[str, Any], merge_dicts]
    # The last reset's contents, as data. Never read for routing.
    cleared_cycle_flags: List[str]

    # --- Memory -------------------------------------------------------------
    patient_summary: Dict[str, Any]
    memory_context: str
    memory_saved: bool
    memory_visit_id: str
    memory_visit_status: str

    # --- Tooling ------------------------------------------------------------
    grounding_facts: Annotated[Dict[str, Any], merge_dicts]
    tool_results: Annotated[Dict[str, Any], merge_tool_results]

    # --- Supervisor (retry / escalation) ------------------------------------
    retry_count: int
    feedback: Annotated[List[str], append_unique]
    needs_human_review: bool
    supervisor_applied: Annotated[List[str], append_unique]
    supervisor_symptom_codes: Annotated[List[str], append_unique]
    route_to: str
    supervisor_decision: str
    escalation_reason: str
    auto_fix_rounds: int
    revalidation_passes: int
    revalidation_stalls: int
    pending_revalidation: bool
    cycle_flag_resets: int
    empty_soap_retries: int

    # --- Observability ------------------------------------------------------
    messages: Annotated[List[AnyMessage], add_messages]
    current_step: str
    timing: Annotated[Dict[str, float], merge_dicts]
    total_seconds: float
    started_at: float

    # --- Fast Path / Slow Path metadata -------------------------------------
    coding_path: str
    coding_path_decision: Dict[str, Any]


def build_initial_state(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str = "",
    mode: str = "fast",
    model_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    use_memory: bool = True,
    started_at: Optional[float] = None,
    physician_feedback: Optional[str] = None,
    visit_context: Optional[Dict[str, Any]] = None,
    specialty: Optional[str] = None,
    is_synthetic: bool = False,
    allow_real_phi: bool = False,
) -> AgentState:
    """Create the initial state for a visit.

    Keeps the same argument surface as the legacy orchestrator so existing
    callers (Streamlit, scripts, tests) keep working unchanged.

    Args:
        visit_context: Optional clinician-provided encounter data (HPI, vitals,
            exam, additional findings). Normalised through
            `agents.visit_context.normalize_visit_context`, so omitted fields
            simply stay absent.
        specialty: Selected medical specialty ID. Defaults to "general".
    """
    from agents.visit_context import normalize_visit_context
    from utils.config import DEFAULT_MODE
    from agents.specialty_profiles import get_default_specialty_id

    return AgentState(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        specialty=specialty or get_default_specialty_id(),
        visit_context=normalize_visit_context(visit_context),
        retrieved_notes=retrieved_notes,
        mode=mode or DEFAULT_MODE,
        model_name=model_name,
        patient_id=patient_id,
        use_memory=use_memory,
        physician_feedback=physician_feedback,
        soap_note={},
        cleaned_soap_note={},
        coding_result={},
        quality_flags=[],
        cleanup_removed=[],
        errors=[],
        cycle_flags={},
        cleared_cycle_flags=[],
        messages=[],
        current_step="start",
        timing={},
        grounding_facts={},
        tool_results={},
        memory_context="",
        memory_saved=False,
        memory_visit_id="",
        memory_visit_status="",
        retry_count=0,
        feedback=[],
        needs_human_review=False,
        supervisor_applied=[],
        supervisor_symptom_codes=[],
        route_to="",
        supervisor_decision="",
        escalation_reason="",
        auto_fix_rounds=0,
        revalidation_passes=0,
        revalidation_stalls=0,
        pending_revalidation=False,
        cycle_flag_resets=0,
        empty_soap_retries=0,
        started_at=time.time() if started_at is None else started_at,
        is_synthetic=is_synthetic,
        allow_real_phi=allow_real_phi,
        coding_path="",
        coding_path_decision={},
    )


__all__ = [
    "CYCLE_FLAG_BUCKETS",
    "AgentState",
    "DETERMINISTIC_STEPS",
    "PIPELINE_STEPS",
    "PIPELINE_VERSION",
    "ROUTE_FINALIZE",
    "ROUTE_HUMAN_REVIEW",
    "ROUTE_RETRY_CODING",
    "ROUTE_RETRY_SCRIBE",
    "ROUTE_REVALIDATE",
    "append_unique",
    "build_initial_state",
    "merge_dicts",
    "merge_tool_results",
    "reset_cycle_flags",
]
