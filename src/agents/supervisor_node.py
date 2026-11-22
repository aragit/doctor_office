"""
supervisor_node - deterministic Supervisor routing for the visit graph.

The Supervisor is a zero-LLM node. It runs after validation and decides the
next step strictly by reading state:

    clean                    -> memory_save (finalize)
    fixable symptom gap      -> apply the missing symptom code via the tool,
                                then re-validate (no LLM involved)
    hallucination            -> retry scribe with feedback
    ungrounded code / CPT    -> retry coding with feedback
    retries exhausted        -> route to human_review, set needs_human_review

Two invariants are enforced here, both about control flow:

- **Stale flags cannot route.** Before any retry, every `cycle_flags` bucket is
  written empty, so the flags of the pass being corrected cannot consume the
  retry budget, trigger a retry the new pass does not need, or escalate a visit
  that has already been fixed. `quality_flags` stays as an audit log and is
  never a routing input.
- **A mutation is always re-validated.** When the Supervisor changes
  `coding_result`, the flags it just decided on describe the *previous* result.
  It clears the affected buckets and sets `pending_revalidation`, which blocks
  finalization until `validation_node` runs against the new codes. Bounded by
  `MAX_REVALIDATIONS` (auto-fix rounds) and `MAX_RETRIES` (LLM passes), so no
  loop can grow without limit.

The `feedback` list is carried in state, so the Scribe/Coding nodes can see
exactly what the previous pass got wrong. Because feedback uses the
`append_unique` reducer, repeated passes never duplicate messages.

All decision logic lives in `supervisor_policy.SupervisorPolicy` so the
routing rules are reusable and unit testable without running the graph or
an LLM.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from agents.graph_state import (
    ROUTE_FINALIZE,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RETRY_CODING,
    ROUTE_RETRY_SCRIBE,
    ROUTE_REVALIDATE,
    AgentState,
    reset_cycle_flags,
)
from agents.supervisor_policy import (
    MAX_RETRIES,
    MAX_REVALIDATIONS,
    SupervisorDecision,
    SupervisorPolicy,
    VisitSupervisorPolicy,
    _decision_flags,
    classify_flags,
    determine_route,
    missing_symptom_codes,
    route_after_validation,
)
from tools.registry import SUPERVISOR_AGENT

logger = logging.getLogger(__name__)


# Default policy instance for the visit pipeline.
_DEFAULT_POLICY = VisitSupervisorPolicy()


def _get_policy(state: AgentState) -> SupervisorPolicy:
    """Extract a policy from state or use the default visit policy.

    Future pipeline variants can inject a custom policy via state["supervisor_policy"].
    """
    policy = state.get("supervisor_policy")
    if isinstance(policy, SupervisorPolicy):
        return policy
    return _DEFAULT_POLICY


def supervisor_node(state: AgentState) -> dict:
    """Inspect validation output and route / correct / escalate.

    This node either (a) fixes a deterministically-fixable issue and asks for
    re-validation, (b) sends targeted feedback to the Scribe or Coding node,
    (c) escalates to human review, or (d) finalizes to memory save.

    Two invariants are enforced here, both about control flow rather than
    clinical content:

    1. **A mutation is always re-validated.** When this node changes
       `coding_result`, the flags it just decided on describe the *previous*
       result. It therefore clears the affected buckets and sets
       `pending_revalidation`, which keeps the Supervisor from finalizing
       until `validation_node` has run against the new codes.
    2. **A retry cannot inherit the previous pass's flags.** Before routing
       back to the Scribe or Coding node, every per-cycle bucket is explicitly
       emptied, so a stale flag cannot consume the retry budget, force a retry
       the new pass does not need, or escalate a visit that has been corrected.

    Args:
        state: Current graph state (must have run validation).

    Returns:
        Partial state update with routing signal, feedback, retry accounting,
        flag-bucket lifecycle, and any deterministic fixes applied.
    """
    started = time.time()

    policy = _get_policy(state)
    flags = _decision_flags(state)
    pending_revalidation = bool(state.get("pending_revalidation", False))
    revalidation_passes = int(state.get("revalidation_passes", 0) or 0)
    revalidation_stalls = int(state.get("revalidation_stalls", 0) or 0)

    decision = policy.determine_route(
        flags,
        int(state.get("retry_count", 0) or 0),
        state.get("coding_result"),
        state.get("chief_complaint", ""),
        pending_revalidation=pending_revalidation,
        revalidation_passes=revalidation_passes,
        revalidation_stalls=revalidation_stalls,
    )

    route_to = decision.route_to
    feedback = decision.feedback
    symptom_codes = decision.symptom_codes
    escalated = decision.escalated

    update: Dict[str, Any] = {
        "route_to": route_to,
        "supervisor_decision": decision.class_,
        "feedback": feedback,
        "current_step": "supervisor",
        "timing": {"supervisor_seconds": round(time.time() - started, 1)},
        "retry_count": int(state.get("retry_count", 0) or 0),
        "needs_human_review": bool(state.get("needs_human_review", False)),
        "messages": [AIMessage(content=f"Supervisor routed to {route_to}.", name="supervisor_node")],
    }

    # Apply deterministic fixes (e.g., symptom-code completion).
    auto_fix = policy.apply_auto_fixes(state, decision)
    mutated = bool(auto_fix) and "coding_result" in auto_fix
    if auto_fix:
        update.update(auto_fix)

    if pending_revalidation and decision.class_ == "revalidate":
        # Nobody cleared the marker, so this re-validation is a repeat demand.
        # Counting them is what makes the loop finite when it happens.
        update["revalidation_stalls"] = revalidation_stalls + 1

    if mutated:
        # The codes changed, so the flags that triggered the fix are stale by
        # definition: empty both affected buckets and demand a fresh
        # validation before anything else may be decided.
        update["cycle_flags"] = {"coding": [], "validation": []}
        update["cleared_cycle_flags"] = list(flags)
        update["cycle_flag_resets"] = (
            int(state.get("cycle_flag_resets", 0) or 0) + 1
        )
        update["auto_fix_rounds"] = (
            int(state.get("auto_fix_rounds", 0) or 0) + 1
        )
        update["pending_revalidation"] = True
        update["messages"] = [AIMessage(
            content=(
                f"Supervisor auto-fixed {len(symptom_codes)} symptom code(s) - "
                f"re-validating before any decision (re-validation "
                f"{revalidation_passes + 1}/{MAX_REVALIDATIONS})."
            ),
            name="supervisor_node",
        )]
    elif decision.class_ in ("retry_scribe", "retry_coding"):
        # The next pass starts from an empty flag view: nothing this pass found
        # may drive the decision that follows the retry.
        update["cycle_flags"] = reset_cycle_flags()
        update["cleared_cycle_flags"] = list(flags)
        update["cycle_flag_resets"] = int(state.get("cycle_flag_resets", 0) or 0) + 1
        update["retry_count"] = update["retry_count"] + 1
        update["messages"] = [AIMessage(
            content=(
                f"Supervisor: retry #{update['retry_count']} (max {MAX_RETRIES}) -> "
                f"{route_to} with {len(feedback)} feedback item(s); "
                f"{len(flags)} stale flag(s) cleared."
            ),
            name="supervisor_node",
        )]

    if escalated:
        update["needs_human_review"] = True
        update["escalation_reason"] = decision.escalation_reason or "max_retries"
        update["quality_flags"] = [_escalation_flag(decision)]
        update["messages"] = [AIMessage(
            content=(
                f"Supervisor: {decision.escalation_reason or 'retries exhausted'} - "
                f"escalation to human review."
            ),
            name="supervisor_node",
        )]

    status = {
        "finalize": "clean - finalizing",
        "revalidate": "auto-corrected - re-validating",
        "retry_scribe": f"retry scribe (count={update['retry_count']})",
        "retry_coding": f"retry coding (count={update['retry_count']})",
        "human_review": "human review",
    }[decision.class_]
    logger.info(
        "supervisor_node: %s (%d current-cycle flag(s), re-validations %d, %s)",
        status, len(flags), revalidation_passes,
        "pending re-validation" if pending_revalidation else "validated result",
    )

    return update


def _escalation_flag(decision: SupervisorDecision) -> str:
    """The quality flag that explains *why* a visit reached a human.

    Names the real cause, so an escalation is never attributed to exhausted LLM
    retries when the deterministic auto-fix was what ran out of road.
    """
    if decision.escalation_reason == "revalidation_exhausted":
        return (
            "Supervisor auto-fix did not resolve the validation gap within "
            f"{MAX_REVALIDATIONS} re-validation(s); visit requires human review"
        )
    return "Max retries reached; visit requires human review"


def human_review_node(state: AgentState) -> dict:
    """Terminal node marking a visit that requires human attention.

    A human-reviewed visit is deliberately NOT written to long-term memory:
    a note a human has not signed should not seed the next visit's context.

    Args:
        state: Current graph state (should have needs_human_review=True).

    Returns:
        Partial state update marking the step and keeping memory unpersisted.
    """
    started = time.time()
    return {
        "current_step": "human_review",
        "needs_human_review": True,
        "memory_saved": False,
        "timing": {"human_review_seconds": round(time.time() - started, 1)},
        "messages": [
            AIMessage(
                content=(
                    "Visit escalated for human review; not written to long-term memory. "
                    "Review the quality flags, correct the note/codes, then sign."
                ),
                name="human_review_node",
            )
        ],
    }


__all__ = [
    "MAX_RETRIES",
    "MAX_REVALIDATIONS",
    "classify_flags",
    "determine_route",
    "human_review_node",
    "missing_symptom_codes",
    "supervisor_node",
    "route_after_validation",
]