"""
Supervisor Policy Abstraction - reusable decision logic for visit pipelines.

This module extracts the zero-LLM Supervisor routing logic into a policy
interface so that different pipeline variants (visit, medication, orders,
inbox) can share the same retry/escalation machinery while plugging in
their own flag classification and auto-fix rules.

The base `SupervisorPolicy` defines the contract. `VisitSupervisorPolicy`
implements the current visit-documentation behaviour exactly as it existed
before this refactor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from agents.graph_state import AgentState

# Import the shared symptom map (canonical codes come from the central module)
from tools.coding_tools import PREFIX_TO_SYMPTOM_CODE, SYMPTOM_CODES
from utils.symptom_codes import (
    AUTO_ADDABLE_PREFIXES,
    symptom_phrase_present,
)


# --- Re-export route constants so callers don't need graph_state ---
from agents.graph_state import (
    CYCLE_FLAG_BUCKETS,
    ROUTE_FINALIZE,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RETRY_CODING,
    ROUTE_RETRY_SCRIBE,
    ROUTE_REVALIDATE,
)

__all__ = [
    "ROUTE_FINALIZE",
    "ROUTE_HUMAN_REVIEW",
    "ROUTE_RETRY_CODING",
    "ROUTE_RETRY_SCRIBE",
    "ROUTE_REVALIDATE",
    "MAX_RETRIES",
    "MAX_REVALIDATIONS",
    "SupervisorDecision",
    "SupervisorPolicy",
    "VisitSupervisorPolicy",
    "classify_flags",
    "determine_route",
    "missing_symptom_codes",
    "_decision_flags",
    "route_after_validation",
]


# Maximum number of LLM re-passes before escalating to human review.
MAX_RETRIES = 2

# Maximum number of empty-SOAP retries before escalating to human review.
MAX_EMPTY_SOAP_RETRIES = 1

# Maximum number of *deterministic* auto-fix rounds the Supervisor may run.
#
# Counted separately from MAX_RETRIES because an auto-fix costs no LLM call: it
# adds a symptom code the patient already stated and then re-validates. One round
# is sufficient by construction - validation derives its symptom-gap flags from
# the chief complaint alone, so every gap it can raise is visible in the first
# validation pass and all of them are completed together. A second round
# therefore means the deterministic fix and validation disagree about the same
# visit, which is a human decision, not another loop.
MAX_REVALIDATIONS = 1

# Quality-flag classification markers (matches run_full_validation output).
_HALLUCINATION_MARKERS = (
    "possible hallucinated diagnosis",
)
_UNGROUNDED_MARKERS = (
    "ungrounded icd-10",
    "tool check:",
)
_CPT_STATUS_MARKERS = (
    "is for new patients",
    "new patient",
    "consider 992",
)
_SYMPTOM_GAP_RE = re.compile(r"no (R\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class SupervisorDecision:
    """Structured decision returned by a SupervisorPolicy.

    Attributes:
        class_: High-level decision class: "finalize" | "revalidate" |
            "retry_scribe" | "retry_coding" | "human_review".
        route_to: Graph node name to execute next.
        feedback: Targeted feedback messages for the next LLM pass (if any).
        escalated: Whether the decision escalates to human review.
        symptom_codes: Deterministically added symptom ICD-10 codes (if any).
        escalation_reason: Why an escalation happened ("max_retries" |
            "revalidation_exhausted"), so the quality flag the node records
            names the real cause instead of always claiming retries ran out.

    Note:
        Supports dict-style access for backwards compatibility with existing tests.
    """
    class_: str
    route_to: str
    feedback: List[str]
    escalated: bool
    symptom_codes: List[str]
    escalation_reason: str = ""

    def __getitem__(self, key: str) -> Any:
        if key == "class":
            return self.class_
        if key == "route_to":
            return self.route_to
        if key == "feedback":
            return self.feedback
        if key == "escalated":
            return self.escalated
        if key == "symptom_codes":
            return self.symptom_codes
        if key == "escalation_reason":
            return self.escalation_reason
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in (
            "class",
            "route_to",
            "feedback",
            "escalated",
            "symptom_codes",
            "escalation_reason",
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@runtime_checkable
class SupervisorPolicy(Protocol):
    """Protocol for Supervisor decision logic.

    Different pipeline types (visit, medication, orders, inbox) implement
    this protocol to plug their own flag classification, auto-fix, and
    retry semantics into the shared supervisor node.
    """

    def classify_flags(self, quality_flags: List[str]) -> Dict[str, List[str]]:
        """Bucket validation flags by which node can fix them.

        Returns a dict with at minimum keys: "scribe", "coding", "symptom",
        "advisory". Implementations may add more buckets.
        """
        ...

    def determine_route(
        self,
        quality_flags: List[str],
        retry_count: int,
        coding_result: Optional[Dict[str, Any]],
        chief_complaint: str = "",
        max_retries: int = MAX_RETRIES,
        *,
        pending_revalidation: bool = False,
        revalidation_passes: int = 0,
        revalidation_stalls: int = 0,
        max_revalidations: int = MAX_REVALIDATIONS,
    ) -> SupervisorDecision:
        """Decide the next step given validation output.

        Pure function - no graph or LLM access - so it is directly unit-testable.
        """
        ...

    def apply_auto_fixes(
        self,
        state: AgentState,
        decision: SupervisorDecision,
    ) -> Dict[str, Any]:
        """Apply any deterministic fixes (e.g. symptom code completion) to state.

        Returns a partial state update dict.
        """
        ...

    def build_retry_feedback(
        self,
        scribe_flags: List[str],
        coding_flags: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Turn flag buckets into short feedback messages for each node.

        Returns (scribe_feedback, coding_feedback).
        """
        ...


# --- Shared pure functions (identical to pre-refactor behaviour) ---


def classify_flags(quality_flags: List[str]) -> Dict[str, List[str]]:
    """Bucket validation flags by which node can fix them.

    Returns:
        Dict with `scribe`, `coding`, `symptom` (fixable deterministically),
        and `advisory` (not retroactive / already handled) buckets.
    """
    buckets: Dict[str, List[str]] = {"scribe": [], "coding": [], "symptom": [], "advisory": []}
    for flag in quality_flags or []:
        lowered = flag.lower()
        if any(m in lowered for m in _HALLUCINATION_MARKERS):
            buckets["scribe"].append(flag)
        elif "soap_empty" in lowered:
            buckets["scribe"].append(flag)
        elif any(m in lowered for m in _UNGROUNDED_MARKERS):
            buckets["coding"].append(flag)
        elif any(m in lowered for m in _CPT_STATUS_MARKERS):
            buckets["coding"].append(flag)
        elif "code was generated" in lowered and "mentions" in lowered:
            buckets["symptom"].append(flag)
        else:
            buckets["advisory"].append(flag)
    return buckets


def missing_symptom_codes(
    quality_flags: List[str],
    chief_complaint: str,
    current_codes: List[str],
) -> List[str]:
    """Find symptom ICD codes that validation flagged as missing and that this
    patient's own complaint justifies.

    Deterministic completion: if the chief complaint says "chest pain" and
    validation says no R07 code was generated, we re-insert
    "R07.9 - Chest pain, unspecified" as a secondary code. This is exactly
    what the previous advisory asked a human coder to consider, and the code
    is patient-stated, so completing it in code is safe and auditable.

    This is the *fallback*, not the primary path: Phase A3 briefs the Coding
    agent with the same candidates so it normally emits the code itself, and
    this function exists for the codes that still fall through. The
    `AUTO_ADDABLE_PREFIXES` allow-list keeps a disease code out of this list
    even if a future flag names one - only signs and symptoms are completed
    automatically, and a diagnosis always comes from the chart.

    Returns:
        List of "CODE - Description" strings to append to secondary_icd10.
    """
    complaint = (chief_complaint or "").lower()
    prefixes = {c[:3].upper() for c in current_codes if c}

    additions: List[str] = []
    for flag in quality_flags or []:
        match = _SYMPTOM_GAP_RE.search(flag)
        if not match:
            continue
        prefix = match.group(1).upper()
        if prefix in prefixes or prefix not in AUTO_ADDABLE_PREFIXES:
            continue
        code = PREFIX_TO_SYMPTOM_CODE.get(prefix)
        # Only complete codes whose symptom phrase is actually present.
        if code and _symptom_phrase_present(prefix, complaint):
            additions.append(SYMPTOM_CODES[code])
            prefixes.add(prefix)
    return additions


def _symptom_phrase_present(prefix: str, complaint: str) -> bool:
    """True when the chief complaint actually contains the symptom.

    Delegates to the central map so any phrase counts, not just the single one
    this used to hard-code per prefix - which is why a complaint saying
    "dyspnea" could raise an R06 gap flag that nothing would then complete.
    """
    return symptom_phrase_present(prefix, complaint)


def _feedback_for(scribe_flags: List[str], coding_flags: List[str]) -> Tuple[List[str], List[str]]:
    """Turn flag buckets into short feedback messages for each node."""
    scribe_feedback = []
    if scribe_flags:
        scribe_feedback.append(
            "Only reference conditions, medications and symptoms that appear in "
            "the patient chart. The previous pass invented a diagnosis - rewrite "
            "the note without it and do not invent any ICD-10 codes."
        )
    for flag in scribe_flags:
        scribe_feedback.append(f"Remove ungrounded diagnosis: {flag}")

    coding_feedback = []
    for flag in coding_flags:
        coding_feedback.append(f"Remove ungrounded diagnosis: {flag}")
    if any("CPT" in f or "new patient" in f.lower() for f in coding_flags):
        coding_feedback.append(
            "CPT level must come from the deterministic 2-of-3 E/M calculator; "
            "do not guess a level. The calculator result is authoritative."
        )
    return scribe_feedback, coding_feedback


def determine_route(
    quality_flags: List[str],
    retry_count: int,
    coding_result: Optional[Dict[str, Any]],
    chief_complaint: str = "",
    max_retries: int = MAX_RETRIES,
    empty_soap_retries: int = 0,
    *,
    pending_revalidation: bool = False,
    revalidation_passes: int = 0,
    revalidation_stalls: int = 0,
    max_revalidations: int = MAX_REVALIDATIONS,
) -> SupervisorDecision:
    """Decide what the Supervisor does next, given the validation output.

    Pure function - no graph or LLM access - so it is directly unit-testable.

    Args:
        quality_flags: Validation quality flags **for the current pass only**.
            The caller is responsible for clearing stale flags (see
            `agents.graph_state.reset_cycle_flags`); this function trusts
            whatever it is given.
        retry_count: How many LLM retries have already happened.
        coding_result: Current coding result (to avoid duplicate symptom codes).
        chief_complaint: Reason for visit (used for symptom auto-completion).
        max_retries: Cap on allowed retries before escalation.
        pending_revalidation: True when the Supervisor itself mutated
            `coding_result` since validation last ran. Finalizing in that state
            would sign off the *pre-fix* result, so the only legal moves are
            re-validating, escalating, or regenerating the codes.
        revalidation_passes: How many validation passes have already run against
            a mutated `coding_result`. This is what `max_revalidations` bounds,
            so the budget is spent by *re-checking*, not by fixing.
        revalidation_stalls: How many re-validations were already demanded
            without the pending marker being cleared. Bounds the loop when the
            node responsible for clearing it does not.
        max_revalidations: Cap on re-validations of a mutated result.

    Returns:
        SupervisorDecision with route, feedback, escalation flag, and symptom codes.
    """
    coding_result = dict(coding_result or {})
    revalidated = int(revalidation_passes or 0)
    budget_spent = revalidated >= int(max_revalidations)

    # 0. A mutated coding_result has never been validated in its current form.
    #    Re-check first, whatever the flags say: they describe the result the fix
    #    just replaced, so no symptom code is added from them either. Finalizing
    #    is not an option here - that is the whole point of the flag.
    if pending_revalidation:
        # Two independent bounds, because either one alone can be starved: the
        # budget counts completed re-checks, the stall count counts demands that
        # nothing consumed.
        if budget_spent or int(revalidation_stalls or 0) >= int(max_revalidations):
            return SupervisorDecision(
                class_="human_review",
                route_to=ROUTE_HUMAN_REVIEW,
                feedback=[
                    f"Codes were auto-corrected and could not be re-validated "
                    f"within {max_revalidations} pass(es); routing to human review "
                    f"rather than finalizing an unchecked result."
                ],
                escalated=True,
                symptom_codes=[],
                escalation_reason="revalidation_exhausted",
            )
        return SupervisorDecision(
            class_="revalidate",
            route_to=ROUTE_REVALIDATE,
            feedback=["Re-validating the auto-corrected codes before finalizing."],
            escalated=False,
            symptom_codes=[],
        )

    secondary = list(coding_result.get("secondary_icd10", []) or [])
    current_codes = [str(coding_result.get("primary_icd10", ""))] + secondary

    buckets = classify_flags(quality_flags or [])
    symptom_codes = missing_symptom_codes(
        quality_flags or [], chief_complaint, current_codes
    )

    # Feedback the Coding node gets in addition to whatever its own flags
    # generated. Used when a symptom gap survives the deterministic layer.
    extra_coding_feedback: List[str] = []

    # 1. Deterministic symptom completion -> re-validate (no LLM pass).
    if symptom_codes and not budget_spent:
        return SupervisorDecision(
            class_="revalidate",
            route_to=ROUTE_REVALIDATE,
            feedback=[f"Added missing symptom code(s): {', '.join(symptom_codes)}"],
            escalated=False,
            symptom_codes=symptom_codes,
        )

    if symptom_codes:
        # The re-validation budget is spent, so the deterministic layer does not
        # get another turn: a gap that outlives it becomes a Coding problem and
        # is handed to the LLM, which is bounded by `max_retries` like any other
        # retry. Without this the two limits could ping-pong forever.
        extra_coding_feedback.append(
            f"Validation still reports a missing symptom code after the "
            f"Supervisor's deterministic completion; emit "
            f"{', '.join(symptom_codes)} explicitly in secondary_icd10."
        )

    scribe_flags, coding_flags = buckets["scribe"], buckets["coding"]

    # Handle soap_empty flag with separate retry budget
    soap_empty_flags = [f for f in scribe_flags if "soap_empty" in f.lower()]
    if soap_empty_flags:
        if int(empty_soap_retries or 0) >= MAX_EMPTY_SOAP_RETRIES:
            return SupervisorDecision(
                class_="human_review",
                route_to=ROUTE_HUMAN_REVIEW,
                feedback=[
                    f"SOAP note still empty after {MAX_EMPTY_SOAP_RETRIES} retry; "
                    f"routing to human review rather than finalizing empty note."
                ],
                escalated=True,
                symptom_codes=[],
                escalation_reason="empty_soap_exhausted",
            )
        # First empty SOAP occurrence -> retry scribe with targeted feedback
        return SupervisorDecision(
            class_="retry_scribe",
            route_to=ROUTE_RETRY_SCRIBE,
            feedback=[
                "The generated SOAP note has all four sections empty (subjective, "
                "objective, assessment, plan). Generate a complete note grounded in "
                "the patient context and chief complaint. Do not leave any section blank."
            ],
            escalated=False,
            symptom_codes=[],
        )

    # 2. A hallucination / ungrounded code needs a regenerated note or codes.
    issue_class: Optional[str] = None
    if scribe_flags:
        issue_class = "retry_scribe"
    elif coding_flags or extra_coding_feedback:
        issue_class = "retry_coding"

    if issue_class is None:
        # 3. Clean (or only advisory already handled) -> finalize.
        return SupervisorDecision(
            class_="finalize",
            route_to=ROUTE_FINALIZE,
            feedback=[],
            escalated=False,
            symptom_codes=[],
        )

    # 4. Retries exhausted -> escalate to a human.
    if int(retry_count or 0) >= max_retries:
        remaining = scribe_flags + coding_flags + extra_coding_feedback
        return SupervisorDecision(
            class_="human_review",
            route_to=ROUTE_HUMAN_REVIEW,
            feedback=[
                f"Max retries reached ({max_retries}); routing to human review. "
                f"Remaining issues: "
                + ("; ".join(remaining) or "see quality flags"),
            ],
            escalated=True,
            symptom_codes=[],
            escalation_reason="max_retries",
        )

    # 5. Retry the responsible node with targeted feedback.
    scribe_feedback, coding_feedback = _feedback_for(scribe_flags, coding_flags)
    if issue_class == "retry_scribe":
        route_to = ROUTE_RETRY_SCRIBE
        feedback = scribe_feedback
    else:
        route_to = ROUTE_RETRY_CODING
        feedback = coding_feedback + extra_coding_feedback

    return SupervisorDecision(
        class_=issue_class,
        route_to=route_to,
        feedback=feedback,
        escalated=False,
        symptom_codes=[],
    )


# --- Visit-specific policy implementation ---


class VisitSupervisorPolicy:
    """Supervisor policy for the visit documentation pipeline.

    This implementation preserves the exact behaviour of the pre-refactor
    `supervisor_node.py` logic, including cycle-flag protection, symptom-code
    auto-completion, retry accounting, and escalation to human review.
    """

    def classify_flags(self, quality_flags: List[str]) -> Dict[str, List[str]]:
        return classify_flags(quality_flags)

    def determine_route(
        self,
        quality_flags: List[str],
        retry_count: int,
        coding_result: Optional[Dict[str, Any]],
        chief_complaint: str = "",
        max_retries: int = MAX_RETRIES,
        *,
        pending_revalidation: bool = False,
        revalidation_passes: int = 0,
        revalidation_stalls: int = 0,
        max_revalidations: int = MAX_REVALIDATIONS,
        empty_soap_retries: int = 0,
    ) -> SupervisorDecision:
        return determine_route(
            quality_flags,
            retry_count,
            coding_result,
            chief_complaint,
            max_retries,
            empty_soap_retries=empty_soap_retries,
            pending_revalidation=pending_revalidation,
            revalidation_passes=revalidation_passes,
            revalidation_stalls=revalidation_stalls,
            max_revalidations=max_revalidations,
        )

    def apply_auto_fixes(
        self,
        state: AgentState,
        decision: SupervisorDecision,
    ) -> Dict[str, Any]:
        """Apply deterministic symptom-code completion on the coding result.

        The completion is idempotent: a code already present in the result is
        never appended twice, so replaying this node (LangGraph re-applies
        updates, and a retry re-enters the Supervisor) cannot inflate the code
        list. Only symptom/sign codes from the central map are ever added -
        never a diagnosis.

        Returns:
            Partial state update. Its presence is what marks the result as
            mutated, which is what forces the fresh validation pass.
        """
        symptom_codes = decision.symptom_codes
        if not symptom_codes:
            return {}

        existing = dict(state.get("coding_result") or {})
        secondary = list(existing.get("secondary_icd10") or [])
        added = [
            code for code in symptom_codes if code not in secondary and code not in existing.get("primary_icd10", "")
        ]
        if not added:
            # Already present - the fix is a no-op, so nothing about the result
            # changed and no re-validation is owed.
            return {}

        updated_coding = {
            **existing,
            "secondary_icd10": secondary + added,
        }
        return {
            "coding_result": updated_coding,
            "supervisor_applied": [
                f"Added symptom code {code} (raised by validation, present in chief complaint)"
                for code in added
            ],
            # Structured provenance for the review UI (Phase A4). Same codes,
            # recorded as data rather than prose so the UI does not parse the
            # sentence above.
            "supervisor_symptom_codes": list(added),
            "tool_results": {
                "supervisor": {
                    "simple_symptom_to_icd_suggestion": {"added": added},
                }
            },
        }

    def build_retry_feedback(
        self,
        scribe_flags: List[str],
        coding_flags: List[str],
    ) -> Tuple[List[str], List[str]]:
        return _feedback_for(scribe_flags, coding_flags)


def _decision_flags(state: AgentState) -> List[str]:
    """Flags from the *current* scribe/cleanup/coding/validation cycle.

    `cycle_flags` is the routing input and `quality_flags` is an audit log: the
    latter accumulates across retries with `append_unique`, so a stale
    hallucination flag from a previous attempt would otherwise re-trigger a retry
    on a corrected note. Every bucket is read, not just the two that happen to be
    populated today, so `reset_cycle_flags` empties exactly what this function
    can see - otherwise clearing a bucket would be theatre.

    The only fallback is a state with no `cycle_flags` at all - a manually seeded
    state or a test - where the accumulated log is the best available evidence.
    """
    cycle = dict(state.get("cycle_flags") or {})
    flags = [
        flag
        for bucket in CYCLE_FLAG_BUCKETS
        for flag in (cycle.get(bucket) or [])
    ]
    if not flags and not cycle:
        flags = list(state.get("quality_flags") or [])
    return flags


def route_after_validation(state: AgentState) -> str:
    """Conditional-edge path function: return the next node name.

    Prefers the `route_to` value set by `supervisor_node` (which also applied
    any symptom-code completion); falls back to recomputing a decision so the
    edge never misroutes on stale state. The recompute passes the same
    re-validation state the node used, so a result the Supervisor mutated can
    never be finalized by the edge either.
    """
    stored = state.get("route_to") or ""
    if stored in (ROUTE_FINALIZE, ROUTE_HUMAN_REVIEW, ROUTE_RETRY_CODING,
                  ROUTE_RETRY_SCRIBE, ROUTE_REVALIDATE):
        return stored
    decision = determine_route(
        _decision_flags(state),
        int(state.get("retry_count", 0) or 0),
        state.get("coding_result"),
        state.get("chief_complaint", ""),
        pending_revalidation=bool(state.get("pending_revalidation", False)),
        revalidation_passes=int(state.get("revalidation_passes", 0) or 0),
        revalidation_stalls=int(state.get("revalidation_stalls", 0) or 0),
    )
    return decision.route_to