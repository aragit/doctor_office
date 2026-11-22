"""
scribe_node - LangGraph node wrapping the Scribe Agent.

Runs the existing Scribe Agent logic (prompting + JSON repair) and returns a
partial state update. All clinical grounding rules, retrieved-note filtering
and the strict ICD-10 mapping rules live in `agents.scribe_agent` and are
therefore unchanged by the tools/memory migration.

Tool use is **deterministic and supervisor-controlled**: the node runs its
scoped toolset in a fixed order and injects the results as a verified facts
block. qwen2.5:1.5b is not asked to decide which tools to call, which keeps
the small model from skipping a grounding step.

Deterministic advisory checks raised here (all non-retrying, so the physician
decides):
- `build_defaulted_content_flag` - a section the model omitted was defaulted empty.
- `build_invented_findings_flag` - a measurement the clinician never supplied.
- `build_specialty_coverage_flag` - documented specialty fields (e.g. ECG notes)
  that the tagged SOAP section does not carry, or a cardiac claim that the chart
  does not support (Phase A2.1).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState
from agents.scribe_agent import generate_soap_note
from agents.specialty_coverage import build_specialty_coverage_flag
from agents.visit_context import (
    find_unprovided_vitals,
    normalize_visit_context,
    visit_context_to_grounding_text,
)
from tools.registry import SCRIBE_AGENT, call_tool

logger = logging.getLogger(__name__)

# Budgets for the prompt's grounding block. Fast mode runs a small model on CPU,
# so the block is capped - but it is capped *per section*, never with one blind
# cut, because the RULES line and the prior-visit history are the two parts a
# truncated cut destroys silently (see `build_grounding_block`).
MAX_FACTS_CHARS = 500
MAX_HISTORY_CHARS = 400

FACT_HEADER = "VERIFIED PATIENT DATA (extracted by tools - authoritative):"
RULES_LINE = (
    "RULES: Only reference conditions, medications and symptoms listed above "
    "or in the Chief Complaint. Any condition not listed is a hallucination."
)

# facts key -> prompt label, in the order they are rendered.
_FACT_LISTS = (
    ("conditions", "Active conditions"),
    ("medications", "Current medications"),
    ("allergies", "Allergies"),
)

# How many entries per fact list to try, largest first, until the facts section
# fits its budget. Overflow is always reported, never silently dropped.
_ITEM_CAPS = (8, 6, 4, 3, 2, 1)

TRUNCATION_MARKER = "[truncated]"


def _gather_grounding_facts(state: AgentState) -> Dict[str, Any]:
    """Run the Scribe node's tools and collect verified chart facts.

    Any tool failure is logged and skipped; a visit must never fail because a
    convenience tool misbehaved.
    """
    patient_context = state["patient_context"]
    facts: Dict[str, Any] = {}

    def _safe(tool_name: str, **kwargs: Any):
        try:
            return call_tool(tool_name, agent=SCRIBE_AGENT, patient_context=patient_context, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Scribe tool %s failed: %s", tool_name, exc)
            return None

    conditions = _safe("get_patient_conditions")
    if conditions is not None:
        facts["conditions"] = conditions

    medications = _safe("get_patient_medications")
    if medications is not None:
        facts["medications"] = medications

    allergies = _safe("get_patient_allergies")
    if allergies is not None:
        facts["allergies"] = allergies

    return facts


def _render_list(values: List[str], max_items: int) -> str:
    """Join fact entries, keeping whole items and marking any omitted tail."""
    entries = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not entries:
        return "NONE DOCUMENTED"
    if len(entries) <= max_items:
        return "; ".join(entries)
    omitted = len(entries) - max_items
    return "; ".join(entries[:max_items]) + f" [{omitted} more]"


def _trim_to_char_limit(text: str, limit: int, marker: str = TRUNCATION_MARKER) -> str:
    """Cut on a word boundary and mark the cut, never mid-word."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    cut = clipped.rfind(" ")
    if cut > limit // 2:
        clipped = clipped[:cut]
    return clipped.rstrip(" .,;:-") + marker


def _render_facts_section(facts: Dict[str, Any], budget: int) -> str:
    """Render the tool-derived facts so the section fits `budget` whole.

    The item cap shrinks (8 -> 1) until the section fits; each reduction reports
    how many entries it omitted. The header and the RULES line are never
    truncated: they are what makes the facts authoritative, so an unusually long
    chart loses detail rather than the instruction that governs it.
    """
    def build(max_items: int) -> str:
        rows = [
            f"- {label}: {_render_list(facts.get(key), max_items)}"
            for key, label in _FACT_LISTS
        ]
        return "\n".join([FACT_HEADER] + rows)

    section = build(_ITEM_CAPS[0])
    for max_items in _ITEM_CAPS:
        section = build(max_items)
        if len(section) + 1 + len(RULES_LINE) <= budget:
            return section

    # Pathological single values (one very long condition string): trim the rows
    # and still emit the RULES line in full.
    return _trim_to_char_limit(section, budget - len(RULES_LINE) - 1)


def _fit_history_block(memory_context: str, limit: int) -> str:
    """Keep the history header and the newest entries that fit `limit`.

    The store renders prior visits oldest-first, so the tail is the most recent
    history: overflow drops the oldest entries instead of cutting mid-sentence,
    and every drop is announced.
    """
    lines = [line for line in str(memory_context or "").splitlines() if line.strip()]
    if not lines:
        return ""

    header, body = lines[0], lines[1:]
    kept: List[str] = []
    used = len(header)
    for line in reversed(body):
        cost = len(line) + (1 if kept else 0)
        if used + cost > limit:
            break
        kept.append(line)
        used += cost
    kept.reverse()

    if len(kept) < len(body):
        omitted = len(body) - len(kept)
        kept.insert(0, f"[{omitted} older history line(s) omitted]")
    return "\n".join([header] + kept)


def build_grounding_block(
    facts: Dict[str, Any], memory_context: str = ""
) -> str:
    """Render tool output as a compact, authoritative prompt block.

    Each section gets its own budget and is trimmed on a whole-line/whole-item
    boundary, so the anti-hallucination RULES line survives intact and prior-visit
    history loses its oldest entries rather than its last sentence.

    Args:
        facts: Output of `_gather_grounding_facts`.
        memory_context: Prior-visit text from long-term memory (optional).

    Returns:
        Prompt text, or an empty string when nothing was derived.
    """
    lines: List[str] = []

    if facts:
        lines.append(_render_facts_section(facts, MAX_FACTS_CHARS))
        lines.append(RULES_LINE)

    if memory_context:
        if lines:
            lines.append("")
        history = _fit_history_block(memory_context, MAX_HISTORY_CHARS)
        if history:
            lines.append(history)

    return "\n".join(lines)


def build_defaulted_content_flag(soap_note: Any) -> List[str]:
    """Flag any SOAP section or ICD-10 list that had to be defaulted to empty.

    The scribe agent now tolerates keys the model omits (``icd10_suggestions``
    being the usual one). That keeps the visit alive, but a note defaulted to
    empty must never look clean, so the Physician Review tab gets an advisory
    flag instead of silently showing no codes.
    """
    gaps: List[str] = []
    if not getattr(soap_note, "icd10_suggestions", None):
        gaps.append("no ICD-10 suggestions")
    for section in ("subjective", "objective", "assessment", "plan"):
        if not str(getattr(soap_note, section, "") or "").strip():
            gaps.append(f"empty {section}")
    if not gaps:
        return []
    return [f"Scribe content incomplete: {', '.join(gaps)} (defaulted by the pipeline)"]


def build_invented_findings_flag(
    soap_note: Any,
    visit_context: Optional[Dict[str, Any]] = None,
    patient_context: str = "",
) -> List[str]:
    """Flag vital signs in the note that the clinician never provided.

    The Scribe is told not to invent exam findings or measurements; this is the
    deterministic backstop for the numeric ones. Values repeated from the chart
    (e.g. last visit's blood pressure) count as provided, so this only fires on
    genuinely unsupported measurements.
    """
    if not visit_context:
        return []

    provided = "\n".join(
        part for part in (visit_context_to_grounding_text(visit_context), patient_context) if part
    )
    if not provided:
        return []

    invented = find_unprovided_vitals(
        f"{soap_note.get('objective', '')}\n{soap_note.get('assessment', '')}", provided
    )
    if not invented:
        return []
    return [
        "Objective contains measurements not provided at this visit: "
        + ", ".join(invented)
        + " - verify against the chart"
    ]


def scribe_node(state: AgentState) -> dict:
    """Generate a SOAP note from patient context + chief complaint.

    Args:
        state: Current graph state.

    Returns:
        Partial state update containing the raw SOAP note, the verified
        grounding facts, per-node timing and an agent trace message.
    """
    started = time.time()

    # 1. Tools first - verified facts ground everything downstream.
    facts = _gather_grounding_facts(state)
    grounding_block = build_grounding_block(facts, state.get("memory_context", ""))

    # 2. Fold the verified facts into the context the Scribe Agent receives.
    #    The tool reads the same context string, so it cannot contradict the
    #    chart - it only makes the grounding explicit and machine-verified.
    augmented_context = state["patient_context"]
    auxiliary: List[str] = []
    if grounding_block:
        auxiliary.append(grounding_block)
    feedback = list(state.get("feedback") or [])
    if feedback:
        auxiliary.append(
            "SUPERVISOR FEEDBACK FROM THE PREVIOUS PASS (fix every item):\n"
            + "\n".join(f"- {item}" for item in feedback)
        )
    physician_feedback = state.get("physician_feedback")
    if physician_feedback:
        auxiliary.append(
            "PHYSICIAN REVISION FEEDBACK (address in this revision):\n"
            + physician_feedback
        )
    if auxiliary:
        augmented_context = f"{augmented_context}\n\n" + "\n\n".join(auxiliary)

    # 3. Encounter data the clinician entered (HPI, vitals, exam, tests).
    #    Passed separately so the prompt can label it as the authoritative
    #    source for Subjective/Objective.
    visit_context = normalize_visit_context(state.get("visit_context"))
    if visit_context:
        logger.info("scribe_node received visit context: %s", sorted(visit_context))

    soap_note = generate_soap_note(
        patient_context=augmented_context,
        retrieved_notes=state.get("retrieved_notes", ""),
        chief_complaint=state["chief_complaint"],
        mode=state.get("mode", "fast"),
        model_name=state.get("model_name"),
        visit_context=visit_context,
        specialty=state.get("specialty"),
    )

    elapsed = round(time.time() - started, 1)
    logger.info("scribe_node completed in %ss (grounding block: %s chars)", elapsed, len(grounding_block))

    quality_flags = build_defaulted_content_flag(soap_note)
    quality_flags += build_invented_findings_flag(
        soap_note.model_dump(),
        visit_context,
        state.get("patient_context", ""),
    )
    quality_flags += build_specialty_coverage_flag(
        soap_note.model_dump(),
        visit_context,
        state.get("patient_context", ""),
    )
    if quality_flags:
        logger.info("scribe_node raised %d advisory flag(s): %s", len(quality_flags), quality_flags)

    return {
        "soap_note": soap_note.model_dump(),
        "grounding_facts": facts,
        "tool_results": {
            SCRIBE_AGENT: {
                "get_patient_conditions": facts.get("conditions", []),
                "get_patient_medications": facts.get("medications", []),
                "get_patient_allergies": facts.get("allergies", []),
            }
        },
        "quality_flags": quality_flags,
        "current_step": "scribe",
        "timing": {"scribe_seconds": elapsed},
        "messages": [
            AIMessage(
                content=(
                    f"Scribe Agent ran {len(facts)} grounding tool(s) and generated a SOAP note "
                    f"with {len(soap_note.icd10_suggestions)} ICD-10 suggestion(s) in {elapsed}s"
                    + (f" using {len(visit_context)} visit-context field(s)." if visit_context else ".")
                ),
                name="scribe_node",
            )
        ],
    }


__all__ = [
    "build_defaulted_content_flag",
    "build_grounding_block",
    "build_invented_findings_flag",
    "scribe_node",
]
