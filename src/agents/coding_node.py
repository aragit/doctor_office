"""
coding_node - LangGraph node wrapping the Coding Agent.

Consumes the *cleaned* SOAP note (never the raw one) so that hallucinated
diagnoses are stripped before any ICD-10/CPT code is generated.

Tool use (deterministic, scoped to this node):
- `utils.symptom_codes.find_symptom_candidates` resolves grounded symptom-code
  candidates (chest pain -> R07.9) from the chief complaint *and* the
  clinician's visit data, suppressing any symptom a more specific charted
  diagnosis already covers. The block is injected as an authoritative prompt
  hint, so Coding normally emits the symptom code itself instead of the
  Supervisor having to add it afterwards (Phase A3).
- `validate_icd10_code` drops malformed / non-existent ICD-10 codes before
  they can reach validation (existence check, local reference table).
- `derive_mdm_domains` + `calculate_em_code` compute the CPT E/M level
  (99212-99215) via the 2-of-3 rule. The LLM's CPT guess is *overridden* by
  this deterministic result: a small model cannot talk itself into a level.
- `validate_icd_against_patient` re-checks every emitted code against the
  chart, independently of the model and of the existing validation layer.
- A narrative check scrubs cardiac diagnoses out of `medical_necessity_note`
  when the chart documents no cardiac diagnosis (Phase A2.1). Advisory: the
  codes are untouched and the visit is not retried.

Fast Path / Slow Path gate:
- Fast Path: deterministic tools only (symptom resolution, ICD validation,
  MDM/CPT calculation). Used when primary diagnosis is clear from chronic
  conditions + chief complaint, symptoms are resolvable, and documentation
  is complete. Skips the LLM Coding Agent call.
- Slow Path: current full Coding agent behavior (kept as-is).

Supervisor feedback (`state.feedback`) is injected verbatim on retry passes so
the agent can repair the exact issue the previous pass caused.
"""

import logging
import time
import re
from typing import Any, Dict, List, NamedTuple, Optional

from langchain_core.messages import AIMessage

from agents.graph_state import AgentState
from agents.coding_agent import generate_coding
from agents.coding_fast_path import (
    should_use_fast_path,
    execute_fast_path,
)
from agents.visit_context import (
    visit_context_to_grounding_text,
    visit_context_to_prompt_block,
)
from tools.registry import CODING_AGENT, call_tool
from tools.em_calculator import calculate_em_code, derive_mdm_domains
from utils.cardiac_claims import (
    has_charted_cardiac_diagnosis,
    scrub_unsupported_cardiac_claims,
)
from utils.symptom_codes import (
    AUTO_ADDABLE_PREFIXES,
    SYMPTOM_CODES,
    find_symptom_candidates,
)

logger = logging.getLogger(__name__)

MAX_SYMPTOM_GUIDANCE = 4

# Pull the ICD code out of a bare code or a "CODE - Description" string.
_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")

# Used when every sentence of the medical necessity note was an unsupported
# cardiac claim: keep a truthful summary rather than an empty field.
NECESSITY_FALLBACK = (
    "Supports the reported symptoms and management of the patient's documented "
    "chronic conditions; no additional diagnosis is supported by the chart."
)

SYMPTOM_HINT_HEADER = (
    "SYMPTOM CODE CANDIDATES (derived by tool from the chief complaint and the "
    "visit data below; each phrase is documented in this encounter):"
)
SYMPTOM_HINT_RULES = (
    "Rules for these codes: include one as SECONDARY for each symptom documented "
    "above; keep them as symptoms only, never upgrade them to a diagnosis code "
    "(chest pain stays R07.x, it does not become I20/I21); and never add a "
    "diagnosis code for a condition that is not in Patient Context."
)
FEEDBACK_HEADER = "SUPERVISOR FEEDBACK FROM THE PREVIOUS PASS (fix every item):"


def build_symptom_hint(candidates: List[Any]) -> str:
    """Render symptom code candidates as a short, authoritative prompt block.

    Args:
        candidates: `SymptomCandidate`s from `utils.symptom_codes`, or plain
            code strings (both are accepted so a stubbed candidate list keeps
            working in tests).

    Returns:
        The prompt block, or an empty string when there is nothing to hint.
    """
    if not candidates:
        return ""
    lines: List[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            lines.append(f"- {candidate}")
        else:
            lines.append(f"- {candidate.description}  [{candidate.source}]")
    return f"{SYMPTOM_HINT_HEADER}\n" + "\n".join(lines) + f"\n{SYMPTOM_HINT_RULES}"


def _symptom_guidance(
    chief_complaint: str,
    patient_context: str,
    visit_context: Any = None,
    already_coded: Optional[List[str]] = None,
) -> List[Any]:
    """Grounded symptom-code candidates for this encounter.

    Scans the chief complaint *and* the clinician's visit data (HPI, exam,
    documented findings), so a symptom written down during the visit is coded
    even when the patient did not open with it. A candidate is suppressed when a
    more specific diagnosis for the same problem is already charted, and when
    the code is already present, so nothing is proposed twice.

    This is the primary symptom-coding path in Phase A3. The Supervisor's
    auto-completion stays as the fallback for whatever the model still omits.
    """
    try:
        return find_symptom_candidates(
            chief_complaint=chief_complaint,
            visit_context=visit_context,
            patient_context=patient_context,
            already_coded=already_coded,
            max_codes=MAX_SYMPTOM_GUIDANCE,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Symptom candidate resolution failed: %s", exc)
        return []


class ExistenceCheck(NamedTuple):
    """Outcome of the ICD-10 existence/shape pre-check.

    Attributes:
        kept: The codes that survived, in their original order.
        flags: One quality flag per dropped code, naming what was removed.
        checked: How many non-empty codes were submitted to the tool.
        dropped: The codes that were removed because they are not well-formed.
    """

    kept: List[str]
    flags: List[str]
    checked: int
    dropped: List[str]


def _existence_precheck(codes: List[str]) -> ExistenceCheck:
    """Drop malformed ICD-10 codes; report what was removed.

    Uses the local reference table (`validate_icd10_code`). Codes that are
    well-formed are left alone for grounding to judge; only structurally invalid
    codes are removed here, so a code that merely is uncatalogued still reaches
    validation with its advisory flag.

    The removal is fail-open: if the tool itself errors the code is kept, because
    silently deleting a legitimate diagnosis is worse than flagging it.

    Returns:
        The surviving codes plus one quality flag per dropped code.
    """
    kept: List[str] = []
    flags: List[str] = []
    dropped: List[str] = []

    for code in codes:
        if not code:
            continue
        try:
            verdict = call_tool(
                "validate_icd10_code", agent=CODING_AGENT, text=code
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("validate_icd10_code failed for %r: %s", code, exc)
            kept.append(code)
            continue
        if verdict.get("valid"):
            kept.append(code)
            continue
        dropped.append(code)
        flags.append(
            f"Coding: dropped malformed ICD-10 code '{code}' "
            f"({verdict.get('reason', 'not well-formed')})"
        )

    return ExistenceCheck(
        kept=kept,
        flags=flags,
        checked=len([code for code in codes if code]),
        dropped=dropped,
    )


def _ungrounded_flags(
    codes: List[str], patient_context: str, chief_complaint: str
) -> List[str]:
    """Re-validate emitted codes with the grounding tool.

    This runs after generation and independently of `utils.validation`, so a
    code has to pass two separate grounding checks to reach the coder.
    """
    flags: List[str] = []
    for code in codes:
        if not code:
            continue
        try:
            verdict = call_tool(
                "validate_icd_against_patient",
                agent=CODING_AGENT,
                patient_context=patient_context,
                code=code,
                chief_complaint=chief_complaint,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("validate_icd_against_patient failed for %r: %s", code, exc)
            continue
        if not verdict.get("valid") and verdict.get("quality_flag"):
            flags.append(f"Tool check: {verdict['quality_flag']}")
    return flags


def _deterministic_cpt(soap_note: Dict[str, Any], patient_context: str) -> Dict[str, Any]:
    """Compute the authoritative CPT E/M code via the 2-of-3 MDM calculator.

    Returns:
        Dict with `cpt_description`, `cpt_code`, `overall_mdm_level`,
        `justification` and `domains` (per-element derivation).
    """
    domains = derive_mdm_domains(soap_note, patient_context)
    em = calculate_em_code(
        domains["problems_level"],
        domains["data_level"],
        domains["risk_level"],
    )
    return {
        "cpt_description": em.get("cpt_description", "99213 - Office visit, established patient, low medical decision making"),
        "cpt_code": em.get("cpt_code", "99213"),
        "overall_mdm_level": em.get("overall_mdm_level", "low"),
        "justification": em.get("justification", ""),
        "domains": domains,
    }


class SymptomCompletion(NamedTuple):
    """Outcome of the deterministic symptom-code completion.

    Attributes:
        codes: The "CODE - Description" strings that were appended.
        flags: One advisory flag per code added, naming the justification.
    """

    codes: List[str]
    flags: List[str]


def _complete_symptom_codes(
    result_dict: Dict[str, Any],
    candidates: List[Any],
) -> SymptomCompletion:
    """Add any grounded symptom code the model left out of the result.

    The prompt hint is the first line of defence; this is the backstop inside the
    Coding node, so a small model's omission cannot turn the Supervisor into the
    primary coder for a documented symptom. Symptom codes only, secondary only,
    never duplicated, advisory only.
    """
    if not candidates:
        return SymptomCompletion([], [])

    emitted = [
        str(entry)
        for entry in [result_dict.get("primary_icd10", "")]
        + list(result_dict.get("secondary_icd10") or [])
        if entry
    ]
    present = {
        match.group(1)[:3].upper()
        for match in (_CODE_RE.search(entry.upper()) for entry in emitted)
        if match
    }

    secondary = list(result_dict.get("secondary_icd10") or [])
    added: List[str] = []
    flags: List[str] = []
    for candidate in candidates:
        prefix = candidate.prefix.upper() if hasattr(candidate, "prefix") else ""
        code = candidate.code if hasattr(candidate, "code") else str(candidate)
        if prefix not in AUTO_ADDABLE_PREFIXES or prefix in present:
            continue
        present.add(prefix)
        description = (
            candidate.description
            if hasattr(candidate, "description")
            else SYMPTOM_CODES.get(code, code)
        )
        secondary.append(description)
        added.append(description)
        source = getattr(candidate, "source", "chief_complaint")
        phrase = getattr(candidate, "phrase", "")
        flags.append(
            f"Coding: added symptom code {description} "
            f"('{phrase}' documented in {source.replace('_', ' ')}); "
            "the Coding pass omitted it"
        )

    if added:
        result_dict["secondary_icd10"] = secondary
    return SymptomCompletion(added, flags)


def _run_slow_path(
    state: AgentState,
    soap_note: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    suggestions: List[Any],
    symptom_hint: str,
    visit_block: str,
    feedback: List[str],
    physician_feedback: Optional[str],
    started: float,
) -> Dict[str, Any]:
    """Execute the Slow Path (full LLM Coding Agent)."""
    # Build coding context with auxiliary information
    coding_context = patient_context
    auxiliary: List[str] = []
    if symptom_hint:
        auxiliary.append(symptom_hint)
    if visit_block:
        auxiliary.append(visit_block)
    if feedback:
        auxiliary.append(FEEDBACK_HEADER + "\n" + "\n".join(f"- {item}" for item in feedback))
    if physician_feedback:
        auxiliary.append(
            "PHYSICIAN REVISION FEEDBACK (address in this revision):\n"
            + physician_feedback
        )
    if auxiliary:
        coding_context = f"{patient_context}\n\n" + "\n\n".join(auxiliary)
        logger.info("coding_node (slow path) auxiliary context: %d item(s)", len(auxiliary))

    # Generate from the cleaned note using LLM
    coding_result = generate_coding(
        soap_note=soap_note,
        patient_context=coding_context,
        mode=state.get("mode", "fast"),
        model_name=state.get("model_name"),
    )
    result_dict = coding_result.model_dump()

    # Deterministic symptom-code completion
    symptom_completion = _complete_symptom_codes(result_dict, suggestions)

    # Deterministic CPT: the calculator result is authoritative
    cpt = _deterministic_cpt(soap_note, patient_context)
    result_dict["cpt_suggestions"] = [cpt["cpt_description"]]
    result_dict["cpt_grounding"] = {
        "deterministic": True,
        "cpt_code": cpt["cpt_code"],
        "overall_mdm_level": cpt["overall_mdm_level"],
        "justification": cpt["justification"],
        "domains": cpt["domains"],
    }

    # Existence pre-check
    primary = result_dict.get("primary_icd10", "")
    secondary = list(result_dict.get("secondary_icd10") or [])
    existence = _existence_precheck([primary] + secondary)
    if existence.dropped:
        result_dict["primary_icd10"] = "" if primary in existence.dropped else primary
        result_dict["secondary_icd10"] = [
            code for code in secondary if code not in existence.dropped
        ]

    # Re-validate what is left
    emitted = [
        code
        for code in [result_dict.get("primary_icd10", "")] + list(result_dict.get("secondary_icd10") or [])
        if code
    ]
    grounding_text = "\n".join(
        part for part in (chief_complaint, visit_context_to_grounding_text(state.get("visit_context")))
        if part
    )
    tool_flags = _ungrounded_flags(emitted, patient_context, grounding_text)

    # Medical necessity narrative cardiac claim scrub
    claim_flags: List[str] = []
    necessity = str(result_dict.get("medical_necessity_note") or "")
    if necessity and not has_charted_cardiac_diagnosis(patient_context):
        scrubbed, claims = scrub_unsupported_cardiac_claims(necessity, has_diagnosis=False)
        if claims:
            result_dict["medical_necessity_note"] = scrubbed or NECESSITY_FALLBACK
            claim_flags.append(
                "Coding: removed unsupported cardiac claim from the medical necessity "
                f"note ({', '.join(claims)}); no cardiac diagnosis is listed in "
                "Patient Context"
            )

    elapsed = round(time.time() - started, 1)
    logger.info(
        "coding_node (slow path) completed in %ss (%d code(s) kept, %d dropped, %d symptom code(s) "
        "added, %d tool flag(s), CPT %s)",
        elapsed, len(emitted), len(existence.dropped), len(symptom_completion.codes),
        len(tool_flags) + len(claim_flags), cpt["cpt_code"],
    )

    # Build tool trace
    tool_trace = {
        "simple_symptom_to_icd_suggestion": {
            "chief_complaint": chief_complaint,
            "suggested_codes": [
                candidate.code for candidate in suggestions[:MAX_SYMPTOM_GUIDANCE]
            ],
            "candidates": [
                {
                    "code": candidate.code,
                    "description": candidate.description,
                    "prefix": candidate.prefix,
                    "phrase": candidate.phrase,
                    "source": candidate.source,
                }
                for candidate in suggestions[:MAX_SYMPTOM_GUIDANCE]
            ],
            "added_to_result": symptom_completion.codes,
        },
        "validate_icd10_code": {
            "codes_checked": existence.checked,
            "malformed_dropped": existence.dropped,
        },
        "derive_mdm_domains": {
            "overall_mdm_level": cpt["overall_mdm_level"],
            "domains": {
                "problems": cpt["domains"]["problems_level"],
                "data": cpt["domains"]["data_level"],
                "risk": cpt["domains"]["risk_level"],
            },
        },
        "calculate_em_code": {
            "cpt_code": cpt["cpt_code"],
            "justification": cpt["justification"],
        },
        "validate_icd_against_patient": {
            "codes_checked": len(emitted),
            "flags": tool_flags,
        },
    }

    flags = symptom_completion.flags + tool_flags + existence.flags + claim_flags
    update: Dict[str, Any] = {
        "coding_result": result_dict,
        "tool_results": {CODING_AGENT: tool_trace},
        "cycle_flags": {"coding": list(flags)},
        "current_step": "coding",
        "timing": {"coding_seconds": elapsed},
        "coding_path": "slow",
        "coding_path_decision": {"reason": "Slow Path: LLM Coding Agent used", "confidence": "high"},
        "messages": [
            AIMessage(
                content=(
                    f"Coding Agent (slow path) produced primary ICD-10 {result_dict.get('primary_icd10')} "
                    f"and CPT {cpt['cpt_code']} (MDM {cpt['overall_mdm_level']}) in {elapsed}s "
                    f"({len(emitted)} code(s) checked against the chart"
                    + (
                        f", {len(existence.dropped)} malformed dropped"
                        if existence.dropped
                        else ""
                    )
                    + (
                        f", {len(symptom_completion.codes)} symptom code(s) completed"
                        if symptom_completion.codes
                        else ""
                    )
                    + "."
                ),
                name="coding_node",
            )
        ],
    }
    if flags:
        update["quality_flags"] = flags
    return update


def _run_fast_path(
    state: AgentState,
    soap_note: Dict[str, Any],
    patient_context: str,
    chief_complaint: str,
    started: float,
    decision_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the Fast Path (deterministic tools only, no LLM)."""
    # Execute fast path deterministic tools
    coding_result, tool_trace, quality_flags = execute_fast_path(
        cleaned_soap_note=soap_note,
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        visit_context=state.get("visit_context"),
        decision_metadata=decision_metadata,
    )

    # Run existence pre-check on fast path results (for consistency)
    primary = coding_result.get("primary_icd10", "")
    secondary = list(coding_result.get("secondary_icd10") or [])
    existence = _existence_precheck([primary] + secondary)
    if existence.dropped:
        coding_result["primary_icd10"] = "" if primary in existence.dropped else primary
        coding_result["secondary_icd10"] = [
            code for code in secondary if code not in existence.dropped
        ]
        quality_flags.extend(existence.flags)

    # Re-validate against patient context
    emitted = [
        code
        for code in [coding_result.get("primary_icd10", "")] + list(coding_result.get("secondary_icd10") or [])
        if code
    ]
    grounding_text = "\n".join(
        part for part in (chief_complaint, visit_context_to_grounding_text(state.get("visit_context")))
        if part
    )
    tool_flags = _ungrounded_flags(emitted, patient_context, grounding_text)
    quality_flags.extend(tool_flags)

    # Medical necessity narrative cardiac claim scrub
    claim_flags: List[str] = []
    necessity = str(coding_result.get("medical_necessity_note") or "")
    if necessity and not has_charted_cardiac_diagnosis(patient_context):
        scrubbed, claims = scrub_unsupported_cardiac_claims(necessity, has_diagnosis=False)
        if claims:
            coding_result["medical_necessity_note"] = scrubbed or NECESSITY_FALLBACK
            claim_flags.append(
                "Coding: removed unsupported cardiac claim from the medical necessity "
                f"note ({', '.join(claims)}); no cardiac diagnosis is listed in "
                "Patient Context"
            )
    quality_flags.extend(claim_flags)

    # Extract CPT info for logging
    cpt_suggestion = coding_result.get("cpt_suggestions", [""])[0]
    cpt_code = cpt_suggestion.split(" - ")[0] if cpt_suggestion else "99213"
    mdm_level = "low"
    if "moderate" in cpt_suggestion.lower():
        mdm_level = "moderate"
    elif "high" in cpt_suggestion.lower():
        mdm_level = "high"
    elif "straightforward" in cpt_suggestion.lower():
        mdm_level = "straightforward"

    # Deterministic CPT grounding (same structure as Slow Path)
    cpt = _deterministic_cpt(soap_note, patient_context)
    coding_result["cpt_grounding"] = {
        "deterministic": True,
        "cpt_code": cpt["cpt_code"],
        "overall_mdm_level": cpt["overall_mdm_level"],
        "justification": cpt["justification"],
        "domains": cpt["domains"],
    }
    # Ensure cpt_suggestions uses the deterministic result
    coding_result["cpt_suggestions"] = [cpt["cpt_description"]]

    elapsed = round(time.time() - started, 1)
    logger.info(
        "coding_node (fast path) completed in %ss (%d code(s) kept, %d dropped, %d tool flag(s), CPT %s)",
        elapsed, len(emitted), len(existence.dropped),
        len(tool_flags) + len(claim_flags), cpt_code,
    )

    # Add fast path marker to tool trace
    tool_trace["fast_path"] = {
        "used": True,
        "reason": decision_metadata.get("reason", ""),
        "confidence": decision_metadata.get("confidence", "high"),
    }

    all_flags = quality_flags
    update: Dict[str, Any] = {
        "coding_result": coding_result,
        "tool_results": {CODING_AGENT: tool_trace},
        "cycle_flags": {"coding": list(all_flags)},
        "current_step": "coding",
        "timing": {"coding_seconds": elapsed},
        "coding_path": "fast",
        "coding_path_decision": decision_metadata,
        "messages": [
            AIMessage(
                content=(
                    f"Coding Agent (fast path) produced primary ICD-10 {coding_result.get('primary_icd10')} "
                    f"and CPT {cpt_code} (MDM {mdm_level}) in {elapsed}s "
                    f"({len(emitted)} code(s) checked against the chart"
                    + (
                        f", {len(existence.dropped)} malformed dropped"
                        if existence.dropped
                        else ""
                    )
                    + "."
                ),
                name="coding_node",
            )
        ],
    }
    if all_flags:
        update["quality_flags"] = all_flags
    return update


def coding_node(state: AgentState) -> dict:
    """Generate ICD-10-CM diagnoses and CPT suggestions.

    Args:
        state: Current graph state, must contain `cleaned_soap_note`.

    Returns:
        Partial state update containing the coding result, tool trace,
        per-node timing and an agent trace message.

    The node now implements a Fast Path / Slow Path gate:
    - Fast Path: deterministic tools only, skips LLM call
    - Slow Path: full LLM Coding Agent (original behavior)
    """
    started = time.time()

    patient_context = state["patient_context"]
    chief_complaint = state["chief_complaint"]
    soap_note = state.get("cleaned_soap_note") or state.get("soap_note") or {}

    # 1. Brief the agent on the documented symptoms so they are not dropped.
    #    Codes the Scribe already suggested are excluded, so the hint can never
    #    push the same code twice, and the visit data is searched as well as
    #    the chief complaint.
    scribe_codes = list(soap_note.get("icd10_suggestions") or [])
    suggestions = _symptom_guidance(
        chief_complaint,
        patient_context,
        visit_context=state.get("visit_context"),
        already_coded=scribe_codes,
    )
    symptom_hint = build_symptom_hint(suggestions)

    # Encounter data the clinician supplied (HPI, exam, tests) supports coding
    # decisions - e.g. a documented exam finding justifies a symptom code.
    visit_block = visit_context_to_prompt_block(state.get("visit_context"))

    # 2. Fold in any Supervisor feedback from a previous failed pass.
    feedback = list(state.get("feedback") or [])
    physician_feedback = state.get("physician_feedback")

    # 3. Fast Path / Slow Path Decision Gate
    use_fast_path, decision_metadata = should_use_fast_path(
        cleaned_soap_note=soap_note,
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        visit_context=state.get("visit_context"),
        grounding_facts=state.get("grounding_facts"),
    )

    # Log the decision
    path_name = "FAST" if use_fast_path else "SLOW"
    logger.info(
        "Coding path decision: %s – %s (confidence: %s)",
        path_name,
        decision_metadata.get("reason", ""),
        decision_metadata.get("confidence", "unknown"),
    )

    # 4. Execute the chosen path
    if use_fast_path:
        return _run_fast_path(
            state=state,
            soap_note=soap_note,
            patient_context=patient_context,
            chief_complaint=chief_complaint,
            started=started,
            decision_metadata=decision_metadata,
        )
    else:
        return _run_slow_path(
            state=state,
            soap_note=soap_note,
            patient_context=patient_context,
            chief_complaint=chief_complaint,
            suggestions=suggestions,
            symptom_hint=symptom_hint,
            visit_block=visit_block,
            feedback=feedback,
            physician_feedback=physician_feedback,
            started=started,
        )


__all__ = ["_symptom_guidance", "build_symptom_hint", "coding_node"]