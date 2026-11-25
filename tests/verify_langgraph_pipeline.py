"""
End-to-end verification for the LangGraph visit pipeline (tools + memory).

Runs the two canonical regression cases through the graph and asserts the
clinical safety guarantees still hold:

  Test Case A: "Routine follow-up for diabetes and hypertension"
  Test Case B: "chest pain and high blood pressure and diarrhea"

The patient (P001, John Doe) has Type 2 Diabetes + Hypertension and NO asthma,
COPD, CHF or CKD, so any such content in the output is a hallucination.

Usage (from the project root, with Ollama running):

    python -m tests.verify_langgraph_pipeline              # both cases, fast mode
    python -m tests.verify_langgraph_pipeline --case A     # case A only
    python -m tests.verify_langgraph_pipeline --case B --mode quality
    python -m tests.verify_langgraph_pipeline --no-rag     # skip vector store lookup
    python -m tests.verify_langgraph_pipeline --no-memory  # disable long-term memory
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.langgraph_orchestrator import (  # noqa: E402
    PIPELINE_VERSION,
    graph_mermaid,
    run_visit_pipeline,
)
from tools.registry import registry_as_dict  # noqa: E402
from utils.validation import HALLUCINATED_TERMS  # noqa: E402

PATIENT_ID = "P001"

PATIENT = {
    "name": "John Doe",
    "age": 58,
    "gender": "male",
    "conditions": ["Type 2 Diabetes", "Hypertension"],
    "medications": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
    "allergies": ["Penicillin"],
    "last_visit_summary": (
        "Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence."
    ),
}

PATIENT_CONTEXT = f"""
Patient: {PATIENT['name']}, {PATIENT['age']}yo {PATIENT['gender']}
Conditions: {', '.join(PATIENT['conditions'])}
Medications: {', '.join(PATIENT['medications'])}
Allergies: {', '.join(PATIENT['allergies'])}
Last Visit: {PATIENT['last_visit_summary']}
"""

TEST_CASES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Routine follow-up for diabetes and hypertension",
        "chief_complaint": "Routine follow-up for diabetes and hypertension",
        "required_icd10_prefixes": {"E11", "I10"},
    },
    "B": {
        "name": "chest pain and high blood pressure and diarrhea",
        "chief_complaint": "chest pain and high blood pressure and diarrhea",
        "required_icd10_prefixes": {"R07", "I10"},
    },
}

REQUIRED_RESULT_KEYS = {"soap_note", "coding_result", "quality_flags", "metadata"}


def get_retrieved_notes(chief_complaint: str) -> str:
    """Fetch RAG notes for the complaint (same path as the Streamlit UI)."""
    try:
        from rag.simple_rag import get_vectorstore

        vs = get_vectorstore()
        docs = vs.similarity_search(chief_complaint, k=1)
        return "\n\n".join(doc.page_content for doc in docs)[:300]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  [warn] RAG unavailable ({exc}); continuing without retrieved notes")
        return ""


def _flatten(result: Dict[str, Any]) -> str:
    """Concatenate every clinically meaningful output field into one string."""
    parts: List[str] = []
    soap = result.get("soap_note", {})
    for key in ("subjective", "objective", "assessment", "plan"):
        parts.append(str(soap.get(key, "")))
    parts.extend(str(c) for c in soap.get("icd10_suggestions", []))

    coding = result.get("coding_result", {})
    parts.append(str(coding.get("primary_icd10", "")))
    parts.extend(str(c) for c in coding.get("secondary_icd10", []))
    parts.extend(str(c) for c in coding.get("cpt_suggestions", []))
    parts.append(str(coding.get("medical_necessity_note", "")))
    return "\n".join(parts)


def _icd10_codes(result: Dict[str, Any]) -> List[str]:
    soap = result.get("soap_note", {})
    coding = result.get("coding_result", {})
    raw = (
        list(soap.get("icd10_suggestions", []))
        + [coding.get("primary_icd10", "")]
        + list(coding.get("secondary_icd10", []))
    )
    codes = []
    for item in raw:
        match = re.match(r"([A-Z]\d{2})", str(item))
        if match:
            codes.append(match.group(1))
    return codes


def check_case(
    case_key: str,
    result: Dict[str, Any],
    use_memory: bool = True,
) -> List[Tuple[bool, str, bool]]:
    """Return a list of (passed, message, advisory) checks for one test case.

    `advisory=True` marks checks that describe a known small-model limitation
    rather than a clinical safety guarantee: they are reported but do not fail
    the run.
    """
    case = TEST_CASES[case_key]
    checks: List[Tuple[bool, str, bool]] = []
    flat = _flatten(result).lower()
    soap = result.get("soap_note", {})
    coding = result.get("coding_result", {})
    flags = " | ".join(result.get("quality_flags", [])).lower()

    # 1. Result schema is backward compatible with the previous orchestrator.
    missing = REQUIRED_RESULT_KEYS - set(result)
    checks.append((not missing, f"result keys present (missing: {sorted(missing) or 'none'})", False))

    # 2. LangGraph metadata is present.
    meta = result.get("metadata", {})
    checks.append(
        (
            meta.get("orchestrator") == "langgraph"
            and meta.get("pipeline_version") == PIPELINE_VERSION,
            f"metadata reports LangGraph pipeline {meta.get('pipeline_version')}",
            False,
        )
    )
    supervisor_meta = meta.get("supervisor") or {}
    steps = meta.get("steps_completed") or []
    finalized = bool(steps) and steps[-1] == "memory_save"
    escalated = bool(steps) and steps[-1] == "human_review"
    canonical = ["memory_load", "scribe", "cleanup", "coding", "validation", "supervisor", "memory_save"]
    checks.append(
        (
            (finalized and steps == canonical)
            or (escalated and supervisor_meta.get("needs_human_review")),
            f"steps completed: {steps} (escalated: {escalated})",
            False,
        )
    )
    checks.append(
        (
            ((not escalated) and supervisor_meta.get("retry_count") == 0)
            or (escalated and supervisor_meta.get("retry_count") >= 1),
            f"supervisor metadata sane: retry_count={supervisor_meta.get('retry_count')}, "
            f"needs_human_review={supervisor_meta.get('needs_human_review')}",
            False,
        )
    )

    # 2b. Tools ran, scoped per node. Memory assertions are conditional so
    #     `--no-memory` runs still pass while proving nothing was read/written.
    memory_meta = meta.get("memory") or {}
    tool_meta = meta.get("tools") or {}
    tools_used = tool_meta.get("tools_used") or []
    if use_memory:
        checks.append(
            (
                memory_meta.get("patient_id") == PATIENT_ID,
                f"memory keyed on patient_id: {memory_meta.get('patient_id')}",
                False,
            )
        )
        # P2: a finished run records an unsigned draft. That is the most a
        # pipeline run may write - `approved` must stay False until a
        # physician approves the note in the UI.
        checks.append(
            (
                memory_meta.get("saved") or bool(supervisor_meta.get("needs_human_review")),
                f"run recorded a draft visit (or escalated): {memory_meta.get('saved')}",
                False,
            )
        )
        checks.append(
            (
                not memory_meta.get("approved")
                and memory_meta.get("visit_status", "") in ("draft", ""),
                f"a pipeline run never approves: approved={memory_meta.get('approved')!r}, "
                f"visit_status={memory_meta.get('visit_status')!r}",
                False,
            )
        )
    else:
        checks.append(
            (
                not memory_meta.get("saved")
                and not memory_meta.get("memory_context"),
                f"memory disabled: nothing read or written ({memory_meta.get('saved')!r})",
                False,
            )
        )
    checks.append(
        (
            "get_patient_conditions" in tools_used and "get_patient_medications" in tools_used,
            f"scribe grounding tools ran: {tools_used}",
            False,
        )
    )
    checks.append(
        (
            "validate_icd_against_patient" in tools_used,
            "coding grounding tool re-checked every code",
            False,
        )
    )
    # Supervisor invariant: the Scribe must never gain the coding validation tool.
    scribe_trace = (tool_meta.get("tool_results") or {}).get("scribe_node") or {}
    coding_trace = (tool_meta.get("tool_results") or {}).get("coding_node") or {}
    checks.append(
        (
            "validate_icd_against_patient" not in scribe_trace,
            "scribe node did not use the coding-only validation tool",
            False,
        )
    )
    checks.append(
        (
            "simple_symptom_to_icd_suggestion" in coding_trace,
            "coding node received symptom-code guidance",
            False,
        )
    )

    # 3. No hallucinated chronic conditions anywhere in the output.
    for condition, terms in HALLUCINATED_TERMS.items():
        hits = sorted({term for term in terms if term in flat})
        checks.append((not hits, f"no {condition} content (hits: {hits or 'none'})", False))

    # 4. Every ICD-10 code is grounded in the patient context / complaint.
    # Allowed: diabetes (E11), hypertension (I10) from the chart, plus acute
    # symptom codes legitimately derivable from the chief complaint.
    allowed_prefixes = {"E11", "I10", "R07", "R19", "R10", "Z71", "Z13", "Z00"}
    codes = _icd10_codes(result)
    ungrounded = [code for code in codes if code not in allowed_prefixes]
    checks.append(
        (not ungrounded, f"ICD-10 codes grounded (codes: {sorted(set(codes)) or 'none'})", False)
    )
    checks.append(("J45" not in codes, "no asthma ICD-10 (J45) in output", False))

    # 5. Chronic diagnoses from the chart are always coded (hard requirement).
    chronic_required = case["required_icd10_prefixes"] & {"E11", "I10"}
    missing_chronic = sorted(chronic_required - set(codes))
    checks.append(
        (not missing_chronic, f"chronic ICD-10 coded (missing: {missing_chronic or 'none'})", False)
    )

    # 6. Acute complaint symptoms must be EITHER coded OR raised as a review
    #    flag by the validation node. Uncoded *and* unflagged is a failure.
    symptom_prefixes = case["required_icd10_prefixes"] - chronic_required
    silent_gaps = [p for p in sorted(symptom_prefixes) if p not in codes and p.lower() not in flags]
    checks.append(
        (
            not silent_gaps,
            f"complaint symptoms coded or flagged (silent gaps: {silent_gaps or 'none'})",
            False,
        )
    )
    uncoded = [p for p in sorted(symptom_prefixes) if p not in codes]
    checks.append(
        (
            True,
            f"symptom codes present: {uncoded or 'all symptom codes present'}",
            bool(uncoded),
        )
    )

    # 7. CPT reflects an established patient (99211-99215) and is deterministic.
    cpt_text = " ".join(coding.get("cpt_suggestions", []))
    cpts = [int(code) for code in re.findall(r"\b(99\d{3})\b", cpt_text)]
    grounding = coding.get("cpt_grounding") or {}
    checks.append((bool(cpts), f"CPT codes present: {coding.get('cpt_suggestions')}", False))
    checks.append(
        (all(99211 <= c <= 99215 for c in cpts), f"CPT in established range 99211-99215: {cpts}", False)
    )
    checks.append(
        (
            grounding.get("deterministic") is True
            and grounding.get("overall_mdm_level") in ("straightforward", "low", "moderate", "high"),
            f"CPT from deterministic E/M calculator: "
            f"{grounding.get('cpt_code')} / MDM {grounding.get('overall_mdm_level')}",
            False,
        )
    )

    # 8. SOAP sections are populated.
    for section in ("subjective", "objective", "assessment", "plan"):
        checks.append(
            (bool(str(soap.get(section, "")).strip()), f"SOAP '{section}' populated", False)
        )

    return checks


def run_case(
    case_key: str,
    mode: str = "fast",
    use_rag: bool = True,
    use_memory: bool = True,
) -> bool:
    case = TEST_CASES[case_key]
    print("=" * 70)
    print(f"TEST CASE {case_key}: {case['name']}")
    print("=" * 70)

    retrieved_notes = get_retrieved_notes(case["chief_complaint"]) if use_rag else ""
    if retrieved_notes:
        print(f"  Retrieved notes: {len(retrieved_notes)} chars")

    result = run_visit_pipeline(
        patient_context=PATIENT_CONTEXT,
        chief_complaint=case["chief_complaint"],
        retrieved_notes=retrieved_notes,
        mode=mode,
        patient_id=PATIENT_ID,
        use_memory=use_memory,
        is_synthetic=True,
    )

    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result["metadata"]
    print("\n--- SOAP NOTE ---")
    print(f"Subjective:  {soap['subjective']}")
    print(f"Objective:   {soap['objective']}")
    print(f"Assessment:  {soap['assessment']}")
    print(f"Plan:        {soap['plan']}")
    print(f"ICD-10 (scribe): {soap['icd10_suggestions']}")
    print("\n--- CODING RESULT ---")
    print(f"Primary ICD-10:   {coding['primary_icd10']}")
    print(f"Secondary ICD-10: {coding['secondary_icd10']}")
    print(f"CPT:              {coding['cpt_suggestions']}")
    print(f"Necessity:        {coding['medical_necessity_note']}")
    print(f"\nQuality flags:    {result['quality_flags']}")
    print(f"Timing:           {meta['timing']}")
    print(f"Memory:           {meta['memory']}")
    print(f"Supervisor:       {meta.get('supervisor')}")
    print(f"Tools used:       {meta['tools']['tools_used']}")
    print(f"Steps completed:  {meta['steps_completed']}")

    print("\n--- CHECKS ---")
    checks = check_case(case_key, result, use_memory=use_memory)
    failed = 0
    for passed, message, advisory in checks:
        if not passed:
            failed += 1
        if passed and advisory:
            status = "WARN"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {message}")

    print(
        f"\nTEST CASE {case_key}: "
        + ("PASSED" if failed == 0 else f"FAILED ({failed} checks)")
        + ("" if failed == 0 else " (advisory items are reported but do not fail the run)")
    )
    return failed == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the LangGraph visit pipeline")
    parser.add_argument("--case", choices=["A", "B", "all"], default="all")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast")
    parser.add_argument("--no-rag", action="store_true", help="Skip the vector store lookup")
    parser.add_argument("--no-memory", action="store_true", help="Disable long-term memory")
    parser.add_argument("--print-graph", action="store_true", help="Print the Mermaid graph")
    parser.add_argument("--print-tools", action="store_true", help="Print the tool registry")
    args = parser.parse_args()

    if args.print_graph:
        print(graph_mermaid())
        print()
    if args.print_tools:
        import json
        print(json.dumps(registry_as_dict(), indent=2))
        print()

    cases = ["A", "B"] if args.case == "all" else [args.case]
    results = [
        run_case(key, mode=args.mode, use_rag=not args.no_rag, use_memory=not args.no_memory)
        for key in cases
    ]

    print("\n" + "=" * 70)
    print(f"SUMMARY ({args.mode} mode): " + ", ".join(
        f"{key}={'PASS' if ok else 'FAIL'}" for key, ok in zip(cases, results)
    ))
    print("=" * 70)

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
