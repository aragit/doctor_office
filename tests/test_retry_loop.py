"""
End-to-end retry / escalation loop test through the REAL compiled graph.

No LLM: the Scribe/Cleanup/Coding nodes are swapped for deterministic stubs,
while the real validation_node-independent decision machinery runs: the real
Supervisor node, the real conditional edge, real reducers, real retry
accounting and real `human_review_node`.

Scenarios:
  1 - validation flags a hallucination once, then the scribe retry
      fixes it -> supervisor retries scribe once, then finalizes.
  2 - validation flags a hallucination every pass -> after MAX_RETRIES
      the supervisor escalates to human_review (nothing persisted).
  3 - (P1) a stale flag from the previous pass cannot force the retry that
      follows it, nor escalate a visit that has already been corrected.
  4 - (P1) the Supervisor's deterministic symptom-code fix is re-validated by
      the REAL validation node before anything is finalized.
  5 - (P1) a gap that survives the fix and the re-validation escalates at the
      retry ceiling instead of looping between fix and validation.
  6 - (P1) a validating node that never acknowledges the fix escalates instead
      of looping between fix and validation forever.

    python -m tests.test_retry_loop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

import agents  # noqa: E402
from agents import langgraph_orchestrator as orch  # noqa: E402
from agents.supervisor_node import MAX_RETRIES  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} -> {detail}")


def _make_validation(always_bad: bool):
    """Real-graph-shaped validation stub: hallucination flag on the first pass
    (or every pass), clean afterwards."""
    calls = {"n": 0}

    def validation_stub(state):
        calls["n"] += 1
        flags = []
        if always_bad or calls["n"] == 1:
            flags = ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"]
        return {
            "quality_flags": flags,
            "cycle_flags": {"validation": list(flags)},
            "current_step": "validation",
            "timing": {"validation_seconds": 0.0},
            "messages": [AIMessage(content="stub validation", name="validation_node")],
        }

    return validation_stub, calls


def _stub_nodes():
    def scribe_stub(state):
        return {
            "soap_note": {
                "subjective": "Routine follow-up.",
                "objective": "Stable.",
                "assessment": "Diabetes and hypertension.",
                "plan": "Continue meds.",
                "icd10_suggestions": ["E11.9 - Type 2 diabetes mellitus without complications"],
            },
            "current_step": "scribe",
            "timing": {"scribe_seconds": 0.0},
            "messages": [AIMessage(content="stub scribe", name="scribe_node")],
        }

    def cleanup_stub(state):
        return {
            "cleaned_soap_note": state.get("soap_note") or {},
            "cleanup_removed": [],
            "cycle_flags": {"cleanup": []},
            "current_step": "cleanup",
            "timing": {"cleanup_seconds": 0.0},
            "messages": [AIMessage(content="stub cleanup", name="cleanup_node")],
        }

    def coding_stub(state):
        return {
            "coding_result": {
                "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
                "secondary_icd10": ["I10 - Essential hypertension"],
                "cpt_suggestions": ["99213 - Office visit, established patient, low medical decision making"],
                "medical_necessity_note": "CPT 99213 supported by E11.9 (diabetes follow-up).",
            },
            "cycle_flags": {"coding": []},
            "current_step": "coding",
            "timing": {"coding_seconds": 0.0},
            "messages": [AIMessage(content="stub coding", name="coding_node")],
        }

    return {"memory_load_node": None, "scribe_node": scribe_stub,
            "cleanup_node": cleanup_stub, "coding_node": coding_stub}


def _run_graph(always_bad: bool, validation_stub, chief_complaint: str = None):
    """Run the real graph with the LLM nodes stubbed.

    `validation_stub=None` keeps the real validation node, which is what the
    post-auto-fix scenarios need: the symptom gap must come from the actual
    validator, not from a canned string.
    """
    original = {name: getattr(orch, name) for name in
                ("memory_load_node", "scribe_node", "cleanup_node", "coding_node",
                 "validation_node", "get_graph")}
    stubs = _stub_nodes()
    try:
        orch.memory_load_node = lambda s: {
            "current_step": "memory_load", "timing": {"memory_load_seconds": 0.0},
            "patient_id": "P001",
            "messages": [AIMessage(content="stub memory", name="memory_load_node")],
        }
        orch.scribe_node = stubs["scribe_node"]
        orch.cleanup_node = stubs["cleanup_node"]
        orch.coding_node = stubs["coding_node"]
        if validation_stub is not None:
            orch.validation_node = validation_stub
        orch.get_graph = lambda: orch.build_graph()
        orch._GRAPH = None
        final = orch.run_visit_pipeline(
            patient_context=(
                "Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes, Hypertension\n"
                "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
                "Allergies: Penicillin\nLast Visit: Blood pressure 148/92. HbA1c 7.8%."
            ),
            chief_complaint=chief_complaint
            or "Routine follow-up for diabetes and hypertension",
            retrieved_notes="",
            mode="fast",
            patient_id="P001",
            use_memory=False,
        )
        return final
    finally:
        for name, value in original.items():
            setattr(orch, name, value)
        orch._GRAPH = None


def test_one_retry_then_finalize():
    print("\nScenario 1: one flag -> one scribe retry -> finalize")
    validation_stub, calls = _make_validation(always_bad=False)
    result = _run_graph(False, validation_stub)

    supervisor = result["metadata"]["supervisor"]
    check("supervisor retried once then finalized", supervisor["retry_count"] == 1, supervisor)
    check("not escalated", supervisor["needs_human_review"] is False, supervisor)
    check("feedback was handed to the retry pass", bool(supervisor["feedback"]), supervisor)
    check(
        "graph ended at memory_save",
        result["metadata"]["graph"] == "memory_save",
        result["metadata"]["graph"],
    )
    check("validation ran at least twice", calls["n"] >= 2, calls)
    check(
        "steps include supervisor before memory_save",
        result["metadata"]["steps_completed"] ==
        ["memory_load", "scribe", "cleanup", "coding", "validation", "supervisor", "memory_save"],
        result["metadata"]["steps_completed"],
    )


def test_escalation_on_persistent_failure():
    print("\nScenario 2: persistent flag -> escalation to human review")
    validation_stub, calls = _make_validation(always_bad=True)
    result = _run_graph(True, validation_stub)

    supervisor = result["metadata"]["supervisor"]
    check("escalated to human review", supervisor["needs_human_review"] is True, supervisor)
    check("retries exhausted at ceiling", supervisor["retry_count"] == MAX_RETRIES, supervisor)
    check(
        "graph ended at human_review",
        result["metadata"]["graph"] == "human_review",
        result["metadata"]["graph"],
    )
    check(
        "human-reviewed visit not persisted",
        result["metadata"]["memory"]["saved"] is False,
        result["metadata"]["memory"],
    )
    check(
        "steps end at human_review",
        result["metadata"]["steps_completed"][-1] == "human_review",
        result["metadata"]["steps_completed"],
    )
    check(
        "escalation flag surfaced in quality_flags",
        any("human review" in str(f).lower() for f in result["quality_flags"]),
        result["quality_flags"],
    )
    check("validation ran for every retry cycle", calls["n"] >= MAX_RETRIES + 1, calls)


# ---------------------------------------------------------------------------
# P1 control flow: stale flags and post-auto-fix re-validation
# ---------------------------------------------------------------------------


def test_stale_flag_does_not_loop_or_escalate():
    """Pass-1 flags must not drive pass 2, however many retries are left."""
    print("\nScenario 3: pass-1 flag -> retry -> pass 2 clean -> finalize")
    # The flag persists in the audit log on every pass (validation always
    # re-appends what it saw), but only pass 1 saw it: this stub reports the
    # hallucination once and is clean afterwards.
    validation_stub, calls = _make_validation(always_bad=False)
    result = _run_graph(False, validation_stub)
    flow = result["metadata"]["control_flow"]

    check("one retry, then finalized", result["metadata"]["supervisor"]["retry_count"] == 1,
          result["metadata"]["supervisor"])
    check("the run was not escalated",
          result["metadata"]["supervisor"]["needs_human_review"] is False)
    check("the graph reached memory_save",
          result["metadata"]["graph"] == "memory_save", result["metadata"]["graph"])
    check("exactly one flag-bucket reset happened", flow["cycle_flag_resets"] == 1, flow)
    check("the cleared flags are reported",
          any("hallucinated" in f for f in flow["cleared_cycle_flags"]),
          flow["cleared_cycle_flags"])
    check("the visit finalized on the fresh validation",
          flow["final_decision"] == "finalize" and flow["final_route"] == "memory_save_node", flow)
    check("no auto-fix and no re-validation were involved",
          not flow["auto_fix_applied"] and flow["revalidations"] == 0, flow)
    check("the audit log still shows what pass 1 found",
          any("hallucinated" in f for f in result["quality_flags"]), result["quality_flags"])
    check("validation ran twice, no more", calls["n"] == 2, calls)


def test_autofix_is_revalidated_before_finalize():
    """The Supervisor's own symptom fix must be validated in its final form."""
    print("\nScenario 4: symptom gap -> auto-fix -> re-validate -> finalize")
    # REAL validation node: the gap comes from the actual validator seeing a
    # chief complaint with no R code, and the Coding stub never emits one.
    result = _run_graph(False, None, chief_complaint="Chest pain x 1 day")
    flow = result["metadata"]["control_flow"]
    coding = result["coding_result"]

    check("the auto-fix was applied", flow["auto_fix_applied"] is True, flow)
    check("the added code is recorded in control flow",
          any("R07.9" in code for code in flow["auto_fix_codes"]), flow)
    check("the added code is in the stored result",
          any("R07.9" in str(code) for code in coding["secondary_icd10"]),
          coding)
    check("the fix was re-validated exactly once", flow["revalidations"] == 1, flow)
    check("nothing is left pending", flow["pending_revalidation"] is False, flow)
    check("the run finalized after the re-validation",
          flow["final_route"] == "memory_save_node" and flow["final_decision"] == "finalize", flow)
    check("the auto-fix cost no LLM retry",
          result["metadata"]["supervisor"]["retry_count"] == 0,
          result["metadata"]["supervisor"])
    check("the finalized run is the one that stored the visit",
          result["metadata"]["graph"] == "memory_save", result["metadata"]["graph"])
    check("no diagnosis code was auto-added",
          all(str(code).strip()[:1] not in ("I", "E", "J")
              for code in flow["auto_fix_codes"]), flow["auto_fix_codes"])
    check("the resolved gap never reappears after the fix",
          result["metadata"]["graph"] == "memory_save", result["metadata"]["graph"])


def test_autofix_that_does_not_settle_escalates():
    """A re-validation that finds a real defect escalates at the retry ceiling."""
    print("\nScenario 5: auto-fix -> re-validate finds a hard failure -> escalate")
    calls = {"n": 0}

    def validation_stub(state):
        """Gap on pass 1 only; a real ungrounded-code failure on every pass after.

        Mirrors what happens when the fix settles the symptom gap but the visit
        has a genuine coding defect underneath it: the fix must not hide it, and
        it must not restart the fix loop. Like the real node, it consumes the
        pending re-validation marker - a stub that does not would be testing a
        broken pipeline.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            flags = ["Chief complaint mentions 'chest pain' but no R07 code was "
                     "generated - review and add a symptom code if clinically appropriate"]
        else:
            flags = ["Tool check: Ungrounded ICD-10 code J45.909 - not in patient "
                     "context or chief complaint"]
        update = {
            "quality_flags": flags,
            "cycle_flags": {"validation": list(flags)},
            "current_step": "validation",
            "timing": {"validation_seconds": 0.0},
            "messages": [AIMessage(content="stub validation", name="validation_node")],
        }
        if state.get("pending_revalidation"):
            update["pending_revalidation"] = False
            update["revalidation_passes"] = int(state.get("revalidation_passes", 0) or 0) + 1
        return update

    result = _run_graph(False, validation_stub, chief_complaint="Chest pain x 1 day")
    flow = result["metadata"]["control_flow"]

    check("the auto-fix still happened first", flow["auto_fix_applied"] is True, flow)
    check("the re-validation still happened", flow["revalidations"] == 1, flow)
    check("the re-validation exposed the real defect",
          result["metadata"]["graph"] == "human_review", result["metadata"]["graph"])
    check("the retries stopped at the ceiling",
          result["metadata"]["supervisor"]["retry_count"] == MAX_RETRIES,
          result["metadata"]["supervisor"])
    check("the escalation is attributed to the retries, not the fix",
          flow["escalation_reason"] == "max_retries", flow)
    check("the fix was not repeated", flow["auto_fix_rounds"] == 1, flow)
    check("nothing was stored for a human-reviewed visit",
          result["metadata"]["memory"]["saved"] is False, result["metadata"]["memory"])
    check("the run terminates: validation ran once per pass, no storm",
          calls["n"] == MAX_RETRIES + 2, calls)


def test_unconsumed_revalidation_escalates_instead_of_looping():
    """A validating node that never consumes the fix must not spin the graph."""
    print("\nScenario 6: pending re-validation never consumed -> bounded escalation")
    calls = {"n": 0}

    def forgetful_validation_stub(state):
        """Reports a gap and never clears `pending_revalidation`.

        A stand-in for a validating node that fails to acknowledge the fix. The
        Supervisor must still terminate the run and escalate, rather than
        re-validate forever until the graph's recursion limit trips.
        """
        calls["n"] += 1
        flags = ["Chief complaint mentions 'chest pain' but no R07 code was generated - review"]
        return {
            "quality_flags": flags,
            "cycle_flags": {"validation": list(flags)},
            "current_step": "validation",
            "timing": {"validation_seconds": 0.0},
            "messages": [AIMessage(content="stub validation", name="validation_node")],
        }

    result = _run_graph(False, forgetful_validation_stub, chief_complaint="Chest pain x 1 day")
    flow = result["metadata"]["control_flow"]

    check("the run terminates by escalating", result["metadata"]["graph"] == "human_review",
          result["metadata"]["graph"])
    check("the repeated demand is counted", flow["revalidation_stalls"] >= 1, flow)
    check("the escalation is attributed to the re-validation, not the retries",
          flow["escalation_reason"] == "revalidation_exhausted", flow)
    check("no LLM retry was charged for a deterministic loop",
          result["metadata"]["supervisor"]["retry_count"] == 0,
          result["metadata"]["supervisor"])
    check("no re-validation completed, because none was acknowledged",
          flow["revalidations"] == 0, flow)
    check("the fix was applied exactly once", flow["auto_fix_rounds"] == 1, flow)
    check("the loop is bounded, not endless", calls["n"] <= 3, calls)
    check("nothing was stored", result["metadata"]["memory"]["saved"] is False,
          result["metadata"]["memory"])


def main():
    print("=" * 70)
    print("RETRY LOOP INTEGRATION TEST (real graph, stub LLM nodes)")
    print("=" * 70)
    test_one_retry_then_finalize()
    test_escalation_on_persistent_failure()
    test_stale_flag_does_not_loop_or_escalate()
    test_autofix_is_revalidated_before_finalize()
    test_autofix_that_does_not_settle_escalates()
    test_unconsumed_revalidation_escalates_instead_of_looping()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())