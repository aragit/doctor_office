"""
P2: pipeline output is a draft; only an explicit Approve creates history.

    python -m tests.test_draft_approval_flow

Pure, no LLM, instant. Covers the graph-level half of the split:

* `memory_save_node` records a draft and echoes the row id, never an approval.
* `human_review_node` / the escalation path records nothing at all.
* `_state_to_result` reports the row as a draft and claims no approval.
* `memory_load_node` reads approved history only, so a draft from the previous
  run is invisible to the Scribe.
* the state initialises the new fields, so a run cannot inherit a stale id.

The store-level half (promotion, idempotence, legacy rows, trimming) lives in
`tests.test_memory`.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.graph_state import build_initial_state  # noqa: E402
from agents.langgraph_orchestrator import _state_to_result  # noqa: E402
from agents import memory_node  # noqa: E402
from agents.memory_node import memory_load_node, memory_save_node  # noqa: E402
from agents.supervisor_node import human_review_node  # noqa: E402
from memory import patient_memory as patient_memory_module  # noqa: E402
from memory.patient_memory import (  # noqa: E402
    VISIT_STATUS_APPROVED,
    VISIT_STATUS_DRAFT,
    JsonPatientMemoryRepository,
    visit_status,
)

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} -> {detail}")


CHART = """Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID
Allergies: Penicillin
Last Visit: Blood pressure 148/92.
"""

CODING = {
    "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
    "secondary_icd10": ["I10 - Essential (primary) hypertension"],
    "cpt_suggestions": ["99213 - Office visit, established patient"],
}
SOAP = {"assessment": "Diabetes, suboptimally controlled.", "plan": "Continue metformin."}


def _use_temp_repo():
    """Point the process-wide memory singleton at a throwaway file."""
    repo = JsonPatientMemoryRepository(path=Path(tempfile.mkdtemp()) / "patients.json")
    patient_memory_module._DEFAULT_REPO = repo
    return repo


def _run_state(**overrides):
    state = build_initial_state(
        CHART,
        "Follow-up for diabetes. Reports fatigue.",
        retrieved_notes="",
        patient_id="P001",
    )
    state.update({
        "cleaned_soap_note": SOAP,
        "coding_result": CODING,
        "quality_flags": [],
    })
    state.update(overrides)
    return state


def test_state_initialises_the_new_fields():
    print("\nState: the draft/approved fields start empty")
    state = build_initial_state(CHART, "cc", "notes")
    check("memory_visit_id starts empty", state["memory_visit_id"] == "", state["memory_visit_id"])
    check("memory_visit_status starts empty", state["memory_visit_status"] == "",
          state["memory_visit_status"])
    check("nothing is recorded at the start", state["memory_saved"] is False, state)


def test_memory_save_records_a_draft():
    print("\nmemory_save_node: records a draft and nothing more")
    repo = _use_temp_repo()
    update = memory_save_node(_run_state())

    check("memory_saved is True", update["memory_saved"] is True, update)
    check("the status it reports is draft", update["memory_visit_status"] == VISIT_STATUS_DRAFT,
          update)
    check("the row id is echoed back", bool(update["memory_visit_id"]), update)

    summary = repo.get_summary("P001")
    check("the stored row is a draft",
          visit_status(summary["visits"][-1]) == VISIT_STATUS_DRAFT, summary["visits"])
    check("no visit is on record yet", summary["visit_count"] == 0, summary)
    check("the row is not approved history", summary["approved_visit_count"] == 0, summary)
    check("the run's own message says it is not history",
          "not history until a physician approves" in update["messages"][-1].content,
          update["messages"][-1].content)


def test_memory_save_never_approves_even_when_asked():
    print("\nmemory_save_node: there is no input that makes it approve")
    repo = _use_temp_repo()
    update = memory_save_node(_run_state())
    row = repo.get_summary("P001")["visits"][-1]

    check("no approved_at on a pipeline row", "approved_at" not in row, row)
    check("the memory nodes import no approval function at all",
          not hasattr(memory_node, "approve_visit"), [n for n in dir(memory_node)])
    check("metadata reports approved=False",
          _state_to_result(_run_state(**{"memory_saved": True,
                                         "memory_visit_id": update["memory_visit_id"],
                                         "memory_visit_status": VISIT_STATUS_DRAFT}),
                           "fast", None)["metadata"]["memory"]["approved"] is False)


def test_escalation_records_nothing():
    print("\nEscalation: nothing is recorded at all")
    _use_temp_repo()
    update = human_review_node(_run_state(needs_human_review=True))
    check("human_review_node persists nothing", update["memory_saved"] is False, update)
    check("and reports no row id", not update.get("memory_visit_id"), update)


def test_memory_disabled_records_nothing():
    print("\nMemory disabled: no row, no id")
    _use_temp_repo()
    update = memory_save_node(_run_state(use_memory=False))
    check("nothing saved", update["memory_saved"] is False, update)
    check("no row id", update["memory_visit_id"] == "", update)
    check("no status", update["memory_visit_status"] == "", update)


def test_a_draft_is_invisible_to_the_next_run():
    print("\nPrior-visit context: a draft is not read by the Scribe")
    repo = _use_temp_repo()

    first = memory_save_node(_run_state())
    draft_id = first["memory_visit_id"]

    load = memory_load_node(_run_state())
    check("the draft is not in the memory context",
          "Follow-up for diabetes" not in load["memory_context"], load["memory_context"])
    check("and no prior visit is counted", load["patient_summary"]["visit_count"] == 0,
          load["patient_summary"])
    check("memory_context_used stays False in the result",
          _state_to_result(_run_state(**load), "fast", None)["metadata"]["memory"][
              "memory_context_used"] is False)

    # Approving it - the physician's action, not the pipeline's - makes the next
    # run see it.
    repo.approve_visit("P001", visit_id=draft_id)
    after = memory_load_node(_run_state())
    check("after approval the visit is read",
          "Follow-up for diabetes" in after["memory_context"], after["memory_context"])
    check("and the prior-visit count is 1",
          after["patient_summary"]["visit_count"] == 1, after["patient_summary"])


def test_result_metadata_reports_the_draft():
    print("\nResult metadata: saved != approved")
    saved = memory_save_node(_run_state())
    result = _state_to_result(
        _run_state(**{
            "memory_saved": True,
            "memory_visit_id": saved["memory_visit_id"],
            "memory_visit_status": VISIT_STATUS_DRAFT,
            "patient_summary": {"visit_count": 0, "approved_visit_count": 0,
                                "draft_visit_count": 1},
        }),
        "fast", None,
    )
    mem = result["metadata"]["memory"]
    check("saved reports the write", mem["saved"] is True, mem)
    check("the row id is carried to the UI", mem["visit_id"] == saved["memory_visit_id"], mem)
    check("the status is carried as draft", mem["visit_status"] == VISIT_STATUS_DRAFT, mem)
    check("approved is False", mem["approved"] is False, mem)
    check("prior_visits counts approved history only", mem["prior_visits"] == 0, mem)
    check("draft_visit_count is reported", mem["draft_visit_count"] == 1, mem)

    # The summary in state was loaded before the write, so the node refreshes it;
    # otherwise the UI would show the store as it looked before this draft.
    refreshed = memory_save_node(_run_state())
    check("the node refreshes the stale load-time summary",
          refreshed["patient_summary"]["draft_visit_count"] == 2,
          refreshed["patient_summary"].get("draft_visit_count"))


def test_a_revision_gets_its_own_row():
    print("\nRequest Changes: each run records its own draft")
    repo = _use_temp_repo()
    first = memory_save_node(_run_state())
    second = memory_save_node(_run_state(chief_complaint="Follow-up for diabetes. Fatigue worse."))

    check("the revision has a different row id",
          second["memory_visit_id"] != first["memory_visit_id"],
          (first["memory_visit_id"], second["memory_visit_id"]))
    check("both rows are kept as drafts",
          repo.get_summary("P001")["draft_visit_count"] == 2, repo.get_summary("P001"))

    repo.approve_visit("P001", visit_id=second["memory_visit_id"])
    approved = [v["chief_complaint"] for v in repo.get_approved_visit_history("P001")]
    check("approving the revision signs only the revision",
          approved == ["Follow-up for diabetes. Fatigue worse."], approved)
    check("the superseded draft is still a draft",
          repo.get_summary("P001")["draft_visit_count"] == 1, repo.get_summary("P001"))


def test_double_approve_is_one_visit():
    print("\nDouble Approve: still one visit")
    repo = _use_temp_repo()
    saved = memory_save_node(_run_state())
    repo.approve_visit("P001", visit_id=saved["memory_visit_id"])
    repo.approve_visit("P001", visit_id=saved["memory_visit_id"])

    summary = repo.get_summary("P001")
    check("one row only", len(summary["visits"]) == 1, summary["visits"])
    check("counted once", summary["visit_count"] == 1, summary)
    check("and it is approved",
          visit_status(summary["visits"][0]) == VISIT_STATUS_APPROVED, summary["visits"][0])


def main():
    print("=" * 70)
    print("P2 DRAFT / APPROVAL FLOW TESTS (pure, no LLM)")
    print("=" * 70)
    try:
        test_state_initialises_the_new_fields()
        test_memory_save_records_a_draft()
        test_memory_save_never_approves_even_when_asked()
        test_escalation_records_nothing()
        test_memory_disabled_records_nothing()
        test_a_draft_is_invisible_to_the_next_run()
        test_result_metadata_reports_the_draft()
        test_a_revision_gets_its_own_row()
        test_double_approve_is_one_visit()
    finally:
        patient_memory_module._DEFAULT_REPO = None

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())