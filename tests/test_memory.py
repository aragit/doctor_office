"""
Long-term patient memory tests - no LLM required.

    python -m tests.test_memory

Uses a temporary store file so the real `data/memory/patients.json` is never
touched by the test run.

The centre of gravity is the draft/approved split: `record_visit` writes an
unsigned draft, and only `approve_visit` turns a visit into longitudinal history.
Most of these tests exist to prove the pipeline cannot do the second one.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory import PatientMemoryStore, parse_patient_context, resolve_patient_id  # noqa: E402
from memory.patient_memory import (  # noqa: E402
    MAX_VISITS_PER_PATIENT,
    VISIT_STATUS_APPROVED,
    VISIT_STATUS_DRAFT,
    visit_status,
)

PASSED, FAILED = [], []

JOHN = """Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%.
"""

JOHN_UPDATED = """Patient: John Doe, 59yo male
Conditions: Type 2 Diabetes, Hypertension, Hyperlipidemia
Medications: Metformin 500mg BID, Lisinopril 10mg daily, Atorvastatin 20mg daily
Allergies: Penicillin
Last Visit: Blood pressure 132/84. HbA1c 7.1%.
"""

CODING = {
    "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
    "secondary_icd10": ["I10 - Essential (primary) hypertension"],
    "cpt_suggestions": ["99213 - Office visit, established patient"],
    "medical_necessity_note": "Follow-up",
}

SOAP = {"assessment": "Diabetes and hypertension well controlled.", "plan": "Continue meds."}


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} -> {detail}")


def test_parsing():
    print("\nPatient profile parsing")
    profile = parse_patient_context(JOHN)
    check("name parsed", profile.name == "John Doe", profile.name)
    check("age parsed", profile.age == 58, profile.age)
    check("gender parsed (female != male)", parse_patient_context("Patient: Mary Smith, 34yo female").gender == "female")
    check("conditions parsed", profile.conditions == ["Type 2 Diabetes", "Hypertension"])
    check("stable id from name", profile.patient_id == "john-doe")
    check("anonymous context still gets an id", parse_patient_context("blah").patient_id.startswith("anon-"))
    check("has_condition", profile.has_condition("diabetes"))
    check("has_condition negative", not profile.has_condition("asthma"))


def test_store_basics(tmp_path: Path):
    print("\nStore basics")
    store = PatientMemoryStore(tmp_path / "patients.json")
    check("unknown patient -> empty summary", store.get_summary("nobody") == {})
    check("unknown patient -> empty history", store.get_visit_history("nobody") == [])
    check("no context block for unknown patient", store.build_context_block("nobody") == "")

    store.ensure_patient("P001", JOHN)
    summary = store.get_summary("P001")
    check("patient created", summary["patient_id"] == "P001")
    check("conditions stored", summary["current"]["conditions"] == ["Type 2 Diabetes", "Hypertension"])
    check("conditions_ever recorded", set(summary["history"]["conditions_ever"]) == {"Type 2 Diabetes", "Hypertension"})
    check("file written to disk", (tmp_path / "patients.json").exists())
    check("file is valid JSON", isinstance(json.loads((tmp_path / "patients.json").read_text()), dict))


def test_persistence_and_reload(tmp_path: Path):
    print("\nPersistence across instances")
    path = tmp_path / "persist.json"
    first = PatientMemoryStore(path)
    first.ensure_patient("P001", JOHN)
    first.record_visit("P001", JOHN, "Routine follow-up for diabetes", CODING, SOAP, [])

    second = PatientMemoryStore(path)  # fresh process simulation
    summary = second.get_summary("P001")
    check("summary survives reload", summary["patient_id"] == "P001")
    check("visit survives reload", second.get_visit_history("P001")[0]["chief_complaint"] == "Routine follow-up for diabetes")
    check("a pipeline run is not a visit on record yet", summary["visit_count"] == 0, summary)
    check("the stored row survives as a draft",
          visit_status(second.get_visit_history("P001")[0]) == VISIT_STATUS_DRAFT)

    second.approve_visit("P001")
    check("approval survives reload too", second.get_summary("P001")["visit_count"] == 1)


def test_visit_history(tmp_path: Path):
    print("\nVisit history")
    store = PatientMemoryStore(tmp_path / "history.json")
    for complaint in ["Follow-up 1: fatigue", "Follow-up 2: dizziness", "Follow-up 3: chest pain"]:
        store.record_visit("P001", JOHN, complaint, CODING, SOAP, [])

    history = store.get_visit_history("P001")
    check("three visits recorded", len(history) == 3, len(history))
    check("newest visit first", history[0]["chief_complaint"] == "Follow-up 3: chest pain", history[0])
    check("codes stored on visit", history[0]["icd10"][0].startswith("E11.9"))

    facts = store.get_longitudinal_facts("P001", limit=2)
    check("drafts are not longitudinal facts", facts == [], facts)

    store.approve_visit("P001", visit_id=history[0]["id"])
    facts = store.get_longitudinal_facts("P001", limit=2)
    check("longitudinal facts formatted after approval", len(facts) == 1 and facts[0].startswith("["), facts)
    check("facts mention the complaint", "chest pain" in facts[0], facts)


def test_history_bound(tmp_path: Path):
    print("\nHistory is bounded")
    store = PatientMemoryStore(tmp_path / "bound.json")
    for i in range(15):
        store.record_visit("P001", JOHN, f"Visit {i}", CODING, SOAP, [])
    summary = store.get_summary("P001")
    check("visit history capped at 10", len(summary["visits"]) == 10, len(summary["visits"]))
    check("15 drafts are still 0 approved visits", summary["visit_count"] == 0, summary)
    check("the drafts that are still stored are counted",
          summary["draft_visit_count"] == 10, summary)

    # Ten approved rows plus a new draft: the draft is the one that has to go,
    # because dropping an approved visit would silently delete history.
    store2 = PatientMemoryStore(tmp_path / "bound2.json")
    for i in range(MAX_VISITS_PER_PATIENT):
        store2.approve_visit(
            "P001",
            visit_id=None if i == 0 else None,
            patient_context=JOHN,
            chief_complaint=f"Approved visit {i}",
            coding_result=CODING,
            soap_note=SOAP,
            quality_flags=[],
        )
    store2.record_visit("P001", JOHN, "New unsigned draft", CODING, SOAP, [])
    summary2 = store2.get_summary("P001")
    check("an eleventh approved row evicts an approved row, not the draft",
          len(summary2["visits"]) == MAX_VISITS_PER_PATIENT, summary2["draft_visit_count"])
    check("the oldest drafts go first",
          summary2["visits"][-1]["chief_complaint"] == "Approved visit 9", summary2["visits"][-1])
    check("approved rows are all still history", summary2["approved_visit_count"] == 10, summary2)


def test_context_block_marks_history_only(tmp_path: Path):
    print("\nContext block marks non-current conditions as historical")
    store = PatientMemoryStore(tmp_path / "ctx.json")
    store.record_visit("P001", JOHN_UPDATED, "New: hyperlipidemia added", CODING, SOAP, [])
    store.approve_visit("P001")
    # Now the live chart no longer lists Hyperlipidemia.
    store.ensure_patient("P001", JOHN)
    block = store.build_context_block("P001")
    check("context block produced", "PATIENT HISTORY" in block, block)
    check("stale condition labelled historical", "Hyperlipidemia" in block and "historical only" in block, block)
    check("the block says it is approved history only",
          "physician-approved visits only" in block, block)


# --- P2: draft vs approved -------------------------------------------------
def test_pipeline_output_is_only_a_draft(tmp_path: Path):
    print("\nP2: a pipeline run records a draft, never history")
    store = PatientMemoryStore(tmp_path / "draft.json")
    record = store.record_visit("P001", JOHN, "Chest pain on exertion", CODING, SOAP, [])

    check("record_visit defaults to draft",
          record["visits"][-1]["status"] == VISIT_STATUS_DRAFT, record["visits"][-1])
    check("the row has no approval timestamp", "approved_at" not in record["visits"][-1])
    check("visit_count stays 0", record["visit_count"] == 0, record)
    check("approved count stays 0", record["approved_visit_count"] == 0, record)
    check("draft count is 1", record["draft_visit_count"] == 1, record)
    check("no last_visit pointer to an unsigned row", record["last_visit"] == {}, record["last_visit"])
    check("approved history is empty",
          store.get_approved_visit_history("P001") == [], store.get_approved_visit_history("P001"))
    check("the scribe sees no longitudinal facts",
          store.get_longitudinal_facts("P001") == [], store.get_longitudinal_facts("P001"))
    check("the scribe context block hides the draft",
          "Chest pain on exertion" not in store.build_context_block("P001"),
          store.build_context_block("P001"))
    check("an explicit approved status is still allowed",
          store.record_visit(
              "P001", JOHN, "Signed elsewhere", CODING, SOAP, [],
              status=VISIT_STATUS_APPROVED,
          )["visit_count"] == 1)


def test_approve_promotes_the_draft_in_place(tmp_path: Path):
    print("\nP2: approve promotes the draft rather than duplicating it")
    store = PatientMemoryStore(tmp_path / "approve.json")
    recorded = store.record_visit("P001", JOHN, "Chest pain on exertion", CODING, SOAP, [])
    visit_id = recorded["visits"][-1]["id"]

    approved = store.approve_visit("P001", visit_id=visit_id)
    check("the same row is promoted", len(approved["visits"]) == 1, approved["visits"])
    check("it is now approved",
          visit_status(approved["visits"][0]) == VISIT_STATUS_APPROVED, approved["visits"][0])
    check("the approval is timestamped", bool(approved["visits"][0].get("approved_at")))
    check("visit_count is now 1", approved["visit_count"] == 1, approved)
    check("draft count is back to 0", approved["draft_visit_count"] == 0, approved)
    check("last_visit points at the approved row",
          approved["last_visit"]["chief_complaint"] == "Chest pain on exertion",
          approved["last_visit"])
    check("the scribe now sees it", store.get_longitudinal_facts("P001") != [])
    check("and it is in the context block",
          "Chest pain on exertion" in store.build_context_block("P001"))

    again = store.approve_visit("P001", visit_id=visit_id)
    check("approving twice does not add a row", len(again["visits"]) == 1, again["visits"])
    check("approving twice does not double-count", again["visit_count"] == 1, again)

    # An Approve with no visit id still has to be a no-op, not a duplicate.
    try:
        store.approve_visit("P001")
        check("an unaddressed approve of a clean store is refused", False, "no error raised")
    except ValueError as exc:
        check("an unaddressed approve of a clean store is refused",
              "no draft visit to approve" in str(exc), str(exc))
    final = store.get_summary("P001")
    check("and it wrote no empty visit",
          len(final["visits"]) == 1 and final["visit_count"] == 1, final)


def test_approve_never_signs_a_different_visit(tmp_path: Path):
    print("\nP2: an unknown visit id does not approve some other draft")
    store = PatientMemoryStore(tmp_path / "mismatch.json")
    store.record_visit("P001", JOHN, "Older draft", CODING, SOAP, [])
    store.record_visit("P001", JOHN, "Newer draft", CODING, SOAP, [])

    # What the app does when run metadata names a row that is no longer there:
    # approve what is on screen, never some other draft that happens to exist.
    record = store.approve_visit(
        "P001",
        visit_id="visit-999",
        patient_context=JOHN,
        chief_complaint="The note on screen",
        coding_result=CODING,
        soap_note=SOAP,
        quality_flags=[],
    )
    check("the stale visit id writes its own approved row",
          record["approved_visit_count"] == 1, record)
    check("and leaves both drafts unsigned", record["draft_visit_count"] == 2, record)
    approved_complaints = [
        visit["chief_complaint"]
        for visit in store.get_approved_visit_history("P001")
    ]
    check("no pre-existing draft was promoted instead",
          approved_complaints == ["The note on screen"], approved_complaints)


def test_revision_supersedes_the_previous_draft(tmp_path: Path):
    print("\nP2: Request Changes produces a new draft that supersedes the old one")
    store = PatientMemoryStore(tmp_path / "revision.json")
    first = store.record_visit("P001", JOHN, "Revision 0", CODING, SOAP, [])
    second = store.record_visit("P001", JOHN, "Revision 1", CODING, SOAP, [])
    check("both runs kept their own row", len(second["visits"]) == 2, second["visits"])
    check("neither is history yet", second["visit_count"] == 0, second)

    newest = second["visits"][-1]["id"]
    approved = store.approve_visit("P001", visit_id=newest)
    check("approving the revision does not approve the draft it replaced",
          approved["approved_visit_count"] == 1, approved)
    check("the superseded row is still a draft",
          visit_status(approved["visits"][0]) == VISIT_STATUS_DRAFT, approved["visits"][0])
    check("the approved row is the one on screen",
          store.get_approved_visit_history("P001")[0]["chief_complaint"] == "Revision 1")
    check("the first row keeps its id", first["visits"][-1]["id"] == approved["visits"][0]["id"])


def test_approve_writes_when_no_draft_exists(tmp_path: Path):
    print("\nP2: an escalated visit has no draft, so approve writes the row")
    store = PatientMemoryStore(tmp_path / "escalated.json")
    record = store.approve_visit(
        "P001",
        visit_id=None,
        patient_context=JOHN,
        chief_complaint="Escalated chest pain",
        coding_result=CODING,
        soap_note=SOAP,
        quality_flags=["Validation: escalated"],
    )
    check("an approved row is written", len(record["visits"]) == 1, record["visits"])
    check("it is approved from the start",
          visit_status(record["visits"][0]) == VISIT_STATUS_APPROVED)
    check("and it counts as history", record["visit_count"] == 1, record)
    check("flags are kept on the row",
          record["visits"][0]["quality_flags"] == ["Validation: escalated"])


def test_reject_is_not_approval(tmp_path: Path):
    print("\nP2: there is no reject that writes, and no reject that approves")
    store = PatientMemoryStore(tmp_path / "reject.json")
    store.record_visit("P001", JOHN, "Rejected note", CODING, SOAP, [])

    check("the store exposes no reject API",
          not hasattr(store, "reject_visit"), dir(store))
    summary = store.get_summary("P001")
    check("nothing about the store changed without an approval",
          summary["visit_count"] == 0 and summary["draft_visit_count"] == 1, summary)
    check("the draft is still unread as history",
          store.get_longitudinal_facts("P001") == [])


def test_legacy_rows_read_as_approved(tmp_path: Path):
    print("\nP2: legacy rows with no status are treated as approved")
    path = tmp_path / "legacy.json"
    legacy = {
        "schema_version": 1,
        "patients": {
            "P001": {
                "patient_id": "P001",
                "name": "John Doe",
                "current": {"conditions": ["Type 2 Diabetes"]},
                "history": {"conditions_ever": ["Type 2 Diabetes"]},
                "visit_count": 1,
                "visits": [
                    {
                        "timestamp": "2026-01-02T09:00:00",
                        "chief_complaint": "Pre-split visit",
                        "icd10": ["E11.9"],
                        "cpt": [],
                        "assessment": "ok",
                        "plan": "ok",
                        "quality_flags": [],
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    store = PatientMemoryStore(path)
    check("a status-less row reads as approved",
          visit_status(store.get_visit_history("P001")[0]) == VISIT_STATUS_APPROVED)
    check("it counts as approved history", store.get_summary("P001")["approved_visit_count"] == 1)
    check("it is visible to the scribe", store.get_longitudinal_facts("P001") != [])
    check("and it is not silently discarded",
          "Pre-split visit" in store.build_context_block("P001"))

    # A legacy count above the stored row count must not be lost on first write.
    legacy["patients"]["P001"]["visits"][0]["status"] = VISIT_STATUS_DRAFT
    path.write_text(json.dumps(legacy), encoding="utf-8")
    migrated = PatientMemoryStore(path)
    migrated.record_visit("P001", JOHN, "New draft", CODING, SOAP, [])
    check("an explicit draft status is honoured over the legacy default",
          migrated.get_summary("P001")["draft_visit_count"] == 2, migrated.get_summary("P001"))
    check("a legacy visit_count is never lowered by a write",
          migrated.get_summary("P001")["visit_count"] == 1, migrated.get_summary("P001"))


def test_unknown_status_is_rejected(tmp_path: Path):
    print("\nP2: an unknown status is refused rather than stored")
    store = PatientMemoryStore(tmp_path / "bad.json")
    try:
        store.record_visit("P001", JOHN, "Bad status", CODING, SOAP, [], status="signed")
        check("record_visit refuses an unknown status", False, "no error raised")
    except ValueError as exc:
        check("record_visit refuses an unknown status", "signed" in str(exc), str(exc))
    check("nothing was written", store.get_summary("P001") == {}, store.get_summary("P001"))


def test_corrupt_file_recovers(tmp_path: Path):
    print("\nCorrupt store file recovers")
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = PatientMemoryStore(path)
    summary = store.get_summary("P001")
    check("corrupt file does not crash", summary == {})
    store.ensure_patient("P001", JOHN)
    check("store is writable again after corruption", store.get_summary("P001")["patient_id"] == "P001")


def test_resolve_patient_id():
    print("\nPatient id resolution")
    check("explicit id wins", resolve_patient_id("P001", JOHN) == "P001")
    check("falls back to name slug", resolve_patient_id(None, JOHN) == "john-doe")
    check("blank id falls back", resolve_patient_id("", JOHN) == "john-doe")


def main():
    print("=" * 70)
    print("PATIENT MEMORY TESTS (no LLM required)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_parsing()
        test_store_basics(tmp_path)
        test_persistence_and_reload(tmp_path)
        test_visit_history(tmp_path)
        test_history_bound(tmp_path)
        test_context_block_marks_history_only(tmp_path)
        test_pipeline_output_is_only_a_draft(tmp_path)
        test_approve_promotes_the_draft_in_place(tmp_path)
        test_approve_never_signs_a_different_visit(tmp_path)
        test_revision_supersedes_the_previous_draft(tmp_path)
        test_approve_writes_when_no_draft_exists(tmp_path)
        test_reject_is_not_approval(tmp_path)
        test_legacy_rows_read_as_approved(tmp_path)
        test_unknown_status_is_rejected(tmp_path)
        test_corrupt_file_recovers(tmp_path)
    test_resolve_patient_id()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
