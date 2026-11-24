"""
Headless UI tests for the Physician Review tab - no LLM calls required.

Run with:

    python -m tests.test_physician_review_ui

These drive the real app script through ``streamlit.testing.v1.AppTest`` with the
pipeline, the vector store and the memory store stubbed, so the "Submit Feedback
& Regenerate" flow is exercised end to end without Ollama or chromadb.

The regressions these lock in:

* when the scribe agent has to default a missing field (typically
  ``icd10_suggestions``), the app must still render the review tab instead of
  crashing;
* Phase A5: the review state machine. Every state says what the note is and
  gates the decision buttons, so a note that is already approved or rejected
  cannot be approved twice, a rejection is never counted as an approval, and a
  regeneration keeps the specialty, the encounter data and the A4 panels.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def _install_stubs(result_overrides=None):
    """Replace the vector store, pipeline and memory store so the UI runs offline.

    The memory store is redirected to a temp file so an Approve can be exercised
    without writing to the real ``data/memory/patients.json``, and every
    ``record_visit`` call is recorded so "no duplicate write" is checkable.

    Returns a dict of recorders, one entry per pipeline call: ``feedback``,
    ``visit_context``, ``specialty``, plus ``record_visit`` (the memory writes)
    and ``repo``.
    """
    import rag.simple_rag as simple_rag

    class _Doc:
        page_content = "Type 2 Diabetes management. HbA1c 7.8%. Continue metformin."

    class _VectorStore:
        def similarity_search(self, query, k=3):
            return [_Doc()]

    simple_rag.get_vectorstore = lambda *a, **k: _VectorStore()

    # A cardiology visit that exercises every Phase A4 surface at once: a symptom
    # code the Coding node completed, a supervisor fallback, a fidelity advisory,
    # a cardiac-claim scrub and an incomplete-note flag.
    result = _result_fixture()
    for key, value in (result_overrides or {}).items():
        if key == "metadata":
            result["metadata"] = {**result["metadata"], **value}
        else:
            result[key] = value

    calls = {"feedback": [], "visit_context": [], "specialty": [],
             "record_visit": [], "approve_visit": []}

    import agents.orchestrator as orch

    def fake_stream(patient_context, chief_complaint, retrieved_notes, mode, model_name=None,
                    patient_id=None, use_memory=True, physician_feedback=None,
                    visit_context=None, specialty=None, is_synthetic=False):
        calls["feedback"].append(physician_feedback)
        calls["visit_context"].append(visit_context)
        calls["specialty"].append(specialty)
        for step in ("scribe", "human_review"):
            snapshot = dict(result)
            snapshot["metadata"] = dict(
                result["metadata"], graph=step, visit_context=dict(visit_context or {})
            )
            yield snapshot

    orch.stream_visit_pipeline = fake_stream

    import memory.patient_memory as patient_memory
    from memory.patient_memory import JsonPatientMemoryRepository

    repo = JsonPatientMemoryRepository(path=Path(tempfile.mkdtemp()) / "patients.json")
    originals = {"record_visit": repo.record_visit, "approve_visit": repo.approve_visit}

    def recording(name):
        def wrapper(*args, **kwargs):
            calls[name].append((args, kwargs))
            return originals[name](*args, **kwargs)
        return wrapper

    repo.record_visit = recording("record_visit")
    repo.approve_visit = recording("approve_visit")
    patient_memory.get_memory_store = lambda *a, **k: repo
    calls["repo"] = repo
    return calls


def _result_fixture():
    """A run result carrying symptom provenance, fidelity and scrub advisories.

    Shaped exactly like `_state_to_result` output, so the UI test and the
    metadata-shape test read the same fixture.
    """
    return {
        "soap_note": {
            "subjective": "Increased fatigue x2 weeks.",
            "objective": "BP 148/92, HbA1c 7.8%.",
            "assessment": "Suboptimal glycemic control.",
            "plan": "Continue metformin 500mg BID. Recheck HbA1c in 3 months.",
            "icd10_suggestions": [],
        },
        "coding_result": {
            "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
            "secondary_icd10": [
                "I10 - Essential hypertension",
                "R07.9 - Chest pain, unspecified",
                "R19.7 - Diarrhea, unspecified",
            ],
            "cpt_suggestions": ["99214"],
            "medical_necessity_note": "Diabetes management with medication review.",
            "cpt_grounding": None,
        },
        "quality_flags": [
            "Scribe content incomplete: no ICD-10 suggestions (defaulted by the pipeline)",
            "Coding: added symptom code R07.9 - Chest pain, unspecified "
            "('chest pain' documented in chief complaint); the Coding pass omitted it",
            "Specialty data documented this visit is not reflected in the OBJECTIVE "
            "section: ECG / Rhythm Notes - carry the documented findings into that section",
            "Cardiac claim not supported by the chart: ischemic etiology, cardiac "
            "diagnostic claim - no cardiac diagnosis is listed in Patient Context, "
            "so describe this as a symptom and plan follow-up instead",
            "Cleanup: Removed unsupported cardiac claim from assessment: cardiac diagnostic claim",
        ],
        "metadata": {
            "orchestrator": "langgraph",
            "mode": "fast",
            "graph": "human_review",
            "timing": {"total_seconds": 1.0},
            "supervisor": {
                "retry_count": 0,
                "needs_human_review": False,
                "applied": [
                    "Added symptom code R19.7 - Diarrhea, unspecified "
                    "(raised by validation, present in chief complaint)"
                ],
                "feedback": [],
            },
            "symptom_coding": {
                "candidates": [
                    {
                        "code": "R07.9",
                        "description": "R07.9 - Chest pain, unspecified",
                        "prefix": "R07",
                        "phrase": "chest pain",
                        "source": "chief_complaint",
                    },
                    {
                        "code": "R19.7",
                        "description": "R19.7 - Diarrhea, unspecified",
                        "prefix": "R19",
                        "phrase": "diarrhea",
                        "source": "chief_complaint",
                    },
                ],
                "coding_added": ["R07.9 - Chest pain, unspecified"],
                "supervisor_added": ["R19.7 - Diarrhea, unspecified"],
            },
            "memory": {"stored": False, "visit_count": 0},
            "tools": {},
        },
    }


def _labels(at, kind):
    try:
        return [str(getattr(e, "value", "")) for e in at.get(kind)]
    except Exception:
        return []


def _all_text(at):
    """Every string the app rendered, across every element kind."""
    return " ".join(
        value
        for kind in (
            "markdown", "write", "subheader", "header", "text", "caption",
            "info", "warning", "error", "success",
        )
        for value in _labels(at, kind)
    )


def _button(at, label):
    """The first button whose label contains `label`, or None."""
    found = [b for b in at.button if label in b.label]
    return found[0] if found else None


def _run_pipeline_button(at):
    return [b for b in at.button if "Run Pipeline" in b.label][0]


def _boot(chief_complaint="Chest pain and diarrhea for three days.", result_overrides=None):
    """Boot the app, run the pipeline once, return the live AppTest and the stubs."""
    from streamlit.testing.v1 import AppTest

    calls = _install_stubs(result_overrides=result_overrides)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app boots without exception", not at.exception, [str(e.value) for e in at.exception])
    at.text_area[0].set_value(chief_complaint).run(timeout=120)
    _run_pipeline_button(at).click().run(timeout=180)
    check("pipeline run renders without exception", not at.exception,
          [str(e.value) for e in at.exception])
    return at, calls


def test_feedback_regenerate_survives_empty_icd10():
    print("\nSubmit Feedback & Regenerate with a defaulted SOAP note")
    from streamlit.testing.v1 import AppTest

    calls = _install_stubs()
    feedback_seen = calls["feedback"]

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app loads without exception", not at.exception, [str(e.value) for e in at.exception])

    complaints = at.text_area
    check("chief complaint input present", bool(complaints))
    complaints[0].set_value("Follow-up for diabetes. Reports fatigue.").run(timeout=120)

    run_buttons = [b for b in at.button if "Run Pipeline" in b.label]
    check("Run Pipeline button present", bool(run_buttons))
    run_buttons[0].click().run(timeout=120)
    check("no exception after the initial run", not at.exception, [str(e.value) for e in at.exception])
    check("initial run succeeded", any("Pipeline Completed Successfully" in s.value for s in at.success),
          [s.value for s in at.success])

    tab_labels = [str(t.label) for t in at.tabs]
    check("Physician Review tab present", any("Physician Review" in l for l in tab_labels), tab_labels)

    body = " ".join(t for kind in ("markdown", "write", "subheader", "header", "text") for t in _labels(at, kind))
    check("review tab shows the note", "Continue metformin 500mg BID" in body)
    check("incomplete-note flag is visible to the physician in plain language",
          any("Note was incomplete" in w.value for w in at.warning),
          [w.value for w in at.warning])
    check("the raw flag text is still available for debugging",
          any("no ICD-10 suggestions" in c.value for c in at.caption)
          or any("no ICD-10 suggestions" in str(getattr(t, "value", "")) for t in at.expander),
          [c.value for c in at.caption])

    request = [b for b in at.button if "Request Changes" in b.label]
    check("Request Changes button present", bool(request))
    request[0].click().run(timeout=120)
    check("no exception after Request Changes", not at.exception, [str(e.value) for e in at.exception])

    feedback_boxes = [w for w in at.text_area if "feedback" in str(w.label).lower()]
    check("feedback text area appears", bool(feedback_boxes), [w.label for w in at.text_area])

    submit = [b for b in at.button if "Submit Feedback & Regenerate" in b.label]
    check("Submit Feedback & Regenerate present", bool(submit), [b.label for b in at.button])
    feedback_boxes[0].set_value("Add ICD-10 codes and clarify the follow-up plan.").run(timeout=120)

    submit = [b for b in at.button if "Submit Feedback & Regenerate" in b.label]
    submit[0].click().run(timeout=180)

    check("NO EXCEPTION after Submit Feedback & Regenerate", not at.exception, [str(e.value) for e in at.exception])
    check("revision banner shown", any("Revision completed" in s.value for s in at.success), [s.value for s in at.success])
    check("no pipeline error banner", not any("Pipeline Error" in e.value for e in at.error), [e.value for e in at.error])
    check("physician feedback reached the pipeline",
          feedback_seen and feedback_seen[-1] == "Add ICD-10 codes and clarify the follow-up plan.", feedback_seen)


def test_rich_visit_context_round_trip():
    """HPI / vitals / exam / findings reach the pipeline and come back for review."""
    print("\nRich visit context: entered -> forwarded -> shown in Physician Review")
    from streamlit.testing.v1 import AppTest

    visit_context_seen = _install_stubs()["visit_context"]

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app loads with the new inputs", not at.exception, [str(e.value) for e in at.exception])

    areas = {str(a.label): a for a in at.text_area}
    inputs_by_key = {str(i.label): i for i in at.text_input}
    check("HPI input present", any("HPI" in l for l in areas), list(areas))
    check("physical exam input present", any("Physical Examination" in l for l in areas), list(areas))
    check("additional findings input present",
          any("Additional Findings" in l for l in areas), list(areas))
    check("structured vitals inputs present",
          all(any(tag in l for l in inputs_by_key) for tag in ("BP", "HR", "Temp", "RR", "SpO")),
          list(inputs_by_key))

    # A chief-complaint-only visit must still work.
    at.text_area[0].set_value("Follow-up for diabetes.").run(timeout=120)
    run = [b for b in at.button if "Run Pipeline" in b.label][0]
    run.click().run(timeout=120)
    check("chief-complaint-only run sends no visit context",
          visit_context_seen and visit_context_seen[-1] == {}, visit_context_seen)

    def _fill():
        """Enter a complete encounter, then re-run so the button reflects it."""
        at.text_area[0].set_value("Follow-up for diabetes. Fatigue x3 weeks.")
        at.text_area(key="vc_hpi").set_value(
            "Fatigue x3 weeks with polydipsia; denies chest pain."
        )
        at.text_input(key="vc_bp").set_value("148/92")
        at.text_input(key="vc_hr").set_value("78")
        at.text_input(key="vc_temp").set_value("98.6")
        at.text_input(key="vc_rr").set_value("16")
        at.text_input(key="vc_spo2").set_value("98")
        at.text_area(key="vc_exam").set_value("Dry mucous membranes. No pedal edema.")
        at.text_area(key="vc_findings").set_value("HbA1c 7.8%")
        return at.run(timeout=120)

    at = _fill()
    check("no exception while filling encounter details", not at.exception,
          [str(e.value) for e in at.exception])

    run = [b for b in at.button if "Run Pipeline" in b.label][0]
    run.click().run(timeout=120)
    check("no exception after the enriched run", not at.exception, [str(e.value) for e in at.exception])

    sent = visit_context_seen[-1] if visit_context_seen else {}
    check("vitals reached the pipeline", sent.get("blood_pressure") == "148/92"
          and sent.get("heart_rate") == "78" and sent.get("spo2") == "98", sent)
    check("HPI reached the pipeline", "polydipsia" in sent.get("hpi", ""), sent)
    check("exam reached the pipeline", "Dry mucous membranes" in sent.get("physical_exam", ""), sent)
    check("findings reached the pipeline", "HbA1c" in sent.get("additional_findings", ""), sent)
    check("blank optional fields are not sent", "weight_kg" not in sent and "bmi" not in sent, sent)

    body = " ".join(t for kind in ("markdown", "write", "subheader", "header", "text")
                    for t in _labels(at, kind))
    check("review shows the visit context heading", "Visit Context Used" in body)
    check("review shows the HPI", "polydipsia" in body, body[-400:])
    check("review shows the vitals with units", "148/92 mmHg" in body, body[-400:])
    check("review shows the exam", "Dry mucous membranes" in body)


def test_a4_symptom_provenance_and_flag_groups():
    """Phase A4: symptom-code origins and grouped flags reach the physician."""
    print("\nPhase A4 surfaces: symptom provenance, fidelity notes, grouped flags")
    from streamlit.testing.v1 import AppTest

    _install_stubs()

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app loads with the A4 panels", not at.exception, [str(e.value) for e in at.exception])

    at.text_area[0].set_value("Chest pain and diarrhea for three days.").run(timeout=120)
    run = [b for b in at.button if "Run Pipeline" in b.label][0]
    run.click().run(timeout=180)
    check("no exception on an A4 run", not at.exception, [str(e.value) for e in at.exception])

    body = " ".join(
        value
        for kind in ("markdown", "write", "subheader", "header", "text", "caption")
        for value in _labels(at, kind)
    )

    # --- 1. Symptom coding section ---------------------------------------
    check("symptom coding section is shown", "Symptom coding" in body, body[-600:])
    check("both candidates are counted",
          "2 candidate(s) considered" in body, body[-600:])
    check("the Coding completion is stated in the header",
          "1 added by Coding" in body, body[-600:])
    check("the Supervisor fallback is stated in the header",
          "1 added by Supervisor" in body, body[-600:])
    check("the compact review lines name each origin",
          "Added by Coding" in body and "Added by Supervisor" in body, body[-900:])
    check("the justifying phrase is shown",
          "chest pain" in body and "diarrhea" in body, body[-900:])
    check("the source of the candidate is shown",
          "chief complaint" in body, body[-900:])

    # The Coding tab renders the same information as a table.
    table_text = " ".join(str(getattr(t, "value", "")) for t in at.table)
    check("the Coding tab has a symptom-coding table",
          "R07.9" in table_text and "R19.7" in table_text, table_text[:400])
    check("the table shows the outcome per candidate",
          "Added by Coding" in table_text and "Added by Supervisor" in table_text,
          table_text[:400])
    check("the table shows where the symptom was found",
          "Found in" in table_text and "chief complaint" in table_text, table_text[:400])

    # --- 2. Grouped quality flags ----------------------------------------
    check("auto-fixed group heading is shown", "Completed automatically" in body, body[-900:])
    check("review group heading is shown", "Review advised" in body, body[-900:])
    check("safety group heading is shown", "Removed by a safety check" in body, body[-900:])
    check("each group explains itself",
          "Added for you" in body and "Only you can judge" in body
          and "Taken out of the note" in body, body[-900:])

    # Buckets render at three levels: auto-fixed = info, review advised =
    # warning, safety scrub = error.
    notices = (
        [i.value for i in at.info]
        + [w.value for w in at.warning]
        + [e.value for e in at.error]
    )
    notices_text = " ".join(notices)
    check("the Coding completion reads in plain language",
          "Added symptom code R07.9" in notices_text, notices_text)
    check("the Supervisor fallback reads in plain language",
          "Added symptom code R19.7" in notices_text, notices_text)
    check("the cardiac scrub reads in plain language",
          "Removed a cardiac diagnosis" in notices_text, notices_text)
    check("the cardiac claim advisory reads in plain language",
          "Cardiac wording not supported by the chart" in notices_text, notices_text)
    check("auto-fixed entries render as information, not errors",
          any("Added symptom code R07.9" in i.value for i in at.info), [i.value for i in at.info])
    check("safety removals render as errors",
          any("Removed a cardiac diagnosis" in e.value for e in at.error),
          [e.value for e in at.error])

    # --- 3. Specialty fidelity notes -------------------------------------
    check("fidelity section is shown", "Specialty field fidelity" in body, body[-900:])
    check("the under-reflected field is named in one line",
          "ECG / Rhythm Notes - not reflected in the Objective section" in notices_text,
          notices_text)

    # --- 4. Raw strings are still reachable -------------------------------
    raw_json = " ".join(str(getattr(j, "value", "")) for j in at.json)
    check("raw flags remain in the Raw JSON tab",
          "no ICD-10 suggestions" in raw_json
          and "ECG / Rhythm Notes" in raw_json,
          raw_json[:300])
    check("raw flags remain in the technical detail expander",
          any("no ICD-10 suggestions" in c.value for c in at.caption),
          [c.value for c in at.caption][:10])


def test_a4_degrades_without_metadata():
    """A result with no A4 metadata must render, not crash."""
    print("\nPhase A4 fallback: a legacy result with no symptom metadata")
    from streamlit.testing.v1 import AppTest

    _install_stubs()
    legacy = _result_fixture()
    legacy["metadata"] = {
        k: v for k, v in legacy["metadata"].items() if k != "symptom_coding"
    }
    legacy["quality_flags"] = ["Scribe content incomplete: no ICD-10 suggestions"]

    import agents.orchestrator as orch

    def fake_stream(*args, **kwargs):
        for step in ("scribe", "human_review"):
            snapshot = dict(legacy)
            snapshot["metadata"] = dict(legacy["metadata"], graph=step)
            yield snapshot

    orch.stream_visit_pipeline = fake_stream

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    at.text_area[0].set_value("Chest pain for three days.").run(timeout=120)
    [b for b in at.button if "Run Pipeline" in b.label][0].click().run(timeout=180)

    check("no exception without symptom metadata", not at.exception,
          [str(e.value) for e in at.exception])
    body = " ".join(
        value
        for kind in ("markdown", "write", "subheader", "header", "text", "caption")
        for value in _labels(at, kind)
    )
    check("an empty symptom section explains itself",
          "No symptom-code candidates were derived" in body, body[-500:])
    check("the rest of the review tab still renders",
          "Note was incomplete" in body or "incomplete" in body.lower(), body[-500:])
    check("no fidelity section when no fidelity flag fired",
          "Specialty field fidelity" not in body, body[-500:])


# ---------------------------------------------------------------------------
# Phase A5: the review state machine
# ---------------------------------------------------------------------------


def test_a5_state_machine_decisions():
    """Every state resolves to one label, one severity and one set of actions."""
    print("\nPhase A5 state machine: states, labels and allowed actions")
    from utils.review_state import (
        allowed_actions,
        approval_preview,
        key_codes,
        regeneration_progress_message,
        review_state,
        revision_summary,
    )

    all_three = ("approve", "request_changes", "reject")

    draft = review_state(None)
    check("a fresh run is awaiting review", draft.key == "awaiting_review", draft)
    check("the draft banner is informational, not a green light", draft.level == "info", draft)
    check("the draft says it is not final", "not final" in draft.detail, draft.detail)
    check("a draft offers all three decisions",
          allowed_actions(draft) == all_three, allowed_actions(draft))
    check("a draft is not terminal", not draft.terminal, draft)

    recorded_draft = review_state(None, draft_recorded=True)
    check("a draft the pipeline recorded says so",
          "unsigned draft" in recorded_draft.detail, recorded_draft.detail)
    check("a recorded draft is explicitly not history",
          not recorded_draft.in_history, recorded_draft)
    check("a recorded draft is marked as recorded", recorded_draft.draft_recorded, recorded_draft)
    check("the draft banner says the scribe will not read it",
          "Scribe will not read it" in recorded_draft.detail, recorded_draft.detail)

    unrecorded = review_state(None)
    check("a run that recorded nothing does not claim a draft",
          not unrecorded.draft_recorded, unrecorded)
    check("and it says nothing is history until approval",
          "until you approve" in unrecorded.detail, unrecorded.detail)

    escalated = review_state(None, needs_human_review=True)
    check("an escalated run is announced as escalated",
          escalated.key == "escalated" and escalated.level == "error", escalated)
    check("an escalated run records nothing at all",
          not escalated.in_history and not escalated.draft_recorded, escalated)
    check("an escalated run is still the physician's to decide",
          allowed_actions(escalated) == all_three, allowed_actions(escalated))

    changes = review_state("changes_requested")
    check("requesting changes is announced", changes.key == "changes_requested", changes)
    check("requesting changes closes every decision",
          allowed_actions(changes) == (), allowed_actions(changes))
    check("requesting changes explains where the feedback goes",
          "next pipeline run" in changes.detail, changes.detail)

    waiting = review_state("changes_requested", feedback_text="Add R07.9 for chest pain.")
    check("written feedback is reported as ready to submit",
          "ready" in waiting.detail, waiting.detail)
    check("empty feedback is not mistaken for submitted feedback",
          "Write what should change" in review_state("changes_requested").detail)

    promoted = review_state("approved", approved=True, draft_recorded=True)
    check("an approval that promoted the pipeline draft is a success",
          promoted.key == "approved_saved"
          and promoted.level == "success", promoted)
    check("it says the draft was promoted, not duplicated",
          "promoted that row rather than writing a second visit" in promoted.detail,
          promoted.detail)
    check("only an approval makes the visit history",
          promoted.in_history, promoted)

    approved_on_screen = review_state("approved", approved=True, draft_recorded=False)
    check("an approval made on the review screen says it was stored",
          "Stored just now" in approved_on_screen.detail, approved_on_screen.detail)
    check("an approved note is terminal", approved_on_screen.terminal, approved_on_screen)
    check("an approved note cannot be approved again",
          "approve" not in allowed_actions(approved_on_screen),
          allowed_actions(approved_on_screen))
    check("an approved note can only be revised or re-run",
          allowed_actions(approved_on_screen) == ("revise",),
          allowed_actions(approved_on_screen))

    unsaved = review_state("approved", approved=False, memory_enabled=False)
    check("approving with memory off is not reported as saved",
          unsaved.key == "approved_not_saved" and unsaved.level == "warning", unsaved)
    check("approving with memory off says why",
          "Long-term memory is off" in unsaved.detail, unsaved.detail)
    check("approving with memory off leaves nothing in history", not unsaved.in_history, unsaved)

    rejected = review_state("rejected", draft_recorded=True)
    check("a rejection is announced as not final",
          rejected.key == "rejected" and "not final" in rejected.label, rejected)
    check("a rejection closes every decision", allowed_actions(rejected) == ("revise",),
          allowed_actions(rejected))
    check("a rejection is never an approval", not rejected.in_history, rejected)
    check("a rejection says the unsigned draft stays",
          "does not delete it" in rejected.detail, rejected.detail)
    check("a rejection says the draft is not history",
          "does not read it as history" in rejected.detail, rejected.detail)
    quiet_reject = review_state("rejected", draft_recorded=False)
    check("a rejection writes nothing of its own",
          "stored nothing as history" in quiet_reject.detail, quiet_reject.detail)

    failed = review_state(None, error="disk full")
    check("a failed save is announced", failed.key == "save_failed", failed)
    check("a failed save reports the reason", "disk full" in failed.detail, failed.detail)
    check("a failed save leaves Approve open to retry",
          "approve" in allowed_actions(failed), allowed_actions(failed))
    check("a failed save is not terminal", not failed.terminal, failed)

    revised = review_state(None, revision_count=1)
    check("a regenerated draft is labelled as a revision",
          "Revised draft (revision 1)" == revised.label, revised.label)
    check("a revised draft is not presented as the previous version",
          "not edited in place" in revised.detail, revised.detail)
    check("a revised draft is still awaiting a decision",
          allowed_actions(revised) == all_three, allowed_actions(revised))
    check("the second revision is numbered from the newest",
          "Revised draft (revision 2)" == review_state(None, revision_count=2).label)

    # --- what an Approve would store -------------------------------------
    result = _result_fixture()
    preview = approval_preview(
        result,
        patient_name="John Doe",
        patient_id="P001",
        chief_complaint="Chest pain and diarrhea for three days.",
        specialty_name="Cardiology",
    )
    check("the confirmation line names the patient",
          "John Doe (P001)" in preview.summary, preview.summary)
    check("the confirmation line quotes the chief complaint",
          "Chest pain and diarrhea" in preview.summary, preview.summary)
    check("the confirmation line lists the key codes",
          "E11.9" in preview.summary and "R07.9" in preview.summary, preview.summary)
    check("the confirmation line names the specialty",
          "Cardiology" in preview.summary, preview.summary)
    check("an ordinary approval persists", preview.will_persist, preview)
    detail = " ".join(preview.detail_lines)
    check("the stored breakdown names the ICD-10 list", "`R19.7`" in detail, detail)
    check("the stored breakdown names the CPT list", "`99214`" in detail, detail)
    check("the stored breakdown says the SOAP is truncated",
          "truncated to 400 characters" in detail, detail)
    check("the stored breakdown admits Subjective/Objective are not stored",
          "Subjective and Objective" in detail and "not part of the stored visit" in detail,
          detail)
    check("the stored breakdown names the rolling history window",
          "10 visits" in detail, detail)

    off = approval_preview(result, patient_name="John Doe", patient_id="P001",
                           memory_enabled=False)
    check("approving with memory off persists nothing", not off.will_persist, off)
    check("approving with memory off warns before the click",
          "Long-term memory is off" in off.warning, off.warning)

    recorded = approval_preview(result, patient_name="John Doe", patient_id="P001",
                                draft_recorded=True)
    check("a pipeline draft is promoted, not duplicated",
          recorded.already_saved and recorded.will_persist, recorded)
    check("the confirmation line says it becomes approved history",
          "promotes this run's draft into approved history" in recorded.summary,
          recorded.summary)
    check("and promises no second visit",
          "No second visit is written" in recorded.summary, recorded.summary)
    check("the breakdown explains the promotion",
          any("Draft promoted, not duplicated" in line for line in recorded.detail_lines),
          recorded.detail_lines)
    check("the breakdown says approving is what creates history",
          any("only ones the" in line for line in recorded.detail_lines),
          recorded.detail_lines)

    off_draft = approval_preview(result, patient_name="John Doe", patient_id="P001",
                                 memory_enabled=False, draft_recorded=True)
    check("a draft with memory off still persists nothing",
          not off_draft.will_persist and off_draft.already_saved, off_draft)

    check("codes are listed primary first and de-duplicated",
          key_codes(result["coding_result"]) == ["E11.9", "I10", "R07.9", "R19.7"],
          key_codes(result["coding_result"]))
    check("a result with no codes degrades to a plain phrase",
          "no codes" in approval_preview({}).summary, approval_preview({}).summary)

    # --- revision provenance --------------------------------------------
    lines = revision_summary(["first ask", "second ask"])
    check("each revision is named, newest first",
          lines[0].startswith("Revision 2") and lines[1].startswith("Revision 1"), lines)
    check("a session with no revisions lists none", revision_summary([]) == [])
    check("the regeneration progress line says the feedback is being used",
          "Regenerating with your feedback" in regeneration_progress_message("add R07.9"),
          regeneration_progress_message("add R07.9"))
    check("a first run gets no regeneration wording",
          regeneration_progress_message(None) == "" and regeneration_progress_message("") == "")


def test_a5_approve_state_and_confirmation():
    """Approve: one confirmation line, one write, then the decisions close."""
    print("\nPhase A5 approve: what is stored, and no double-approve")
    at, calls = _boot()

    body = _all_text(at)
    check("a fresh draft is announced as awaiting review", "Awaiting review" in body, body[-400:])

    approve = _button(at, "Approve & Save")
    request = _button(at, "Request Changes")
    reject = _button(at, "Reject")
    check("the approve confirmation names the patient", "John Doe (P001)" in body, body[-700:])
    check("the approve confirmation quotes the chief complaint",
          "Chest pain and diarrhea for three days" in body, body[-700:])
    check("the approve confirmation lists the key codes",
          "E11.9, I10, R07.9" in body, body[-700:])
    check("the approve confirmation names the specialty",
          "General / Internal Medicine" in body, body[-700:])
    check("the approve confirmation warns about the truncated SOAP",
          "truncated to 400 characters" in body, body[-900:])
    check("all three decisions are enabled on a draft",
          not approve.disabled and not request.disabled and not reject.disabled)
    check("no revise action is offered before a decision", _button(at, "Revise") is None)

    approve.click().run(timeout=120)
    check("approving does not crash", not at.exception, [str(e.value) for e in at.exception])
    body = _all_text(at)
    check("approval is announced as a success state",
          "Approved & saved" in body, body[-400:])
    check("the approval says when the visit was written",
          "Stored just now" in body, body[-400:])
    check("approval goes through approve_visit, the only path to history",
          len(calls["approve_visit"]) == 1, calls["approve_visit"])
    # P2: this fixture's run recorded nothing, so approve_visit writes the
    # approved row itself - through record_visit, with the status spelled out.
    check("the write happens exactly once",
          len(calls["record_visit"]) == 1, calls["record_visit"])
    check("and it is written as approved, never as a bare draft",
          calls["record_visit"][0][1].get("status") == "approved",
          calls["record_visit"][0][1].get("status"))
    args, kwargs = calls["approve_visit"][0]
    check("the write is keyword-only, as approve_visit requires", not args, args)
    repo = calls["repo"]
    check("the visit is approved history after the click",
          repo.get_summary("P001").get("visit_count") == 1, repo.get_summary("P001"))
    check("the UI says the visit is in patient history",
          "In patient history:** Yes" in body, body[-900:])
    check("the write carries the reviewed chief complaint",
          (kwargs.get("chief_complaint") or "").startswith("Chest pain and diarrhea"),
          kwargs.get("chief_complaint"))
    check("the write carries the reviewed codes",
          "E11.9" in (kwargs.get("coding_result") or {}).get("primary_icd10", ""),
          kwargs.get("coding_result"))
    check("approve is closed after success", _button(at, "Approve & Save").disabled)
    check("request changes is closed after success",
          _button(at, "Request Changes").disabled)
    check("reject is closed after success", _button(at, "Reject").disabled)
    revise = _button(at, "Revise")
    check("a revise action replaces the closed decisions",
          revise is not None and not revise.disabled, [b.label for b in at.button])


def test_a5_approve_with_memory_off_warns():
    """Approve with long-term memory off says it stores nothing."""
    print("\nPhase A5 approve with memory disabled")
    at, calls = _boot()

    memory_toggle = [t for t in at.toggle]
    check("the long-term memory toggle is present", bool(memory_toggle))
    memory_toggle[0].set_value(False).run(timeout=120)

    body = _all_text(at)
    check("the warning appears before the approval is clicked",
          "Long-term memory is off" in body or "will not persist" in body, body[-700:])
    check("the confirmation line says nothing will be stored",
          "Approving stores nothing" in body, body[-700:])

    _button(at, "Approve & Save").click().run(timeout=120)
    check("approving with memory off does not crash", not at.exception,
          [str(e.value) for e in at.exception])
    body = _all_text(at)
    check("the state is approved but not saved",
          "Approved - not stored" in body, body[-400:])
    check("the banner explains that nothing was persisted",
          "not in long-term memory" in body, body[-400:])
    check("no visit was written", not calls["record_visit"], calls["record_visit"])
    check("approve is still closed after the decision",
          _button(at, "Approve & Save").disabled)


def test_a5_reject_state_and_recovery():
    """Reject: nothing is written, and the review is not stuck."""
    print("\nPhase A5 reject: not final, nothing stored, recoverable")
    at, calls = _boot()

    _button(at, "Reject").click().run(timeout=120)
    check("rejecting does not crash", not at.exception, [str(e.value) for e in at.exception])
    body = _all_text(at)
    check("rejection is announced as not final",
          "Rejected - not final" in body, body[-400:])
    check("the banner says nothing became history",
          "rejecting it stored nothing as history" in body, body[-400:])
    check("reject never calls record_visit", not calls["record_visit"], calls["record_visit"])
    check("approve is closed after a rejection", _button(at, "Approve & Save").disabled)
    check("request changes is closed after a rejection",
          _button(at, "Request Changes").disabled)
    check("reject is closed after a rejection", _button(at, "Reject").disabled)
    check("a revise action is offered instead", _button(at, "Revise") is not None)

    at.text_area[0].set_value("Blood pressure follow-up.").run(timeout=120)
    _run_pipeline_button(at).click().run(timeout=180)
    body = _all_text(at)
    check("a new run clears the rejection", "Rejected - not final" not in body, body[-400:])
    check("a new run is awaiting review again", "Awaiting review" in body, body[-400:])
    check("decisions are open again after a new run",
          not _button(at, "Approve & Save").disabled and not _button(at, "Reject").disabled)


def test_a5_reject_is_honest_about_a_recorded_draft():
    """A rejected note the pipeline recorded as a draft must say so."""
    print("\nPhase A5 reject honesty: the pipeline's draft cannot be undone")
    at, _calls = _boot(result_overrides={
        "metadata": {"memory": {"patient_id": "P001", "prior_visits": 0,
                                "memory_context_used": False, "saved": True,
                                "visit_id": "visit-1", "visit_status": "draft"}}
    })

    check("the draft banner says the run recorded an unsigned draft",
          "unsigned draft" in _all_text(at), _all_text(at)[-600:])
    _button(at, "Reject").click().run(timeout=120)
    body = _all_text(at)
    check("the rejection says the unsigned draft survives",
          "does not delete it" in body, body[-400:])
    check("the rejection says the draft is not history",
          "does not read it as history" in body, body[-400:])
    check("rejecting never approves anything",
          "not in the patient" in body or "not in the patient's history" in body, body[-400:])


def test_a5_request_changes_regenerates_without_losing_context():
    """Request Changes: labelled feedback, a real re-run, and nothing dropped."""
    print("\nPhase A5 request changes: feedback -> regeneration -> preserved context")
    from streamlit.testing.v1 import AppTest

    calls = _install_stubs()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app boots with the A5 widgets", not at.exception, [str(e.value) for e in at.exception])

    # A cardiology visit with encounter data, so "the same facts" is testable.
    at.selectbox(key="specialty_select").set_value("cardiology")
    at = at.run(timeout=120)
    check("switching specialty renders its own fields",
          at.text_area(key="specialty_cardiology_chest_pain_characteristics") is not None,
          [w.label for w in at.text_area])

    at.text_area[0].set_value("Chest pain on exertion for one week.")
    at.text_area(key="specialty_cardiology_chest_pain_characteristics").set_value(
        "Substernal pressure, radiates to the left arm, 7/10, relieved by rest."
    )
    at.text_area(key="specialty_cardiology_ecg_rhythm_notes").set_value(
        "Sinus rhythm 78 bpm, nonspecific ST-T changes in V4-V6."
    )
    at.text_area(key="vc_hpi").set_value("Chest pain walking to the mailbox.")
    at.text_input(key="vc_bp").set_value("148/92")
    at.text_input(key="vc_hr").set_value("78")
    at.text_area(key="vc_exam").set_value("No pedal edema. Heart sounds regular.")
    at = at.run(timeout=120)
    _run_pipeline_button(at).click().run(timeout=180)
    check("the enriched A5 run renders", not at.exception, [str(e.value) for e in at.exception])

    body = _all_text(at)
    check("the review shows the selected specialty", "**Specialty:** Cardiology" in body,
          body[-900:])
    check("the review shows the specialty field the clinician entered",
          "radiates to the left arm" in body, body[-900:])
    check("the review shows the encounter data",
          "148/92 mmHg" in body and "Chest pain walking to the mailbox" in body,
          body[-900:])

    _button(at, "Request Changes").click().run(timeout=120)
    check("requesting changes does not crash", not at.exception,
          [str(e.value) for e in at.exception])
    body = _all_text(at)
    check("the state is announced as changes requested",
          "Changes requested" in body, body[-500:])
    check("the banner says the feedback goes into the next pipeline run",
          "Scribe + Coding" in body, body[-600:])
    check("approve is closed while a revision is pending",
          _button(at, "Approve & Save").disabled)
    check("reject is closed while a revision is pending", _button(at, "Reject").disabled)

    feedback_boxes = [w for w in at.text_area if "feedback" in str(w.label).lower()]
    check("the feedback box is labelled as going into the next run",
          bool(feedback_boxes)
          and "next run" in str(feedback_boxes[0].label)
          and "Scribe + Coding" in str(feedback_boxes[0].label),
          [w.label for w in at.text_area])

    submit = _button(at, "Submit Feedback & Regenerate")
    check("an empty feedback cannot be submitted", submit.disabled, submit.label)
    feedback_boxes[0].set_value("Carry the ECG findings into the Objective section.").run(timeout=120)
    submit = _button(at, "Submit Feedback & Regenerate")
    check("submitting becomes available once feedback is written", not submit.disabled)
    submit.click().run(timeout=180)

    check("the regeneration does not crash", not at.exception,
          [str(e.value) for e in at.exception])
    check("the revision is reported as completed",
          any("Revision completed" in s.value for s in at.success), [s.value for s in at.success])
    check("the feedback reached the pipeline",
          calls["feedback"][-1] == "Carry the ECG findings into the Objective section.",
          calls["feedback"])
    check("the pipeline ran twice: the draft and the revision",
          len(calls["feedback"]) == 2, calls["feedback"])

    body = _all_text(at)
    check("the new draft is announced as a revised draft, not the old one",
          "Revised draft (revision 1)" in body, body[-600:])
    check("the new draft says the previous version was not edited",
          "not edited in place" in body, body[-600:])
    check("the new draft says the feedback that produced it",
          "Carry the ECG findings" in body, body[-900:])
    check("the regeneration reports itself as a new draft",
          any("new draft written from your feedback" in s.value for s in at.success),
          [s.value for s in at.success])

    check("the specialty survived the regeneration",
          calls["specialty"] == ["cardiology", "cardiology"], calls["specialty"])
    check("the encounter data survived the regeneration unchanged",
          calls["visit_context"][0] == calls["visit_context"][-1]
          and calls["visit_context"][-1].get("blood_pressure") == "148/92"
          and "ecg_rhythm_notes" in (calls["visit_context"][-1].get("specialty_fields") or {}),
          calls["visit_context"])
    check("the revised draft still shows the specialty",
          "**Specialty:** Cardiology" in body, body[-900:])
    check("the revised draft still shows the encounter data",
          "148/92 mmHg" in body and "Chest pain walking to the mailbox" in body, body[-900:])
    check("the revised draft still shows the specialty field",
          "Sinus rhythm 78 bpm" in body, body[-900:])
    check("the revised draft still shows the A4 symptom provenance",
          "Symptom coding" in body and "2 candidate(s) considered" in body, body[-1200:])
    check("the revised draft still shows the A4 flag groups",
          "Completed automatically" in body and "Removed by a safety check" in body,
          body[-1200:])
    check("the revised draft is a fresh decision",
          not _button(at, "Approve & Save").disabled
          and not _button(at, "Reject").disabled)


def main():
    print("=" * 70)
    print("PHYSICIAN REVIEW UI TESTS (no LLM required)")
    print("=" * 70)
    test_feedback_regenerate_survives_empty_icd10()
    test_rich_visit_context_round_trip()
    test_a4_symptom_provenance_and_flag_groups()
    test_a4_degrades_without_metadata()
    test_a5_state_machine_decisions()
    test_a5_approve_state_and_confirmation()
    test_a5_approve_with_memory_off_warns()
    test_a5_reject_state_and_recovery()
    test_a5_reject_is_honest_about_a_recorded_draft()
    test_a5_request_changes_regenerates_without_losing_context()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())