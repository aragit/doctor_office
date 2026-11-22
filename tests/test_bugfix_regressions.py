"""
Focused regression tests for four verified bugs - no LLM required.

Run with:

    python -m tests.test_bugfix_regressions

Each test locks in one bug that shipped in the pipeline:

1. ``get_llm(model_name=...)`` raised ``TypeError``. Both agents call the factory
   that way, so any run that pinned an explicit model (the documented
   ``run_visit_pipeline(..., model_name="gemma3:4b")`` path) died before the
   first token.
2. The Physician Review "Approve & Save" button called
   ``store.record_visit(patient_id, visit_record)``. ``record_visit`` expects
   ``patient_context`` as its second argument, so the visit dict was parsed as
   chart text and the write failed with ``AttributeError`` - swallowed into
   "Failed to save to memory". An approved note was never stored.
3. The Coding Agent flagged malformed ICD-10 codes as "dropped" but never
   removed them, so structurally invalid codes reached the physician.
4. The Scribe's grounding block was cut by a single blind 500-character slice,
   which truncated the anti-hallucination RULES line mid-sentence on a realistic
   chart.
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


# --- 1. get_llm(model_name=...) ---------------------------------------------


def test_get_llm_model_name():
    """The LLM factory accepts an explicit model and still honours the modes."""
    print("\nget_llm(model_name=...) no longer raises TypeError")
    import inspect

    from utils.config import MODEL_PRESETS
    from utils.llm import get_llm

    params = inspect.signature(get_llm).parameters
    check("get_llm accepts model_name", "model_name" in params, list(params))
    check("model_name is optional", params["model_name"].default is None)
    check("mode stays first for backward compatibility", list(params)[0] == "mode", list(params))

    try:
        llm = get_llm(model_name="gemma3:4b", temperature=0.05, num_predict=32)
    except TypeError as exc:
        check("get_llm(model_name=...) works", False, exc)
        return
    check("get_llm(model_name=...) works", True)
    check("explicit model is used verbatim", llm.model_name == "gemma3:4b", llm.model_name)
    check("other kwargs still apply", llm.max_tokens == 32, llm.max_tokens)

    check("mode default still resolves", get_llm().model_name == MODEL_PRESETS["fast"], get_llm().model_name)
    check("quality mode still resolves",
          get_llm(mode="quality").model_name == MODEL_PRESETS["quality"])
    check("mode works alongside num_predict",
          get_llm("quality", 0.0, 256).model_name == MODEL_PRESETS["quality"])
    check("blank model_name falls back to the mode",
          get_llm(mode="fast", model_name="   ").model_name == MODEL_PRESETS["fast"])
    check("model_name overrides mode",
          get_llm(mode="fast", model_name="llama3.2:3b").model_name == "llama3.2:3b")

    # The call sites that raised the TypeError in production.
    from agents.coding_agent import create_coding_chain
    from agents.scribe_agent import create_scribe_chain

    def _pinned_model(chain):
        """The model name baked into a `prompt | llm | parser` chain."""
        return next(
            (getattr(step, "model", None) or getattr(step, "model_name", None) for step in chain.steps
             if getattr(step, "model", None) or getattr(step, "model_name", None)),
            None,
        )

    for name, factory in (("create_scribe_chain", create_scribe_chain),
                          ("create_coding_chain", create_coding_chain)):
        try:
            chain = factory(mode="fast", model_name="gemma3:4b")
        except TypeError as exc:
            check(f"{name}(model_name=...) builds", False, exc)
            continue
        check(f"{name}(model_name=...) builds", True)
        check(f"{name} pins the explicit model",
              _pinned_model(chain) == "gemma3:4b", _pinned_model(chain))

        mode_chain = factory(mode="quality")
        check(f"{name} still honours the mode",
              _pinned_model(mode_chain) == MODEL_PRESETS["quality"], _pinned_model(mode_chain))


# --- 3. malformed ICD-10 codes are actually removed -------------------------


def test_malformed_icd10_dropped():
    """A structurally invalid code must not survive into the coding result."""
    print("\nCoding Agent removes malformed ICD-10 codes")
    import importlib

    from agents.coding_agent import CodingResult

    # `agents.coding_node` as an attribute is the node *function* (the package
    # re-export shadows the submodule), so patch the module object explicitly.
    coding_node = importlib.import_module("agents.coding_node")

    original_generate = coding_node.generate_coding
    try:
        coding_node.generate_coding = lambda **kwargs: CodingResult(
            primary_icd10="E11.9999 - bogus diabetes",
            secondary_icd10=[
                "I10 - Essential (primary) hypertension",
                "A1 - nonsense",
                "E11.9 - Type 2 diabetes mellitus without complications",
            ],
            cpt_suggestions=[],
            medical_necessity_note="Diabetes follow-up.",
        )
        update = coding_node.coding_node({
            "patient_context": (
                "Patient: John Doe, 58yo male\n"
                "Conditions: Type 2 Diabetes, Hypertension\n"
                "Medications: Metformin 500mg BID, Lisinopril 10mg daily"
            ),
            "chief_complaint": "Follow-up for diabetes and hypertension",
            "cleaned_soap_note": {
                "subjective": "s", "objective": "o", "assessment": "a",
                "plan": "p", "icd10_suggestions": [],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original_generate

    result = update["coding_result"]
    primary = result["primary_icd10"]
    secondary = list(result["secondary_icd10"])
    check("malformed primary is removed", primary == "", repr(primary))
    check("malformed secondary is removed",
          not any("nonsense" in c for c in secondary), secondary)
    check("valid codes are kept",
          "I10 - Essential (primary) hypertension" in secondary
          and "E11.9 - Type 2 diabetes mellitus without complications" in secondary,
          secondary)
    check("no code reaches the output unparsed",
          all("nonsense" not in c and "9999" not in c for c in [primary] + secondary),
          [primary] + secondary)

    flags = update.get("quality_flags") or []
    check("dropped primary is flagged",
          any("E11.9999" in f and "dropped malformed" in f for f in flags), flags)
    check("dropped secondary is flagged",
          any("A1 - nonsense" in f for f in flags), flags)
    check("every flag records a real removal",
          all("dropped malformed" in f for f in flags), flags)

    trace = update["tool_results"]["coding_node"]["validate_icd10_code"]
    check("codes_checked counts each code once", trace["codes_checked"] == 4, trace)
    check("malformed_dropped lists the removed codes",
          trace["malformed_dropped"] == ["E11.9999 - bogus diabetes", "A1 - nonsense"],
          trace)
    check("kept + dropped == checked",
          trace["codes_checked"] == 2 + len(trace["malformed_dropped"]), trace)

    # A clean result must be left completely alone.
    original_generate = coding_node.generate_coding
    try:
        coding_node.generate_coding = lambda **kwargs: CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[],
            medical_necessity_note="ok",
        )
        clean = coding_node.coding_node({
            "patient_context": "Patient: John Doe\nConditions: Type 2 Diabetes, Hypertension",
            "chief_complaint": "Follow-up for diabetes",
            "cleaned_soap_note": {"subjective": "s", "objective": "o",
                                  "assessment": "a", "plan": "p",
                                  "icd10_suggestions": []},
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original_generate

    clean_result = clean["coding_result"]
    check("a valid primary survives untouched",
          clean_result["primary_icd10"].startswith("E11.9"), clean_result["primary_icd10"])
    check("a valid secondary list is unchanged",
          clean_result["secondary_icd10"] == ["I10 - Essential (primary) hypertension"],
          clean_result["secondary_icd10"])
    check("no drop flags on a clean result",
          not any("dropped" in f for f in (clean.get("quality_flags") or [])),
          clean.get("quality_flags"))
    check("clean codes_checked is not double counted",
          clean["tool_results"]["coding_node"]["validate_icd10_code"]["codes_checked"] == 2,
          clean["tool_results"]["coding_node"]["validate_icd10_code"])


# --- 4. grounding block keeps its critical lines ----------------------------


def test_grounding_block_truncation():
    """A long chart must lose detail, not the RULES line or the history header."""
    print("\nGrounding block truncates per section, never mid-sentence")
    from agents.scribe_node import (
        MAX_FACTS_CHARS,
        MAX_HISTORY_CHARS,
        RULES_LINE,
        build_grounding_block,
    )

    facts = {
        "conditions": ["Type 2 Diabetes", "Hypertension", "CKD stage 3",
                       "Hyperlipidemia", "Hypothyroidism", "Osteoarthritis",
                       "GERD", "Osteoporosis"],
        "medications": ["Metformin 500mg BID", "Lisinopril 10mg daily",
                        "Atorvastatin 20mg qhs", "Levothyroxine 75mcg",
                        "Omeprazole 20mg", "Alendronate 70mg weekly",
                        "Amlodipine 5mg", "Furosemide 40mg"],
        "allergies": ["Penicillin", "Sulfa drugs", "Latex", "Codeine",
                      "Iodinated contrast", "Shellfish"],
    }
    memory = (
        "PATIENT HISTORY (reference only, from previous runs):\n"
        "Previously recorded (historical only, NOT current diagnoses): Asthma\n"
        "Recent visits:\n"
        "  - [2026-01-20] Visit: quarterly check, no acute symptoms | Codes: E11.9, I10\n"
        "  - [2026-02-24] Visit: medication review, mild fatigue | Codes: E11.9"
    )

    block = build_grounding_block(facts, memory)
    facts_section = block.split("PATIENT HISTORY")[0]

    check("RULES line survives a long chart", RULES_LINE in block, block[-200:])
    check("RULES line is not cut mid-sentence",
          "Any condition not listed is a hallucination." in block)
    check("RULES is the last line of the facts section",
          facts_section.strip().endswith("hallucination."), facts_section[-120:])
    check("facts section stays within its budget",
          len(facts_section.strip()) <= MAX_FACTS_CHARS,
          len(facts_section.strip()))
    check("omitted entries are reported, not silently dropped",
          "[4 more]" in block, block)
    check("header still states authority",
          "extracted by tools - authoritative" in block)
    check("history header survives", "PATIENT HISTORY" in block)
    check("newest visit survives", "2026-02-24" in block, block[-260:])
    check("history keeps the 'historical only' framing",
          "historical only, NOT current diagnoses" in block)

    # Overflowing history drops the OLDEST lines and says so.
    long_history = "PATIENT HISTORY (reference only, from previous runs):\nRecent visits:\n" + "\n".join(
        f"  - [2026-01-{i:02d}] Visit: lengthy narrative about encounter {i} "
        f"with codes E11.9, I10 and additional narrative detail"
        for i in range(1, 15)
    )
    trimmed = build_grounding_block({"conditions": ["Type 2 Diabetes"],
                                     "medications": ["Metformin"],
                                     "allergies": []}, long_history)
    history_block = trimmed.split("PATIENT HISTORY")[1]
    check("history is bounded", len(history_block) <= MAX_HISTORY_CHARS + 80, len(history_block))
    check("omitted history is announced", "omitted" in history_block, history_block[:200])
    check("the newest visit is the one kept", "2026-01-14" in history_block, history_block[-200:])
    check("the oldest visit is the one dropped", "2026-01-01" not in history_block)
    check("no history line is cut mid-word",
          all(line.rstrip().endswith(("detail", ")", ".", "]")) or "Codes" in line
              for line in history_block.strip().splitlines()[1:]),
          history_block)
    check("RULES still intact with an overlong history", RULES_LINE in trimmed)

    # Small inputs behave exactly as before (existing structural expectations).
    small = {"conditions": ["Type 2 Diabetes"], "medications": ["Metformin 500mg BID"],
             "allergies": []}
    small_block = build_grounding_block(small)
    check("short chart keeps every condition", "Type 2 Diabetes" in small_block, small_block)
    check("empty allergy list is explicit", "Allergies: NONE DOCUMENTED" in small_block)
    check("short chart needs no truncation marker", "more]" not in small_block, small_block)
    check("no facts and no memory yields nothing", build_grounding_block({}) == "")
    check("history alone renders", "PATIENT HISTORY" in build_grounding_block({}, memory))
    check("short chart stays compact", len(small_block) < MAX_FACTS_CHARS, len(small_block))


# --- 2. Approve & Save writes the visit -------------------------------------


def _stub_offline_app(result_overrides=None):
    """Stub the vector store, the pipeline and the memory store.

    The fake pipeline does what the real one does at the end of a run: it records
    a **draft**, and reports that row's id in metadata. Approve must then promote
    that exact row rather than write a second visit, which is only observable if
    the stub behaves the same way.

    Returns (temp_repo, {write method: calls}) so the test can assert exactly
    what Approve & Save wrote without touching the real patient memory file.
    """
    import rag.simple_rag as simple_rag

    class _Doc:
        page_content = "Type 2 Diabetes management. HbA1c 7.8%. Continue metformin."

    class _VectorStore:
        def similarity_search(self, query, k=3):
            return [_Doc()]

    simple_rag.get_vectorstore = lambda *a, **k: _VectorStore()

    result = {
        "soap_note": {
            "subjective": "Increased fatigue x2 weeks.",
            "objective": "BP 148/92, HbA1c 7.8%.",
            "assessment": "Suboptimal glycemic control.",
            "plan": "Continue metformin 500mg BID. Recheck HbA1c in 3 months.",
            "icd10_suggestions": ["E11.9 - Type 2 diabetes mellitus without complications"],
        },
        "coding_result": {
            "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
            "secondary_icd10": ["I10 - Essential hypertension"],
            "cpt_suggestions": ["99214 - Office visit, established patient, moderate MDM"],
            "medical_necessity_note": "Diabetes management with medication review.",
            "cpt_grounding": None,
        },
        "quality_flags": [],
        "metadata": {
            "orchestrator": "langgraph",
            "mode": "fast",
            "graph": "memory_save",
            "timing": {"total_seconds": 1.0},
            "supervisor": {"retry_count": 0, "needs_human_review": False,
                           "applied": [], "feedback": []},
            "memory": {"patient_id": "P001", "prior_visits": 0,
                       "memory_context_used": False, "saved": True},
            "tools": {},
        },
    }
    for key, value in (result_overrides or {}).items():
        if key == "metadata":
            result["metadata"].update(value)
        else:
            result[key] = value

    import agents.orchestrator as orch

    import memory.patient_memory as patient_memory
    from memory.patient_memory import JsonPatientMemoryRepository

    repo = JsonPatientMemoryRepository(path=Path(tempfile.mkdtemp()) / "patients.json")
    calls = {"record_visit": [], "approve_visit": []}
    originals = {
        "record_visit": repo.record_visit,
        "approve_visit": repo.approve_visit,
    }

    def recording(name):
        def wrapper(*args, **kwargs):
            calls[name].append((args, kwargs))
            return originals[name](*args, **kwargs)
        return wrapper

    repo.record_visit = recording("record_visit")
    repo.approve_visit = recording("approve_visit")

    # The pipeline's own memory-save step: a draft, echoed back with its id.
    def fake_stream(patient_context, chief_complaint, retrieved_notes, mode, model_name=None,
                    patient_id=None, use_memory=True, physician_feedback=None,
                    visit_context=None, specialty=None):
        snapshot = dict(result)
        metadata = dict(result["metadata"], visit_context=dict(visit_context or {}))
        escalated = bool((metadata.get("supervisor") or {}).get("needs_human_review"))
        if not escalated and use_memory:
            record = repo.record_visit(
                patient_id=patient_id or "P001",
                patient_context=patient_context,
                chief_complaint=chief_complaint,
                coding_result=snapshot["coding_result"],
                soap_note=snapshot["soap_note"],
                quality_flags=snapshot.get("quality_flags") or [],
                status="draft",
            )
            metadata["memory"] = dict(
                metadata.get("memory") or {},
                visit_id=(record.get("visits") or [{}])[-1].get("id", ""),
            )
        snapshot["metadata"] = metadata
        yield snapshot

    orch.stream_visit_pipeline = fake_stream
    patient_memory.get_memory_store = lambda *a, **k: repo
    return repo, calls


def test_approve_and_save_writes_the_visit(escalated: bool):
    """Approve & Save must reach the store with the documented signature."""
    print(f"\nApprove & Save persists the visit (escalated={escalated})")
    from streamlit.testing.v1 import AppTest

    overrides = (
        {
            "metadata": {
                "graph": "human_review",
                "supervisor": {"retry_count": 2, "needs_human_review": True,
                               "applied": [], "feedback": ["max retries"]},
                "memory": {"patient_id": "P001", "prior_visits": 0,
                           "memory_context_used": False, "saved": False},
            }
        }
        if escalated
        else {}
    )
    repo, calls = _stub_offline_app(overrides)

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(timeout=120)
    check("app loads without exception", not at.exception, [str(e.value) for e in at.exception])

    at.text_area[0].set_value("Follow-up for diabetes. Reports fatigue.").run(timeout=120)
    [b for b in at.button if "Run Pipeline" in b.label][0].click().run(timeout=120)
    check("run completed", not at.exception, [str(e.value) for e in at.exception])

    approve = [b for b in at.button if "Approve & Save" in b.label]
    check("Approve & Save button present", bool(approve), [b.label for b in at.button])
    approve[0].click().run(timeout=120)

    check("no exception when approving", not at.exception, [str(e.value) for e in at.exception])
    check("no save-failure banner",
          not any("Failed to save" in e.value for e in at.error), [e.value for e in at.error])

    # Approve is the only route to history, so it is the only route that may
    # call approve_visit. Nothing on this screen may write through record_visit.
    check("approve went through approve_visit", len(calls["approve_visit"]) == 1, calls)
    # The only direct draft write is the pipeline's own. An escalated approval
    # has no patient record to promote into, so approve_visit writes the row
    # itself - visible here as a status="approved" record_visit, never as a
    # second row.
    pipeline_writes = [c for c in calls["record_visit"] if c[1].get("status") == "draft"]
    check("the pipeline's draft is the only unsigned write",
          len(pipeline_writes) == (0 if escalated else 1), calls["record_visit"])
    check("and no unsigned write was ever approved",
          not any(c[1].get("status") not in (None, "draft", "approved")
                  for c in calls["record_visit"]), calls["record_visit"])

    args, kwargs = calls["approve_visit"][0]
    check("patient_id passed by keyword", kwargs.get("patient_id") == "P001", kwargs)
    context = kwargs.get("patient_context")
    check("patient_context is chart text, not a dict", isinstance(context, str), type(context))
    check("chart text names the patient", "John Doe" in context, context)
    check("chart text carries the conditions", "Type 2 Diabetes" in context, context)
    check("chief complaint comes from the reviewed run",
          kwargs.get("chief_complaint") == "Follow-up for diabetes. Reports fatigue.",
          kwargs.get("chief_complaint"))
    check("soap note forwarded", "metformin" in (kwargs.get("soap_note") or {}).get("plan", "").lower(),
          kwargs.get("soap_note"))
    check("coding result forwarded",
          (kwargs.get("coding_result") or {}).get("primary_icd10", "").startswith("E11.9"),
          kwargs.get("coding_result"))
    check("quality flags forwarded as a list",
          isinstance(kwargs.get("quality_flags"), list), kwargs.get("quality_flags"))
    check("no positional args (signature-safe)", not args, args)

    if escalated:
        check("no draft row was addressed - there was none",
              kwargs.get("visit_id") is None, kwargs.get("visit_id"))
    else:
        check("the run's own draft row is the one approved",
              kwargs.get("visit_id") == "visit-1", kwargs.get("visit_id"))

    summary = repo.get_summary("P001")
    check("the visit is now approved history", summary.get("visit_count") == 1, summary)
    check("and there is no leftover draft", summary.get("draft_visit_count") == 0, summary)
    check("stored codes are the reviewed ones",
          summary.get("visits", [{}])[0].get("icd10") == [
              "E11.9 - Type 2 diabetes mellitus without complications",
              "I10 - Essential hypertension",
          ], summary.get("visits"))
    check("stored complaint matches the run",
          summary.get("visits", [{}])[0].get("chief_complaint")
          == "Follow-up for diabetes. Reports fatigue.", summary.get("visits"))
    check("exactly one visit row exists either way",
          len(summary.get("visits", [])) == 1, summary.get("visits"))


def main():
    print("=" * 70)
    print("BUGFIX REGRESSION TESTS (no LLM required)")
    print("=" * 70)
    test_get_llm_model_name()
    test_malformed_icd10_dropped()
    test_grounding_block_truncation()
    test_approve_and_save_writes_the_visit(escalated=True)
    test_approve_and_save_writes_the_visit(escalated=False)

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())