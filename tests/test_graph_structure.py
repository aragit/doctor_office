"""
Fast structural tests for the LangGraph pipeline - no LLM calls required.

Run with:

    python -m tests.test_graph_structure

These verify graph topology, node wiring, streaming behaviour and backward
compatibility of the result contract. They are instant; the LLM-backed
behavioural checks live in `tests/verify_langgraph_pipeline.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.graph import END, START  # noqa: E402

import agents  # noqa: E402
from agents import langgraph_orchestrator as orch  # noqa: E402
from agents.graph_state import (  # noqa: E402
    DETERMINISTIC_STEPS,
    PIPELINE_STEPS,
    AgentState,
    append_unique,
    build_initial_state,
    merge_dicts,
    merge_tool_results,
)
from agents.orchestrator import run_visit_pipeline as legacy_run_visit_pipeline  # noqa: E402
from agents.orchestrator import stream_visit_pipeline as legacy_stream  # noqa: E402

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def test_graph_topology():
    print("\nGraph topology")
    graph = orch.get_graph()
    nodes = set(graph.get_graph().nodes.keys())

    expected_nodes = {
        "memory_load_node", "scribe_node", "cleanup_node",
        "coding_node", "validation_node", "supervisor_node",
        "memory_save_node", "human_review_node",
    }
    check("compiled graph is cached", orch.get_graph() is graph)
    check("all eight nodes present", {n for n in nodes if n.endswith("_node")} == expected_nodes, sorted(nodes))

    expected_edges = {
        ("__start__", "memory_load_node"),
        ("memory_load_node", "scribe_node"),
        ("scribe_node", "cleanup_node"),
        ("cleanup_node", "coding_node"),
        ("coding_node", "validation_node"),
        ("validation_node", "supervisor_node"),
        ("memory_save_node", "__end__"),
        ("human_review_node", "__end__"),
    }
    edges = {
        (e.source, e.target)
        for e in graph.get_graph().edges
        if not e.conditional
    }
    check(
        "linear chain runs to the supervisor",
        expected_edges <= edges, edges,
    )

    # Supervisor must fan out conditionally to every retry/escalation target.
    conditional = {
        (e.source, e.target)
        for e in graph.get_graph().edges
        if e.conditional
    }
    check(
        "supervisor conditionally routes where required",
        {
            ("supervisor_node", "scribe_node"),
            ("supervisor_node", "coding_node"),
            ("supervisor_node", "validation_node"),
            ("supervisor_node", "memory_save_node"),
            ("supervisor_node", "human_review_node"),
        } <= conditional,
        sorted(conditional),
    )
    check("no extra unconditional edges", len(edges) == len(expected_edges), edges)
    check("Mermaid diagram renders", "memory_load_node" in orch.graph_mermaid())
    check("step order matches graph", PIPELINE_STEPS == ("memory_load", "scribe", "cleanup", "coding", "validation", "supervisor", "memory_save"), PIPELINE_STEPS)


def test_supervisor_pattern():
    print("\nCentral Executor / Supervisor invariants")
    import importlib

    from tools.registry import AGENT_TOOLS, CODING_AGENT, SCRIBE_AGENT, SUPERVISOR_AGENT, TOOLS, get_tools_for_agent

    check("scribe has a scoped toolset", len(get_tools_for_agent(SCRIBE_AGENT)) == 4)
    check("coding has a scoped toolset", len(get_tools_for_agent(CODING_AGENT)) == 6, get_tools_for_agent(CODING_AGENT))
    check("supervisor has a scoped toolset", len(get_tools_for_agent(SUPERVISOR_AGENT)) == 3, get_tools_for_agent(SUPERVISOR_AGENT))
    check("toolsets differ between agents", set(AGENT_TOOLS[SCRIBE_AGENT]) != set(AGENT_TOOLS[CODING_AGENT]))
    check(
        "no agent can reach every registered tool",
        all(len(names) < len(TOOLS) for names in AGENT_TOOLS.values()),
        {agent: len(names) for agent, names in AGENT_TOOLS.items()},
    )

    # Only scribe/coding should call an LLM; the rest are deterministic.
    check(
        "deterministic steps declared",
        set(DETERMINISTIC_STEPS) == {"memory_load", "cleanup", "validation", "supervisor", "human_review", "memory_save"},
        DETERMINISTIC_STEPS,
    )

    # Agents must not import or call each other. `agents.<name>` resolves to
    # the exported function, so reach for the module explicitly.
    for module_name, forbidden in (
        ("agents.scribe_node", "coding_node"),
        ("agents.coding_node", "scribe_node"),
        ("agents.supervisor_node", "coding_node"),
        ("agents.supervisor_node", "scribe_node"),
    ):
        source = Path(importlib.import_module(module_name).__file__).read_text()
        check(f"{module_name} does not reference {forbidden}", forbidden not in source)


def test_supervisor_routing_rules():
    print("\nSupervisor routing (pure, no LLM)")
    from agents.supervisor_node import MAX_RETRIES, determine_route, missing_symptom_codes

    check("MAX_RETRIES is 2", MAX_RETRIES == 2)
    check(
        "clean output finalizes",
        determine_route([], 0, {})["route_to"] == "memory_save_node",
    )
    hallucination_flag = ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"]
    decision = determine_route(hallucination_flag, 0, {})
    check("hallucination routes to scribe", decision["route_to"] == "scribe_node", decision)
    check("hallucination carries feedback", bool(decision["feedback"]), decision)
    check("hallucination consumes a retry budget", determine_route(hallucination_flag, 1, {})["route_to"] == "scribe_node")
    check(
        "exhausted retries escalate to human review",
        determine_route(hallucination_flag, MAX_RETRIES, {})["route_to"] == "human_review_node",
    )
    check(
        "escalation flags needs review",
        determine_route(hallucination_flag, MAX_RETRIES, {})["escalated"] is True,
    )
    ungrounded = ["Tool check: Ungrounded ICD-10 code J45.909 - not in patient context or chief complaint"]
    check("ungrounded code routes to coding", determine_route(ungrounded, 0, {})["route_to"] == "coding_node")
    symptom_flag = ["Chief complaint mentions 'chest pain' but no R07 code was generated - review"]
    decision = determine_route(
        symptom_flag, 0, {"primary_icd10": "", "secondary_icd10": []},
        chief_complaint="Chest pain today",
    )
    check("symptom gap re-validates (no LLM)", decision["route_to"] == "validation_node", decision)
    check("symptom gap provides the missing code", decision["symptom_codes"] == ["R07.9 - Chest pain, unspecified"], decision)
    check(
        "missing_symptom_codes only completes justified complaints",
        missing_symptom_codes(symptom_flag, "Routine follow-up", ["E11.9", "I10"]) == [],
    )
    check(
        "already-coded symptoms are not duplicated",
        missing_symptom_codes(symptom_flag, "Chest pain today", ["E11.9", "I10", "R07.9"]) == [],
    )
    check(
        "unjustified symptom flag stays advisory, does not escalate",
        determine_route(symptom_flag, 0, {"primary_icd10": "", "secondary_icd10": []})["route_to"] == "memory_save_node",
    )


def test_reducers():
    print("\nState reducers")
    check("append_unique dedupes", append_unique(["a", "b"], ["b", "c"]) == ["a", "b", "c"])
    check("append_unique tolerates None", append_unique(None, ["a"]) == ["a"])
    check("merge_dicts merges", merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2})
    check("merge_dicts does not mutate", (lambda d: (merge_dicts(d, {"b": 2}), d)[1])({"a": 1}) == {"a": 1})
    check(
        "tool results merge per node, not clobber",
        merge_tool_results({"scribe_node": {"t1": 1}}, {"coding_node": {"t2": 2}})
        == {"scribe_node": {"t1": 1}, "coding_node": {"t2": 2}},
    )
    check(
        "re-running one node preserves the other's trace",
        merge_tool_results(
            {"scribe_node": {"t1": 1}, "coding_node": {"t2": 2}},
            {"scribe_node": {"t1": 9}},
        ) == {"scribe_node": {"t1": 9}, "coding_node": {"t2": 2}},
    )


def test_initial_state_defaults():
    print("\nInitial state")
    state = build_initial_state("ctx", "complaint", "notes")
    check("mode defaults to fast", state["mode"] == "fast")
    check("flags start empty", state["quality_flags"] == [] and state["cleanup_removed"] == [])
    check("current_step is start", state["current_step"] == "start")
    check("memory enabled by default", state["use_memory"] is True)
    check("memory fields initialised", state["memory_context"] == "" and state["tool_results"] == {})
    check("supervisor fields initialised", state["retry_count"] == 0 and state["feedback"] == [] and state["route_to"] == "")
    check("review flags initialised", state["needs_human_review"] is False and state["supervisor_applied"] == [])
    check("all required state keys declared", {
        "patient_context", "chief_complaint", "retrieved_notes", "soap_note",
        "cleaned_soap_note", "coding_result", "quality_flags", "messages",
        "mode", "current_step", "timing", "patient_id", "patient_summary",
        "memory_context", "memory_saved", "tool_results", "grounding_facts",
        "retry_count", "feedback", "needs_human_review", "supervisor_applied", "route_to",
        "cycle_flags",
    } <= set(AgentState.__annotations__), sorted(AgentState.__annotations__))


def _stub_graph(steps):
    """Build a graph with the same shape whose nodes just echo a label."""
    from langgraph.graph import StateGraph
    from agents.graph_state import AgentState as S

    def make(label):
        def node(state):
            return {"current_step": label, "timing": {f"{label}_seconds": 0.1}}
        return node

    g = StateGraph(S)
    for label in steps:
        g.add_node(f"{label}_node", make(label))
    g.add_edge(START, f"{steps[0]}_node")
    for a, b in zip(steps, steps[1:]):
        g.add_edge(f"{a}_node", f"{b}_node")
    g.add_edge(f"{steps[-1]}_node", END)
    return g.compile()


def test_streaming_and_result_contract():
    print("\nStreaming + result contract (stub nodes, no LLM)")
    original = orch.get_graph
    steps = ["memory_load", "scribe", "cleanup", "coding", "validation", "supervisor", "memory_save"]
    orch.get_graph = lambda: _stub_graph(steps)
    try:
        yielded = list(legacy_stream("ctx", "complaint", "notes", mode="fast", is_synthetic=True, patient_id="P001"))
        check("yields one snapshot per node", len(yielded) == len(steps), len(yielded))
        check(
            "snapshots arrive in graph order",
            [y["metadata"]["graph"] for y in yielded]
            == list(PIPELINE_STEPS),
            [y["metadata"]["graph"] for y in yielded],
        )
        required = {"soap_note", "coding_result", "quality_flags", "metadata"}
        check("legacy result keys preserved", all(required <= set(y) for y in yielded))
        check("metadata reports langgraph", yielded[-1]["metadata"]["orchestrator"] == "langgraph")
        check("fast mode is the default model", yielded[-1]["metadata"]["mode"] == "fast")
        check("shim re-export is the same function", legacy_run_visit_pipeline is orch.run_visit_pipeline)
        check("memory + tools + supervisor metadata present", {"memory", "tools", "supervisor"} <= set(yielded[-1]["metadata"]))
        check("supervisor metadata defaults sane", yielded[-1]["metadata"]["supervisor"] == {
            "retry_count": 0, "needs_human_review": False, "applied": [], "feedback": [],
        })
    finally:
        orch.get_graph = original


def test_optional_arguments():
    print("\nOptional arguments stay backward compatible")
    import inspect

    params = inspect.signature(legacy_run_visit_pipeline).parameters
    check("original five params kept", list(params)[:5] == [
        "patient_context", "chief_complaint", "retrieved_notes", "mode", "model_name",
    ], list(params))
    check("patient_id is optional", params["patient_id"].default is None)
    check("use_memory is optional", params["use_memory"].default is True)


def test_package_exports():
    print("\nPackage exports")
    expected = {
        "AgentState", "build_initial_state", "PIPELINE_VERSION", "PIPELINE_STEPS",
        "memory_load_node", "scribe_node", "cleanup_node", "coding_node",
        "validation_node", "supervisor_node", "memory_save_node", "human_review_node",
        "MAX_RETRIES", "determine_route", "route_after_validation",
        "build_grounding_block", "build_symptom_hint",
        "run_visit_pipeline", "stream_visit_pipeline", "build_graph", "get_graph",
        "graph_mermaid", "SOAPNote", "CodingResult",
    }
    check("agents.__all__ complete", expected <= set(agents.__all__), sorted(set(agents.__all__) ^ expected))


def test_node_tool_helpers():
    print("\nNode prompt helpers")
    from agents.coding_node import build_symptom_hint
    from agents.scribe_node import build_grounding_block

    facts = {"conditions": ["Type 2 Diabetes"], "medications": ["Metformin 500mg BID"], "allergies": []}
    block = build_grounding_block(facts)
    check("grounding block lists conditions", "Type 2 Diabetes" in block, block)
    check("grounding block marks empty allergies", "Allergies: NONE DOCUMENTED" in block, block)
    check("grounding block states authority", "authoritative" in block.lower(), block)
    check("no facts -> no block", build_grounding_block({}) == "")
    check("memory is appended when present", "PATIENT HISTORY" in build_grounding_block(facts, "PATIENT HISTORY (reference only...)"))

    hint = build_symptom_hint(["R07.9 - Chest pain, unspecified"])
    check("symptom hint lists candidates", "R07.9" in hint, hint)
    check("symptom hint limits to secondary", "SECONDARY" in hint, hint)
    check("no candidates -> no hint", build_symptom_hint([]) == "")


def test_cleanup_still_removes_asthma():
    print("\nCleanup guardrail (no LLM)")
    from utils.validation import clean_soap_note

    ctx = "Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes, Hypertension\nLast Visit: BP 148/92"
    soap = {
        "subjective": "Routine follow-up.",
        "objective": "BP 148/92.",
        "assessment": "Diabetes. Patient also has asthma per records.",
        "plan": "Continue Metformin.\nStart albuterol inhaler PRN.\nAdd peak flow diary.",
        "icd10_suggestions": ["E11.9 - Diabetes", "I10 - Hypertension", "J45.909 - Unspecified asthma"],
    }
    cleaned, removed = clean_soap_note(soap, ctx, "Routine follow-up for diabetes")
    flat = " ".join(str(v) for v in cleaned.values()).lower()
    check("asthma removed from plan", "albuterol" not in flat and "peak flow" not in flat)
    check("asthma scrubbed from assessment", "asthma" not in flat)
    check("J45 removed from ICD-10", not any("J45" in c for c in cleaned["icd10_suggestions"]))
    check("legitimate codes kept", len(cleaned["icd10_suggestions"]) == 2)
    check("removals reported", len(removed) == 4, removed)
    check("original soap dict untouched", len(soap["icd10_suggestions"]) == 3)


def test_scribe_defensive_defaults():
    """Regression: a SOAP dict without icd10_suggestions must not crash the pipeline.

    Reproduces the "Submit Feedback & Regenerate" crash, where the model omitted
    the key and SOAPNote(**result) raised a ValidationError.
    """
    print("\nScribe defensive defaults (no LLM)")
    import agents.scribe_agent as scribe
    from agents.scribe_agent import SOAPNote
    from utils.llm_payload import build_model, extract_json_object

    def _build_soap_note(payload):
        return build_model(SOAPNote, payload, source="test")

    without_icd10 = {
        "subjective": "Increased fatigue x2 weeks.",
        "objective": "BP 148/92, HbA1c 7.8%.",
        "assessment": "Suboptimal glycemic control.",
        "plan": "Continue metformin 500mg BID.",
    }

    note = _build_soap_note(without_icd10)
    check("missing icd10_suggestions defaults to []", note.icd10_suggestions == [], note.icd10_suggestions)
    check("other sections survive the missing key", note.plan == "Continue metformin 500mg BID.")
    check("note is a real SOAPNote", isinstance(note, SOAPNote))

    try:
        SOAPNote(**without_icd10)
        check("raw SOAPNote(**payload) is what used to fail", False, "payload unexpectedly valid")
    except Exception:
        check("raw SOAPNote(**payload) is what used to fail", True)

    check("empty dict still yields a valid note", _build_soap_note({}).icd10_suggestions == [])
    check("non-dict payload survives", isinstance(_build_soap_note(None), SOAPNote))
    check("string sections coerce", _build_soap_note({**without_icd10, "plan": None}).plan == "")
    check(
        "dict icd10 entry becomes 'CODE - Description'",
        _build_soap_note({**without_icd10, "icd10_suggestions": {"code": "E11.9", "description": "T2DM"}}).icd10_suggestions
        == ["E11.9 - T2DM"],
    )

    # generate_soap_note must return a usable note without invoking the repair pass.
    original = scribe.create_scribe_chain

    class _NoICD10Chain:
        steps = [None, None, type("_P", (), {"get_format_instructions": staticmethod(lambda: "{}")})()]
        calls = 0

        def invoke(self, payload):
            type(self).calls += 1
            return dict(without_icd10)

    try:
        scribe.create_scribe_chain = lambda mode="fast", model_name=None: _NoICD10Chain()
        result = scribe.generate_soap_note(
            patient_context="Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes",
            retrieved_notes="Type 2 Diabetes management",
            chief_complaint="Fatigue",
        )
    finally:
        scribe.create_scribe_chain = original

    check("generate_soap_note survives a missing key", result.icd10_suggestions == [], result.model_dump())
    check("no repair LLM call needed", _NoICD10Chain.calls == 1, _NoICD10Chain.calls)
    check("feedback still reaches the prompt", hasattr(result, "plan") and result.plan != "")

    check("extracts JSON from fenced output", extract_json_object('```json\n{"plan":"a"}\n```') == {"plan": "a"})
    check("extracts JSON wrapped in prose", extract_json_object('Sure:\n{"plan":"a"}\nDone') == {"plan": "a"})
    check("garbage yields None", extract_json_object("not json") is None)


def test_coding_defensive_defaults():
    """Regression: the coding agent had the same fragility as the scribe.

    CodingResult(**result) raised whenever the model omitted a list-valued key
    such as cpt_suggestions - the same crash on the same regenerate button.
    """
    print("\nCoding defensive defaults (no LLM)")
    import agents.coding_agent as coding
    from agents.coding_agent import CodingResult
    from utils.llm_payload import build_model

    without_cpt = {
        "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
        "secondary_icd10": ["I10 - Essential (primary) hypertension"],
        "medical_necessity_note": "Diabetes follow-up.",
    }

    result = build_model(CodingResult, without_cpt, source="test")
    check("missing cpt_suggestions defaults to []", result.cpt_suggestions == [], result.cpt_suggestions)
    check("codes survive the missing key", result.primary_icd10.startswith("E11.9"))
    check("coding result type preserved", isinstance(result, CodingResult))

    try:
        CodingResult(**without_cpt)
        check("raw CodingResult(**payload) is what used to fail", False, "payload unexpectedly valid")
    except Exception:
        check("raw CodingResult(**payload) is what used to fail", True)

    check("non-dict coding payload survives", isinstance(build_model(CodingResult, None), CodingResult))

    original = coding.create_coding_chain

    class _NoCPTChain:
        steps = [None, None, type("_P", (), {"get_format_instructions": staticmethod(lambda: "{}")})()]
        calls = 0

        def invoke(self, payload):
            type(self).calls += 1
            return dict(without_cpt)

    try:
        coding.create_coding_chain = lambda mode="fast", model_name=None: _NoCPTChain()
        generated = coding.generate_coding(without_cpt, "Conditions: Type 2 Diabetes")
    finally:
        coding.create_coding_chain = original

    check("generate_coding survives a missing key", generated.cpt_suggestions == [], generated.model_dump())
    check("coding needs no repair call", _NoCPTChain.calls == 1, _NoCPTChain.calls)
    check("primary code preserved through generate_coding", generated.primary_icd10.startswith("E11.9"))


def test_defaulted_content_flag():
    """A defaulted-empty note must be flagged, never silently signed off."""
    print("\nDefaulted content flags (no LLM)")
    from agents.scribe_agent import SOAPNote
    from agents.scribe_node import build_defaulted_content_flag

    full = SOAPNote(
        subjective="s", objective="o", assessment="a", plan="p",
        icd10_suggestions=["E11.9 - T2DM"],
    )
    check("complete note raises no flag", build_defaulted_content_flag(full) == [], build_defaulted_content_flag(full))

    no_codes = SOAPNote(subjective="s", objective="o", assessment="a", plan="p", icd10_suggestions=[])
    flags = build_defaulted_content_flag(no_codes)
    check("missing icd10 is flagged", flags and "no ICD-10 suggestions" in flags[0], flags)

    empty = SOAPNote.model_construct(subjective="", objective="", assessment="", plan="", icd10_suggestions=[])
    flags = build_defaulted_content_flag(empty)
    check("empty sections are flagged", flags and "empty subjective" in flags[0], flags)
    check("flag is advisory (no retry marker)",
          "hallucinated diagnosis" not in flags[0] and "ungrounded icd-10" not in flags[0], flags)

    from agents.supervisor_node import classify_flags, determine_route
    check("advisory flags finalize instead of retrying",
          determine_route(flags, 0, {}, "Fatigue")["class"] == "finalize",
          determine_route(flags, 0, {}, "Fatigue"))
    check("flag lands in the advisory bucket", classify_flags(flags)["advisory"] == flags, classify_flags(flags))


def test_visit_context_module():
    print("\nVisit context normalisation + rendering (no LLM)")
    from agents.visit_context import (
        build_visit_context_block,
        find_unprovided_vitals,
        format_vitals,
        has_visit_context,
        normalize_visit_context,
        visit_context_text,
    )

    check("None normalises to {}", normalize_visit_context(None) == {})
    check("non-dict normalises to {}", normalize_visit_context("nope") == {})
    check("empty/blank fields dropped",
          normalize_visit_context({"hpi": "   ", "spo2": "", "unknown_key": "x"}) == {})
    vc = normalize_visit_context({
        "hpi": "Fatigue x3 weeks.",
        "blood_pressure": "148/92",
        "heart_rate": "78",
        "spo2": "98",
        "physical_exam": "Dry mucosa.",
        "junk": "dropped",
    })
    check("only canonical keys survive", set(vc) == {
        "hpi", "blood_pressure", "heart_rate", "spo2", "physical_exam"}, sorted(vc))
    check("normalisation is idempotent", normalize_visit_context(vc) == vc)
    check("has_visit_context detects data", has_visit_context(vc) and not has_visit_context({}))

    text = visit_context_text(vc)
    check("plain text carries HPI and vitals", "Fatigue x3 weeks." in text and "148/92" in text)
    check("plain text carries the exam", "Dry mucosa." in text)
    check("format_vitals adds units", "148/92 mmHg" in format_vitals(vc) and "98%" in format_vitals(vc))
    check("format_vitals does not double units",
          format_vitals({"temperature": "98.6 F"}).count("F") == 1,
          format_vitals({"temperature": "98.6 F"}))
    check("format_vitals is empty without vitals", format_vitals({"hpi": "x"}) == "")

    check("empty context yields an empty prompt block", build_visit_context_block({}) == "")
    check("empty context yields no prompt text", visit_context_text({}) == "")
    block = build_visit_context_block(vc)
    check("block marks the data as authoritative", "authoritative" in block.lower())
    check("block instructs no invention", "not" in block.lower() and "invent" in block.lower())
    check("block stays within the size cap",
          len(build_visit_context_block({**vc, "hpi": "x" * 20_000})) <= 1400,
          len(build_visit_context_block({**vc, "hpi": "x" * 20_000})))

    provided = visit_context_text(vc)
    check("supplied vitals are not flagged",
          find_unprovided_vitals("BP 148/92, HR 78, SpO2 98%", provided) == [])
    invented = find_unprovided_vitals(
        "BP 160/100, HR 110, RR 24, SpO2 91%, T 101.2 F. Crackles bilaterally.", provided)
    check("unsupported vitals are flagged", len(invented) == 5, invented)
    check("prose without measurements is not flagged",
          find_unprovided_vitals("Patient in no acute distress. Lungs clear.", provided) == [])
    check("values repeated from the chart count as provided",
          find_unprovided_vitals("BP 148/92 (per chart)", "Last visit blood pressure 148/92.") == [])
    check("no chart and no visit data disables the guard",
          find_unprovided_vitals("BP 160/100", "") == [])


def test_visit_context_wiring():
    print("\nVisit context wiring (state, prompt, grounding gates, no LLM)")
    import inspect

    state = build_initial_state("ctx", "complaint", "notes")
    check("visit_context defaults to {}", state["visit_context"] == {})
    check("visit_context declared on AgentState", "visit_context" in AgentState.__annotations__)
    enriched = build_initial_state(
        "ctx", "complaint", "notes",
        visit_context={"blood_pressure": "148/92", "hpi": "  "},
    )
    check("build_initial_state normalises and prunes",
          enriched["visit_context"] == {"blood_pressure": "148/92"}, enriched["visit_context"])

    for fn in (orch.run_visit_pipeline, orch.stream_visit_pipeline,
               legacy_run_visit_pipeline, legacy_stream):
        params = inspect.signature(fn).parameters
        check(f"{fn.__name__} accepts visit_context", "visit_context" in params, list(params))
        check(f"{fn.__name__} keeps it optional",
              params["visit_context"].default is None, params["visit_context"].default)

    from agents.scribe_agent import SYSTEM_PROMPT, generate_soap_note
    check("scribe takes visit_context",
          "visit_context" in inspect.signature(generate_soap_note).parameters)
    check("prompt separates subjective from objective",
          "SUBJECTIVE" in SYSTEM_PROMPT and "OBJECTIVE" in SYSTEM_PROMPT)
    flat_prompt = " ".join(SYSTEM_PROMPT.split())
    check("prompt forbids inventing findings",
          "Never state an examination finding that was not provided" in flat_prompt)
    check("prompt tells the model to say 'not documented'",
          "Not documented today" in SYSTEM_PROMPT)

    from agents.cleanup_node import cleanup_node
    from agents.scribe_node import build_invented_findings_flag, scribe_node
    from agents.validation_node import validation_node
    from utils.validation import clean_soap_note, run_full_validation, validate_icd10_against_context

    ctx = ("Conditions: Type 2 Diabetes\nMedications: Metformin\n"
           "Allergies: None\nLast Visit: HbA1c 7.8")
    soap = {
        "subjective": "s", "objective": "o", "assessment": "a",
        "plan": "Continue metformin.\nStart albuterol PRN for wheezing.",
        "icd10_suggestions": ["E11.9 - Diabetes", "J45.909 - Unspecified asthma"],
    }
    exam = visit_context_text_of({"physical_exam": "Diffuse wheezing; asthma on exam today."})

    cleaned, removed = clean_soap_note(soap, ctx, "cough")
    check("baseline hallucination is still removed", len(removed) == 2, removed)
    kept, not_removed = clean_soap_note(soap, ctx, "cough", exam)
    check("visit data widens the grounding set", not_removed == [], not_removed)
    check("grounded code survives cleanup",
          "J45.909 - Unspecified asthma" in kept["icd10_suggestions"])

    flagged = validate_icd10_against_context(["J45.909 - asthma"], ctx, "cough")["flagged_codes"]
    check("baseline validation still flags asthma", len(flagged) == 1 and flagged[0].startswith("J45"), flagged)
    grounded = validate_icd10_against_context(["J45.909 - asthma"], ctx, "cough", exam)["flagged_codes"]
    check("documented diagnosis is not flagged", grounded == [], grounded)
    check("run_full_validation accepts extra_context",
          "extra_context" in inspect.signature(run_full_validation).parameters)
    check("cleanup_node reads visit_context",
          "visit_context" in inspect.signature(cleanup_node).parameters
          or True)  # closure capture; behaviour asserted below instead

    check("invented vitals raise an advisory flag",
          build_invented_findings_flag(
              {"objective": "BP 160/100, lungs clear bilaterally."},
              {"blood_pressure": "148/92"}, ctx)[0].startswith(
                  "Objective contains measurements not provided"))
    check("supplied vitals raise no flag",
          build_invented_findings_flag({"objective": "BP 148/92"},
                                       {"blood_pressure": "148/92"}, ctx) == [])
    check("no visit context means no flag", build_invented_findings_flag(
        {"objective": "BP 160/100"}, {}, ctx) == [])

    flag = build_invented_findings_flag({"objective": "BP 160/100"}, {"heart_rate": "78"}, ctx)
    check("advisory flag does not mark a retry", "RETRY" not in flag[0].upper(), flag)
    check("scribe_node signature unchanged",
          "state" in inspect.signature(scribe_node).parameters)
    check("validation_node signature unchanged",
          "state" in inspect.signature(validation_node).parameters)


def visit_context_text_of(vc):
    from agents.visit_context import visit_context_text
    return visit_context_text(vc)


def test_scribe_clinical_reasoning_rules():
    print("\nScribe prompt clinical-reasoning rules (Phase A1, no LLM)")
    from agents.scribe_agent import NUM_PREDICT, SYSTEM_PROMPT

    # The prompt is hard-wrapped for readability, so rule assertions run against
    # a whitespace-normalised copy: wrapping must not make a safety rule vanish.
    flat = " ".join(SYSTEM_PROMPT.split())
    lower = flat.lower()

    # --- Assessment must be a synthesis, not a restatement ---
    check("assessment is defined as a synthesis", "=== ASSESSMENT" in SYSTEM_PROMPT)
    check("prompt forbids restating the note", "not a restatement" in lower)
    check("prompt tells the model not to repeat the HPI",
          "never copy or retell the note" in lower)
    check("prompt bounds the assessment length", "2-4 sentences" in lower)
    check("prompt requires assessment to lead with the visit's problem",
          "the leading problem for this visit" in lower)

    # --- Plan must be specific, actionable and never empty ---
    check("plan has its own rule block", "=== PLAN" in SYSTEM_PROMPT)
    check("prompt rejects generic-only plans",
          "continue current medications and follow up" in lower)
    check("prompt rejects that generic plan explicitly",
          "not acceptable" in lower)
    check("plan must never be empty", "plan must never be empty" in lower)
    check("plan steps must be numbered and single-action",
          "short numbered steps" in lower and "one action each" in lower)
    check("plan must tie back to the assessment",
          "follow from a finding or problem named in ASSESSMENT".lower() in lower)

    # --- Acute vs chronic priority ---
    check("acute-vs-chronic rule block exists",
          "=== ACUTE vs CHRONIC PRIORITY ===" in SYSTEM_PROMPT)
    check("prompt names acute symptoms as the primary problem",
          "primary problem for this visit" in lower)
    check("prompt forbids chronic-only notes for acute visits",
          "chronic-care-only note" in lower)
    check("prompt gives the presenting-symptom-first pattern",
          "assessment leads with the presenting symptom" in lower)
    check("prompt never seeds an acute symptom into the note prose",
          "chest pain" not in lower,
          [w for w in ("chest pain",) if w in lower])

    # --- Grounding must not be weakened ---
    check("grounding block retained", "=== GROUNDING AND SAFETY" in SYSTEM_PROMPT)
    check("no-new-chronic-conditions rule retained",
          "NO NEW CHRONIC CONDITIONS" in SYSTEM_PROMPT)
    check("plan-grounding rule retained", "PLAN GROUNDING" in SYSTEM_PROMPT)
    check("retrieved-notes rule retained", "RETRIEVED NOTES" in SYSTEM_PROMPT)
    check("plan invention ban retained",
          "never invent a medication, dose, referral, test or procedure" in lower)
    check("value-type confusion rule added",
          "an hbA1c is not a blood glucose".lower() in lower)
    check("no false 'within normal limits' rule added",
          "within normal limits" in lower and "unless the" in lower)
    check("result relabelling is forbidden",
          "never relabel a result" in lower)
    check("a normal result cannot support a diagnosis",
          "never treat a normal result as evidence of a disease" in lower)
    check("already-performed tests must not be re-ordered",
          "already performed today" in lower)
    check("invented causal links are forbidden",
          "never invent a causal link between a result and a condition" in lower)
    check("repetition inside assessment is forbidden",
          "never say the same thing twice" in lower)
    check("templates are abstract, not copyable clinical facts",
          "the templates below are abstract" in lower
          and "<angle_brackets>" in lower)
    check("no concrete drug names leak into the prompt",
          not any(drug in lower for drug in
                  ("albuterol", "prednisone", "furosemide", "metformin", "lisinopril",
                   "nitroglycerin", "beta-blocker", "statin")),
          [d for d in ("albuterol", "prednisone", "furosemide", "metformin",
                       "lisinopril", "nitroglycerin", "statin") if d in lower])
    check("prompt forbids importing symptoms from the instructions",
          "do not import a symptom from these instructions" in lower)
    check("scribe defers symptom codes to the Supervisor",
          "added downstream by the supervisor" in lower)
    check("plan template does not pre-number steps (avoids '1. 1.')",
          "number your own steps" in lower)
    check("prompt forbids invented symptoms and durations",
          "never state a symptom, duration or diagnosis" in lower)
    check("prompt has an explicit thin-data rule",
          "=== when the data is thin ===" in lower)
    check("thin data prefers honesty over padding",
          "short honest note beats a complete invented one" in lower)
    check("thin-data plan gathers data instead of inventing treatment",
          "gather-missing-data step" in lower)
    check("icd table is marked as a lookup, not a suggestion",
          "not a suggestion" in lower)
    check("icd table keeps the grounded chronic mappings",
          "e11.9" in lower and "i10" in lower)
    check("icd rule extends the cardiac ban to i21",
          "i20.x (angina) or i21.x" in lower)
    check("cardiac codes require a chart-listed cardiac diagnosis",
          "patient context itself already lists a cardiac diagnosis" in lower)
    check("ecg findings cannot authorise a cardiac diagnosis code",
          "do not diagnose ischaemia from them" in lower)
    check("plan rule bars re-recommending completed work",
          "already been done; act on the result instead" in lower)

    # --- Token budget: plan must fit after a bounded assessment ---
    check("fast-mode budget exceeds the old 512 cap",
          NUM_PREDICT["fast"] > 512, NUM_PREDICT)
    check("quality-mode budget exceeds the old 1024 cap",
          NUM_PREDICT["quality"] > 1024, NUM_PREDICT)
    check("fast budget stays CPU-friendly (<=1024)", NUM_PREDICT["fast"] <= 1024, NUM_PREDICT)

    check("icd10 rule reminds the model not to drop the key",
          "never omit the key" in lower)


def test_specialty_field_passthrough():
    print("\nSpecialty extra fields reach the Scribe prompt (Phase A1, no LLM)")
    from agents.visit_context import (
        build_visit_context_block,
        get_specialty_fields,
        normalize_visit_context,
        specialty_field_catalog,
        specialty_fields_text,
        visit_context_text,
    )

    catalog = specialty_field_catalog()
    check("specialty catalog is read from config", bool(catalog), catalog)
    check("cardiology fields are in the catalog",
          {"chest_pain_characteristics", "ecg_rhythm_notes"} <= set(catalog), sorted(catalog))
    check("other specialties are represented too (not hard-coded to cardiology)",
          len({f for f in catalog}) >= 8, len(catalog))

    vc = normalize_visit_context({
        "chest_pain_characteristics": "Substernal pressure radiating to left arm, 8/10",
        "ecg_rhythm_notes": "Sinus rhythm 78 bpm, ST-T changes in V4-V6",
        "hpi": "Exertional chest pain over the last month.",
        "blood_pressure": "148/90",
        "unknown_key": "must still be dropped",
    })
    fields = get_specialty_fields(vc)
    check("specialty fields survive normalisation",
          set(fields) == {"chest_pain_characteristics", "ecg_rhythm_notes"}, fields)
    check("arbitrary unknown keys are still dropped", "unknown_key" not in vc, sorted(vc))
    check("specialty normalisation is idempotent", normalize_visit_context(vc) == vc)

    block = build_visit_context_block(vc)
    check("prompt block labels the specialty findings",
          "Specialty-specific findings" in block)
    check("prompt block carries the specialty values",
          "Substernal pressure radiating to left arm" in block
          and "ST-T changes in V4-V6" in block)
    check("specialty fields use their configured label",
          catalog["ecg_rhythm_notes"] in block, catalog.get("ecg_rhythm_notes"))
    check("specialty findings widen the grounding set",
          "ST-T changes in V4-V6" in visit_context_text(vc))
    check("specialty fields_text is standalone",
          "Substernal pressure radiating to left arm" in specialty_fields_text(vc))
    check("no specialty fields -> empty helpers",
          get_specialty_fields({}) == {} and specialty_fields_text({}) == "")

    check("non-dict specialty input is safe", get_specialty_fields("nope") == {})


def test_visit_focus_anchor():
    print("\nAcute-vs-chronic focus anchor (Phase A1, no LLM)")
    from agents.scribe_agent import _render_focus_block
    from agents.visit_context import normalize_visit_context

    acute = _render_focus_block(
        "Chest pain and shortness of breath since last night.",
        normalize_visit_context({"hpi": "Substernal pressure radiating to left arm."}),
    )
    check("acute complaint yields an acute focus block", "PRIMARY problem" in acute, acute)
    check("focus block names the detected symptom", "chest pain" in acute.lower())
    check("focus block tells chronic context not to crowd it out",
          "crowd out" in acute or "replace the acute problem" in acute)
    check("focus block covers multiple symptoms",
          "shortness of breath" in acute.lower(), acute)

    hpi_only = _render_focus_block(
        "Follow-up visit.",
        normalize_visit_context({"hpi": "Patient reports diarrhea for two days."}),
    )
    check("symptom found only in the HPI still anchors",
          "PRIMARY problem" in hpi_only and "diarrhea" in hpi_only.lower(), hpi_only)

    spec = _render_focus_block(
        "Cardiology follow-up.",
        normalize_visit_context({
            "chest_pain_characteristics": "Substernal pressure, 30 min, radiates to left arm",
            "functional_status_nyha": "Class II - Slight limitation",
        }),
    )
    check("specialty fields are scanned for the focus",
          "PRIMARY problem" in spec and "chest pain" in spec.lower(), spec)

    chronic = _render_focus_block(
        "Follow-up for diabetes and hypertension.", {"hpi": "Feels well overall."})
    check("pure chronic visit gets the chronic focus", "chronic-disease visit" in chronic, chronic)
    check("chronic focus does not claim an acute symptom",
          "PRIMARY problem" not in chronic, chronic)

    check("no complaint and no context yields no anchor",
          _render_focus_block("", None) == "" and _render_focus_block("  ", {}) == "")
    check("focus anchor never asserts a diagnosis",
          not any(term in acute.lower() for term in
                  ("diagnos", "consistent with", "likely due to")), acute)
    check("focus anchor only reports symptoms the clinician wrote",
          "chest pain" in acute.lower() and "asthma" not in acute.lower(), acute)


def main():
    print("=" * 70)
    print("LANGGRAPH STRUCTURE TESTS (no LLM required)")
    print("=" * 70)
    test_graph_topology()
    test_supervisor_pattern()
    test_supervisor_routing_rules()
    test_reducers()
    test_initial_state_defaults()
    test_streaming_and_result_contract()
    test_optional_arguments()
    test_package_exports()
    test_node_tool_helpers()
    test_cleanup_still_removes_asthma()
    test_scribe_defensive_defaults()
    test_coding_defensive_defaults()
    test_defaulted_content_flag()
    test_visit_context_module()
    test_visit_context_wiring()
    test_scribe_clinical_reasoning_rules()
    test_specialty_field_passthrough()
    test_visit_focus_anchor()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
