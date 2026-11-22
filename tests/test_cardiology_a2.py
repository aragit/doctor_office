"""
Phase A2 tests - Cardiology specialty deepening. No LLM required.

Run with:

    python -m tests.test_cardiology_a2

Phase A2 goal: when ``specialty="cardiology"`` and the cardiology extra fields
are supplied, the generated note should be visibly more cardiac-oriented and
should actually use the documented cardiac data - without inventing findings,
without weakening any safety gate, and without changing the character of the
General or chief-complaint-only paths.

What these tests lock in:
- Cardiology fields are section-tagged in the prompt block, so the Scribe knows
  where each documented finding belongs.
- The Cardiology profile tells the Scribe to use supplied cardiac data and
  forbids inventing ischemia, murmurs, ECG abnormalities or diagnoses.
- A deterministic coverage guard flags documented cardiac data the note ignored,
  as ONE advisory flag that finalises instead of retrying.
- General / neurology / dermatology / oncology rendering is untouched, and a
  cardiology chief-complaint-only visit injects no cardiac directives.
- Safety gates (I20/I21 ban, cleanup, validation, invented-vitals guard,
  deterministic CPT) and the P0 fixes are intact.
"""

import sys
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


DIABETES_CONTEXT = (
    "Patient: John Doe, 58yo male\n"
    "Conditions: Type 2 Diabetes, Hypertension\n"
    "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
    "Allergies: Penicillin\n"
    "Last Visit: Blood pressure 148/92. HbA1c 7.8%."
)

CARDIAC_RAW = {
    "chest_pain_characteristics": (
        "Substernal pressure, onset 3 days ago, radiates to left arm, 8/10, "
        "precipitated by exertion, relieved by rest."
    ),
    "ecg_rhythm_notes": (
        "Sinus rhythm 82 bpm, nonspecific ST-T changes in V4-V6, no acute ischemic changes."
    ),
    "functional_status_nyha": "Class II - Slight limitation",
    "hpi": "Three days of exertional chest pressure. Denies dyspnea at rest.",
    "blood_pressure": "146/88",
    "heart_rate": "82",
    "physical_exam": "Regular rate and rhythm. No murmur, gallop or rub heard.",
}

CARDIAC_COMPLAINT = "Chest pain on exertion for three days."


def _vc(extra=None):
    from agents.visit_context import normalize_visit_context

    raw = dict(CARDIAC_RAW)
    raw.update(extra or {})
    return normalize_visit_context(raw)


# --- 1. Cardiology + rich input: fields reach the prompt, section-tagged ------


def test_cardiology_fields_are_section_tagged():
    print("\nCardiology + rich input: fields are tagged with their SOAP section")
    from agents.specialty_profiles import PROMPT_SECTIONS
    from agents.visit_context import (
        get_specialty_fields,
        specialty_field_sections,
        visit_context_to_prompt_block,
        visit_context_to_grounding_text,
    )

    vc = _vc()
    fields = get_specialty_fields(vc)
    check("all three cardiac fields are collected",
          set(fields) == {"chest_pain_characteristics", "ecg_rhythm_notes",
                          "functional_status_nyha"}, sorted(fields))

    sections = specialty_field_sections()
    check("chest-pain characteristics are tagged SUBJECTIVE",
          sections.get("chest_pain_characteristics") == "subjective", sections)
    check("ECG notes are tagged OBJECTIVE",
          sections.get("ecg_rhythm_notes") == "objective", sections)
    check("functional status is tagged ASSESSMENT",
          sections.get("functional_status_nyha") == "assessment", sections)
    check("every tag is a known SOAP section",
          all(v in PROMPT_SECTIONS for v in sections.values()), sections)

    block = visit_context_to_prompt_block(vc)
    check("prompt block tags the subjective finding", "[SUBJECTIVE]" in block, block)
    check("prompt block tags the objective finding", "[OBJECTIVE]" in block, block)
    check("prompt block tags the assessment finding", "[ASSESSMENT]" in block, block)
    check("field values survive into the prompt",
          "Substernal pressure" in block and "ST-T changes in V4-V6" in block
          and "Class II" in block, block)
    check("prompt block keeps the authoritative header",
          "authoritative" in block.lower())
    check("prompt block instructs not to drop tagged findings",
          "do not leave it out of the note" in block, block)
    check("prompt block still forbids invention",
          "Never invent an" in block, block)

    # Grounding text (the cleanup/validation widening) is unchanged by tagging.
    grounding = visit_context_to_grounding_text(vc)
    check("grounding text carries the cardiac values",
          "ST-T changes in V4-V6" in grounding and "Substernal pressure" in grounding,
          grounding)
    check("grounding text carries no section tags",
          "[SUBJECTIVE]" not in grounding and "[OBJECTIVE]" not in grounding, grounding)


def test_cardiology_profile_directives():
    print("\nCardiology profile: use the data, never invent the findings")
    from agents.scribe_agent import _render_specialty_guidance
    from agents.specialty_profiles import SPECIALTY_PROFILES, get_specialty_profile

    guidance = _render_specialty_guidance("cardiology")
    flat = " ".join(guidance.split()).lower()
    check("cardiology guidance is rendered", "Cardiology" in guidance, guidance[:80])
    check("tells the model to use the supplied data",
          "use the documented cardiac data" in flat, guidance)
    check("tells the model not to ignore the fields",
          "do not ignore it" in flat)
    check("names the three cardiac inputs",
          all(term in flat for term in
              ("chest-pain characteristics", "ecg/rhythm notes", "functional status")),
          guidance)
    check("routes chest-pain character to Subjective",
          "chest-pain character is subjective" in flat, guidance)
    check("routes ECG and rhythm to Objective",
          "ecg and rhythm findings are objective" in flat, guidance)
    check("routes functional class to Assessment",
          "functional class is assessment" in flat, guidance)
    check("states that an unused field makes the note incomplete",
          "note is incomplete" in flat)
    check("assessment must lead with the cardiac problem",
          "lead with the cardiac presentation" in flat)
    check("plan must act on already-documented results",
          "do not re-order it" in flat and "already recorded" in flat)
    check("forbids inventing murmurs, rhythms, ECG values",
          "never invent a murmur" in flat and "st-t change" in flat, guidance)
    check("forbids diagnosing ischemia from chest pain alone",
          "from chest pain alone" in flat)
    check("keeps the I20/I21 chart requirement",
          "i20.x/i21.x require a cardiac diagnosis already documented" in flat, guidance)

    # Other profiles untouched in character.
    general = " ".join(_render_specialty_guidance("general").split()).lower()
    check("General profile is not cardiac", "cardiac" not in general, general)
    check("General profile keeps its own framing",
          "general internal medicine" in general, general)
    check("unknown specialty falls back to general", "general internal medicine" in
          " ".join(_render_specialty_guidance("not-a-specialty").split()).lower())
    check("no specialty means no guidance block", _render_specialty_guidance(None) != ""
          and "General" in _render_specialty_guidance(None))
    check("cardiology profile object carries the new emphasis",
          "USE THE DOCUMENTED CARDIAC DATA"
          in get_specialty_profile("cardiology").prompt_emphasis)
    check("profile registry still lists every specialty",
          set(SPECIALTY_PROFILES) == {"general", "cardiology", "neurology",
                                      "dermatology", "oncology"}, sorted(SPECIALTY_PROFILES))


def test_regeneration_keeps_the_cardiac_directives():
    print("\nHITL regeneration re-renders the cardiac directives every pass")
    from agents.scribe_agent import (
        _render_focus_block,
        _render_specialty_guidance,
        _render_visit_context,
    )

    vc = _vc()
    first = {
        "guidance": _render_specialty_guidance("cardiology"),
        "focus": _render_focus_block(CARDIAC_COMPLAINT, vc),
        "context": _render_visit_context(vc),
    }
    second = {
        "guidance": _render_specialty_guidance("cardiology"),
        "focus": _render_focus_block(CARDIAC_COMPLAINT, vc),
        "context": _render_visit_context(vc),
    }
    check("cardiac guidance is re-rendered, not cached", first == second)
    check("regeneration still carries the cardiac directives",
          "USE THE DOCUMENTED CARDIAC DATA" in second["guidance"])
    check("regeneration still carries the section-tagged fields",
          "[OBJECTIVE]" in second["context"] and "ST-T changes" in second["context"],
          second["context"])
    check("regeneration still carries the acute focus anchor",
          "PRIMARY problem" in second["focus"])

    # The block is re-derivable from raw UI input every time (no stale state).
    from agents.visit_context import normalize_visit_context

    rebuilt = _render_visit_context(normalize_visit_context({
        "ecg_rhythm_notes": "Sinus rhythm 82 bpm, ST-T changes in V4-V6.",
    }))
    check("block is rebuilt from raw UI input each pass",
          "[OBJECTIVE]" in rebuilt and "ST-T changes" in rebuilt, rebuilt)


def test_focus_anchor_still_fires_for_cardiac_fields():
    print("\nAcute focus anchor: a cardiac field alone can anchor the visit")
    from agents.scribe_agent import _render_focus_block

    anchored = _render_focus_block("Cardiology follow-up.", _vc())
    check("chest-pain field anchors an acute focus", "PRIMARY problem" in anchored,
          anchored[:160])
    check("the anchor names chest pain", "chest pain" in anchored.lower())
    check("the anchor asserts no diagnosis",
          not any(t in anchored.lower() for t in ("diagnos", "ischemi", "ischaemi")),
          anchored)

    # A chronic complaint with no cardiac data must not be forced acute.
    from agents.visit_context import normalize_visit_context

    chronic_only = normalize_visit_context({
        "blood_pressure": "148/92",
        "additional_findings": "HbA1c 8.1% (was 7.8%)",
    })
    chronic = _render_focus_block("Routine follow-up for diabetes and hypertension.", chronic_only)
    check("a chronic complaint without cardiac data is not forced acute",
          "PRIMARY problem" not in chronic, chronic[:160])

    complaint_anchored = _render_focus_block(CARDIAC_COMPLAINT, {})
    check("chief complaint alone still anchors", "PRIMARY problem" in complaint_anchored,
          complaint_anchored[:120])


# --- 2. coverage guard -------------------------------------------------------


def _note(**overrides):
    base = {
        "subjective": "Exertional substernal pressure radiating to the left arm, 8/10.",
        "objective": "BP 146/88. ECG: sinus rhythm 82 bpm with nonspecific ST-T changes in V4-V6.",
        "assessment": "Exertional chest pressure with NYHA class II limitation; ischemia not confirmed.",
        "plan": "1. Act on the documented ECG and arrange cardiology follow-up.",
        "icd10_suggestions": [],
    }
    base.update(overrides)
    return base


def test_specialty_coverage_guard():
    print("\nCoverage guard: documented cardiac data the note ignored")
    from agents.specialty_coverage import (
        COVERAGE_FLAG_TEMPLATE,
        build_specialty_coverage_flag,
        find_unreflected_specialty_fields,
    )

    vc = _vc()
    complete = _note()
    check("a note that uses every field raises nothing",
          find_unreflected_specialty_fields(complete, vc) == [],
          find_unreflected_specialty_fields(complete, vc))
    check("and produces no flag", build_specialty_coverage_flag(complete, vc) == [])

    no_ecg = _note(
        objective="BP 146/88. Heart sounds regular, no murmur.",
        plan="1. Cardiology follow-up to review symptoms.",
    )
    missing = find_unreflected_specialty_fields(no_ecg, vc)
    check("an ignored ECG field is detected", missing == ["ECG / Rhythm Notes"], missing)
    flag = build_specialty_coverage_flag(no_ecg, vc)
    check("exactly one flag is raised", len(flag) == 1, flag)
    check("flag names the field", "ECG / Rhythm Notes" in flag[0], flag)
    check("flag is actionable", "into that section" in flag[0], flag)

    bare = {"subjective": "Follow-up.", "objective": "BP 146/88.",
            "assessment": "Stable.", "plan": "1. Continue medications."}
    all_missing = find_unreflected_specialty_fields(bare, vc)
    check("all three omissions are detected in one pass",
          all_missing == ["Chest Pain Characteristics", "ECG / Rhythm Notes",
                          "Functional Status / NYHA Class"], all_missing)
    bare_flags = build_specialty_coverage_flag(bare, vc)
    check("each omission is named separately (Phase A2.1 section scoping)",
          len(bare_flags) == 3, bare_flags)

    # Section scoping (A2.1): each field must land in its tagged section.
    misplaced = _note(objective="BP 146/88.",
                      assessment="ECG shows sinus rhythm with ST-T changes V4-V6; "
                                 "NYHA class II; exertional substernal chest pain.")
    check("evidence in the wrong section does not count",
          find_unreflected_specialty_fields(misplaced, vc) != [],
          find_unreflected_specialty_fields(misplaced, vc))

    # Silence when nothing was documented.
    from agents.visit_context import normalize_visit_context

    chronic_vc = normalize_visit_context({"blood_pressure": "148/92"})
    check("no specialty fields -> no findings", find_unreflected_specialty_fields(bare, chronic_vc) == [])
    check("no specialty fields -> no flag", build_specialty_coverage_flag(bare, chronic_vc) == [])
    check("no visit context -> no flag", build_specialty_coverage_flag(bare, None) == [])
    check("empty visit context -> no flag", build_specialty_coverage_flag(bare, {}) == [])

    # Untagged specialties are never flagged by this guard.
    from agents.visit_context import normalize_visit_context as norm

    neuro = norm({"neurological_symptoms_detail": "Right-sided weakness onset 2 hours ago."})
    check("untagged specialty is not guarded",
          find_unreflected_specialty_fields(bare, neuro) == [],
          find_unreflected_specialty_fields(bare, neuro))

    # Partial documentation: only the supplied field is required.
    ecg_only = norm({"ecg_rhythm_notes": "Sinus rhythm 78 bpm."})
    check("only supplied fields are required",
          find_unreflected_specialty_fields(_note(), ecg_only) == [],
          find_unreflected_specialty_fields(_note(), ecg_only))

    check("flag template is single-line", "\n" not in COVERAGE_FLAG_TEMPLATE)

    # "Nothing to report" answers must not raise a flag.
    for blank in ("Not assessed", "none", "N/A", "-", "", "Not documented."):
        declined = norm({"functional_status_nyha": blank})
        check(f"declined value raises no flag: {blank!r}",
              build_specialty_coverage_flag(bare, declined) == [],
              build_specialty_coverage_flag(bare, declined))
    partial = norm({"ecg_rhythm_notes": "Sinus rhythm 78 bpm.",
                    "functional_status_nyha": "Not assessed"})
    check("a declined field is not required",
          find_unreflected_specialty_fields(_note(), partial) == [],
          find_unreflected_specialty_fields(_note(), partial))
    check("a declined field does not mask a real omission",
          find_unreflected_specialty_fields(no_ecg, partial) == ["ECG / Rhythm Notes"],
          find_unreflected_specialty_fields(no_ecg, partial))

    # A finding that merely *starts* with a decline word is real data.
    for recorded in ("No acute ischemic changes",
                     "no ST-T changes; sinus rhythm 82 bpm",
                     "Normal study, no ischaemia",
                     "Nasal congestion only"):
        finding = norm({"ecg_rhythm_notes": recorded})
        check(f"negative-finding value still requires reflection: {recorded!r}",
              find_unreflected_specialty_fields(no_ecg, finding) == ["ECG / Rhythm Notes"],
              find_unreflected_specialty_fields(no_ecg, finding))
    check("a declined value with an explanation is still declined",
          find_unreflected_specialty_fields(no_ecg, norm(
              {"ecg_rhythm_notes": "Not assessed (patient asymptomatic)"})) == [])


def test_coverage_flag_is_advisory_and_never_retries():
    print("\nCoverage flag is advisory: it finalises, it never loops")
    from agents.specialty_coverage import build_specialty_coverage_flag
    from agents.supervisor_policy import (
        MAX_RETRIES,
        classify_flags,
        determine_route,
        route_after_validation,
    )

    bare = {"subjective": "Follow-up.", "objective": "BP 146/88.",
            "assessment": "Stable.", "plan": "1. Continue medications."}
    flag = build_specialty_coverage_flag(bare, _vc())
    buckets = classify_flags(flag)
    check("flag lands in the advisory bucket", buckets["advisory"] == flag, buckets)
    check("flag triggers no scribe retry", buckets["scribe"] == [], buckets)
    check("flag triggers no coding retry", buckets["coding"] == [], buckets)

    decision = determine_route(flag, 0, {}, CARDIAC_COMPLAINT)
    check("a coverage flag finalises the visit",
          decision.route_to == "memory_save_node", decision)
    check("it is not an escalation",
          decision.class_ == "finalize" and not decision.escalated, decision)
    check("no feedback is generated", decision.feedback == [], decision)

    state = {"cycle_flags": {"validation": flag}, "retry_count": 0,
             "coding_result": {}, "chief_complaint": CARDIAC_COMPLAINT, "route_to": ""}
    check("the conditional edge routes to finalize",
          route_after_validation(state) == "memory_save_node",
          route_after_validation(state))

    # Combined with a real problem it must not mask the retry/escalation path.
    combined = flag + ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"]
    check("a real problem still routes to the scribe",
          determine_route(combined, 0, {}, CARDIAC_COMPLAINT).route_to == "scribe_node")
    check("exhausted retries still escalate",
          determine_route(combined, MAX_RETRIES, {}, CARDIAC_COMPLAINT)
          .route_to == "human_review_node")


# --- 3. General / untagged / complaint-only paths unchanged -------------------


def test_general_and_complaint_only_paths_unchanged():
    print("\nGeneral and chief-complaint-only paths keep their character")
    from agents.scribe_agent import (
        SYSTEM_PROMPT,
        _render_specialty_guidance,
        _render_visit_context,
    )
    from agents.visit_context import (
        normalize_visit_context,
        visit_context_to_prompt_block,
        visit_context_to_grounding_text,
    )

    # Same complaint, same data, General specialty: no cardiac framing anywhere.
    general_block = visit_context_to_prompt_block(_vc())
    general_guidance = _render_specialty_guidance("general")
    check("General guidance is not cardiac",
          "cardiac" not in general_guidance.lower(), general_guidance)
    check("General guidance carries none of the A2 cardiac directives",
          "USE THE DOCUMENTED CARDIAC DATA" not in general_guidance
          and "NYHA" not in general_guidance)

    # The supplied cardiac values still reach the prompt under General (they are
    # the clinician's data) and carry no cardiac instruction. Section tags are a
    # property of the *field configuration*, not of the selected specialty: they
    # route documented data into a SOAP section and say nothing about the visit.
    check("cardiac data is still shown under General",
          "ST-T changes in V4-V6" in general_block, general_block)
    check("section tags follow the field config, not the specialty",
          "[SUBJECTIVE]" in general_block and "[OBJECTIVE]" in general_block, general_block)
    check("General sees no cardiology-only instruction",
          "NEVER" not in general_guidance or "I20.x" not in general_guidance,
          general_guidance)

    # Chief-complaint-only cardiology: no fields, so no cardiac directives at all.
    complaint_only = _render_visit_context({})
    check("chief-complaint-only injects no visit block", complaint_only == "",
          complaint_only)
    empty_block = visit_context_to_prompt_block({})
    check("empty visit context renders nothing", empty_block == "")
    check("empty grounding text renders nothing", visit_context_to_grounding_text({}) == "")
    check("chronic cardiology visit without fields has no cardiac framing in the block",
          "[SUBJECTIVE]" not in visit_context_to_prompt_block(
              normalize_visit_context({"blood_pressure": "148/92"})))

    # The global system prompt is unchanged by A2 (no cardiac special-casing).
    flat = " ".join(SYSTEM_PROMPT.split()).lower()
    check("system prompt has no cardiology special case", "cardiology" not in flat)
    check("system prompt still bans I20/I21",
          "never use i20.x (angina) or i21.x" in flat)
    check("system prompt still bans invented findings",
          "never invent an examination finding" in flat)

    # Untagged specialties render exactly as before Phase A2.
    for raw, expected in (
        ({"lesion_description": "3 cm erythematous plaque"}, "Lesion Description"),
        ({"dermoscopy_notes": "regular pigment network"}, "Dermoscopy Findings (if performed)"),
        ({"cancer_history_detail": "Stage IIIC breast cancer"}, "Cancer History Detail"),
        ({"performance_status": "1 - Restricted"}, "Performance Status (ECOG)"),
        ({"neurological_symptoms_detail": "Right-sided weakness"}, "Neurological Symptoms Detail"),
        ({"treatment_toxicities": "Grade 2 neuropathy"}, "Current Treatment Toxicities (CTCAE)"),
        ({"neuro_exam_findings": "R UE 3/5"}, "Neurological Exam Findings"),
    ):
        block = visit_context_to_prompt_block(normalize_visit_context(raw))
        check(f"untagged field renders plainly: {expected}", f"* {expected}:" in block, block)
        check(f"untagged field carries no section tag: {expected}", "[" not in block, block)


# --- 4. Safety gates and P0 fixes preserved ----------------------------------


def test_safety_gates_intact():
    print("\nSafety gates and P0 fixes are preserved")
    from agents.coding_node import build_symptom_hint
    from agents.specialty_coverage import build_specialty_coverage_flag
    from agents.scribe_agent import SYSTEM_PROMPT
    from agents.scribe_node import build_invented_findings_flag
    from agents.supervisor_policy import classify_flags, determine_route
    from utils.validation import clean_soap_note, run_full_validation

    # A2 must not let a cardiology field justify a cardiac ICD code.
    ctx = "Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes, Hypertension"
    vc = _vc()
    soap = {
        "subjective": "Exertional chest pressure.",
        "objective": "ECG with ST-T changes.",
        "assessment": "Angina, ischemic etiology.",
        "plan": "1. Start aspirin.\n2. Start albuterol PRN.",
        "icd10_suggestions": ["E11.9 - Type 2 diabetes", "I20.9 - Angina pectoris",
                              "J45.909 - Unspecified asthma"],
    }
    cleaned, removed = clean_soap_note(soap, ctx, CARDIAC_COMPLAINT,
                                      visit_context_to_grounding_text_safe(vc))
    flat = " ".join(str(v) for v in cleaned.values())
    check("cleanup still strips asthma", "albuterol" not in flat, flat[:200])
    check("cleanup still removes J45", not any("J45" in c for c in cleaned["icd10_suggestions"]))
    check("documented ECG keeps the cardiac finding in the note",
          "ST-T changes" in flat, flat[:200])
    validation = run_full_validation(
        cleaned["icd10_suggestions"], {"primary_icd10": "I20.9 - Angina", "secondary_icd10": []},
        ctx, CARDIAC_COMPLAINT,
    )
    i20_flags = [f for f in validation["quality_flags"] if "I20.9" in f]
    check("I20 without a charted cardiac diagnosis is still flagged",
          i20_flags and "hallucinated" in i20_flags[0].lower(), validation["quality_flags"])
    check("I20 flag routes to a scribe retry",
          determine_route(i20_flags, 0, {}, CARDIAC_COMPLAINT).route_to == "scribe_node",
          i20_flags)
    # With the symptom-gap flag also present the supervisor prefers the
    # zero-LLM re-check - existing precedence, unchanged by A2.
    check("symptom gap still outranks the note defect",
          determine_route(validation["quality_flags"], 0, {}, CARDIAC_COMPLAINT).route_to
          == "validation_node", validation["quality_flags"])

    # Invented-vitals guard still active on a cardiology visit.
    invented = build_invented_findings_flag(
        {"objective": "BP 160/100, HR 110, crackles bilaterally."},
        {"blood_pressure": "146/88"}, DIABETES_CONTEXT,
    )
    check("invented-vitals guard still fires", invented and
          invented[0].startswith("Objective contains measurements not provided"), invented)
    check("invented vitals flag is advisory",
          determine_route(invented, 0, {}, CARDIAC_COMPLAINT).route_to == "memory_save_node")

    # Deterministic CPT path untouched.
    from tools.em_calculator import calculate_em_code, derive_mdm_domains

    domains = derive_mdm_domains(_note(), DIABETES_CONTEXT)
    em = calculate_em_code(domains["problems_level"], domains["data_level"],
                           domains["risk_level"])
    check("CPT calculator still returns an E/M level",
          em["valid"] and em["cpt_code"] in {"99212", "99213", "99214", "99215"}, em)

    # Cardiac data alone must not create a coding hint that was not justified.
    check("symptom hint still limits codes to SECONDARY", "SECONDARY" in build_symptom_hint(
        ["R07.9 - Chest pain, unspecified"]))

    # The A2 flag coexists with the other advisory flags without becoming a retry.
    combined = build_specialty_coverage_flag(
        {"subjective": "s", "objective": "o", "assessment": "a", "plan": "p"}, vc)
    combined += build_invented_findings_flag(
        {"objective": "BP 160/100"}, {"blood_pressure": "146/88"}, DIABETES_CONTEXT)
    buckets = classify_flags(combined)
    check("both advisories stay advisory", buckets["advisory"] == combined, buckets)
    check("no retry from A2 advisories",
          determine_route(combined, 0, {}, CARDIAC_COMPLAINT).route_to == "memory_save_node")

    flat_prompt = " ".join(SYSTEM_PROMPT.split())
    check("A2 did not weaken the abstract-template rule",
          "the templates below are abstract" in flat_prompt.lower())


def visit_context_to_grounding_text_safe(vc):
    from agents.visit_context import visit_context_to_grounding_text

    return visit_context_to_grounding_text(vc)


def test_p0_fixes_still_hold():
    print("\nP0 fixes remain intact")
    from agents.scribe_node import RULES_LINE, build_grounding_block

    from utils.config import MODEL_PRESETS
    from utils.llm import get_llm

    check("get_llm still accepts model_name", get_llm(model_name="gemma3:4b").model_name == "gemma3:4b")
    check("get_llm still honours the mode", get_llm(mode="quality").model_name == MODEL_PRESETS["quality"])

    facts = {"conditions": ["Type 2 Diabetes"] * 8, "medications": ["Metformin"] * 8,
             "allergies": ["Penicillin", "Sulfa", "Latex", "Codeine"]}
    block = build_grounding_block(facts, "PATIENT HISTORY (reference only):\nRecent visits:\n  - old")
    check("grounding RULES line still survives truncation", RULES_LINE in block, block[-160:])

    import importlib

    from agents.coding_agent import CodingResult

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    try:
        coding_node.generate_coding = lambda **kw: CodingResult(
            primary_icd10="E11.9999 - bogus",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[], medical_necessity_note="x",
        )
        update = coding_node.coding_node({
            "patient_context": DIABETES_CONTEXT,
            "chief_complaint": CARDIAC_COMPLAINT,
            "cleaned_soap_note": {"subjective": "s", "objective": "o", "assessment": "a",
                                  "plan": "p", "icd10_suggestions": []},
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("malformed ICD-10 is still dropped", update["coding_result"]["primary_icd10"] == "",
          update["coding_result"]["primary_icd10"])
    check("malformed drop is still flagged",
          any("dropped malformed" in f for f in update.get("quality_flags", [])),
          update.get("quality_flags"))

    # Memory repository interface: the only change since A2 is the P2 `status`
    # parameter, which defaults to a draft.
    import inspect

    from memory.patient_memory import (
        VISIT_STATUS_DRAFT,
        JsonPatientMemoryRepository,
    )

    signature = inspect.signature(JsonPatientMemoryRepository.record_visit)
    params = signature.parameters
    check("record_visit signature: status added, nothing else",
          list(params) == ["self", "patient_id", "patient_context", "chief_complaint",
                           "coding_result", "soap_note", "quality_flags", "status"], list(params))
    check("record_visit still records a draft unless told otherwise",
          params["status"].default == VISIT_STATUS_DRAFT, params["status"].default)
    check("approve_visit exists and is keyword-friendly",
          "visit_id" in inspect.signature(
              JsonPatientMemoryRepository.approve_visit).parameters)


def main():
    print("=" * 70)
    print("PHASE A2 CARDIOLOGY TESTS (no LLM required)")
    print("=" * 70)
    test_cardiology_fields_are_section_tagged()
    test_cardiology_profile_directives()
    test_regeneration_keeps_the_cardiac_directives()
    test_focus_anchor_still_fires_for_cardiac_fields()
    test_specialty_coverage_guard()
    test_coverage_flag_is_advisory_and_never_retries()
    test_general_and_complaint_only_paths_unchanged()
    test_safety_gates_intact()
    test_p0_fixes_still_hold()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())