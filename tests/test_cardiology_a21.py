"""
Phase A2.1 tests - Cardiology narrative guardrails. No LLM required.

Run with:

    python -m tests.test_cardiology_a21

Phase A2.1 exists because of a real UI run: with diabetes + hypertension on the
chart, a normal-sinus ECG with nonspecific ST-T changes in the specialty fields,
and NYHA class II, the pipeline produced correct codes (E11.9, I10, R07.9, no
I20/I21) but a note that claimed "most consistent with ischemic etiology",
planned "further evaluation for ischemic heart disease", repeated the ischemic
framing in the medical-necessity text, gave a one-line Subjective that ignored
the documented chest-pain characteristics, and an Objective that recorded vitals
only instead of the documented ECG.

These tests lock in the three fixes: a hard ban on unsupported cardiac claims
(prompt + deterministic scrub), section-scoped specialty field fidelity, and a
coding-side necessity scrub - with no new retry loops and no change to the
existing code-level, cleanup, validation, CPT or P0 behaviour.
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


# The chart from the failing UI run: no cardiac diagnosis at all.
DIABETES_CHART = (
    "Patient: John Doe, 58yo male\n"
    "Conditions: Type 2 Diabetes, Hypertension\n"
    "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
    "Allergies: Penicillin\n"
    "Last Visit: BP 148/92. HbA1c 7.8%."
)
CARDIAC_CHART = DIABETES_CHART + "\nConditions: Coronary artery disease, prior MI (2019)."

UI_FINDINGS = {
    "chest_pain_characteristics": (
        "Substernal pressure, onset 3 days ago, radiates to left arm, 8/10, "
        "precipitated by exertion, relieved by rest."
    ),
    "ecg_rhythm_notes": (
        "Sinus rhythm 82 bpm, nonspecific ST-T changes in V4-V6, no acute ischemic changes."
    ),
    "functional_status_nyha": "Class II - Slight limitation",
    "hpi": "Three days of exertional chest pressure.",
    "blood_pressure": "146/88",
    "physical_exam": "Regular rate and rhythm. No murmur, gallop or rub heard.",
}
COMPLAINT = "Chest pain on exertion for three days."

# The narrative that failed in the UI.
BAD_NOTE = {
    "subjective": "Chest pain x 1 day.",
    "objective": "BP 146/88, HR 82. HbA1c 8.1%.",
    "assessment": "The chest pain is most consistent with ischemic etiology given "
                  "the exertional character.",
    "plan": "1. Further evaluation for ischemic heart disease.\n2. Return in 2 weeks.",
    "icd10_suggestions": ["E11.9 - Type 2 diabetes mellitus without complications",
                          "I10 - Essential (primary) hypertension",
                          "R07.9 - Chest pain, unspecified"],
}
# What the note should look like.
GOOD_NOTE = {
    "subjective": "Substernal exertional pressure for three days, radiating to the left "
                  "arm, 8/10 severity, precipitated by exertion and relieved by rest.",
    "objective": "BP 146/88 mmHg, HR 82 bpm. ECG: sinus rhythm 82 bpm with nonspecific "
                 "ST-T changes in V4-V6, no acute ischemic changes. No murmur heard.",
    "assessment": "Exertional substernal pressure with NYHA class II limitation. "
                  "Nocturnal symptoms absent. Ischaemia is not established by today's data.",
    "plan": "1. Continue current medications.\n2. Correlate symptoms with the "
            "documented ECG findings at follow-up.\n3. Return precautions for worsening "
            "pain.\n4. Risk-factor control for diabetes and hypertension.",
    "icd10_suggestions": ["E11.9 - Type 2 diabetes mellitus without complications",
                          "I10 - Essential (primary) hypertension",
                          "R07.9 - Chest pain, unspecified"],
}


def _vc():
    from agents.visit_context import normalize_visit_context

    return normalize_visit_context(dict(UI_FINDINGS))


# --- 1. banned cardiac claims -------------------------------------------------


def test_claim_detector():
    print("\nCardiac claim detector: flags diagnoses, allows findings and negations")
    from utils.cardiac_claims import (
        find_unsupported_cardiac_claims,
        has_charted_cardiac_diagnosis,
    )

    must_flag = [
        "The chest pain is most consistent with ischemic etiology.",
        "1. Further evaluation for ischemic heart disease.",
        "2. If ischemic etiology is suspected, order a troponin.",
        "Presentation is consistent with angina.",
        "Findings highly suspicious for ischemia.",
        "Symptoms concerning for myocardial infarction.",
        "Assess for ACS.",
        "Likely unstable angina given the presentation.",
        "Correlated with coronary artery disease.",
        "Findings consistent with ischaemia.",
        "Etiology is ischemic.",
        "Chest pain raises concern for acute coronary syndrome.",
        # Phrasings the 1.5b model actually produced in the live run.
        "Chest pain is most consistent with a new or acute presentation of angina.",
        "Symptoms are consistent with an acute presentation of ischemia.",
        "The patient is likely to be suffering from angina.",
        "Findings in keeping with an acute coronary event.",
    ]
    for text in must_flag:
        found = find_unsupported_cardiac_claims(text)
        check(f"flags: {text[:56]}", bool(found), found)

    must_allow = [
        # The clinician's own documented ECG findings must survive untouched.
        "ECG: sinus rhythm 82 bpm with nonspecific ST-T changes in V4-V6, no acute ischemic changes.",
        "No acute ischemic changes identified on today's ECG.",
        "Rhythm is normal sinus, not ischemic.",
        "Patient denies angina today.",
        # Risk framing is explicitly allowed, just without the conclusion.
        "Diabetes and hypertension raise lifetime risk of ischemic heart disease.",
        "Risk-factor modification targets ischemic heart disease risk.",
        # Ordinary clinical prose.
        "Chest pain is a symptom of unknown cause.",
        "Blood pressure is elevated from baseline.",
        "The etiology of the pain is unclear.",
        "No murmur, gallop or rub heard.",
        "Presentation consistent with the absence of ischemic changes.",
        "ECG shows sinus rhythm; no evidence of ischemia or acute coronary event.",
        "Findings consistent with a benign musculoskeletal cause.",
    ]
    for text in must_allow:
        found = find_unsupported_cardiac_claims(text)
        check(f"allows: {text[:56]}", not found, found)

    check("diabetes/hypertension chart has no cardiac diagnosis",
          not has_charted_cardiac_diagnosis(DIABETES_CHART))
    check("a charted cardiac diagnosis is detected",
          has_charted_cardiac_diagnosis(CARDIAC_CHART))
    check("no chart text means no cardiac diagnosis",
          not has_charted_cardiac_diagnosis(""))
    check("charted diagnosis unlocks cardiac narrative",
          not find_unsupported_cardiac_claims(
              "Most consistent with ischemic etiology and unstable angina.",
              has_diagnosis=True,
          ))
    check("visit-context-only wording does not unlock it",
          bool(find_unsupported_cardiac_claims(
              "Most consistent with ischemic etiology.", has_diagnosis=False)))
    check("ECG text in the chart is not a cardiac diagnosis",
          not has_charted_cardiac_diagnosis(
              "Tests: ECG - sinus rhythm, nonspecific ST-T changes, no acute ischemic changes."))


def test_cleanup_scrubs_claims():
    print("\nCleanup removes unsupported cardiac claims from the note")
    from utils.validation import clean_soap_note

    cleaned, removed = clean_soap_note(dict(BAD_NOTE), DIABETES_CHART, COMPLAINT)
    flat = " ".join(str(v) for v in cleaned.values()).lower()

    check("'ischemic etiology' no longer in the assessment",
          "ischemic etiology" not in str(cleaned["assessment"]).lower(),
          cleaned["assessment"])
    check("'ischemic heart disease' no longer in the plan",
          "ischemic heart disease" not in str(cleaned["plan"]).lower(), cleaned["plan"])
    check("no unsupported cardiac claim survives anywhere",
          not any(phrase in flat for phrase in (
              "ischemic etiology", "ischemic heart disease", "consistent with angina")),
          flat)
    check("the removal is reported",
          any("unsupported cardiac claim" in item for item in removed), removed)
    check("the reported removal names the claim",
          any("ischemic etiology" in item for item in removed), removed)
    check("the input note is not mutated", "ischemic etiology" in BAD_NOTE["assessment"])

    # Codes and the honest parts of the note are untouched.
    check("ICD-10 suggestions are untouched by the narrative scrub",
          cleaned["icd10_suggestions"] == BAD_NOTE["icd10_suggestions"],
          cleaned["icd10_suggestions"])
    check("R07.9 survives", any("R07.9" in code for code in cleaned["icd10_suggestions"]))
    check("no I20/I21 is introduced",
          not any(code.startswith(("I20", "I21")) for code in cleaned["icd10_suggestions"]))
    check("an unflagged plan step is kept",
          "Return in 2 weeks" in cleaned["plan"], cleaned["plan"])

    # A good note is untouched.
    untouched, removed_good = clean_soap_note(dict(GOOD_NOTE), DIABETES_CHART, COMPLAINT)
    check("a compliant note is left alone", not removed_good, removed_good)
    check("a compliant note keeps its ECG finding",
          "sinus rhythm" in untouched["objective"].lower())
    check("a compliant note keeps its risk-factor plan",
          "Risk-factor control" in untouched["plan"])

    # A charted cardiac diagnosis means nothing is scrubbed.
    kept, removed_cardiac = clean_soap_note(
        {**BAD_NOTE, "assessment": "Most consistent with unstable angina."},
        CARDIAC_CHART, COMPLAINT,
    )
    check("a charted cardiac diagnosis is left alone",
          "angina" in kept["assessment"].lower() and not removed_cardiac, removed_cardiac)

    # The clinician's own ECG wording is never scrubbed.
    ecg_note = {
        "subjective": "Exertional substernal pain.",
        "objective": "ECG: sinus rhythm 82 bpm, nonspecific ST-T changes in V4-V6, "
                     "no acute ischemic changes.",
        "assessment": "Presentation under evaluation.",
        "plan": "1. Follow up.",
    }
    ecg_cleaned, ecg_removed = clean_soap_note(ecg_note, DIABETES_CHART, COMPLAINT)
    check("documented ECG wording survives cleanup",
          "no acute ischemic changes" in ecg_cleaned["objective"].lower(), ecg_removed)


def test_cleanup_flags_are_advisory():
    print("\nThe claim removal does not create a retry loop")
    from agents.cleanup_node import cleanup_node
    from agents.supervisor_policy import MAX_RETRIES, classify_flags, determine_route
    from utils.validation import clean_soap_note

    update = cleanup_node({
        "patient_context": DIABETES_CHART,
        "chief_complaint": COMPLAINT,
        "soap_note": dict(BAD_NOTE),
    })
    flags = update.get("quality_flags") or []
    check("cleanup raised a flag for the removal", flags, flags)
    buckets = classify_flags(flags)
    check("the removal flag is advisory, not a retry trigger",
          buckets["scribe"] == [] and buckets["coding"] == [], buckets)
    decision = determine_route(flags, 0, {}, COMPLAINT)
    check("it finalises instead of retrying",
          decision.route_to == "memory_save_node" and decision.class_ == "finalize", decision)
    check("no retry feedback is sent", decision.feedback == [], decision)
    check("it does not mask a real hallucination",
          determine_route(
              flags + ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"],
              0, {}, COMPLAINT,
          ).route_to == "scribe_node")
    check("escalation still works when retries are exhausted",
          determine_route(flags, MAX_RETRIES, {}, COMPLAINT).route_to == "memory_save_node")

    # An empty assessment after scrubbing is reported by the existing advisory flag.
    from agents.scribe_node import build_defaulted_content_flag
    from agents.scribe_agent import SOAPNote

    scrubbed_only = clean_soap_note(
        {"subjective": "s", "objective": "o",
         "assessment": "Most consistent with ischemic etiology.", "plan": "p",
         "icd10_suggestions": []},
        DIABETES_CHART, COMPLAINT,
    )[0]
    emptied_flags = build_defaulted_content_flag(SOAPNote(**scrubbed_only))
    check("an emptied assessment is reported as incomplete",
          any("empty assessment" in flag for flag in emptied_flags), emptied_flags)


# --- 2. field fidelity --------------------------------------------------------


def test_section_scoped_fidelity():
    print("\nSpecialty field fidelity is checked per SOAP section")
    from agents.specialty_coverage import (
        build_specialty_coverage_flag,
        find_unreflected_specialty_fields,
    )

    vc = _vc()
    missing = find_unreflected_specialty_fields(BAD_NOTE, vc)
    check("thin Subjective is caught",
          "Chest Pain Characteristics" in missing, missing)
    check("Objective without the ECG is caught",
          "ECG / Rhythm Notes" in missing, missing)
    check("Assessment without functional status is caught",
          "Functional Status / NYHA Class" in missing, missing)
    check("all three omissions are reported together", len(missing) == 3, missing)

    check("a compliant note reports nothing",
          find_unreflected_specialty_fields(GOOD_NOTE, vc) == [],
          find_unreflected_specialty_fields(GOOD_NOTE, vc))
    check("a compliant note raises no flag",
          build_specialty_coverage_flag(GOOD_NOTE, vc, DIABETES_CHART) == [])

    flags = build_specialty_coverage_flag(BAD_NOTE, vc, DIABETES_CHART)
    fidelity_flags = [f for f in flags if "Specialty data" in f]
    claim_flags = [f for f in flags if "Cardiac claim" in f]
    check("one flag per omitted field", len(fidelity_flags) == 3, fidelity_flags)
    check("each flag names its section",
          any("SUBJECTIVE section" in f for f in fidelity_flags)
          and any("OBJECTIVE section" in f for f in fidelity_flags)
          and any("ASSESSMENT section" in f for f in fidelity_flags), fidelity_flags)
    check("each flag names its field",
          any("Chest Pain Characteristics" in f for f in fidelity_flags)
          and any("ECG / Rhythm Notes" in f for f in fidelity_flags)
          and any("NYHA" in f for f in fidelity_flags), fidelity_flags)
    check("the claims in the failing note are also reported",
          len(claim_flags) == 2, claim_flags)

    # Section scoping: evidence in the wrong section does not satisfy the check.
    misplaced = {
        "subjective": "Chest pain x 1 day.",
        "objective": "BP 146/88 mmHg, HR 82 bpm.",
        "assessment": "Exertional chest pressure. NYHA class II limitation. ECG: sinus "
                      "rhythm 82 bpm with nonspecific ST-T changes in V4-V6.",
        "plan": "1. Follow up.",
    }
    still_missing = find_unreflected_specialty_fields(misplaced, vc)
    check("ECG text in the Assessment does not satisfy the Objective check",
          "ECG / Rhythm Notes" in still_missing, still_missing)
    check("a one-line Subjective is not accepted as fidelity",
          "Chest Pain Characteristics" in still_missing, still_missing)
    check("NYHA in the right section is accepted",
          "Functional Status / NYHA Class" not in still_missing, still_missing)

    # The clinician's own words count, even when paraphrased.
    paraphrased = {
        "subjective": "Patient reports substernal pressure on exertion, radiating to the "
                      "left arm, rated 8/10, easing with rest.",
        "objective": "ECG today: normal sinus rhythm at 82 bpm, with nonspecific ST-T "
                     "change in V4-V6.",
        "assessment": "Exertional pressure with functional class II limitation.",
        "plan": "1. Follow up.",
    }
    check("paraphrased descriptors satisfy the fidelity check",
          find_unreflected_specialty_fields(paraphrased, vc) == [],
          find_unreflected_specialty_fields(paraphrased, vc))

    # The live-run wording: symptom name plus two weak markers, no descriptors.
    thin_real = {
        "subjective": "Chest pain x 1 day; dyspnea at rest; three days of exertional "
                      "chest pressure; denies dyspnea at rest.",
        "objective": "BP 146/88 mmHg, HR 82 bpm. Sinus rhythm 82 bpm with nonspecific "
                     "ST-T changes in V4-V6.",
        "assessment": "Exertional chest pressure with NYHA class II limitation.",
        "plan": "1. Follow up.",
    }
    check("symptom name plus 'exertional' is not fidelity",
          find_unreflected_specialty_fields(thin_real, vc)
          == ["Chest Pain Characteristics"],
          find_unreflected_specialty_fields(thin_real, vc))

    # Different words, same recorded descriptors -> still fidelity.
    reworded = {
        "subjective": "Substernal discomfort on exertion, radiates to the left arm, "
                      "alleviated by rest.",
        "objective": "Electrocardiogram today: normal sinus at 82 bpm, nonspecific "
                     "ST-T change in V4-V6.",
        "assessment": "Functional class II limitation.",
        "plan": "1. Follow up.",
    }
    check("reworded descriptors still count",
          find_unreflected_specialty_fields(reworded, vc) == [],
          find_unreflected_specialty_fields(reworded, vc))

    # Silence when nothing was supplied or nothing was declined.
    from agents.visit_context import normalize_visit_context

    check("no specialty fields -> silent",
          find_unreflected_specialty_fields(BAD_NOTE, normalize_visit_context(
              {"blood_pressure": "148/92"})) == [])
    check("no visit context -> silent",
          find_unreflected_specialty_fields(BAD_NOTE, None) == [])
    check("complaint-only cardiology -> silent",
          find_unreflected_specialty_fields(BAD_NOTE, normalize_visit_context({})) == [])
    check("a declined field is not required",
          find_unreflected_specialty_fields(
              BAD_NOTE, normalize_visit_context(
                  {"functional_status_nyha": "Not assessed"})) == [],
          find_unreflected_specialty_fields(BAD_NOTE, normalize_visit_context(
              {"functional_status_nyha": "Not assessed"})))
    check("a negative-finding ECG value is still required",
          find_unreflected_specialty_fields(BAD_NOTE, normalize_visit_context(
              {"ecg_rhythm_notes": "No acute ischemic changes"}))
          == ["ECG / Rhythm Notes"])
    check("untagged specialties are never flagged",
          find_unreflected_specialty_fields(BAD_NOTE, normalize_visit_context(
              {"lesion_description": "3 cm erythematous plaque"})) == [])


def test_overlap_calibration():
    print("\nFidelity calibration: separation between carried and dropped descriptors")
    from agents.specialty_coverage import _overlap_threshold, _token_keys
    from agents.visit_context import normalize_visit_context

    def carries(section_text, value):
        vc = normalize_visit_context({"chest_pain_characteristics": value})
        note = {"subjective": section_text, "objective": "", "assessment": "", "plan": ""}
        return find_unreflected(note, vc)

    from agents.specialty_coverage import find_unreflected_specialty_fields as find_unreflected

    recorded = ("Substernal pressure, onset 3 days ago, radiates to left arm, 8/10, "
                "precipitated by exertion, relieved by rest.")
    for label, text, expected in (
        ("verbatim", "Substernal exertional pressure radiating to the left arm, 8/10, "
                     "precipitated by exertion and relieved by rest.", False),
        ("paraphrased", "Patient reports substernal pressure on exertion, radiating to "
                        "the left arm, rated 8/10, easing with rest.", False),
        ("reworded", "Substernal discomfort on exertion, radiates to the left arm, "
                     "alleviated by rest.", False),
        ("live-run wording", "Chest pain x 1 day; three days of exertional chest "
                             "pressure; denies dyspnea at rest.", True),
        ("symptom only", "Chest pain x 1 day.", True),
        ("complaint wording", "Chest pain on exertion for three days.", True),
    ):
        check(f"{label} -> {'flagged' if expected else 'accepted'}",
              bool(carries(text, recorded)) == expected, carries(text, recorded))

    # A short recorded value must stay satisfiable.
    check("a short recorded value is not over-demanded",
          carries("Radiating to the left arm", "Radiates to left arm") == [],
          carries("Radiating to the left arm", "Radiates to left arm"))
    check("a short recorded value still has to be reflected",
          carries("Pain is present today.", "Radiates to left arm") == ["Chest Pain Characteristics"])

    # The threshold never exceeds the number of recorded stems.
    for value in ("Radiates to left arm", "Class II - Slight limitation",
                  "Sinus rhythm 82 bpm", recorded,
                  "Substernal pressure, onset 3 days ago, radiates to left arm, 8/10 "
                  "severity, precipitated by exertion climbing stairs, relieved by rest "
                  "within 10 minutes, no diaphoresis, no nausea"):
        terms = _token_keys(value)
        check(f"threshold is reachable for {len(terms)} stems",
              0 < _overlap_threshold(terms) <= len(terms),
              (_overlap_threshold(terms), len(terms)))


def test_coverage_guard_claim_backstop():
    print("\nCoverage guard catches a cardiac claim that survived cleanup")
    from agents.specialty_coverage import build_specialty_coverage_flag

    vc = _vc()
    surviving = {**GOOD_NOTE, "assessment": "Most consistent with ischemic etiology."}
    flags = build_specialty_coverage_flag(surviving, vc, DIABETES_CHART)
    claim_flags = [f for f in flags if "Cardiac claim" in f]
    check("a surviving claim is flagged", len(claim_flags) == 1, flags)
    check("the flag names the claim", "ischemic etiology" in claim_flags[0], claim_flags)
    check("the flag states why", "no cardiac diagnosis" in claim_flags[0].lower(),
          claim_flags)
    check("no claim flag when the chart documents one",
          not [f for f in build_specialty_coverage_flag(
              surviving, vc, CARDIAC_CHART) if "Cardiac claim" in f])
    check("no claim flag without the chart text",
          not [f for f in build_specialty_coverage_flag(surviving, vc) if "Cardiac claim" in f])

    # A documented ECG negation must not excuse a claim in another section.
    mixed = {
        "subjective": "Exertional substernal pressure.",
        "objective": "ECG: sinus rhythm 82 bpm, no acute ischemic changes.",
        "assessment": "Symptoms are most consistent with angina.",
        "plan": "1. Follow up.",
    }
    mixed_flags = [f for f in build_specialty_coverage_flag(mixed, vc, DIABETES_CHART)
                   if "Cardiac claim" in f]
    check("a negation in another section does not excuse a claim",
          len(mixed_flags) == 1, mixed_flags)

    # Advisory routing.
    from agents.supervisor_policy import classify_flags, determine_route

    buckets = classify_flags(flags)
    check("every guard flag is advisory",
          buckets["scribe"] == [] and buckets["coding"] == [] and buckets["symptom"] == [],
          buckets)
    check("the visit still finalises",
          determine_route(flags, 0, {}, COMPLAINT).route_to == "memory_save_node")


# --- 3. coding-side medical necessity ----------------------------------------


def test_coding_necessity_scrub():
    print("\nCoding: medical necessity text must not claim a cardiac diagnosis")
    import importlib

    from agents.coding_agent import CodingResult

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    try:
        coding_node.generate_coding = lambda **kwargs: CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=["I10 - Essential (primary) hypertension",
                             "R07.9 - Chest pain, unspecified"],
            cpt_suggestions=[],
            medical_necessity_note=(
                "Chest pain in a patient with diabetes and hypertension supports "
                "further evaluation for ischemic heart disease. Chronic disease "
                "management continues."
            ),
        )
        update = coding_node.coding_node({
            "patient_context": DIABETES_CHART,
            "chief_complaint": COMPLAINT,
            "cleaned_soap_note": {"subjective": "s", "objective": "o",
                                  "assessment": "a", "plan": "p",
                                  "icd10_suggestions": []},
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    necessity = update["coding_result"]["medical_necessity_note"]
    check("the ischemic claim is gone from the necessity text",
          "ischemic heart disease" not in necessity.lower(), necessity)
    check("the honest part of the necessity text is kept",
          "Chronic disease management continues" in necessity, necessity)
    check("the necessity text is never empty", bool(necessity.strip()))

    flags = update.get("quality_flags") or []
    check("the scrub is reported", any("unsupported cardiac claim" in f for f in flags), flags)
    from agents.supervisor_policy import classify_flags, determine_route

    buckets = classify_flags(flags)
    check("the necessity scrub is advisory",
          buckets["coding"] == [] and buckets["scribe"] == [], buckets)
    check("it does not retry coding",
          determine_route(flags, 0, {}, COMPLAINT).route_to == "memory_save_node")

    check("codes are unchanged by the necessity scrub",
          update["coding_result"]["secondary_icd10"] ==
          ["I10 - Essential (primary) hypertension", "R07.9 - Chest pain, unspecified"],
          update["coding_result"]["secondary_icd10"])
    check("the CPT is still deterministic",
          (update["coding_result"].get("cpt_grounding") or {}).get("cpt_code")
          in {"99212", "99213", "99214", "99215"},
          update["coding_result"].get("cpt_grounding"))

    # Nothing is scrubbed when the chart documents a cardiac diagnosis.
    try:
        coding_node.generate_coding = lambda **kwargs: CodingResult(
            primary_icd10="I20.9 - Angina pectoris",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[],
            medical_necessity_note="Symptoms are consistent with angina.",
        )
        cardiac = coding_node.coding_node({
            "patient_context": CARDIAC_CHART,
            "chief_complaint": COMPLAINT,
            "cleaned_soap_note": {"subjective": "s", "objective": "o",
                                  "assessment": "a", "plan": "p",
                                  "icd10_suggestions": []},
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("a charted cardiac diagnosis keeps its narrative",
          "angina" in cardiac["coding_result"]["medical_necessity_note"].lower(),
          cardiac["coding_result"]["medical_necessity_note"])
    check("no scrub flag when the diagnosis is charted",
          not any("cardiac claim" in f for f in (cardiac.get("quality_flags") or [])),
          cardiac.get("quality_flags"))


# --- 4. prompt-level rules ----------------------------------------------------


def test_prompt_rules():
    print("\nPrompt rules: explicit bans and field fidelity")
    from agents.coding_agent import CODING_SYSTEM_PROMPT
    from agents.scribe_agent import SYSTEM_PROMPT, _render_specialty_guidance

    flat = " ".join(SYSTEM_PROMPT.split())
    low = flat.lower()
    check("system prompt bans unsupported cardiac diagnoses",
          "no unsupported cardiac diagnosis" in low, low[-400:])
    check("system prompt names the banned phrase 'ischemic etiology'",
          "ischemic etiology" in low)
    check("system prompt names the banned phrase 'ischemic heart disease'",
          "ischemic heart disease" in low)
    check("system prompt bans myocardial infarction / ACS",
          "myocardial infarction" in low and "acs" in low)
    check("system prompt still allows risk factors as risk factors",
          "risk factors" in low and "return precautions" in low)
    check("A2.1 did not weaken the I20/I21 chart requirement",
          "never use i20.x (angina) or i21.x" in low)
    check("A2.1 did not seed a symptom into the prompt", "chest pain" not in low)

    guidance = " ".join(_render_specialty_guidance("cardiology").split())
    glow = guidance.lower()
    check("cardiology profile lists the banned phrases",
          all(phrase in glow for phrase in (
              "ischemic etiology", "ischemic heart disease",
              "consistent with ischemia/ischaemia", "consistent with angina")),
          guidance)
    check("cardiology profile forbids naming angina at all",
          "name angina at all" in glow, guidance)
    check("cardiology profile forbids MI/ACS implication",
          "myocardial infarction, acs" in glow or "myocardial infarction, acs or coronary" in glow,
          guidance)
    check("cardiology profile states the exception",
          "unless the same diagnosis is already listed in patient context" in glow, guidance)
    check("cardiology profile allows symptom + risk-factor wording",
          "allowed:" in glow and "risk factors" in glow, guidance)
    check("cardiology profile rejects a thin subjective",
          "'chest pain x 1 day' is not an acceptable substitute" in glow, guidance)
    check("cardiology profile requires the ECG in the objective",
          "objective must record the supplied ecg/rhythm findings" in glow, guidance)
    check("cardiology profile requires functional class in the assessment",
          "assessment must use the documented functional class" in glow, guidance)
    check("cardiology profile still forbids invented findings",
          "never invent a murmur" in glow)

    general = " ".join(_render_specialty_guidance("general").split()).lower()
    check("General profile is still not cardiac", "cardiac" not in general, general)

    coding = " ".join(CODING_SYSTEM_PROMPT.split())
    clow = coding.lower()
    check("coding prompt cites only coded symptoms",
          "never explain the visit as ischemic" in clow, coding)
    check("coding prompt keeps the R07.x framing",
          "r07" in clow)
    check("coding prompt still forbids I20.x without documented ischemia",
          "never use i20.x (angina) without documented cardiac ischemia" in clow)


# --- 5. compatibility and P0 -------------------------------------------------


def test_compatibility_and_p0():
    print("\nCompatibility, code-level safety and P0 fixes")
    from agents.scribe_node import RULES_LINE, build_grounding_block
    from agents.specialty_profiles import get_specialty_profile
    from agents.visit_context import visit_context_to_prompt_block
    from utils.config import MODEL_PRESETS
    from utils.llm import get_llm
    from utils.validation import run_full_validation

    # I20/I21 code bans unchanged.
    ctx = "Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes, Hypertension"
    validation = run_full_validation(
        ["I20.9 - Angina pectoris"], {"primary_icd10": "", "secondary_icd10": []},
        ctx, COMPLAINT,
    )
    check("I20 without a charted cardiac diagnosis is still flagged",
          any("hallucinated" in f.lower() for f in validation["quality_flags"]),
          validation["quality_flags"])
    validation_charted = run_full_validation(
        ["I20.9 - Angina pectoris"], {"primary_icd10": "", "secondary_icd10": []},
        DIABETES_CHART + "\nConditions: Angina pectoris.", COMPLAINT,
    )
    check("I20 with a charted cardiac diagnosis is still allowed",
          not any("hallucinated" in f.lower() for f in validation_charted["quality_flags"]),
          validation_charted["quality_flags"])
    check("I21 with a charted cardiac diagnosis is still allowed",
          not any("hallucinated" in f.lower() for f in run_full_validation(
              ["I21.4 - Non-ST elevation myocardial infarction"],
              {"primary_icd10": "", "secondary_icd10": []},
              DIABETES_CHART + "\nConditions: Angina pectoris.", COMPLAINT,
          )["quality_flags"]))
    check("R07.9 remains the symptom code",
          not any("hallucinated" in f.lower() for f in run_full_validation(
              ["R07.9 - Chest pain, unspecified"], {"primary_icd10": "", "secondary_icd10": []},
              ctx, COMPLAINT)["quality_flags"]))

    # Chief-complaint-only and General paths inject nothing new.
    from agents.visit_context import normalize_visit_context

    check("empty visit context renders no block", visit_context_to_prompt_block({}) == "")
    check("chief-complaint-only cardiology has no cardiac block",
          "[SUBJECTIVE]" not in visit_context_to_prompt_block(normalize_visit_context({})))
    check("general profile unchanged in length",
          len(get_specialty_profile("general").prompt_emphasis) == 359)

    # P0 fixes.
    check("P0: get_llm(model_name=...) works",
          get_llm(model_name="gemma3:4b").model_name == "gemma3:4b")
    check("P0: get_llm(mode=...) still works", get_llm(mode="quality").model_name == MODEL_PRESETS["quality"])
    facts = {"conditions": ["Type 2 Diabetes"] * 8, "medications": ["Metformin"] * 8,
             "allergies": ["Penicillin", "Sulfa", "Latex", "Codeine"]}
    check("P0: grounding RULES line still survives truncation",
          RULES_LINE in build_grounding_block(
              facts, "PATIENT HISTORY (reference only):\nRecent visits:\n  - old"))

    import importlib

    from agents.coding_agent import CodingResult

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    try:
        coding_node.generate_coding = lambda **kwargs: CodingResult(
            primary_icd10="E11.9999 - bogus",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[], medical_necessity_note="Diabetes follow-up.",
        )
        update = coding_node.coding_node({
            "patient_context": DIABETES_CHART,
            "chief_complaint": COMPLAINT,
            "cleaned_soap_note": {"subjective": "s", "objective": "o",
                                  "assessment": "a", "plan": "p",
                                  "icd10_suggestions": []},
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("P0: malformed ICD-10 is still dropped",
          update["coding_result"]["primary_icd10"] == "",
          update["coding_result"]["primary_icd10"])
    check("P0: the malformed drop is still flagged",
          any("dropped malformed" in f for f in (update.get("quality_flags") or [])))

    # Memory repository interface: P2 added `status` (default: draft) and
    # `approve_visit`; the stored row shape is otherwise unchanged.
    import inspect

    from memory.patient_memory import (
        VISIT_STATUS_DRAFT,
        JsonPatientMemoryRepository,
    )

    signature = inspect.signature(JsonPatientMemoryRepository.record_visit)
    params = signature.parameters
    check("P0: record_visit signature: only `status` was added",
          list(params) == ["self", "patient_id", "patient_context", "chief_complaint",
                           "coding_result", "soap_note", "quality_flags", "status"], list(params))
    check("P0: the default is still a draft",
          params["status"].default == VISIT_STATUS_DRAFT, params["status"].default)

    # Deterministic CPT untouched.
    from tools.em_calculator import calculate_em_code

    em = calculate_em_code("moderate", "low", "low")
    check("P0: deterministic CPT calculator unchanged",
          em["valid"] and em["cpt_code"] == "99213", em)


def main():
    print("=" * 70)
    print("PHASE A2.1 CARDIOLOGY NARRATIVE TESTS (no LLM required)")
    print("=" * 70)
    test_claim_detector()
    test_cleanup_scrubs_claims()
    test_cleanup_flags_are_advisory()
    test_section_scoped_fidelity()
    test_overlap_calibration()
    test_coverage_guard_claim_backstop()
    test_coding_necessity_scrub()
    test_prompt_rules()
    test_compatibility_and_p0()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())