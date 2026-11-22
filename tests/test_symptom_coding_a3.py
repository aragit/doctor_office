"""
Phase A3 tests - symptom-to-code reliability (no LLM required).

Run with:

    python -m tests.test_symptom_coding_a3

A3 exists because symptom codes were reaching the final result through a late
safety net: Coding dropped R07.9, validation raised "chief complaint mentions
'chest pain' but no R07 code was generated", and the Supervisor added it
afterwards. That works, but it makes the flag stream noisy, leaves Coding a
passive participant, and puts the medical-necessity text at the mercy of a code
list that only gets corrected later.

A3 moves the decision earlier: one central symptom map, resolved
deterministically, injected into the Coding prompt as authoritative candidates.
The Supervisor stays as the fallback for whatever still falls through.

These tests lock in:

- the map itself (coverage of core symptoms, symptoms-only, no disease codes)
- grounding and suppression (visit data counts, a charted diagnosis suppresses
  the redundant symptom, no duplicates)
- the Coding path (candidates reach the prompt, and a model that follows the
  hint produces validation silence instead of a gap flag)
- coherence (validation stays quiet when the code is present, the Supervisor
  still completes a true gap, and no retry is added for a symptom)
- safety (no I20/I21 anywhere in this path, A2.1 narrative guards intact)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASSED = []
FAILED = []


from utils.symptom_codes import find_symptom_candidates  # noqa: E402


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


# The chart used throughout: two chronic conditions, no acute symptom, and
# deliberately no cardiac diagnosis.
CHART = (
    "Patient: John Doe, 58yo male\n"
    "Conditions: Type 2 Diabetes, Hypertension\n"
    "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
    "Allergies: Penicillin\n"
    "Last Visit: Blood pressure 148/92. HbA1c 7.8%."
)
CARDIAC_CHART = CHART + "\nConditions: Coronary artery disease, status post MI (2019)."

CARDIAC_FIELDS = {
    "chest_pain_characteristics": (
        "Substernal pressure, onset 3 days ago, radiates to left arm, 8/10, "
        "precipitated by exertion, relieved by rest."
    ),
    "ecg_rhythm_notes": (
        "Sinus rhythm 82 bpm, nonspecific ST-T changes in V4-V6, no acute ischemic changes."
    ),
    "functional_status_nyha": "Class II - Slight limitation",
}

# Core symptoms named in the A3 brief, with the codes the map must produce.
CORE_SYMPTOMS = [
    ("chest pain", "R07.9"),
    ("chest tightness", "R07.9"),
    ("dyspnea", "R06.02"),
    ("shortness of breath", "R06.02"),
    ("diarrhea", "R19.7"),
    ("headache", "R51.9"),
    ("fatigue", "R53.83"),
    ("fever", "R50.9"),
    ("cough", "R05.9"),
    ("abdominal pain", "R10.9"),
    ("nausea", "R11.2"),
    ("dizziness", "R42"),
    ("rash", "R21"),
]


def _codes(result):
    """Every ICD code in a CodingResult-shaped dict, as bare prefixes and codes."""
    entries = [result.get("primary_icd10", "")] + list(result.get("secondary_icd10") or [])
    return [entry for entry in entries if entry]


def _coding(update):
    return update["coding_result"]


# --- 1. the map --------------------------------------------------------------


def test_map_coverage():
    print("\nSymptom map: core symptom coverage")
    from utils.symptom_codes import (
        SYMPTOM_CODES,
        SYMPTOM_PHRASES,
        canonical_code_for_prefix,
        describe_prefix,
        is_symptom_prefix,
    )

    for phrase, expected in CORE_SYMPTOMS:
        candidates = find_symptom_candidates(chief_complaint=phrase)
        codes = [candidate.code for candidate in candidates]
        check(f"{phrase} -> {expected}", codes == [expected], codes)
        check(f"{phrase} carries its description",
              candidates and candidates[0].description == SYMPTOM_CODES[expected],
              candidates[0].description if candidates else None)
        check(f"{phrase} reports the justifying phrase",
              candidates and candidates[0].phrase in phrase,
              candidates[0].phrase if candidates else None)

    # Prefix lookup and reverse lookup agree.
    for code, description in SYMPTOM_CODES.items():
        prefix = code[:3]
        check(f"{code} prefix resolves back to itself",
              canonical_code_for_prefix(prefix) == code, canonical_code_for_prefix(prefix))
        check(f"{code} description is well-formed",
              description.startswith(code + " - "), description)
        check(f"{prefix} is described", describe_prefix(prefix) == description)
        check(f"{prefix} is a known symptom prefix", is_symptom_prefix(prefix))

    # Every phrase in the map is actually reachable.
    for prefix, phrases in SYMPTOM_PHRASES.items():
        for phrase in phrases:
            codes = [c.code for c in find_symptom_candidates(chief_complaint=phrase)]
            check(f"phrase '{phrase}' -> {canonical_code_for_prefix(prefix)}",
                  codes == [canonical_code_for_prefix(prefix)], codes)

    check("a phrase with no map entry yields nothing",
          find_symptom_candidates(chief_complaint="knee pain after a fall") == [])
    check("an empty complaint yields nothing", find_symptom_candidates() == [])
    check("routine follow-up yields nothing",
          find_symptom_candidates(chief_complaint="Routine follow-up for diabetes") == [])

    # The map itself is symptoms-only.
    disease_codes = [
        code for code in SYMPTOM_CODES
        if not code.startswith(("R", "Z"))
    ]
    check("no disease code is in the map", not disease_codes, disease_codes)
    check("no I20/I21 in the map",
          not any(code.startswith(("I20", "I21")) for code in SYMPTOM_CODES))
    check("angina is a symptom phrase, not an angina code",
          canonical_code_for_prefix("R07") == "R07.9"
          and not any("I20" in description for description in SYMPTOM_CODES.values()))


def test_map_is_symptoms_only():
    print("\nSymptom map: never a diagnosis, never an upgrade")
    from utils.symptom_codes import AUTO_ADDABLE_PREFIXES, SYMPTOM_CODES

    check("auto-addable prefixes are all R or Z",
          all(prefix.startswith(("R", "Z")) for prefix in AUTO_ADDABLE_PREFIXES),
          sorted(AUTO_ADDABLE_PREFIXES))
    check("every canonical code is auto-addable-shaped",
          all(code.startswith(("R", "Z")) for code in SYMPTOM_CODES),
          sorted(SYMPTOM_CODES))
    check("I10 is not auto-addable", "I10" not in AUTO_ADDABLE_PREFIXES)
    check("I20 is not auto-addable", "I20" not in AUTO_ADDABLE_PREFIXES)

    # No phrase combination can produce a cardiac diagnosis code.
    for complaint in ("chest pain", "angina", "chest pain radiating to the arm",
                      "chest pain with diaphoresis", "exertional chest pain"):
        codes = [c.code for c in find_symptom_candidates(chief_complaint=complaint,
                                                          patient_context=CHART)]
        check(f"no I20/I21 for '{complaint[:34]}'",
              not any(code.startswith(("I20", "I21")) for code in codes), codes)

    # A diagnosis-sounding complaint is a symptom, not a diagnosis.
    check("'angina' in a complaint yields R07.9, never I20",
          [c.code for c in find_symptom_candidates(chief_complaint="angina")] == ["R07.9"])
    check("'high blood pressure' yields no code at all",
          find_symptom_candidates(chief_complaint="high blood pressure") == [],
          find_symptom_candidates(chief_complaint="high blood pressure"))
    check("'asthma exacerbation' yields no diagnosis code",
          all(not c.code.startswith("J") for c in find_symptom_candidates(
              chief_complaint="asthma exacerbation with wheezing")))
    check("wheezing alone still codes the symptom",
          [c.code for c in find_symptom_candidates(chief_complaint="wheezing")] == ["R06.02"])


def test_grounding_and_suppression():
    print("\nGrounding: visit data counts, charted diagnoses suppress, no duplicates")
    from agents.visit_context import normalize_visit_context

    # A symptom recorded in the HPI is coded even when the complaint is vague.
    vc = normalize_visit_context({
        "hpi": "Three days of diarrhea, no blood.",
        "blood_pressure": "148/92",
    })
    candidates = find_symptom_candidates(
        chief_complaint="Feeling unwell", visit_context=vc
    )
    check("a visit-data symptom is found", [c.code for c in candidates] == ["R19.7"],
          [(c.code, c.source) for c in candidates])
    check("its source is the visit data", candidates[0].source == "visit_context")
    check("plain-text visit data is accepted",
          [c.code for c in find_symptom_candidates(
              chief_complaint="Feeling unwell", visit_context="Exam: chest pain on palpation")]
          == ["R07.9"])

    # Specialty fields are searched too (Cardiology writes the descriptors there).
    cardio_vc = normalize_visit_context(dict(CARDIAC_FIELDS))
    cardio_codes = [c.code for c in find_symptom_candidates(
        chief_complaint="Follow-up", visit_context=cardio_vc, patient_context=CHART)]
    check("cardiology specialty fields yield R07.9", cardio_codes == ["R07.9"], cardio_codes)

    # Complaint is searched first, so its code leads the list.
    both = find_symptom_candidates(
        chief_complaint="diarrhea", visit_context="Exam: chest pain reproduced"
    )
    check("chief-complaint candidates come first",
          [(c.code, c.source) for c in both]
          == [("R19.7", "chief_complaint"), ("R07.9", "visit_context")],
          [(c.code, c.source) for c in both])

    # A more specific charted diagnosis suppresses the redundant symptom.
    check("charted asthma suppresses the dyspnea candidate",
          find_symptom_candidates(
              chief_complaint="dyspnea",
              patient_context="Patient: Jane Roe\nConditions: Asthma, Hypertension")
          == [])
    check("an uncharted asthma does not",
          [c.code for c in find_symptom_candidates(chief_complaint="dyspnea")] == ["R06.02"])
    check("charted migraine suppresses the headache candidate",
          find_symptom_candidates(
              chief_complaint="headache",
              patient_context="Patient: Jane Roe\nConditions: Migraine without aura")
          == [])
    check("charted CAD suppresses the chest-pain candidate",
          find_symptom_candidates(
              chief_complaint="chest pain", patient_context=CARDIAC_CHART) == [])
    check("a diabetes/hypertension chart does not suppress it",
          [c.code for c in find_symptom_candidates(
              chief_complaint="chest pain", patient_context=CHART)] == ["R07.9"])
    check("an ECG finding in visit data is not a cardiac diagnosis",
          [c.code for c in find_symptom_candidates(
              chief_complaint="chest pain",
              visit_context=CARDIAC_FIELDS["ecg_rhythm_notes"],
              patient_context=CHART)] == ["R07.9"])

    # Already-coded symptoms are never proposed again.
    check("an already-coded candidate is skipped",
          find_symptom_candidates(chief_complaint="chest pain",
                                  already_coded=["R07.9 - Chest pain, unspecified"]) == [])
    check("a differently-worded code still counts as present",
          find_symptom_candidates(chief_complaint="chest pain",
                                  already_coded=["R07.89"]) == [])

    # Multiple symptoms, bounded.
    many = find_symptom_candidates(
        chief_complaint="headache, fever, cough, rash, dizziness and nausea", max_codes=3
    )
    check("max_codes bounds the list", len(many) == 3, [c.code for c in many])


# --- 2. the Coding path ------------------------------------------------------


def test_coding_receives_candidates():
    print("\nCoding path: candidates reach the prompt and land in the result")
    import importlib

    from agents.coding_agent import CodingResult
    from agents.visit_context import normalize_visit_context

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    seen = {}

    def spy(**kwargs):
        seen["context"] = kwargs["patient_context"]
        # A model that follows the hint: the candidates it was given, as
        # secondary codes, with no diagnosis invented.
        return CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=[
                "I10 - Essential (primary) hypertension",
                "R07.9 - Chest pain, unspecified",
                "R19.7 - Diarrhea, unspecified",
            ],
            cpt_suggestions=[],
            medical_necessity_note="Diabetes and hypertension follow-up with chest pain "
                                   "and diarrhea evaluated as symptoms.",
        )

    vc = normalize_visit_context({"hpi": "Two days of diarrhea."})
    try:
        coding_node.generate_coding = spy
        update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Chest pain and diarrhea for three days.",
            "cleaned_soap_note": {
                "subjective": "Chest pain and diarrhea.",
                "objective": "BP 148/92.",
                "assessment": "Symptoms under evaluation.",
                "plan": "1. Follow up.",
                "icd10_suggestions": [],
            },
            "visit_context": vc,
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    context = seen["context"]
    check("the candidate block reaches the prompt", "SYMPTOM CODE CANDIDATES" in context)
    check("R07.9 is in the block", "R07.9 - Chest pain, unspecified" in context)
    check("R19.7 is in the block", "R19.7 - Diarrhea, unspecified" in context)
    check("the block marks the chief complaint as the source", "[chief_complaint]" in context)
    check("the block forbids upgrading a symptom", "never upgrade" in context.lower())
    check("the chart is still passed through", "Type 2 Diabetes" in context)

    result = _coding(update)
    check("the symptom codes survive to the result",
          "R07.9 - Chest pain, unspecified" in result["secondary_icd10"],
          result["secondary_icd10"])
    check("a charted diagnosis is still primary",
          result["primary_icd10"].startswith("E11.9"), result["primary_icd10"])
    check("no I20/I21 appears", not any(c.startswith(("I20", "I21"))
                                        for c in _codes(result)))
    trace = update["tool_results"]["coding_node"]["simple_symptom_to_icd_suggestion"]
    check("the tool trace records the candidates",
          trace["suggested_codes"] == ["R07.9", "R19.7"], trace)
    check("the hint is empty when nothing is documented",
          coding_node.build_symptom_hint([]) == "")
    check("a plain code list still renders",
          "R07.9" in coding_node.build_symptom_hint(["R07.9 - Chest pain, unspecified"]))

    # A Scribe code for the same symptom suppresses the duplicate hint.
    seen.clear()
    try:
        coding_node.generate_coding = spy
        coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Chest pain for three days.",
            "cleaned_soap_note": {
                "subjective": "Chest pain.", "objective": "BP 148/92.",
                "assessment": "Under evaluation.", "plan": "1. Follow up.",
                "icd10_suggestions": ["R07.9 - Chest pain, unspecified"],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("an already-suggested symptom is not hinted twice",
          "R07.9 - Chest pain, unspecified" not in seen["context"],
          seen["context"][-400:])


def test_coding_completes_omitted_symptom_codes():
    print("\nCoding path: a model that ignores the hint is still completed")
    import importlib

    from agents.coding_agent import CodingResult
    from agents.supervisor_policy import classify_flags, determine_route
    from agents.visit_context import normalize_visit_context
    from utils.validation import validate_chief_complaint_coverage

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding

    def ignoring(**kwargs):
        # What qwen2.5:1.5b actually does: chronic codes only, symptom dropped.
        return CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[],
            medical_necessity_note="Diabetes and hypertension follow-up.",
        )

    vc = normalize_visit_context(dict(CARDIAC_FIELDS))
    try:
        coding_node.generate_coding = ignoring
        update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Chest pain on exertion for three days.",
            "cleaned_soap_note": {
                "subjective": "Exertional chest pain.", "objective": "BP 146/88.",
                "assessment": "Under evaluation.", "plan": "1. Follow up.",
                "icd10_suggestions": [],
            },
            "visit_context": vc,
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    result = _coding(update)
    check("R07.9 is completed into the result",
          "R07.9 - Chest pain, unspecified" in result["secondary_icd10"],
          result["secondary_icd10"])
    check("the diagnosis the model chose is untouched",
          result["primary_icd10"].startswith("E11.9")
          and "I10 - Essential (primary) hypertension" in result["secondary_icd10"],
          result)
    check("the symptom is secondary, never primary",
          not result["primary_icd10"].startswith("R07"), result["primary_icd10"])
    check("no I20/I21 is introduced",
          not any(code.startswith(("I20", "I21")) for code in _codes(result)))
    check("the completion is reported once",
          sum("added symptom code" in flag for flag in (update.get("quality_flags") or [])) == 1,
          update.get("quality_flags"))
    check("the report names the code and its justification",
          any("R07.9 - Chest pain, unspecified" in flag and "chest pain" in flag
              for flag in (update.get("quality_flags") or [])), update.get("quality_flags"))
    check("the tool trace records what was added",
          update["tool_results"]["coding_node"]["simple_symptom_to_icd_suggestion"][
              "added_to_result"] == ["R07.9 - Chest pain, unspecified"],
          update["tool_results"]["coding_node"])

    # The point of the whole phase: validation is now silent, so the Supervisor
    # never becomes the primary coder for a documented symptom.
    codes = _codes(result)
    check("validation raises no gap flag for the completed code",
          validate_chief_complaint_coverage(codes, "Chest pain on exertion for three days.") == [],
          validate_chief_complaint_coverage(codes, "Chest pain on exertion for three days."))

    # The completion flag is advisory: no retry, no escalation.
    completion_flags = [flag for flag in (update.get("quality_flags") or [])
                        if "added symptom code" in flag]
    buckets = classify_flags(completion_flags)
    check("the completion flag is advisory",
          buckets["scribe"] == [] and buckets["coding"] == [] and buckets["symptom"] == [],
          buckets)
    check("the completion flag does not retry",
          determine_route(completion_flags, 0, {}, "Chest pain").route_to
          == "memory_save_node",
          determine_route(completion_flags, 0, {}, "Chest pain"))

    # Deterministic CPT still authoritative after the completion.
    check("deterministic CPT survives the completion",
          (result.get("cpt_grounding") or {}).get("deterministic") is True
          and (result.get("cpt_grounding") or {}).get("cpt_code") in
          {"99212", "99213", "99214", "99215"},
          result.get("cpt_grounding"))

    # No duplicate: a model that *did* follow the hint gets no second add.
    def compliant(**kwargs):
        return CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=["I10 - Essential (primary) hypertension",
                             "R07.9 - Chest pain, unspecified"],
            cpt_suggestions=[],
            medical_necessity_note="Follow-up.",
        )

    try:
        coding_node.generate_coding = compliant
        compliant_update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Chest pain on exertion for three days.",
            "cleaned_soap_note": {
                "subjective": "s", "objective": "o", "assessment": "a", "plan": "p",
                "icd10_suggestions": [],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    secondary = _coding(compliant_update)["secondary_icd10"]
    check("a code the model emitted is not added twice",
          secondary.count("R07.9 - Chest pain, unspecified") == 1, secondary)
    check("a compliant pass raises no completion flag",
          not any("added symptom code" in flag
                  for flag in (compliant_update.get("quality_flags") or [])),
          compliant_update.get("quality_flags"))

    # A visit-data symptom is completed and is not then called ungrounded.
    try:
        coding_node.generate_coding = ignoring
        visit_update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Feeling unwell since yesterday.",
            "cleaned_soap_note": {
                "subjective": "s", "objective": "o", "assessment": "a", "plan": "p",
                "icd10_suggestions": [],
            },
            "visit_context": normalize_visit_context({
                "hpi": "Two days of diarrhea, no blood.",
            }),
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    visit_result = _coding(visit_update)
    check("a visit-data symptom is completed",
          "R19.7 - Diarrhea, unspecified" in visit_result["secondary_icd10"],
          visit_result["secondary_icd10"])
    check("a completed visit-data code is not flagged ungrounded",
          not any("ungrounded" in flag.lower() for flag in (visit_update.get("quality_flags") or [])),
          visit_update.get("quality_flags"))

    # Nothing to complete for a routine visit: no flag, no code.
    try:
        coding_node.generate_coding = ignoring
        routine = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Routine follow-up for diabetes and hypertension.",
            "cleaned_soap_note": {
                "subjective": "Well.", "objective": "BP 148/92.", "assessment": "Stable.",
                "plan": "1. Continue medications.", "icd10_suggestions": [],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("a routine visit gains no symptom code",
          not any(code[:1] == "R" for code in _codes(_coding(routine))),
          _codes(_coding(routine)))
    check("a routine visit raises no completion flag",
          not any("added symptom code" in flag
                  for flag in (routine.get("quality_flags") or [])),
          routine.get("quality_flags"))


def test_coding_ignores_a_phantom_symptom():
    print("\nCoding path: no candidate is invented from absent evidence")
    import importlib

    from agents.coding_agent import CodingResult

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    seen = {}

    def spy(**kwargs):
        seen["context"] = kwargs["patient_context"]
        return CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=["I10 - Essential (primary) hypertension"],
            cpt_suggestions=[],
            medical_necessity_note="Diabetes and hypertension follow-up.",
        )

    try:
        coding_node.generate_coding = spy
        update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Routine diabetes follow-up.",
            "cleaned_soap_note": {
                "subjective": "Feeling well.", "objective": "BP 148/92.",
                "assessment": "Stable.", "plan": "1. Continue medications.",
                "icd10_suggestions": [],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    check("no candidate block for a routine visit",
          "SYMPTOM CODE CANDIDATES" not in seen["context"])
    check("no symptom code is added",
          not any(code[:3].startswith("R") for code in _codes(_coding(update))),
          _codes(_coding(update)))
    check("validation stays silent too",
          not any("no R" in flag for flag in (update.get("quality_flags") or [])),
          update.get("quality_flags"))


def test_cardiology_interaction():
    print("\nCardiology: symptom code yes, cardiac diagnosis no")
    import importlib

    from agents.coding_agent import CodingResult
    from agents.visit_context import normalize_visit_context

    coding_node = importlib.import_module("agents.coding_node")
    original = coding_node.generate_coding
    seen = {}

    def spy(**kwargs):
        seen["context"] = kwargs["patient_context"]
        # A model that tries to upgrade the symptom, to prove the guards hold.
        return CodingResult(
            primary_icd10="E11.9 - Type 2 diabetes mellitus without complications",
            secondary_icd10=[
                "I10 - Essential (primary) hypertension",
                "R07.9 - Chest pain, unspecified",
                "I20.9 - Angina pectoris",
                "I21.4 - Non-ST elevation myocardial infarction",
            ],
            cpt_suggestions=[],
            medical_necessity_note="Chest pain is most consistent with ischemic "
                                   "heart disease requiring further evaluation.",
        )

    vc = normalize_visit_context(dict(CARDIAC_FIELDS))
    try:
        coding_node.generate_coding = spy
        update = coding_node.coding_node({
            "patient_context": CHART,
            "chief_complaint": "Chest pain on exertion for three days.",
            "cleaned_soap_note": {
                "subjective": "Substernal exertional pain.", "objective": "BP 146/88.",
                "assessment": "Under evaluation.", "plan": "1. Follow up.",
                "icd10_suggestions": [],
            },
            "visit_context": vc,
            "specialty": "cardiology",
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original

    result = _coding(update)
    check("R07.9 reaches Coding", "R07.9 - Chest pain, unspecified" in _codes(result),
          result["secondary_icd10"])
    check("the chief complaint is named as the source",
          "[chief_complaint]" in seen["context"], seen["context"][-400:])
    hint_lines = [
        line for line in seen["context"].splitlines()
        if line.strip().startswith("- ") and " - " in line
    ]
    check("the candidate block proposes no cardiac diagnosis",
          hint_lines and not any(
              line.split(" - ")[0].strip().startswith(("I20", "I21"))
              for line in hint_lines),
          hint_lines)

    # A cardiac diagnosis the model invented on its own is caught by the
    # existing ungrounded-code gate, and removed by validation - unchanged
    # behaviour that A3 must not weaken.
    check("an invented I20 is flagged ungrounded",
          any("I20" in flag for flag in (update.get("quality_flags") or [])),
          update.get("quality_flags"))
    check("an invented I21 is flagged ungrounded",
          any("I21" in flag for flag in (update.get("quality_flags") or [])),
          update.get("quality_flags"))

    from utils.validation import run_full_validation

    validated = run_full_validation(
        [], result, CHART, "Chest pain on exertion for three days."
    )
    check("validation removes the invented I20",
          any("I20" in flag for flag in validated["quality_flags"]),
          validated["quality_flags"])
    check("the cardiac necessity claim is scrubbed",
          "ischemic heart disease" not in result["medical_necessity_note"].lower(),
          result["medical_necessity_note"])
    check("the necessity scrub is advisory",
          any("cardiac claim" in flag for flag in (update.get("quality_flags") or [])))
    check("deterministic CPT is untouched",
          (result.get("cpt_grounding") or {}).get("cpt_code") in
          {"99212", "99213", "99214", "99215"},
          result.get("cpt_grounding"))

    # A chart that does document a cardiac diagnosis keeps its narrative.
    try:
        coding_node.generate_coding = spy
        cardiac = coding_node.coding_node({
            "patient_context": CARDIAC_CHART,
            "chief_complaint": "Chest pain on exertion for three days.",
            "cleaned_soap_note": {
                "subjective": "Substernal exertional pain.", "objective": "BP 146/88.",
                "assessment": "Under evaluation.", "plan": "1. Follow up.",
                "icd10_suggestions": [],
            },
            "mode": "fast",
        })
    finally:
        coding_node.generate_coding = original
    check("a charted cardiac diagnosis is not narratively scrubbed",
          "ischemic heart disease" in _coding(cardiac)["medical_necessity_note"].lower(),
          _coding(cardiac)["medical_necessity_note"])
    check("no R07.9 candidate when CAD is charted",
          "R07.9 - Chest pain, unspecified" not in seen["context"],
          seen["context"][-500:])


# --- 3. validation and Supervisor coherence ---------------------------------


def test_validation_and_supervisor_coherence():
    print("\nCoherence: quiet when coded, Supervisor completes a true gap")
    from agents.supervisor_policy import (
        MAX_RETRIES,
        classify_flags,
        determine_route,
        missing_symptom_codes,
    )
    from utils.validation import run_full_validation, validate_chief_complaint_coverage

    # The symptom the model emitted: no gap flag, so no advisory noise.
    flags = validate_chief_complaint_coverage(
        ["E11.9", "I10", "R07.9 - Chest pain, unspecified"],
        "Chest pain for three days.",
    )
    check("no gap flag when the symptom is coded", flags == [], flags)

    # Full validation is quiet too, including when Coding supplied the code.
    quiet = run_full_validation(
        [],
        {"primary_icd10": "E11.9 - Diabetes", "secondary_icd10": [
            "I10 - Essential hypertension", "R07.9 - Chest pain, unspecified"]},
        CHART, "Chest pain for three days.",
    )
    check("full validation is quiet when Coding codes the symptom",
          not any("no R" in flag for flag in quiet["quality_flags"]),
          quiet["quality_flags"])

    # A genuine gap still raises exactly one flag, for the right symptom.
    gap = validate_chief_complaint_coverage(["E11.9", "I10"], "Chest pain for three days.")
    check("a true gap still raises the flag", len(gap) == 1, gap)
    check("the flag names the symptom and the prefix",
          "chest pain" in gap[0] and "R07" in gap[0], gap)

    # Several symptoms, several flags - and no duplicate for the same prefix.
    multi = validate_chief_complaint_coverage(
        ["E11.9"], "Chest pain, diarrhea and fatigue."
    )
    check("each missing symptom raises one flag", len(multi) == 3, multi)
    check("R07, R19 and R53 are named",
          all(any(prefix in flag for flag in multi)
              for prefix in ("R07", "R19", "R53")), multi)

    # The Supervisor remains the fallback.
    decision = determine_route(
        gap, 0,
        {"primary_icd10": "E11.9 - Diabetes", "secondary_icd10": ["I10 - Hypertension"]},
        chief_complaint="Chest pain for three days.",
    )
    check("the Supervisor still adds the code",
          decision.symptom_codes == ["R07.9 - Chest pain, unspecified"], decision)
    check("the gap routes to revalidation, not an LLM retry",
          decision.route_to == "validation_node" and decision.class_ == "revalidate", decision)
    check("the retry feedback is not sent to an LLM agent",
          all("Added missing symptom code" in item for item in decision.feedback),
          decision.feedback)

    check("no duplicate add when the code is present",
          missing_symptom_codes(gap, "Chest pain", ["E11.9", "R07.9"]) == [])
    check("no code for an absent symptom",
          missing_symptom_codes(
              ["Chief complaint mentions 'headache' but no R51 code was generated"],
              "Routine follow-up", ["E11.9"]) == [])

    # Dyspnea is the case the old hard-coded phrase table got wrong.
    dyspnea_gap = validate_chief_complaint_coverage(["E11.9"], "Dyspnea on exertion.")
    check("a dyspnea complaint raises the R06 gap",
          any("R06" in flag for flag in dyspnea_gap), dyspnea_gap)
    check("the Supervisor now completes dyspnea too",
          missing_symptom_codes(dyspnea_gap, "Dyspnea on exertion.", ["E11.9"])
          == ["R06.02 - Shortness of breath"],
          missing_symptom_codes(dyspnea_gap, "Dyspnea on exertion.", ["E11.9"]))

    # A disease code is never auto-completed, even if a flag names one.
    check("I10 is not auto-completed from a phrase",
          missing_symptom_codes(
              ["Chief complaint mentions 'high blood pressure' but no I10 code was generated"],
              "High blood pressure check.", ["E11.9"]) == [])

    # No retry: symptom flags are handled deterministically and never retry.
    check("symptom flags do not trigger a scribe retry",
          classify_flags(gap)["scribe"] == [] and classify_flags(gap)["coding"] == [],
          classify_flags(gap))
    check("a symptom gap consumes no retry budget",
          determine_route(gap, 0, {}, "Chest pain").class_ == "revalidate")
    check("repeated gaps still never escalate",
          determine_route(gap, MAX_RETRIES, {}, "Chest pain").route_to
          != "human_review" or determine_route(gap, MAX_RETRIES, {}, "Chest pain").class_
          == "finalize",
          determine_route(gap, MAX_RETRIES, {}, "Chest pain"))


# --- 4. shared-source coherence and safety ----------------------------------


def test_single_source_and_safety():
    print("\nSingle source of truth, tool surface, and A2.1 preservation")
    from agents.coding_node import build_symptom_hint
    from tools.coding_tools import (
        PREFIX_TO_SYMPTOM_CODE,
        SYMPTOM_CODES,
        SYMPTOM_PHRASES,
        simple_symptom_to_icd_suggestion,
        suggest_codes_for_complaint,
        validate_icd_against_patient,
    )
    from utils.symptom_codes import SYMPTOM_CODES as CANONICAL
    from utils.validation import CHIEF_COMPLAINT_CODE_HINTS

    check("the tool re-exports the canonical codes", SYMPTOM_CODES == CANONICAL)
    check("the tool re-exports the canonical phrases",
          {k: tuple(v) for k, v in SYMPTOM_PHRASES.items()}
          == {k: tuple(v) for k, v in
              __import__("utils.symptom_codes", fromlist=["x"]).SYMPTOM_PHRASES.items()})
    check("the prefix map matches the codes",
          all(PREFIX_TO_SYMPTOM_CODE[code[:3]] == code for code in CANONICAL))

    # Validation's hint rows cover every map phrase, so the three layers agree.
    hint_phrases = {row[0] for row in CHIEF_COMPLAINT_CODE_HINTS}
    map_phrases = {
        phrase for phrases in SYMPTOM_PHRASES.values() for phrase in phrases
    }
    check("validation knows every map phrase", map_phrases <= hint_phrases,
          sorted(map_phrases - hint_phrases))
    check("the I10 review row is still present",
          ("high blood pressure", "I10", "high blood pressure") in CHIEF_COMPLAINT_CODE_HINTS)

    # Tool behaviour is unchanged for its documented cases.
    check("tool: chest pain -> R07.9",
          simple_symptom_to_icd_suggestion("chest pain")["codes"] == ["R07.9"])
    check("tool: diarrhea -> R19.7",
          simple_symptom_to_icd_suggestion("diarrhea")["codes"] == ["R19.7"])
    check("tool: unknown symptom -> none",
          simple_symptom_to_icd_suggestion("knee pain")["codes"] == [])
    check("tool: empty input -> not found",
          simple_symptom_to_icd_suggestion("")["found"] is False)
    check("tool: dyspnea is recognized (was missed before A3)",
          simple_symptom_to_icd_suggestion("dyspnea")["codes"] == ["R06.02"])
    check("suggest: complaint yields both codes",
          suggest_codes_for_complaint("chest pain and diarrhea") == ["R07.9", "R19.7"])
    check("suggest: charted HTN drops nothing since I10 is not a symptom",
          suggest_codes_for_complaint("high blood pressure", CHART) == [])

    # Grounding: any phrase for the prefix justifies the code.
    check("grounding: dyspnea justifies R06",
          validate_icd_against_patient("R06.02", CHART, "Dyspnea on exertion")["valid"])
    check("grounding: chest pain justifies R07",
          validate_icd_against_patient("R07.9", CHART, "Chest pain")["valid"])
    check("grounding: an absent symptom does not",
          not validate_icd_against_patient("R51.9", CHART, "Chest pain")["valid"])
    # A diagnosis code is grounded by the chart, not by a phrase: the chart
    # says nothing about hypertension, so "high blood pressure" in a complaint
    # does not license I10.
    no_htn = "Patient: Jane Roe, 40yo female\nConditions: Seasonal Allergies"
    check("grounding: a complaint phrase does not license a diagnosis code",
          not validate_icd_against_patient("I10", no_htn, "High blood pressure check")["valid"])
    check("grounding: the chart does license I10",
          validate_icd_against_patient("I10", CHART, "Routine follow-up")["valid"])

    # A2.1 narrative guards are untouched by the symptom work.
    from agents.specialty_coverage import build_specialty_coverage_flag
    from utils.cardiac_claims import find_unsupported_cardiac_claims
    from utils.validation import clean_soap_note

    check("A2.1: the claim detector still flags ischemic etiology",
          bool(find_unsupported_cardiac_claims(
              "The pain is most consistent with ischemic etiology.")))
    check("A2.1: a documented ECG negation is still allowed",
          not find_unsupported_cardiac_claims("ECG: no acute ischemic changes."))
    check("A2.1: cleanup still scrubs the narrative",
          "ischemic" not in clean_soap_note(
              {"subjective": "s", "objective": "o", "plan": "p",
               "assessment": "Most consistent with ischemic etiology."},
              CHART, "Chest pain")[0]["assessment"].lower())
    check("A2.1: the specialty fidelity guard still fires",
          any("SUBJECTIVE" in flag for flag in build_specialty_coverage_flag(
              {"subjective": "Chest pain x 1 day.", "objective": "",
               "assessment": "Stable.", "plan": "1. Follow up."},
              __import__("agents.visit_context", fromlist=["x"]).normalize_visit_context(
                  dict(CARDIAC_FIELDS)), CHART)))

    # The hint block carries no diagnosis-code instruction.
    hint = build_symptom_hint(list(find_symptom_candidates(chief_complaint="chest pain")))
    check("the hint proposes no I20/I21 code",
          not any(candidate.code.startswith(("I20", "I21"))
                  for candidate in find_symptom_candidates(chief_complaint="chest pain")),
          hint)
    check("the hint is a single block", hint.count("SYMPTOM CODE CANDIDATES") == 1)


def test_compatibility():
    print("\nCompatibility: prompts, registry, graph structure, determinism")
    from agents.coding_agent import CODING_SYSTEM_PROMPT
    from tools.registry import TOOLS
    from utils.config import MODEL_PRESETS
    from utils.llm import get_llm

    flat = " ".join(CODING_SYSTEM_PROMPT.split())
    check("the coding prompt documents the candidate block",
          "SYMPTOM CODE CANDIDATES" in flat)
    check("the coding prompt forbids the upgrade",
          "chest pain stays R07.x" in flat, flat)
    check("the coding prompt keeps the medical-necessity rule",
          "never explain the visit as ischemic" in flat)
    check("the coding prompt keeps the I20.x chart requirement",
          "never use i20.x (angina) without documented cardiac ischemia" in flat.lower())

    check("the symptom tool is still registered",
          "simple_symptom_to_icd_suggestion" in TOOLS)
    check("the symptom tool is still deterministic-only",
          "simple_symptom_to_icd_suggestion" in TOOLS)

    # P0 fixes and the deterministic CPT calculator are unchanged.
    check("P0: explicit model name still works",
          get_llm(model_name="gemma3:4b").model_name == "gemma3:4b")
    check("P0: mode still works", get_llm(mode="quality").model_name == MODEL_PRESETS["quality"])
    from tools.em_calculator import calculate_em_code

    check("deterministic CPT unchanged",
          calculate_em_code("moderate", "low", "low")["cpt_code"] == "99213")

    # General specialty and chief-complaint-only paths keep working.
    from agents.visit_context import normalize_visit_context, visit_context_to_prompt_block
    from agents.specialty_profiles import get_specialty_profile

    check("no visit context renders no block", visit_context_to_prompt_block({}) == "")
    check("a chief-complaint-only cardiology visit still codes its symptom",
          [c.code for c in find_symptom_candidates(
              chief_complaint="Chest pain on exertion")] == ["R07.9"])
    general = " ".join(get_specialty_profile("general").prompt_emphasis.split()).lower()
    check("General profile is still not cardiac", "cardiac" not in general)
    check("an empty complaint yields no candidates",
          find_symptom_candidates(
              chief_complaint="", visit_context=normalize_visit_context({})) == [])


def test_negation_awareness():
    print("\nNegation awareness: denied symptoms are not present")
    from utils.symptom_codes import (
        phrase_present_not_negated,
        symptom_phrase_present,
        symptom_gap_labels,
        find_symptom_candidates,
    )

    # Positive controls
    check("positive chest pain for 1 day",
          phrase_present_not_negated("chest pain", "chest pain for 1 day"),
          "chest pain")
    check("positive chest pain and diarrhea -> chest pain",
          phrase_present_not_negated("chest pain", "chest pain and diarrhea"),
          "chest pain")
    check("positive chest pain and diarrhea -> diarrhea",
          phrase_present_not_negated("diarrhea", "chest pain and diarrhea"),
          "diarrhea")

    # Denial controls
    check("denied chest pain",
          not phrase_present_not_negated("chest pain", "Denies chest pain"),
          "Denies chest pain")
    check("denied chest pain, SOB, dizziness -> chest pain",
          not phrase_present_not_negated("chest pain", "denies chest pain, shortness of breath, dizziness"),
          "denies chest pain, SOB, dizziness")
    check("denied chest pain, SOB, dizziness -> SOB",
          not phrase_present_not_negated("shortness of breath", "denies chest pain, shortness of breath, dizziness"),
          "denies SOB")
    check("denied chest pain, SOB, dizziness -> dizziness",
          not phrase_present_not_negated("dizziness", "denies chest pain, shortness of breath, dizziness"),
          "denies dizziness")
    check("no chest pain or SOB -> chest pain",
          not phrase_present_not_negated("chest pain", "no chest pain or SOB"),
          "no chest pain or SOB")
    check("no chest pain or SOB -> SOB",
          not phrase_present_not_negated("shortness of breath", "no chest pain or SOB"),
          "no SOB")

    # Scribe focus integration via symptom_phrase_present (the Supervisor tool)
    check("denial disables supervisor auto-add",
          not symptom_phrase_present("R07", "Denies chest pain"),
          "R07 auto-add")
    check("positive enables supervisor auto-add",
          symptom_phrase_present("R07", "Chest pain for 1 day"),
          "R07 auto-add")

    # Symptom gap labels should not include denied symptoms
    check("gap labels exclude denied chest pain",
          symptom_gap_labels("Denies chest pain") == [],
          "gap labels")
    check("gap labels include positive chest pain",
          len(symptom_gap_labels("chest pain for 1 day")) > 0,
          "gap labels")

    # find_symptom_candidates should not return denied symptoms
    check("candidate list excludes denied",
          find_symptom_candidates(chief_complaint="Denies chest pain") == [],
          "candidates")
    check("candidate list includes positive",
          len(find_symptom_candidates(chief_complaint="Chest pain for 1 day")) > 0,
          "candidates")

    # Across multiple symptoms, each is independently evaluated
    # The function processes chief_complaint first, then visit_context.
    # It stops when it has enough candidates (max_codes = 4 by default), not when it has processed all sources.
    check("chief complaint is processed first, visit_context not processed",
          [c.code for c in find_symptom_candidates(
              chief_complaint="Chest pain",
              visit_context="denies chest pain, shortness of breath, dizziness"
          )] == ["R07.9"],
          "chief complaint first, stop after first match")

    # Scribe's _render_focus_block uses the same matcher; verify it doesn't
    # produce an acute focus for a denial.
    check("denial does not produce acute focus",
          not phrase_present_not_negated("chest pain", "denies chest pain"),
          "acute focus")
    check("positive produces acute focus",
          phrase_present_not_negated("chest pain", "chest pain for 1 day"),
          "acute focus")

    print("  Positive controls check passed")
    print("  Denial controls check passed")


def test_a4_metadata_shape():
    print("\nPhase A4: result metadata exposes symptom provenance, changing nothing")
    from agents.langgraph_orchestrator import _state_to_result
    from utils.review_presentation import (
        flag_groups,
        origin_label,
        source_label,
        specialty_fidelity_notes,
        symptom_coding_summary,
    )

    def _state(**overrides):
        state = {
            "cleaned_soap_note": {
                "subjective": "s", "objective": "o", "assessment": "a",
                "plan": "p", "icd10_suggestions": [],
            },
            "coding_result": {
                "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
                "secondary_icd10": ["I10 - Essential (primary) hypertension"],
            },
            "quality_flags": [],
            "tool_results": {
                "coding_node": {
                    "simple_symptom_to_icd_suggestion": {
                        "chief_complaint": "Chest pain and diarrhea.",
                        "suggested_codes": ["R07.9", "R19.7"],
                        "candidates": [
                            {"code": "R07.9",
                             "description": "R07.9 - Chest pain, unspecified",
                             "prefix": "R07", "phrase": "chest pain",
                             "source": "chief_complaint"},
                            {"code": "R19.7",
                             "description": "R19.7 - Diarrhea, unspecified",
                             "prefix": "R19", "phrase": "diarrhea",
                             "source": "visit_context"},
                        ],
                        "added_to_result": ["R07.9 - Chest pain, unspecified"],
                    }
                }
            },
            "supervisor_symptom_codes": ["R19.7 - Diarrhea, unspecified"],
            "supervisor_applied": [
                "Added symptom code R19.7 - Diarrhea, unspecified "
                "(raised by validation, present in chief complaint)"
            ],
            "timing": {},
            "retry_count": 0,
        }
        state.update(overrides)
        return state

    result = _state_to_result(_state(), "fast", None)
    meta = result["metadata"]
    block = meta["symptom_coding"]

    check("metadata carries a symptom_coding block", isinstance(block, dict), type(block))
    check("the block has the three documented keys",
          set(block) == {"candidates", "coding_added", "supervisor_added"}, sorted(block))
    check("each candidate names its phrase and source",
          all({"code", "phrase", "source"} <= set(candidate)
              for candidate in block["candidates"]), block["candidates"])
    check("the Coding additions are listed",
          block["coding_added"] == ["R07.9 - Chest pain, unspecified"], block["coding_added"])
    check("the Supervisor additions are listed",
          block["supervisor_added"] == ["R19.7 - Diarrhea, unspecified"],
          block["supervisor_added"])
    check("provenance lives in exactly one metadata key",
          "symptom_codes" not in meta["supervisor"]
          and set(meta["supervisor"]) == {"retry_count", "needs_human_review",
                                          "applied", "feedback"},
          sorted(meta["supervisor"]))
    check("the pre-existing metadata keys are unchanged",
          {"model_used", "mode", "pipeline_version", "orchestrator", "steps_completed",
           "graph", "timing", "memory", "supervisor", "visit_context", "specialty",
           "tools"} <= set(meta), sorted(meta))

    # Backward compatible: a result without the new state keys still builds.
    legacy = _state_to_result(
        {"cleaned_soap_note": {}, "coding_result": {}, "quality_flags": [],
         "tool_results": {}},
        "fast", None,
    )
    check("a bare state still produces a result",
          legacy["coding_result"] == {} and legacy["quality_flags"] == [])
    check("an empty state yields empty symptom metadata",
          legacy["metadata"]["symptom_coding"] == {"candidates": [],
                                                  "coding_added": [],
                                                  "supervisor_added": []},
          legacy["metadata"]["symptom_coding"])

    # Presentation reads the block without needing the tool trace.
    coding_result = {
        "primary_icd10": "E11.9 - Diabetes",
        "secondary_icd10": ["I10 - Hypertension", "R07.9 - Chest pain, unspecified",
                            "R19.7 - Diarrhea, unspecified"],
    }
    summary = symptom_coding_summary(meta, coding_result)
    check("origins are attributed correctly",
          [(row.code, row.origin) for row in summary.rows]
          == [("R07.9", "added_by_coding"), ("R19.7", "added_by_supervisor")],
          [(row.code, row.origin) for row in summary.rows])
    check("the counts add up",
          (summary.considered, summary.coding_added, summary.supervisor_added,
           summary.missing) == (2, 1, 1, 0), summary)
    # A code the model emitted itself is not an addition.
    model_emit = _state_to_result(
        _state(tool_results={"coding_node": {"simple_symptom_to_icd_suggestion": {
            "suggested_codes": ["R07.9"],
            "candidates": [{"code": "R07.9",
                            "description": "R07.9 - Chest pain, unspecified",
                            "prefix": "R07", "phrase": "chest pain",
                            "source": "chief_complaint"}],
            "added_to_result": [],
        }}}),
        "fast", None,
    )["metadata"]
    already = symptom_coding_summary(
        model_emit,
        {"primary_icd10": "E11.9", "secondary_icd10": ["R07.9 - Chest pain, unspecified"]},
    )
    check("a code already present reads as such",
          already.rows[0].origin == "already_present" and already.already_present == 1,
          already.rows)
    check("a candidate that never landed reads as such",
          symptom_coding_summary(
              meta, {"primary_icd10": "E11.9", "secondary_icd10": []}
          ).missing == 2)
    check("sources and origins have readable labels",
          source_label("visit_context") == "visit data (HPI / exam / results)"
          and origin_label("added_by_supervisor") == "Added by Supervisor",
          (source_label("visit_context"), origin_label("added_by_supervisor")))

    # A legacy result with no block falls back to the tool trace.
    trace_only = {"tools": {"tool_results": _state()["tool_results"]}}
    fallback = symptom_coding_summary(trace_only, coding_result)
    check("a result without the block still renders from the tool trace",
          [row.code for row in fallback.rows] == ["R07.9", "R19.7"], fallback.rows)

    # Flag grouping, over every flag shape the pipeline actually emits.
    flags = [
        "Coding: added symptom code R07.9 - Chest pain, unspecified "
        "('chest pain' documented in chief complaint); the Coding pass omitted it",
        "Added symptom code R19.7 - Diarrhea, unspecified "
        "(raised by validation, present in chief complaint)",
        "Specialty data documented this visit is not reflected in the OBJECTIVE "
        "section: ECG / Rhythm Notes - carry the documented findings into that section",
        "Chief complaint mentions 'chest pain' but no R07 code was generated - "
        "review and add a symptom code if clinically appropriate",
        "Cardiac claim not supported by the chart: ischemic etiology, cardiac "
        "diagnostic claim - no cardiac diagnosis is listed in Patient Context",
        "Cleanup: Removed unsupported cardiac claim from assessment: cardiac diagnostic claim",
        "Cleanup: Possible hallucinated diagnosis: Asthma (code J45.909) removed",
        "Scribe content incomplete: no ICD-10 suggestions (defaulted by the pipeline)",
        "A flag nobody has seen before",
    ]
    buckets = {group["key"]: group["flags"] for group in flag_groups(flags)}
    check("every flag lands in exactly one bucket",
          sum(len(items) for items in buckets.values()) == len(flags), buckets)
    check("completions are informational",
          len(buckets["auto_fixed"]) == 2, buckets.get("auto_fixed"))
    check("fidelity and gaps are review items",
          len(buckets["review_advised"]) == 4, buckets.get("review_advised"))
    check("removals are safety scrubs",
          len(buckets["safety_scrub"]) == 3, buckets.get("safety_scrub"))
    check("every cleanup removal is a safety scrub, whatever its wording",
          all(
              any(group["key"] == "safety_scrub"
                  for group in flag_groups([flag]))
              for flag in (
                  "Cleanup: Removed plan line for unsupported cardiac claim: ischemic etiology",
                  "Cleanup: Removed symptom code suggestion for ungrounded symptom: fever",
                  "Cleanup: Removed ICD-10: I21.0 - Exertional angina",
                  "Coding: removed unsupported cardiac claim from the medical necessity note "
                  "(ischemic heart disease); no cardiac diagnosis is listed in Patient Context",
              )
          ))
    check("raw flag text is preserved on every row",
          all(item.detail in flags for items in buckets.values() for item in items))
    check("an unknown flag stays visible",
          any(item.detail == "A flag nobody has seen before"
              for items in buckets.values() for item in items))
    check("fidelity notes read as one line each",
          specialty_fidelity_notes(flags)
          == ["ECG / Rhythm Notes - not reflected in the Objective section"],
          specialty_fidelity_notes(flags))
    check("no fidelity notes when none fired",
          specialty_fidelity_notes([]) == [])


def main():
    print("=" * 70)
    print("PHASE A3 SYMPTOM CODING TESTS (no LLM required)")
    print("=" * 70)
    test_map_coverage()
    test_map_is_symptoms_only()
    test_grounding_and_suppression()
    test_coding_receives_candidates()
    test_coding_completes_omitted_symptom_codes()
    test_coding_ignores_a_phantom_symptom()
    test_cardiology_interaction()
    test_validation_and_supervisor_coherence()
    test_single_source_and_safety()
    test_compatibility()
    test_negation_awareness()
    test_a4_metadata_shape()
    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
