"""
Individual tool tests - no LLM required, runs in under a second.

    python -m tests.test_tools
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import (  # noqa: E402
    AGENT_TOOLS,
    CODING_AGENT,
    SCRIBE_AGENT,
    SUPERVISOR_AGENT,
    TOOLS,
    calculate_em_code,
    call_tool,
    cpt_for_mdm,
    derive_mdm_domains,
    get_patient_allergies,
    get_patient_conditions,
    get_patient_medications,
    get_patient_profile,
    get_tools_for_agent,
    registry_as_dict,
    search_clinical_notes,
    simple_symptom_to_icd_suggestion,
    suggest_codes_for_complaint,
    validate_codes_against_patient,
    validate_icd10_code,
    validate_icd_against_patient,
)

PASSED, FAILED = [], []

JOHN = """Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%.
"""

MARY = """Patient: Mary Smith, 34yo female
Conditions: Asthma, Seasonal Allergies
Medications: Albuterol inhaler PRN, Fluticasone nasal spray
Allergies: None
Last Visit: Mild asthma exacerbation.
"""


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} -> {detail}")


def test_patient_tools():
    print("\nPatient tools")
    check("get_patient_conditions", get_patient_conditions(JOHN) == ["Type 2 Diabetes", "Hypertension"])
    check("get_patient_medications", get_patient_medications(JOHN) == ["Metformin 500mg BID", "Lisinopril 10mg daily"])
    check("get_patient_allergies", get_patient_allergies(JOHN) == ["Penicillin"])
    check("'None' allergies -> empty list", get_patient_allergies(MARY) == [])
    check("empty context -> empty list", get_patient_conditions("") == [])
    check("garbage context -> empty list", get_patient_conditions("no labels here") == [])
    profile = get_patient_profile(JOHN)
    check("profile has id/name/age", profile["patient_id"] == "john-doe" and profile["name"] == "John Doe" and profile["age"] == 58)


def test_grounding_tool():
    print("\nvalidate_icd_against_patient")
    check("E11.9 grounded", validate_icd_against_patient("E11.9", JOHN)["valid"])
    check("I10 grounded", validate_icd_against_patient("I10 - Essential hypertension", JOHN)["valid"])
    check("J45.909 rejected for John", not validate_icd_against_patient("J45.909", JOHN)["valid"])
    check("J45.909 accepted for Mary", validate_icd_against_patient("J45.909", MARY)["valid"])
    check("I20 rejected without cardiac dx", not validate_icd_against_patient("I20", JOHN)["valid"])
    check("R07.9 rejected without chest pain", not validate_icd_against_patient("R07.9", JOHN, "routine follow-up")["valid"])
    check("R07.9 accepted with chest pain", validate_icd_against_patient("R07.9", JOHN, "chest pain today")["valid"])
    check("unparseable input flagged, not raised", not validate_icd_against_patient("not a code", JOHN)["valid"])
    rejection = validate_icd_against_patient("J45.909", JOHN)
    check("rejection carries a quality flag", bool(rejection["quality_flag"]))


def test_symptom_tool():
    print("\nsimple_symptom_to_icd_suggestion")
    check("chest pain -> R07.9", simple_symptom_to_icd_suggestion("chest pain")["codes"] == ["R07.9"])
    check("diarrhea -> R19.7", simple_symptom_to_icd_suggestion("diarrhea")["codes"] == ["R19.7"])
    check("unknown symptom -> none", simple_symptom_to_icd_suggestion("knee pain")["codes"] == [])
    check("empty input -> none", simple_symptom_to_icd_suggestion("")["found"] is False)

    print("\nsuggest_codes_for_complaint")
    check(
        "complaint yields chest pain + diarrhea",
        suggest_codes_for_complaint("chest pain and high blood pressure and diarrhea", JOHN) == ["R07.9", "R19.7"],
        suggest_codes_for_complaint("chest pain and high blood pressure and diarrhea", JOHN),
    )
    check(
        "charted hypertension drops redundant I10",
        suggest_codes_for_complaint("high blood pressure", JOHN) == [],
        suggest_codes_for_complaint("high blood pressure", JOHN),
    )
    check(
        "routine follow-up yields nothing",
        suggest_codes_for_complaint("Routine follow-up for diabetes and hypertension", JOHN) == [],
    )


def test_batch_validation():
    print("\nvalidate_codes_against_patient")
    result = validate_codes_against_patient(
        ["E11.9 - Diabetes", "J45.909 - Asthma", "R07.9 - Chest pain"],
        JOHN,
        "chest pain",
    )
    check("valid codes identified", result["valid_codes"] == ["E11.9 - Diabetes", "R07.9 - Chest pain"], result["valid_codes"])
    check("invalid codes identified", result["invalid_codes"] == ["J45.909 - Asthma"], result["invalid_codes"])
    check("flags produced for invalid code", len(result["flags"]) == 1, result["flags"])


def test_registry_policy():
    print("\nTool registry / agent policy")
    check("all required tools registered", {
        "get_patient_conditions", "get_patient_medications",
        "search_clinical_notes", "validate_icd_against_patient",
        "calculate_em_code", "validate_icd10_code", "derive_mdm_domains",
    } <= set(TOOLS), sorted(TOOLS))
    check("optional tools registered", {"get_patient_allergies", "simple_symptom_to_icd_suggestion"} <= set(TOOLS))

    scribe = {t.name for t in get_tools_for_agent(SCRIBE_AGENT)}
    coding = {t.name for t in get_tools_for_agent(CODING_AGENT)}
    check("scribe can read the chart", "get_patient_conditions" in scribe, scribe)
    check("scribe can search notes", "search_clinical_notes" in scribe, scribe)
    check("scribe CANNOT validate codes", "validate_icd_against_patient" not in scribe, scribe)
    check("coding can validate codes", "validate_icd_against_patient" in coding, coding)
    check("coding can calculate CPT", "calculate_em_code" in coding and "derive_mdm_domains" in coding, coding)
    check("coding can check ICD existence", "validate_icd10_code" in coding, coding)
    check("coding cannot read raw notes", "search_clinical_notes" not in coding, coding)

    super_ = {t.name for t in get_tools_for_agent(SUPERVISOR_AGENT)}
    check("supervisor is registered as an agent", SUPERVISOR_AGENT == "supervisor_node")
    check("supervisor tools are deterministic only", super_ <= {"calculate_em_code", "validate_icd10_code", "simple_symptom_to_icd_suggestion"}, super_)

    try:
        call_tool("validate_icd_against_patient", agent=SCRIBE_AGENT, patient_context=JOHN, code="J45.909")
        check("policy violation raises PermissionError", False, "no exception")
    except PermissionError:
        check("policy violation raises PermissionError", True)

    check("unknown agent gets no tools", get_tools_for_agent("rogue_node") == [])
    check("registry snapshot lists owners", "scribe_node" in registry_as_dict()["search_clinical_notes"]["allowed_for"])
    check("every agent policy references real tools", all(n in TOOLS for names in AGENT_TOOLS.values() for n in names))


def test_em_calculator():
    print("\nDeterministic E/M calculator (99212-99215, 2-of-3 MDM)")
    check("straightforward x3 -> 99212", calculate_em_code("straightforward", "straightforward", "straightforward")["cpt_code"] == "99212")
    check("low x3 -> 99213", calculate_em_code("low", "low", "low")["cpt_code"] == "99213")
    check("moderate x3 -> 99214", calculate_em_code("moderate", "moderate", "moderate")["cpt_code"] == "99214")
    check("high x3 -> 99215", calculate_em_code("high", "high", "high")["cpt_code"] == "99215")

    # 2-of-3 rule: two high + one low still lands on high.
    check("two-of-three lifts to high", calculate_em_code("high", "high", "low")["overall_mdm_level"] == "high")
    check("two-of-three drops to moderate", calculate_em_code("moderate", "high", "low")["overall_mdm_level"] == "moderate")
    check("two-of-three with split low/moderate -> moderate", calculate_em_code("moderate", "low", "moderate")["overall_mdm_level"] == "moderate")

    result = calculate_em_code("low", "moderate", "low")
    check("result is valid", result["valid"] is True)
    check("justification is provided", bool(result["justification"]))

    bad = calculate_em_code("low", "banana", "high")
    check("invalid level -> valid False, no raise", bad["valid"] is False and "error" in bad)
    check("cpt_for_mdm maps levels", cpt_for_mdm("moderate").startswith("99214"))

    domains = derive_mdm_domains(
        {
            "subjective": "Chest pain today, plus diarrhea.",
            "objective": "BP 150/94.",
            "assessment": "HTN, diabetes follow-up.",
            "plan": "Start new medication. Recheck labs and x-ray.",
        },
        JOHN,
    )
    check("derive_mdm_domains returns three levels", {
        "problems_level", "data_level", "risk_level",
    } <= set(domains), domains)
    check("problems derive from conditions + symptoms", domains["problems_level"] in ("low", "moderate"), domains)
    check("empty note stays a straightforward baseline", derive_mdm_domains({})["risk_level"] == "straightforward")


def test_icd_validator():
    print("\nvalidate_icd10_code (existence check)")
    check("E11.9 exists", validate_icd10_code("E11.9")["exists"] is True)
    check("bare I10 exists", validate_icd10_code("I10")["exists"] is True)
    check("known prefix passes valid", validate_icd10_code("J45.909 - Unspecified asthma")["valid"] is True)
    check("unknown prefix flagged", validate_icd10_code("ZZZ.99")["plausible"] is False)
    check("malformed code invalid", validate_icd10_code("12AB")["valid"] is False)
    check("no code -> invalid", validate_icd10_code("")["valid"] is False)
    check("too few digits invalid", validate_icd10_code("E1.9")["valid"] is False)
    verdict = validate_icd10_code("E11.9")
    check("valid code carries a reason", bool(verdict["reason"]))
    check("never raises on garbage", isinstance(validate_icd10_code(12345)["valid"], bool))

    call_via_registry = call_tool("validate_icd10_code", text="R07.9", agent=SUPERVISOR_AGENT)
    check("supervisor may call validate_icd10_code", call_via_registry["exists"] is True, call_via_registry)


def test_call_tool_injection():
    print("\ncall_tool argument injection")
    check(
        "patient_context auto-injected",
        call_tool("get_patient_conditions", patient_context=JOHN) == ["Type 2 Diabetes", "Hypertension"],
    )
    check(
        "explicit kwargs passed through",
        call_tool("simple_symptom_to_icd_suggestion", symptom="chest pain")["codes"] == ["R07.9"],
    )


def test_search_tool_offline():
    print("\nsearch_clinical_notes (degrades gracefully)")
    result = search_clinical_notes("diabetes management", patient_id=None, k=1)
    check("returns a dict contract", set(["notes", "hits", "patient_context", "patient_found"]) <= set(result))
    check("unknown patient -> patient_found False", result["patient_found"] is False)
    check("empty query does not crash", search_clinical_notes("")["notes"] == "" or True)


def main():
    print("=" * 70)
    print("TOOL TESTS (no LLM required)")
    print("=" * 70)
    test_patient_tools()
    test_grounding_tool()
    test_symptom_tool()
    test_batch_validation()
    test_registry_policy()
    test_em_calculator()
    test_icd_validator()
    test_call_tool_injection()
    test_search_tool_offline()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
