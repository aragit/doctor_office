"""
Supervisor routing / retry / escalation tests - pure, no LLM, instant.

    python -m tests.test_supervisor

Covers the decision logic that the Supervisor node applies after validation:
flag classification, symptom-code auto-completion, retry accounting, and
escalation to human review at the retry limit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.supervisor_node import (  # noqa: E402
    MAX_RETRIES,
    MAX_REVALIDATIONS,
    classify_flags,
    determine_route,
    missing_symptom_codes,
    supervisor_node,
    human_review_node,
)
from agents.graph_state import (  # noqa: E402
    PIPELINE_VERSION,
    ROUTE_FINALIZE,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RETRY_CODING,
    ROUTE_RETRY_SCRIBE,
    ROUTE_REVALIDATE,
    build_initial_state,
)

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} -> {detail}")


def test_flag_classification():
    print("\nFlag classification")
    hallucination = "Possible hallucinated diagnosis: Asthma (code J45.909) removed"
    ungrounded = "Tool check: Ungrounded ICD-10 code J45.909 - not in patient context or chief complaint"
    cpt_status = "CPT 99202 is for new patients; patient appears established. Consider 99213-99215."
    symptom = "Chief complaint mentions 'chest pain' but no R07 code was generated - review"
    advisory1 = "Cleanup: Removed plan line: start albuterol"
    advisory2 = "Coding: dropped malformed ICD-10 code 'E119'"
    flags = [hallucination, ungrounded, cpt_status, symptom, advisory1, advisory2]
    buckets = classify_flags(flags)
    check("hallucination -> scribe", hallucination in buckets["scribe"], buckets["scribe"])
    check("ungrounded tool check -> coding", ungrounded in buckets["coding"], buckets["coding"])
    check("CPT status mismatch -> coding", cpt_status in buckets["coding"], buckets["coding"])
    check("symptom gap -> symptom", symptom in buckets["symptom"], buckets["symptom"])
    check("unknown / advisory left untouched", len(buckets["advisory"]) >= 2, buckets)
    check("does not mutate on empty", classify_flags(None) == {"scribe": [], "coding": [], "symptom": [], "advisory": []})


def test_routing_decisions():
    print("\nRouting decisions")
    hallucination = ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"]
    ungrounded = ["Tool check: Ungrounded ICD-10 code J45.909 - not in patient context"]

    for flags, retries, expected in (
        ([], 0, ROUTE_FINALIZE),
        (hallucination, 0, ROUTE_RETRY_SCRIBE),
        (hallucination, 1, ROUTE_RETRY_SCRIBE),
        (hallucination, MAX_RETRIES, ROUTE_HUMAN_REVIEW),
        (ungrounded, 0, ROUTE_RETRY_CODING),
        (ungrounded, MAX_RETRIES, ROUTE_HUMAN_REVIEW),
    ):
        decision = determine_route(flags, retries, {})
        check(
            f"flags={len(flags)} retries={retries} -> {expected}",
            decision["route_to"] == expected,
            decision,
        )

    decision = determine_route(hallucination, 0, {})
    check("scribe retry carries feedback", bool(decision["feedback"]) and decision["class"] == "retry_scribe", decision)
    check("scribe retry increments reviewer expectation", decision["route_to"] == ROUTE_RETRY_SCRIBE)

    escalated = determine_route(ungrounded, MAX_RETRIES, {})
    check("escalation class is human_review", escalated["class"] == "human_review")
    check("escalation marks escalated", escalated["escalated"] is True)


def test_symptom_autocomplete():
    print("\nSymptom-code auto-completion")
    flag = ["Chief complaint mentions 'chest pain' but no R07 code was generated - review"]
    decision = determine_route(
        flag, 0, {"primary_icd10": "E11.9 - Diabetes", "secondary_icd10": ["I10 - Hypertension"]},
        chief_complaint="chest pain and high blood pressure and diarrhea",
    )
    check("routes back to validation (re-check, no LLM)", decision["route_to"] == ROUTE_REVALIDATE, decision)
    check("provides canonical R07.9", decision["symptom_codes"] == ["R07.9 - Chest pain, unspecified"], decision)
    check("route class is revalidate", decision["class"] == "revalidate")

    # Diarrhea + chest pain -> both R07 and R19 complete.
    both = [
        "Chief complaint mentions 'chest pain' but no R07 code was generated - review",
        "Chief complaint mentions 'diarrhea' but no R19 code was generated - review",
    ]
    decision = determine_route(
        both, 0, {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
        chief_complaint="chest pain and high blood pressure and diarrhea",
    )
    check("multi-symptom completion returns both codes", decision["symptom_codes"] == [
        "R07.9 - Chest pain, unspecified",
        "R19.7 - Diarrhea, unspecified",
    ], decision)

    check(
        "missing_symptom_codes empty for a clean complaint",
        missing_symptom_codes([], "Routine follow-up", ["E11.9", "I10"]) == [],
    )
    check(
        "does not duplicate existing code",
        missing_symptom_codes(flag, "Chest pain today", ["E11.9", "I10", "R07.9"]) == [],
    )
    check(
        "does not invent a code for an absent symptom",
        missing_symptom_codes(
            ["Chief complaint mentions 'headache' but no R51 code was generated - review"],
            "Routine follow-up", ["E11.9", "I10"],
        ) == [],
    )


def test_supervisor_node_update():
    print("\nsupervisor_node state update")
    hall_unlocked = build_initial_state("ctx", "cc", "notes")
    hall_unlocked.update({
        "quality_flags": ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"],
        "retry_count": 0,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(hall_unlocked)
    check("routes to scribe on hallucination", update["route_to"] == ROUTE_RETRY_SCRIBE, update)
    check("increments retry_count", update["retry_count"] == 1, update)
    check("adds feedback for the scribe", bool(update["feedback"]), update)
    check("stays in deterministic step set", update["current_step"] == "supervisor")
    check("does not flag human review yet", update["needs_human_review"] is False)

    exhausted = build_initial_state("ctx", "cc", "notes")
    exhausted.update({
        "quality_flags": ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"],
        "retry_count": MAX_RETRIES,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(exhausted)
    check("routes to human review at limit", update["route_to"] == ROUTE_HUMAN_REVIEW, update)
    check("sets needs_human_review", update["needs_human_review"] is True)
    check("keeps retry_count at ceiling", update["retry_count"] == MAX_RETRIES)

    clean = build_initial_state("ctx", "cc", "notes")
    clean.update({
        "quality_flags": [],
        "retry_count": 0,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(clean)
    check("clean visit finalizes", update["route_to"] == ROUTE_FINALIZE, update)
    check("clean visit unchanged retry", update["retry_count"] == 0)
    check("no feedback for clean visit", update["feedback"] == [])

    # Symptom auto-completion path: modify coding_result + append supervisor_applied.
    gap = build_initial_state("ctx", "chest pain today", "notes")
    gap.update({
        "quality_flags": ["Chief complaint mentions 'chest pain' but no R07 code was generated - review"],
        "retry_count": 0,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(gap)
    check("auto-completed coding_result gains R07.9",
          "R07.9 - Chest pain, unspecified" in update["coding_result"]["secondary_icd10"],
          update["coding_result"])
    check("supervisor_applied records the fix", bool(update.get("supervisor_applied")), update)
    check("routes to validation for re-check", update["route_to"] == ROUTE_REVALIDATE, update)
    check("no retry consumed for a deterministic fix", update["retry_count"] == 0, update)


def test_human_review_node():
    print("\nhuman_review_node")
    state = build_initial_state("ctx", "cc", "notes")
    state.update({"needs_human_review": True})
    update = human_review_node(state)
    check("marks current_step human_review", update["current_step"] == "human_review")
    check("flags needs_human_review", update["needs_human_review"] is True)
    check("never persists a human-review visit", update["memory_saved"] is False)


def test_stale_flag_does_not_repeat_retry():
    print("\nStale-flag immunity (current cycle wins)")
    # After a scribe retry the note is clean, but the *audit* trail still holds
    # the pass-1 hallucination flag. The Supervisor must judge the current
    # cycle (clean) and finalize instead of looping.
    retried_pass = build_initial_state("ctx", "cc", "notes")
    retried_pass.update({
        "quality_flags": ["Possible hallucinated diagnosis: Asthma (code J45.909) removed"],
        "cycle_flags": {"coding": [], "validation": []},
        "retry_count": 1,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(retried_pass)
    check("clean current cycle finalizes", update["route_to"] == ROUTE_FINALIZE, update)
    check("no retry consumed for a clean cycle", update["retry_count"] == 1, update)

    # Regression: an ungrounded coding flag from pass N must not re-fire once
    # coding/validation report a clean cycle on pass N+1.
    second_pass = build_initial_state("ctx", "cc", "notes")
    second_pass.update({
        "quality_flags": ["Tool check: Ungrounded ICD-10 code J45.909 - not in patient context"],
        "cycle_flags": {"coding": [], "validation": []},
        "retry_count": 1,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(second_pass)
    check("coding stale flag does not re-loop", update["route_to"] == ROUTE_FINALIZE, update)


# ---------------------------------------------------------------------------
# P1 control flow: stale-flag reset and post-auto-fix re-validation
# ---------------------------------------------------------------------------


def test_retry_clears_the_previous_pass_flags():
    """A retry must hand the next pass an empty flag view, not an inherited one."""
    print("\nP1 stale flags: a retry empties the routing view")
    from agents.supervisor_policy import _decision_flags

    pass1_flag = "Possible hallucinated diagnosis: Asthma (code J45.909) removed"
    stale = build_initial_state("ctx", "cc", "notes")
    stale.update({
        # The audit log keeps every pass's flags, forever.
        "quality_flags": [pass1_flag],
        "cycle_flags": {"scribe": [pass1_flag], "coding": [], "validation": []},
        "retry_count": 0,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })

    update = supervisor_node(stale)
    check("the flag still routes the retry", update["route_to"] == ROUTE_RETRY_SCRIBE, update)
    check("every bucket of the next cycle starts empty",
          update["cycle_flags"] == {
              "scribe": [], "cleanup": [], "coding": [], "validation": [],
          }, update.get("cycle_flags"))
    check("the cleared flags are kept as data, not as routing input",
          update["cleared_cycle_flags"] == [pass1_flag], update.get("cleared_cycle_flags"))
    check("the reset is counted", update["cycle_flag_resets"] == 1, update)
    check("the retry still consumes budget",
          update["retry_count"] == 1 and update["route_to"] == ROUTE_RETRY_SCRIBE, update)
    check("the audit log is left alone", "quality_flags" not in update, update)

    # The proof that matters: the state a retry leaves behind cannot route on
    # the pass it just corrected, even though quality_flags still holds the flag.
    after_reset = dict(stale, **{
        k: v for k, v in update.items() if k in ("cycle_flags", "retry_count", "cycle_flag_resets")
    })
    check("the reset state exposes no flags to the Supervisor",
          _decision_flags(after_reset) == [], _decision_flags(after_reset))
    check("the reset state finalizes instead of escalating",
          supervisor_node(after_reset)["route_to"] == ROUTE_FINALIZE,
          supervisor_node(after_reset))

    # A pass that finds nothing must not fall through to the accumulated log.
    exhausted = build_initial_state("ctx", "cc", "notes")
    exhausted.update({
        "quality_flags": [pass1_flag],
        "cycle_flags": dict(update["cycle_flags"]),
        "retry_count": MAX_RETRIES,
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    decision = supervisor_node(exhausted)
    check("stale flags cannot escalate a visit at the retry ceiling",
          decision["route_to"] == ROUTE_FINALIZE, decision)
    check("no extra retry is charged", decision["retry_count"] == MAX_RETRIES, decision)


def test_autofix_requires_revalidation():
    """A mutated coding_result may never be finalized on the flags that predate it."""
    print("\nP1 auto-fix: re-validation is required before finalize")
    gap_flag = "Chief complaint mentions 'chest pain' but no R07 code was generated - review"

    # 1. The gap routes to a deterministic fix.
    gapped = build_initial_state("ctx", "chest pain today", "notes")
    gapped.update({
        "quality_flags": [gap_flag],
        "cycle_flags": {"validation": [gap_flag], "coding": []},
        "coding_result": {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]},
    })
    update = supervisor_node(gapped)
    check("the gap is auto-completed, not sent to an LLM",
          update["route_to"] == ROUTE_REVALIDATE, update)
    check("the auto-fix is recorded as such",
          update["supervisor_decision"] == "revalidate", update)
    check("the fix demands a fresh validation",
          update["pending_revalidation"] is True, update)
    check("the auto-fix round is counted", update["auto_fix_rounds"] == 1, update)
    check("the flags that triggered the fix are cleared",
          update["cycle_flags"] == {"coding": [], "validation": []}, update.get("cycle_flags"))
    check("the fix consumed no LLM retry", update["retry_count"] == 0, update)
    check("R07.9 is appended to the codes",
          update["coding_result"]["secondary_icd10"] ==
          ["I10", "R07.9 - Chest pain, unspecified"], update["coding_result"])

    # 2. Replaying the node (LangGraph re-applies updates) must not double-add.
    replay = dict(gapped, **{k: v for k, v in update.items() if k != "messages"})
    replay_update = supervisor_node(replay)
    check("replaying the fix does not duplicate the code",
          replay_update.get("coding_result") is None
          or replay_update["coding_result"]["secondary_icd10"] ==
          replay["coding_result"]["secondary_icd10"], replay_update.get("coding_result"))
    check("a no-op fix does not demand another re-validation",
          "pending_revalidation" not in replay_update, replay_update)

    # 3. The state between the fix and the validation cannot finalize.
    midway = dict(replay, pending_revalidation=True, cycle_flags={"coding": [], "validation": []})
    midway_decision = supervisor_node(midway)
    check("an unvalidated fix routes to validation again, never to memory",
          midway_decision["route_to"] == ROUTE_REVALIDATE, midway_decision)
    check("it is not escalated and not finalized",
          midway_decision["route_to"] not in (ROUTE_FINALIZE, ROUTE_HUMAN_REVIEW),
          midway_decision)

    # 4. Budget: with the re-validation pass already spent, escalate instead of
    #    looping, and say so for the right reason.
    spent = dict(midway, revalidation_passes=MAX_REVALIDATIONS)
    spent_decision = supervisor_node(spent)
    check("a mutated result is never finalized once the budget is spent",
          spent_decision["route_to"] == ROUTE_HUMAN_REVIEW, spent_decision)
    check("the escalation names the re-validation, not the LLM retries",
          spent_decision["escalation_reason"] == "revalidation_exhausted", spent_decision)
    check("the escalation is marked on the state",
          spent_decision["needs_human_review"] is True, spent_decision)


def test_autofix_revalidation_uses_real_validation():
    """The re-validation pass must run the real validator on the fixed codes."""
    print("\nP1 auto-fix: the real re-validation resolves the gap it fixed")
    from agents.validation_node import validation_node

    context = (
        "Patient: John Doe, 58yo male\nConditions: Type 2 Diabetes, Hypertension\n"
        "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
        "Allergies: Penicillin\nLast Visit: Blood pressure 148/92."
    )
    gap_flag = "Chief complaint mentions 'chest pain' but no R07 code was generated - review"

    before = build_initial_state(context, "Chest pain x 1 day", "notes")
    before.update({
        "soap_note": {"subjective": "Chest pain x 1 day.", "objective": "BP 148/92.",
                      "assessment": "Chest pain, undifferentiated.",
                      "plan": "ECG today.", "icd10_suggestions": ["E11.9 - Type 2 diabetes"]},
        "cleaned_soap_note": {"subjective": "Chest pain x 1 day.", "objective": "BP 148/92.",
                              "assessment": "Chest pain, undifferentiated.",
                              "plan": "ECG today.", "icd10_suggestions": ["E11.9 - Type 2 diabetes"]},
        "coding_result": {
            "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
            "secondary_icd10": ["I10 - Essential hypertension"],
            "cpt_suggestions": ["99213 - Office visit, established patient, low MDM"],
        },
        "quality_flags": [gap_flag],
        "cycle_flags": {"validation": [gap_flag], "coding": []},
    })

    fix = supervisor_node(before)
    fixed_state = dict(before, **{
        "coding_result": fix["coding_result"],
        "pending_revalidation": True,
        "auto_fix_rounds": 1,
        "supervisor_symptom_codes": fix["supervisor_symptom_codes"],
    })

    first = validation_node(before)
    check("the gap is raised before the fix",
          any("no R07 code" in f for f in first["quality_flags"]), first["quality_flags"])
    check("an ordinary pass owes no re-validation",
          "pending_revalidation" not in first, first)
    check("an ordinary pass runs no integrity check",
          not any("Post-fix" in f for f in first["quality_flags"]), first["quality_flags"])

    second = validation_node(fixed_state)
    check("the gap is not re-raised after the fix",
          not any("no R07 code" in f for f in second["quality_flags"]),
          second["quality_flags"])
    check("the added code is structurally valid, so nothing is flagged",
          not any("Post-fix" in f for f in second["quality_flags"]),
          [f for f in second["quality_flags"] if "Post-fix" in f])
    check("the re-validation is counted", second["revalidation_passes"] == 1, second)
    check("the re-validation clears the pending marker",
          second["pending_revalidation"] is False, second)
    check("the added code passes grounding",
          not any("Ungrounded" in f for f in second["quality_flags"]),
          [f for f in second["quality_flags"] if "Ungrounded" in f])

    final = supervisor_node(dict(fixed_state, **{
        "cycle_flags": second["cycle_flags"],
        "pending_revalidation": False,
        "revalidation_passes": second["revalidation_passes"],
    }))
    check("a validated fix finalizes", final["route_to"] == ROUTE_FINALIZE, final)
    check("the finalize is recorded as such",
          final["supervisor_decision"] == "finalize", final)


def test_post_fix_integrity_flags():
    """Structural and duplication checks that run only on a mutated result."""
    print("\nP1 auto-fix: post-fix code-list integrity")
    from utils.validation import MAX_POST_FIX_SYMPTOM_CODES, validate_post_fix_integrity

    clean = {
        "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
        "secondary_icd10": ["I10 - Essential hypertension", "R07.9 - Chest pain, unspecified"],
    }
    check("a well-formed list raises nothing",
          validate_post_fix_integrity(clean, ["R07.9 - Chest pain, unspecified"]) == [])
    check("an empty result raises nothing", validate_post_fix_integrity({}, []) == [])

    duplicated = {
        "primary_icd10": "E11.9 - Diabetes",
        "secondary_icd10": ["R07.9 - Chest pain, unspecified",
                             "R07.9 - Chest pain, unspecified"],
    }
    flags = validate_post_fix_integrity(duplicated, [])
    check("an exact duplicate is caught",
          any("duplicate ICD-10 code" in f and "R07.9" in f for f in flags), flags)

    malformed = {"primary_icd10": "E11.9", "secondary_icd10": ["chest pain"]}
    flags = validate_post_fix_integrity(malformed, [])
    check("a malformed entry is caught",
          any("malformed ICD-10 entry" in f for f in flags), flags)

    missing = validate_post_fix_integrity(clean, ["R19.7 - Diarrhea, unspecified"])
    check("a reported addition that did not land is caught",
          any("R19.7" in f and "absent" in f for f in missing), missing)

    flooded = {
        "primary_icd10": "E11.9",
        "secondary_icd10": [f"R0{i}.9 - symptom {i}" for i in range(1, 6)],
    }
    flags = validate_post_fix_integrity(flooded, [])
    check("a runaway symptom list is caught",
          any(f"more than {MAX_POST_FIX_SYMPTOM_CODES}" in f for f in flags), flags)

    check("integrity flags route to coding, not to an advisory",
          all("Post-fix" not in f or "re-validation" in f for f in
              validate_post_fix_integrity(duplicated, [])), "flag shape")


def test_spent_revalidation_budget_hands_the_gap_to_coding():
    """A gap that outlives the deterministic layer goes to the LLM, bounded."""
    print("\nP1 auto-fix: budget spent -> bounded coding retry")
    gap_flag = "Chief complaint mentions 'chest pain' but no R07 code was generated - review"
    coding = {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]}

    spent = build_initial_state("ctx", "chest pain today", "notes")
    spent.update({
        "quality_flags": [gap_flag],
        "cycle_flags": {"validation": [gap_flag], "coding": []},
        "revalidation_passes": MAX_REVALIDATIONS,
        "coding_result": coding,
    })
    decision = supervisor_node(spent)
    check("the deterministic layer is not used again",
          not decision.get("supervisor_symptom_codes"), decision)
    check("the gap is routed to the Coding agent",
          decision["route_to"] == ROUTE_RETRY_CODING, decision)
    check("the feedback asks for the code explicitly",
          any("R07.9" in f for f in decision["feedback"]), decision["feedback"])
    check("the retry is charged, not the fix budget",
          decision["retry_count"] == 1, decision)

    exhausted = dict(spent, retry_count=MAX_RETRIES)
    decision = supervisor_node(exhausted)
    check("a persistent gap still terminates at MAX_RETRIES",
          decision["route_to"] == ROUTE_HUMAN_REVIEW, decision)
    check("that escalation is attributed to the retries",
          decision["escalation_reason"] == "max_retries", decision)

    # An advisory-only flag stays advisory: no new routing appears.
    advisory = build_initial_state("ctx", "Routine follow-up", "notes")
    advisory.update({
        "quality_flags": ["Chief complaint mentions 'high blood pressure' but no I10 "
                          "code was generated - review"],
        "cycle_flags": {"validation": ["Chief complaint mentions 'high blood pressure' but "
                                       "no I10 code was generated - review"], "coding": []},
        "coding_result": coding,
    })
    check("a non-auto-addable gap still does not route",
          supervisor_node(advisory)["route_to"] == ROUTE_FINALIZE,
          supervisor_node(advisory))


def test_supervisor_escalates_on_empty_soap():
    """Empty SOAP note triggers retry, then human review on second occurrence."""
    from agents.supervisor_policy import MAX_EMPTY_SOAP_RETRIES

    empty_soap_flag = "soap_empty: All SOAP sections (subjective, objective, assessment, plan) are empty"
    coding = {"primary_icd10": "E11.9", "secondary_icd10": ["I10"]}

    # First occurrence: should retry scribe
    first = build_initial_state("ctx", "cc", "notes")
    first.update({
        "quality_flags": [empty_soap_flag],
        "cycle_flags": {"validation": [empty_soap_flag], "coding": [], "scribe": []},
        "coding_result": coding,
        "empty_soap_retries": 0,
    })
    update = supervisor_node(first)
    assert update["route_to"] == ROUTE_RETRY_SCRIBE, f"Expected retry_scribe, got {update['route_to']}"
    assert update.get("empty_soap_retries") == 1, f"Expected empty_soap_retries=1, got {update.get('empty_soap_retries')}"
    assert any("empty" in f.lower() for f in update["feedback"]), "Feedback should mention empty sections"

    # Second occurrence (retries exhausted): should escalate to human review
    second = build_initial_state("ctx", "cc", "notes")
    second.update({
        "quality_flags": [empty_soap_flag],
        "cycle_flags": {"validation": [empty_soap_flag], "coding": [], "scribe": []},
        "coding_result": coding,
        "empty_soap_retries": MAX_EMPTY_SOAP_RETRIES,
    })
    update = supervisor_node(second)
    assert update["route_to"] == ROUTE_HUMAN_REVIEW, f"Expected human_review, got {update['route_to']}"
    assert update.get("escalation_reason") == "empty_soap_exhausted", f"Expected empty_soap_exhausted, got {update.get('escalation_reason')}"
    assert update["needs_human_review"] is True, "needs_human_review should be True"


def test_version_and_constants():
    print("\nVersion and constants")
    check("pipeline version bumped", PIPELINE_VERSION == "3.1-supervisor-revalidate", PIPELINE_VERSION)
    check("retry ceiling exported", MAX_RETRIES == 2)
    check("re-validation ceiling exported", MAX_REVALIDATIONS == 1)


def main():
    print("=" * 70)
    print("SUPERVISOR TESTS (pure, no LLM)")
    print("=" * 70)
    test_flag_classification()
    test_routing_decisions()
    test_symptom_autocomplete()
    test_supervisor_node_update()
    test_human_review_node()
    test_stale_flag_does_not_repeat_retry()
    test_retry_clears_the_previous_pass_flags()
    test_autofix_requires_revalidation()
    test_autofix_revalidation_uses_real_validation()
    test_post_fix_integrity_flags()
    test_spent_revalidation_budget_hands_the_gap_to_coding()
    test_supervisor_escalates_on_empty_soap()
    test_version_and_constants()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())