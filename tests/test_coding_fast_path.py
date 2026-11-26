"""
Unit tests for the Fast Path / Slow Path decision gate (coding_fast_path.py).

Tests cover:
- Clear chronic DM + HTN follow-up -> should be Fast Path
- New/uncertain symptoms or incomplete documentation -> should be Slow Path
- Edge case with only symptom codes -> should decide reasonably
"""

import pytest
from agents.coding_fast_path import (
    should_use_fast_path,
    execute_fast_path,
    MAX_FAST_PATH_SYMPTOM_CODES,
)


# Test patient context for P001 (John Doe) - chronic DM + HTN
PATIENT_CONTEXT_P001 = """
Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence.
"""

# Clean SOAP note for a routine follow-up
CLEAN_SOAP_ROUTINE = {
    "subjective": "Patient reports increased fatigue and occasional dizziness. History of Type 2 Diabetes and Hypertension. Occasional polyuria and fatigue reported previously.",
    "objective": "BP 150/94. HbA1c 7.8%. BMI 31.2. Medications: Metformin 500mg BID, Lisinopril 10mg daily.",
    "assessment": "Patient presents with persistent hyperglycemia despite Metformin and hypertension. BP uncontrolled. Considering medication intensification for diabetes and further optimization of hypertension management.",
    "plan": "Continue Metformin 500mg BID. Continue Lisinopril 10mg daily. Monitor BP and glucose twice daily. Encourage dietary modifications and regular exercise. Follow-up in 4 weeks for BP and repeat HbA1c. Consider GLP-1 agonist or SGLT2 inhibitor for diabetes intensification.",
    "icd10_suggestions": [
        "E11.9 - Type 2 diabetes mellitus without complications",
        "I10 - Essential (primary) hypertension"
    ]
}

# Chief complaint for routine follow-up
CHIEF_COMPLAINT_ROUTINE = "Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."

# Chief complaint with acute symptoms (should trigger Slow Path)
CHIEF_COMPLAINT_ACUTE = "Chest pain and shortness of breath for 2 hours"

# Chief complaint with uncertainty phrases
CHIEF_COMPLAINT_UNCERTAIN = "Rule out myocardial infarction, new onset chest pain"

# Strong uncertainty phrases that MUST force Slow Path
CHIEF_COMPLAINT_RULE_OUT = "Rule out pneumonia"
CHIEF_COMPLAINT_POSSIBLE = "Possible pneumonia, needs further workup"
CHIEF_COMPLAINT_SUSPECTED = "Suspected PE, differential includes pneumonia"
CHIEF_COMPLAINT_WORKING_DX = "Working diagnosis: rule out MI"
CHIEF_COMPLAINT_CANNOT_RULE_OUT = "Cannot rule out appendicitis"
CHIEF_COMPLAINT_FURTHER_EVAL = "Further evaluation needed for chest pain"
CHIEF_COMPLAINT_UNDIFFERENTIATED = "Undifferentiated abdominal pain, unclear etiology"
CHIEF_COMPLAINT_FIRST_PRESENTATION = "First presentation of chest pain"

# Words that should NOT force Slow Path in chronic context
CHIEF_COMPLAINT_ACUTE_WORD = "Follow-up for diabetes and hypertension. Acute fatigue noted."
CHIEF_COMPLAINT_WORSENING = "Follow-up for diabetes. Worsening fatigue and dizziness."
CHIEF_COMPLAINT_REFERRAL = "Follow-up for hypertension. Referral to cardiology for BP management."
CHIEF_COMPLAINT_MONITOR = "Follow-up for diabetes. Monitor HbA1c and adjust medications."
CHIEF_COMPLAINT_ADJUST = "Follow-up for HTN. Continue lisinopril, increase dose if needed."
CHIEF_COMPLAINT_MILD_MODERATE = "Follow-up for diabetes. Mild fatigue, moderate dizziness."
CHIEF_COMPLAINT_STABLE_UNCONTROLLED = "Follow-up for HTN. BP uncontrolled but stable."

# Scribe-generated SOAP with soft uncertainty language (should NOT force Slow Path)
CLEAN_SOAP_WITH_SCRIBE_UNCERTAINTY = {
    "subjective": "Patient reports increased fatigue and occasional dizziness. History of Type 2 Diabetes and Hypertension.",
    "objective": "BP 150/94. HbA1c 7.8%. BMI 31.2. Medications: Metformin 500mg BID, Lisinopril 10mg daily.",
    "assessment": "Patient possibly has persistent hyperglycemia despite Metformin. Hypertension may be uncontrolled. Diabetes possibly worsening.",
    "plan": "Continue Metformin 500mg BID. Continue Lisinopril 10mg daily. Monitor BP and glucose. Follow-up in 4 weeks.",
    "icd10_suggestions": [
        "E11.9 - Type 2 diabetes mellitus without complications",
        "I10 - Essential (primary) hypertension"
    ]
}

# Incomplete SOAP (missing assessment)
CLEAN_SOAP_INCOMPLETE = {
    "subjective": "Patient reports fatigue.",
    "objective": "BP 140/90.",
    "assessment": "",
    "plan": "Follow up in 2 weeks.",
    "icd10_suggestions": []
}

# Visit context with additional findings
VISIT_CONTEXT_WITH_EXAM = {
    "hpi": "Patient reports increased fatigue for 2 weeks.",
    "physical_exam": "BP 150/94. Heart regular. Lungs clear.",
    "additional_findings": "HbA1c 7.8%. BMP normal.",
    "vitals": {"blood_pressure": "150/94", "heart_rate": "78"},
    "specialty_fields": {}
}


class TestShouldUseFastPath:
    """Tests for the should_use_fast_path decision function."""

    def test_clear_chronic_dm_htn_followup_fast_path(self):
        """Case A: Clear chronic DM + HTN follow-up -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"
        assert metadata["confidence"] == "high"
        assert metadata["deterministic_primary_icd"] is not None
        assert "E11" in metadata["deterministic_primary_icd"] or "I10" in metadata["deterministic_primary_icd"]
        assert metadata["deterministic_cpt"] is not None
        assert len(metadata["missing_items"]) == 0
        assert "FAST" in metadata["reason"] or "deterministically" in metadata["reason"]

    def test_acute_chest_pain_slow_path(self):
        """Case B: Acute chest pain -> should be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ACUTE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert metadata["confidence"] == "low"
        assert len(metadata["missing_items"]) > 0
        assert any("uncertainty" in item.lower() or "acute" in item.lower() or "symptom" in item.lower()
                   for item in metadata["missing_items"])

    def test_uncertainty_phrases_slow_path(self):
        """Case B: Uncertainty phrases (rule out, new onset) -> should be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_UNCERTAIN,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert metadata["confidence"] == "low"
        assert len(metadata["missing_items"]) > 0
        assert any("uncertainty" in item.lower() or "rule out" in item.lower() or "new onset" in item.lower()
                   for item in metadata["missing_items"])

    def test_incomplete_soap_slow_path(self):
        """Case B: Incomplete SOAP (empty assessment) -> should be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_INCOMPLETE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert metadata["confidence"] == "low"
        assert len(metadata["missing_items"]) > 0
        assert any("assessment" in item.lower() for item in metadata["missing_items"])

    def test_no_chronic_conditions_slow_path(self):
        """Case: Patient with no recognized chronic conditions -> should be Slow Path."""
        patient_context_no_chronic = """
        Patient: Jane Smith, 30yo female
        Conditions: None
        Medications: None
        Allergies: None
        Last Visit: None
        """
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=patient_context_no_chronic,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert metadata["confidence"] == "low"
        assert any("chronic" in item.lower() for item in metadata["missing_items"])

    def test_symptom_only_chief_complaint_slow_path(self):
        """Case: Chief complaint with only acute symptoms, no chronic match -> Slow Path."""
        patient_context_healthy = """
        Patient: Bob Wilson, 25yo male
        Conditions: None
        Medications: None
        Allergies: None
        Last Visit: None
        """
        chief_complaint_symptom = "Headache and dizziness for 3 days"
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=patient_context_healthy,
            chief_complaint=chief_complaint_symptom,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert metadata["confidence"] == "low"

    def test_strong_uncertainty_rule_out_slow_path(self):
        """Strong uncertainty phrase 'rule out' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_RULE_OUT,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("rule out" in item.lower() for item in metadata["missing_items"])

    def test_strong_uncertainty_possible_slow_path(self):
        """Strong uncertainty phrase 'possible' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_POSSIBLE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("possible" in item.lower() for item in metadata["missing_items"])

    def test_strong_uncertainty_suspected_slow_path(self):
        """Strong uncertainty phrase 'suspected' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_SUSPECTED,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("suspect" in item.lower() for item in metadata["missing_items"])

    def test_strong_uncertainty_differential_slow_path(self):
        """Strong uncertainty phrase 'differential' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_WORKING_DX,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("differential" in item.lower() or "working diagnosis" in item.lower() or "rule out" in item.lower()
                   for item in metadata["missing_items"])

    def test_strong_uncertainty_cannot_rule_out_slow_path(self):
        """Strong uncertainty phrase 'cannot rule out' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_CANNOT_RULE_OUT,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("cannot rule out" in item.lower() for item in metadata["missing_items"])

    def test_strong_uncertainty_further_eval_slow_path(self):
        """Strong uncertainty phrase 'further evaluation needed' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_FURTHER_EVAL,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("further evaluation" in item.lower() for item in metadata["missing_items"])

    def test_strong_uncertainty_undifferentiated_slow_path(self):
        """Strong uncertainty phrase 'undifferentiated'/'unclear etiology' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_UNDIFFERENTIATED,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("undifferentiated" in item.lower() or "unclear etiology" in item.lower() or "etiology unclear" in item.lower()
                   for item in metadata["missing_items"])

    def test_strong_uncertainty_first_presentation_slow_path(self):
        """Strong uncertainty phrase 'first presentation' -> MUST be Slow Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_FIRST_PRESENTATION,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is False, f"Expected Slow Path, got Fast Path: {metadata['reason']}"
        assert any("first presentation" in item.lower() or "first episode" in item.lower()
                   for item in metadata["missing_items"])

    def test_allowed_word_acute_in_chronic_context_fast_path(self):
        """Word 'acute' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ACUTE_WORD,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_worsening_in_chronic_context_fast_path(self):
        """Word 'worsening' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_WORSENING,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_referral_in_chronic_context_fast_path(self):
        """Word 'referral' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_REFERRAL,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_monitor_in_chronic_context_fast_path(self):
        """Word 'monitor' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_MONITOR,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_adjust_in_chronic_context_fast_path(self):
        """Word 'adjust'/'increase'/'decrease' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ADJUST,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_mild_moderate_in_chronic_context_fast_path(self):
        """Words 'mild'/'moderate'/'severe' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_MILD_MODERATE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_allowed_word_stable_uncontrolled_in_chronic_context_fast_path(self):
        """Words 'stable'/'uncontrolled' in chronic follow-up context -> should be Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_STABLE_UNCONTROLLED,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"

    def test_scribe_generated_uncertainty_language_fast_path(self):
        """Scribe-generated soft uncertainty ('possibly', 'may be') in SOAP should NOT force Slow Path
        when original inputs clearly indicate chronic follow-up."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_WITH_SCRIBE_UNCERTAINTY,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )
        assert use_fast is True, f"Expected Fast Path, got Slow Path: {metadata['reason']}"
        assert metadata["confidence"] == "high"
        assert len(metadata["missing_items"]) == 0

    def test_fast_path_metadata_structure(self):
        """Verify decision_metadata has all required fields."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        # Check all required keys exist
        required_keys = [
            "reason",
            "confidence",
            "missing_items",
            "deterministic_primary_icd",
            "deterministic_secondary_icds",
            "deterministic_cpt",
            "decision_flags",
        ]
        for key in required_keys:
            assert key in metadata, f"Missing required key: {key}"

        # Check types
        assert isinstance(metadata["reason"], str)
        assert metadata["confidence"] in ("high", "medium", "low")
        assert isinstance(metadata["missing_items"], list)
        assert isinstance(metadata["deterministic_secondary_icds"], list)
        assert isinstance(metadata["decision_flags"], list)

    def test_execute_fast_path_output_shape(self):
        """Verify execute_fast_path returns the expected output shape."""
        # First get a fast path decision
        use_fast, decision_metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is True, "Precondition: should be Fast Path"

        # Execute fast path
        coding_result, tool_trace, quality_flags = execute_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
            decision_metadata=decision_metadata,
        )

        # Check coding_result has all required fields (same as Slow Path)
        required_result_keys = [
            "primary_icd10",
            "secondary_icd10",
            "cpt_suggestions",
            "medical_necessity_note",
        ]
        for key in required_result_keys:
            assert key in coding_result, f"Missing required result key: {key}"

        # Check types
        assert isinstance(coding_result["primary_icd10"], str)
        assert isinstance(coding_result["secondary_icd10"], list)
        assert isinstance(coding_result["cpt_suggestions"], list)
        assert len(coding_result["cpt_suggestions"]) > 0
        assert isinstance(coding_result["medical_necessity_note"], str)

        # Check tool_trace has expected structure
        assert "simple_symptom_to_icd_suggestion" in tool_trace
        assert "validate_icd10_code" in tool_trace
        assert "derive_mdm_domains" in tool_trace
        assert "calculate_em_code" in tool_trace
        assert "validate_icd_against_patient" in tool_trace
        assert tool_trace.get("fast_path", {}).get("used") is True

        # Check quality_flags is a list
        assert isinstance(quality_flags, list)


class TestFastPathEdgeCases:
    """Edge case tests for the Fast Path logic."""

    def test_multiple_chronic_conditions_priority(self):
        """When multiple chronic conditions match, diabetes should be primary for DM+HTN."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint="Follow-up for diabetes and hypertension",
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is True
        # Primary should be one of the chronic conditions
        primary = metadata["deterministic_primary_icd"]
        assert primary is not None
        assert "E11" in primary or "I10" in primary

    def test_symptom_codes_added(self):
        """Verify symptom codes (fatigue, dizziness) are resolved in Fast Path."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is True
        secondary = metadata["deterministic_secondary_icds"]
        # Should have R53 (fatigue) and/or R42 (dizziness)
        assert len(secondary) > 0
        secondary_str = " ".join(secondary)
        assert "R53" in secondary_str or "R42" in secondary_str or "Fatigue" in secondary_str or "Dizziness" in secondary_str

    def test_cpt_calculation_present(self):
        """Verify CPT is calculated deterministically."""
        use_fast, metadata = should_use_fast_path(
            cleaned_soap_note=CLEAN_SOAP_ROUTINE,
            patient_context=PATIENT_CONTEXT_P001,
            chief_complaint=CHIEF_COMPLAINT_ROUTINE,
            visit_context=VISIT_CONTEXT_WITH_EXAM,
        )

        assert use_fast is True
        cpt = metadata["deterministic_cpt"]
        assert cpt is not None
        assert "9921" in cpt  # Should be an established patient CPT code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])