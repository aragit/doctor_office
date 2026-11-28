#!/usr/bin/env python3
"""
Diagnose latency anomaly - check if soap_empty retry loop caused 283s runtime.

This script runs the pipeline with a mocked LLM that returns an empty Subjective
but valid Objective/Assessment/Plan, to trigger the soap_empty validation flag
and observe if the retry loop activates.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage
from agents import langgraph_orchestrator as orch
from agents.supervisor_node import MAX_EMPTY_SOAP_RETRIES


def build_empty_subjective_soap():
    """Returns a SOAP note with empty subjective but valid other sections.
    
    NOTE: The validation's soap_empty flag only triggers when ALL 4 sections
    are empty. This test case has only subjective empty, so it should NOT
    trigger the soap_empty retry loop.
    """
    return {
        "subjective": "",  # EMPTY - but other sections have content
        "objective": "BP 138/88. HR 72. BMI 31.2. No acute distress.",
        "assessment": "Type 2 Diabetes and Hypertension follow-up. HbA1c 7.8% indicates suboptimal control. Blood pressure at goal on current regimen. Fatigue likely multifactorial - consider anemia, thyroid, or medication side effect.",
        "plan": "1. Check CBC, TSH, B12 for fatigue workup. 2. Continue Metformin 500mg BID and Lisinopril 10mg daily. 3. Reinforce diet/exercise counseling. 4. Follow-up in 3 months with repeat HbA1c.",
        "icd10_suggestions": [
            "E11.9 - Type 2 diabetes mellitus without complications",
            "I10 - Essential (primary) hypertension"
        ]
    }


def build_fully_empty_soap():
    """Returns a SOAP note with ALL four sections empty.
    
    This SHOULD trigger the soap_empty validation flag and retry loop.
    """
    return {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": "",
        "icd10_suggestions": []
    }


def make_scribe_stub(soap_note):
    """Create a scribe stub that returns the given SOAP note."""
    def scribe_stub(state):
        return {
            "soap_note": soap_note,
            "current_step": "scribe",
            "timing": {"scribe_seconds": 0.0},
            "messages": [AIMessage(content="stub scribe (empty subjective)", name="scribe_node")],
        }
    return scribe_stub


def make_cleanup_stub():
    def cleanup_stub(state):
        soap = state.get("soap_note") or {}
        return {
            "cleaned_soap_note": soap,
            "cleanup_removed": [],
            "cycle_flags": {"cleanup": []},
            "current_step": "cleanup",
            "timing": {"cleanup_seconds": 0.0},
            "messages": [AIMessage(content="stub cleanup", name="cleanup_node")],
        }
    return cleanup_stub

# it2 test 
def make_coding_stub():
    def coding_stub(state):
        return {
            "coding_result": {
                "primary_icd10": "E11.9 - Type 2 diabetes mellitus without complications",
                "secondary_icd10": ["I10 - Essential (primary) hypertension"],
                "cpt_suggestions": ["99213 - Office visit, established patient, low medical decision making"],
                "medical_necessity_note": "CPT 99213 supported by E11.9 (diabetes follow-up).",
            },
            "cycle_flags": {"coding": []},
            "current_step": "coding",
            "timing": {"coding_seconds": 0.0},
            "messages": [AIMessage(content="stub coding", name="coding_node")],
        }
    return coding_stub


def make_memory_load_stub():
    def memory_load_stub(state):
        return {
            "current_step": "memory_load",
            "timing": {"memory_load_seconds": 0.0},
            "patient_id": "P001",
            "messages": [AIMessage(content="stub memory", name="memory_load_node")],
        }
    return memory_load_stub

#main run 
def run_diagnosis():
    """Run the pipeline with mocked LLM to diagnose soap_empty retry behavior."""
    
    print("=" * 70)
    print("LATENCY ANOMALY DIAGNOSIS - soap_empty retry loop check")
    print("=" * 70)
    print(f"MAX_EMPTY_SOAP_RETRIES = {MAX_EMPTY_SOAP_RETRIES}")
    print()
    
    # Test Case 1: Only subjective empty (should NOT trigger soap_empty)
    print("=" * 70)
    print("TEST CASE 1: Only subjective empty (other sections have content)")
    print("=" * 70)
    test_soap_1 = build_empty_subjective_soap()
    print("Test SOAP note:")
    print(f"  subjective: '{test_soap_1['subjective']}' (EMPTY)")
    print(f"  objective:  '{test_soap_1['objective'][:60]}...'")
    print(f"  assessment: '{test_soap_1['assessment'][:60]}...'")
    print(f"  plan:       '{test_soap_1['plan'][:60]}...'")
    print()
    
    result_1 = _run_test(test_soap_1, "Test Case 1 (partial empty)")
    
    # Test Case 2: All four sections empty (SHOULD trigger soap_empty)
    print()
    print("=" * 70)
    print("TEST CASE 2: ALL FOUR sections empty (should trigger soap_empty)")
    print("=" * 70)
    test_soap_2 = build_fully_empty_soap()
    print("Test SOAP note:")
    print(f"  subjective: '{test_soap_2['subjective']}' (EMPTY)")
    print(f"  objective:  '{test_soap_2['objective']}' (EMPTY)")
    print(f"  assessment: '{test_soap_2['assessment']}' (EMPTY)")
    print(f"  plan:       '{test_soap_2['plan']}' (EMPTY)")
    print()
    
    result_2 = _run_test(test_soap_2, "Test Case 2 (fully empty)")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test Case 1 (partial empty): empty_soap_retries = {result_1['empty_soap_retries']}, retry_count = {result_1['retry_count']}")
    print(f"Test Case 2 (fully empty):   empty_soap_retries = {result_2['empty_soap_retries']}, retry_count = {result_2['retry_count']}")
    print()
    
    if result_2['empty_soap_retries'] > 0:
        print("✅ CONFIRMED: Fully empty SOAP triggers soap_empty retry loop")
        print("   This could explain the 283s runtime anomaly if the LLM produced empty notes")
    else:
        print("❌ NOT CONFIRMED: Even fully empty SOAP did not trigger retry loop")
    
    print("=" * 70)
    
    return result_2  # Return the more critical test case


def _run_test(soap_note, test_name):
    """Run a single test with the given SOAP note."""
    
    # Save original functions
    original = {
        name: getattr(orch, name) for name in
        ("memory_load_node", "scribe_node", "cleanup_node", "coding_node",
         "validation_node", "get_graph", "supervisor_node")
    }
    
    try:
        # Install stubs
        orch.memory_load_node = make_memory_load_stub()
        orch.scribe_node = make_scribe_stub(soap_note)
        orch.cleanup_node = make_cleanup_stub()
        orch.coding_node = make_coding_stub()
        # Keep REAL validation_node and supervisor_node to see actual routing
        orch.get_graph = lambda: orch.build_graph()
        orch._GRAPH = None
        
        # Patient context for P001 (John Doe, Type 2 Diabetes + Hypertension)
        patient_context = (
            "Patient: John Doe, 58yo male\n"
            "Conditions: Type 2 Diabetes, Hypertension\n"
            "Medications: Metformin 500mg BID, Lisinopril 10mg daily\n"
            "Allergies: Penicillin\n"
            "Last Visit: Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence."
        )
        
        chief_complaint = "Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
        
        print(f"Running {test_name}...")
        start = time.time()
        
        result = orch.run_visit_pipeline(
            patient_context=patient_context,
            chief_complaint=chief_complaint,
            retrieved_notes="",
            mode="fast",
            patient_id="P001",
            use_memory=False,
            is_synthetic=True,
        )
        
        elapsed = time.time() - start
        print(f"Pipeline completed in {elapsed:.1f}s (mocked)")
        print()
        
        # Extract key metadata
        meta = result["metadata"]
        supervisor = meta["supervisor"]
        control_flow = meta["control_flow"]
        steps = meta["steps_completed"]
        
        print("NODE EXECUTION SEQUENCE:")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        print()
        
        print("SUPERVISOR RETRY STATE:")
        print(f"   retry_count:           {supervisor['retry_count']}")
        print(f"   empty_soap_retries:    {control_flow.get('empty_soap_retries', 'NOT IN CONTROL_FLOW')}")
        print(f"   needs_human_review:    {supervisor['needs_human_review']}")
        print(f"   final_route:           {control_flow['final_route']}")
        print(f"   final_decision:        {control_flow['final_decision']}")
        print(f"   escalation_reason:     {control_flow['escalation_reason']}")
        print()
        
        print("CYCLE FLAG RESETS:")
        print(f"   cycle_flag_resets:     {control_flow['cycle_flag_resets']}")
        print(f"   cleared_cycle_flags:   {control_flow['cleared_cycle_flags']}")
        print()
        
        print("QUALITY FLAGS RAISED:")
        for flag in result["quality_flags"]:
            print(f"   - {flag}")
        print()
        
        empty_soap_retries = control_flow.get('empty_soap_retries', 0)
        
        return {
            "empty_soap_retries": empty_soap_retries,
            "retry_count": supervisor['retry_count'],
            "steps_completed": steps,
            "final_route": control_flow['final_route'],
            "final_decision": control_flow['final_decision'],
            "quality_flags": result["quality_flags"],
        }
        
    finally:
        # Restore original functions
        for name, value in original.items():
            setattr(orch, name, value)
        orch._GRAPH = None


if __name__ == "__main__":
    result = run_diagnosis()
    sys.exit(0 if result["empty_soap_retries"] > 0 else 1)