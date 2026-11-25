import time
import os

os.environ["ALLOW_REAL_PHI"] = "false"

# Ensure src is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.langgraph_orchestrator import run_visit_pipeline

patient_context = """
Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence.
"""

start = time.time()
result = run_visit_pipeline(
    patient_context=patient_context,
    chief_complaint="Follow-up for diabetes and hypertension",
    retrieved_notes="BP 148/92, HbA1c 7.8%, fatigue, dizziness",
    patient_id="P001",
    is_synthetic=True,
    mode="fast"
)
end = time.time()

print(f"TOTAL RUNTIME: {end - start:.1f} seconds")

# Print the SOAP Note text to check quality
print("\n=== SUBJECTIVE ===")
print(result.get("soap_note", {}).get("subjective", "MISSING"))

print("\n=== OBJECTIVE ===")
print(result.get("soap_note", {}).get("objective", "MISSING"))

print("\n=== ASSESSMENT ===")
print(result.get("soap_note", {}).get("assessment", "MISSING"))

print("\n=== PLAN ===")
print(result.get("soap_note", {}).get("plan", "MISSING"))

# Print the Codes
print("\n=== ICD-10 CODES ===")
print(f"Primary: {result.get('coding_result', {}).get('primary_icd10', 'MISSING')}")
print(f"Secondary: {result.get('coding_result', {}).get('secondary_icd10', [])}")

print("\n=== CPT CODES ===")
print(result.get("coding_result", {}).get("cpt_suggestions", []))

print("\n=== QUALITY FLAGS ===")
for flag in result.get("quality_flags", []):
    print(f"  - {flag}")

print("\n=== METADATA ===")
meta = result.get("metadata", {})
print(f"Mode: {meta.get('mode')}")
print(f"Steps completed: {meta.get('steps_completed')}")
print(f"Timing: {meta.get('timing')}")
print(f"Total seconds: {meta.get('timing', {}).get('total_seconds', 'N/A')}")
print(f"Supervisor retry_count: {meta.get('supervisor', {}).get('retry_count')}")
print(f"Supervisor needs_human_review: {meta.get('supervisor', {}).get('needs_human_review')}")
print(f"Supervisor applied: {meta.get('supervisor', {}).get('applied')}")