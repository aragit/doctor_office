import sys
import time
sys.path.insert(0, 'src')

from data.synthetic.load_patients import load_internist_patients
from agents.orchestrator import run_visit_pipeline

patients = load_internist_patients()
test_ids = ['P001', 'P002', 'P003', 'P004']

for pid in test_ids:
    p = [p for p in patients if p['patient_id'] == pid][0]
    print(f'=== {pid} {p["patient_name"]} ===')
    print(f'Expected Path: {p["expected_path"].upper()}')
    
    start = time.time()
    result = run_visit_pipeline(
        patient_context=p['patient_context'],
        chief_complaint=p['chief_complaint'],
        retrieved_notes='',
        mode='fast',
        patient_id=p['patient_id'],
        use_memory=False,
        visit_context=p['visit_context'],
        is_synthetic=True,
    )
    elapsed = time.time() - start
    
    meta = result.get('metadata', {}) or {}
    supervisor = meta.get('supervisor', {}) or {}
    coding = result.get('coding_result', {}) or {}
    quality_flags = result.get('quality_flags', []) or []
    
    print(f'Time: {elapsed:.1f}s')
    print(f'Supervisor Decision: {supervisor.get("decision", "N/A")}')
    print(f'Primary ICD-10: {coding.get("primary_icd10")}')
    print(f'Secondary ICD-10: {coding.get("secondary_icd10")}')
    print(f'CPT: {coding.get("cpt_suggestions", ["N/A"])[0]}')
    print(f'Quality Flags: {len(quality_flags)}')
    for flag in quality_flags:
        print(f'  - {flag[:120]}')
    print(f'Retries: {supervisor.get("retry_count", 0)}')
    print(f'Needs Human Review: {supervisor.get("needs_human_review", False)}')
    print()