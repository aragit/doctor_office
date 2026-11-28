"""Regression tests for demo patient visit-context data flow.

The "Load Patient" button loads `visit_context` (today's encounter data) into
the UI widgets, and the Run Pipeline button passes that dict to the Scribe
pipeline. These tests lock in that the data survives all the way into the
prompt block the Scribe agent actually sees - a truncation regression there
made the Objective say "Not documented today" even when vitals/labs were
loaded.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.visit_context import build_visit_context_block  # noqa: E402

FIXTURE = ROOT / "data" / "synthetic" / "internist_patients.json"


def test_all_demo_patient_vitals_and_findings_reach_prompt_block():
    """Every demo patient's vitals, HPI, exam and labs must survive rendering."""
    patients = json.loads(FIXTURE.read_text())
    assert patients

    for patient in patients:
        block = build_visit_context_block(patient["visit_context"])
        assert "[visit context truncated]" not in block, patient["patient_id"]
        for key in ("blood_pressure", "heart_rate", "temperature",
                    "respiratory_rate", "spo2", "weight_kg", "bmi"):
            assert str(patient["visit_context"][key]) in block, (patient["patient_id"], key)
        assert patient["visit_context"]["hpi"] in block, patient["patient_id"]
        assert patient["visit_context"]["physical_exam"] in block, patient["patient_id"]
        assert patient["visit_context"]["additional_findings"] in block, patient["patient_id"]


def test_p001_labs_and_vitals_are_grounded():
    """P001's specific values must be present in the rendered prompt block."""
    patients = json.loads(FIXTURE.read_text())
    p001 = next(p for p in patients if p["patient_id"] == "P001")
    block = build_visit_context_block(p001["visit_context"])

    assert "150/94" in block
    assert "78 bpm" in block
    assert "98.2" in block
    assert "95.3" in block
    assert "31.2" in block
    assert "HbA1c 7.8%" in block
    assert "Na 139" in block
    assert "LDL 118" in block


def test_empty_visit_context_still_yields_no_block():
    """No encounter data means no block - 'Not documented today' stays accurate."""
    assert build_visit_context_block({}) == ""
    assert build_visit_context_block(None) == ""
    assert build_visit_context_block({"hpi": "   ", "blood_pressure": ""}) == ""
