"""
Helper module to load synthetic Internist patient dataset.
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def load_internist_patients() -> List[Dict[str, Any]]:
    """
    Load the synthetic Internist patient dataset.

    Returns:
        List of 10 patient dictionaries with fields:
        - patient_id, patient_name, is_synthetic, specialty, chief_complaint
        - patient_context, visit_context, expected_path, demo_notes
    """
    data_path = Path(__file__).parent / "internist_patients.json"
    with open(data_path, "r") as f:
        return json.load(f)


def get_fast_path_patients() -> List[Dict[str, Any]]:
    """Return only patients expected to take Fast Path."""
    return [p for p in load_internist_patients() if p["expected_path"] == "fast"]


def get_slow_path_patients() -> List[Dict[str, Any]]:
    """Return only patients expected to take Slow Path."""
    return [p for p in load_internist_patients() if p["expected_path"] == "slow"]


def get_patient_by_id(patient_id: str) -> Dict[str, Any]:
    """Get a specific patient by ID (e.g., 'P001')."""
    for p in load_internist_patients():
        if p["patient_id"] == patient_id:
            return p
    raise ValueError(f"Patient {patient_id} not found")


if __name__ == "__main__":
    patients = load_internist_patients()
    print(f"Loaded {len(patients)} patients:")
    for p in patients:
        print(f"  {p['patient_id']} {p['patient_name']:20s} -> {p['expected_path'].upper()} : {p['demo_notes']}")