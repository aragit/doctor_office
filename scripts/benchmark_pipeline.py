#!/usr/bin/env python3
"""
benchmark_pipeline.py
---------------------
Reproducible latency benchmark for the Doctor Office MVP.

Focus case (from screenshots – the 333.6 s run):
  Patient : John Doe (P001)
  Specialty: General / Internal Medicine
  Chief complaint + visit context matching the fatigue / DM / HTN case.

Usage:
  # From project root (with the app dependencies installed)
  python scripts/benchmark_pipeline.py

  # Or with custom iterations / mode
  python scripts/benchmark_pipeline.py --iterations 3 --mode fast

  # Save raw results
  python scripts/benchmark_pipeline.py --output results/benchmark_333s_case.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helper: extract CPT code from coding_result (handles both Fast/Slow path)
# ---------------------------------------------------------------------------
def _extract_cpt_summary(coding_result: Dict[str, Any]) -> Optional[str]:
    """Extract CPT code/description from coding_result dict.

    Handles multiple possible keys:
    - cpt_suggestions: list of "CODE - Description" (both paths)
    - cpt_code: just the code number (Fast Path decision metadata)
    - cpt_description: full "CODE - Description" (Fast Path decision metadata)
    """
    if not coding_result:
        return None

    # Slow Path / standard: cpt_suggestions is a list of "CODE - Description"
    cpt_suggestions = coding_result.get("cpt_suggestions")
    if cpt_suggestions and isinstance(cpt_suggestions, list) and len(cpt_suggestions) > 0:
        return cpt_suggestions[0]

    # Fast Path decision metadata may have these directly
    if coding_result.get("cpt_description"):
        return coding_result["cpt_description"]
    if coding_result.get("cpt_code"):
        return coding_result["cpt_code"]

    return None


# ---------------------------------------------------------------------------
# Make sure we can import from src/ when running from project root or scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.orchestrator import run_visit_pipeline
    from utils.config import get_model_name, LLM_BASE_URL, LLM_MODEL_NAME
except ImportError as e:
    print("ERROR: Could not import project modules.")
    print("Make sure you run this from the project root and have installed dependencies:")
    print("  pip install -r requirements.txt   (or  pip install -e .)")
    print(f"Import error: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Fixed benchmark case – matches the screenshot that took 333.6 s
# ---------------------------------------------------------------------------
BENCHMARK_CASE = {
    "patient_id": "P001",
    "patient_name": "John Doe",
    "is_synthetic": True,
    "specialty": "general_internal_medicine",  # adjust if your specialty id differs
    "chief_complaint": "Follow-up for diabetes and hypertension. Fatigue for 3 weeks.",
    "patient_context": """
Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence.
""".strip(),
    "visit_context": {
        "hpi": "Fatigue for 3 weeks. Mild to moderate. No associated chest pain, shortness of breath, or other symptoms reported.",
        "blood_pressure": "148/92",
        "heart_rate": "78",
        "temperature": "98.4",
        "respiratory_rate": "16",
        "spo2": "97",
        "weight_kg": "72.5",
        "bmi": "26.4",
        "physical_exam": "Dry mucous membranes. No peripheral edema. Foot skin intact, sensation intact to monofilament.",
        "additional_findings": "HbA1c 7.8% (was 7.2). Urine microalbumin 42 mg/g. CXR: no infiltrate.",
        "vitals_note": "Seated BP, repeat 148/90. Pain 3/10 with movement.",
    },
    "retrieved_notes": "",
    "mode": "fast",
    "use_memory": True,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_single_iteration(
    mode: str = "fast",
    use_memory: bool = True,
) -> Dict[str, Any]:
    """Run one full pipeline iteration and return structured timing + metadata."""
    start = time.perf_counter()

    result = run_visit_pipeline(
        patient_context=BENCHMARK_CASE["patient_context"],
        chief_complaint=BENCHMARK_CASE["chief_complaint"],
        retrieved_notes=BENCHMARK_CASE["retrieved_notes"],
        mode=mode,
        patient_id=BENCHMARK_CASE["patient_id"],
        use_memory=use_memory,
        visit_context=BENCHMARK_CASE["visit_context"],
        specialty=BENCHMARK_CASE.get("specialty"),
        is_synthetic=True,
    )

    total_seconds = round(time.perf_counter() - start, 2)

    metadata = result.get("metadata") or {}
    timing = metadata.get("timing") or {}
    supervisor = metadata.get("supervisor") or {}
    control_flow = metadata.get("control_flow") or {}
    symptom_coding = metadata.get("symptom_coding") or {}
    memory = metadata.get("memory") or {}

    usage = (
        metadata.get("token_usage")
        or metadata.get("usage")
        or result.get("usage")
        or {}
    )

    return {
        "total_seconds": total_seconds,
        "timing": {
            "total_seconds": total_seconds,
            "memory_load_seconds": timing.get("memory_load_seconds"),
            "scribe_seconds": timing.get("scribe_seconds"),
            "cleanup_seconds": timing.get("cleanup_seconds"),
            "coding_seconds": timing.get("coding_seconds"),
            "validation_seconds": timing.get("validation_seconds"),
            "supervisor_seconds": timing.get("supervisor_seconds"),
            "memory_save_seconds": timing.get("memory_save_seconds"),
            "human_review_seconds": timing.get("human_review_seconds"),
        },
        "model_used": metadata.get("model_used") or get_model_name(mode),
        "mode": mode,
        "pipeline_version": metadata.get("pipeline_version"),
        "steps_completed": metadata.get("steps_completed") or [],
        "quality_flags": result.get("quality_flags") or [],
        "supervisor": {
            "retry_count": supervisor.get("retry_count", 0),
            "needs_human_review": supervisor.get("needs_human_review", False),
            "applied": supervisor.get("applied") or [],
            "feedback": supervisor.get("feedback") or [],
        },
        "control_flow": {
            "auto_fix_applied": control_flow.get("auto_fix_applied", False),
            "auto_fix_rounds": control_flow.get("auto_fix_rounds", 0),
            "revalidations": control_flow.get("revalidations", 0),
            "final_route": control_flow.get("final_route"),
            "final_decision": control_flow.get("final_decision"),
            "escalation_reason": control_flow.get("escalation_reason"),
        },
        "symptom_coding": {
            "candidates": symptom_coding.get("candidates") or [],
            "coding_added": symptom_coding.get("coding_added") or [],
            "supervisor_added": symptom_coding.get("supervisor_added") or [],
        },
        "memory": {
            "patient_id": memory.get("patient_id"),
            "visit_id": memory.get("visit_id"),
            "visit_status": memory.get("visit_status"),
            "approved_visit_count": memory.get("approved_visit_count"),
            "draft_visit_count": memory.get("draft_visit_count"),
            "memory_context_used": memory.get("memory_context_used"),
        },
        "token_usage": usage,
        "coding_result_summary": {
            "primary_icd10": (result.get("coding_result") or {}).get("primary_icd10"),
            "secondary_count": len((result.get("coding_result") or {}).get("secondary_icd10") or []),
            "cpt": _extract_cpt_summary(result.get("coding_result") or {}),
        },
        "raw_result_keys": sorted(result.keys()),
    }


def print_summary(results: List[Dict[str, Any]], case_name: str) -> None:
    """Pretty-print a human-readable summary."""
    totals = [r["total_seconds"] for r in results]
    scribe_times = [r["timing"].get("scribe_seconds") or 0 for r in results]
    coding_times = [r["timing"].get("coding_seconds") or 0 for r in results]

    print("\n" + "=" * 72)
    print(f"BENCHMARK SUMMARY — {case_name}")
    print("=" * 72)
    print(f"Iterations          : {len(results)}")
    print(f"Model / Mode        : {results[0].get('model_used')} / {results[0].get('mode')}")
    print(f"LLM_BASE_URL        : {LLM_BASE_URL}")
    print(f"LLM_MODEL_NAME      : {LLM_MODEL_NAME}")
    print("-" * 72)
    print(f"Total time (s)      : median={statistics.median(totals):.1f}  "
          f"min={min(totals):.1f}  max={max(totals):.1f}  mean={statistics.mean(totals):.1f}")
    print(f"Scribe time (s)     : median={statistics.median(scribe_times):.1f}  "
          f"min={min(scribe_times):.1f}  max={max(scribe_times):.1f}")
    print(f"Coding time (s)     : median={statistics.median(coding_times):.1f}  "
          f"min={min(coding_times):.1f}  max={max(coding_times):.1f}")
    print("-" * 72)

    last = results[-1]
    print("Last run details:")
    print(f"  Quality flags     : {len(last['quality_flags'])} → {last['quality_flags'][:3]}{'...' if len(last['quality_flags']) > 3 else ''}")
    print(f"  Supervisor retries: {last['supervisor']['retry_count']}")
    print(f"  Needs human review: {last['supervisor']['needs_human_review']}")
    print(f"  Auto-fix applied  : {last['control_flow']['auto_fix_applied']} "
          f"(rounds={last['control_flow']['auto_fix_rounds']})")
    print(f"  Final decision    : {last['control_flow'].get('final_decision')}")
    print(f"  Primary ICD-10    : {last['coding_result_summary'].get('primary_icd10')}")
    print(f"  CPT               : {last['coding_result_summary'].get('cpt')}")
    print(f"  Visit status      : {last['memory'].get('visit_status')} "
          f"(drafts={last['memory'].get('draft_visit_count')}, "
          f"approved={last['memory'].get('approved_visit_count')})")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Doctor Office latency benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--mode", default="fast", choices=["fast", "quality"], help="LLM mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable long-term memory")
    parser.add_argument("--output", type=str, default=None, help="Save full JSON results to this path")
    parser.add_argument("--warmup", action="store_true", help="Run one discarded warmup iteration first")
    args = parser.parse_args()

    print(f"[{_now_iso()}] Starting benchmark")
    print(f"Case          : {BENCHMARK_CASE['patient_name']} ({BENCHMARK_CASE['patient_id']})")
    print(f"Chief complaint: {BENCHMARK_CASE['chief_complaint'][:80]}...")
    print(f"Iterations    : {args.iterations}")
    print(f"Mode          : {args.mode}")
    print(f"Use memory    : {not args.no_memory}")
    print(f"LLM endpoint  : {LLM_BASE_URL}")
    print()

    if args.warmup:
        print("Running warmup iteration (discarded)...")
        try:
            run_single_iteration(mode=args.mode, use_memory=not args.no_memory)
            print("Warmup done.\n")
        except Exception as e:
            print(f"Warmup failed (continuing anyway): {e}\n")

    results: List[Dict[str, Any]] = []
    for i in range(1, args.iterations + 1):
        print(f"--- Iteration {i}/{args.iterations} ---")
        try:
            res = run_single_iteration(mode=args.mode, use_memory=not args.no_memory)
            results.append(res)
            print(f"  → {res['total_seconds']:.1f}s  "
                  f"(scribe={res['timing'].get('scribe_seconds')}s, "
                  f"coding={res['timing'].get('coding_seconds')}s)")
        except Exception as e:
            print(f"  ERROR on iteration {i}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No successful iterations. Aborting.")
        sys.exit(1)

    print_summary(results, case_name="P001 Diabetes+HTN+Fatigue (333.6s reference case)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark_meta": {
                "timestamp": _now_iso(),
                "case": BENCHMARK_CASE,
                "iterations": args.iterations,
                "mode": args.mode,
                "use_memory": not args.no_memory,
                "llm_base_url": LLM_BASE_URL,
                "llm_model_name": LLM_MODEL_NAME,
            },
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
