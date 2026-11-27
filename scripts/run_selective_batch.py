#!/usr/bin/env python3
"""
run_selective_batch.py
----------------------
Selective batch evaluation runner for demo patients.

Runs a subset of patients from data/synthetic/internist_patients.json
sequentially and produces a structured comparison report (JSON + Markdown).

Usage:
  python scripts/run_selective_batch.py
  python scripts/run_selective_batch.py --patients P001,P002,P007,P005 --mode fast --warmup --output results/selective_demo_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.orchestrator import run_visit_pipeline
    from utils.config import is_known_synthetic_patient
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    sys.exit(1)


DEFAULT_PATIENTS = ["P001", "P002", "P007", "P005"]


def load_internist_patients() -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "synthetic" / "internist_patients.json"
    with open(path) as f:
        return json.load(f)


def filter_patients(patients: List[Dict[str, Any]], patient_ids: List[str]) -> List[Dict[str, Any]]:
    id_set = set(patient_ids)
    return [p for p in patients if p["patient_id"] in id_set]


def run_single_patient(patient: Dict[str, Any], mode: str = "fast", warmup: bool = False) -> Dict[str, Any]:
    """Run pipeline for one patient and extract structured metrics."""
    if warmup:
        # Run once without recording
        run_visit_pipeline(
            patient_context=patient["patient_context"],
            chief_complaint=patient["chief_complaint"],
            retrieved_notes="",
            mode=mode,
            patient_id=patient["patient_id"],
            use_memory=True,
            visit_context=patient["visit_context"],
            specialty=patient.get("specialty", "general_internal_medicine"),
            is_synthetic=True,
        )

    start = time.perf_counter()
    result = run_visit_pipeline(
        patient_context=patient["patient_context"],
        chief_complaint=patient["chief_complaint"],
        retrieved_notes="",
        mode=mode,
        patient_id=patient["patient_id"],
        use_memory=True,
        visit_context=patient["visit_context"],
        specialty=patient.get("specialty", "general_internal_medicine"),
        is_synthetic=True,
    )
    total_seconds = round(time.perf_counter() - start, 2)

    metadata = result.get("metadata") or {}
    timing = metadata.get("timing") or {}
    supervisor = metadata.get("supervisor") or {}
    control_flow = metadata.get("control_flow") or {}
    memory = metadata.get("memory") or {}
    coding = result.get("coding_result") or {}
    soap = result.get("soap_note") or {}

    return {
        "patient_id": patient["patient_id"],
        "patient_name": patient["patient_name"],
        "expected_path": patient["expected_path"],
        "actual_path": metadata.get("coding_path", "unknown"),
        "path_matched": metadata.get("coding_path") == patient["expected_path"],
        "total_seconds": total_seconds,
        "timing": {
            "memory_load_seconds": timing.get("memory_load_seconds"),
            "scribe_seconds": timing.get("scribe_seconds"),
            "cleanup_seconds": timing.get("cleanup_seconds"),
            "coding_seconds": timing.get("coding_seconds"),
            "validation_seconds": timing.get("validation_seconds"),
            "supervisor_seconds": timing.get("supervisor_seconds"),
            "memory_save_seconds": timing.get("memory_save_seconds"),
        },
        "coding": {
            "primary_icd10": coding.get("primary_icd10"),
            "secondary_icd10": coding.get("secondary_icd10", []),
            "cpt_suggestions": coding.get("cpt_suggestions", []),
            "has_cpt_grounding": bool(coding.get("cpt_grounding")),
            "cpt_grounding": coding.get("cpt_grounding"),
        },
        "governance": {
            "retry_count": supervisor.get("retry_count", 0),
            "empty_soap_retries": metadata.get("empty_soap_retries", 0),
            "needs_human_review": supervisor.get("needs_human_review", False),
            "final_decision": control_flow.get("final_decision"),
            "quality_flags": result.get("quality_flags", []),
        },
        "soap_preview": {
            "subjective": soap.get("subjective", "")[:300],
            "objective": soap.get("objective", "")[:300],
            "assessment": soap.get("assessment", "")[:300],
            "plan": soap.get("plan", "")[:300],
        },
        "demo_notes": patient.get("demo_notes", ""),
    }


def generate_markdown_report(results: List[Dict[str, Any]], mode: str) -> str:
    """Generate a clean Markdown comparison table."""
    lines = []
    lines.append(f"# Selective Demo Batch Report")
    lines.append(f"**Mode:** {mode}  ")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ")
    lines.append(f"**Patients evaluated:** {len(results)}  ")
    lines.append("")

    # Summary table
    lines.append("## Summary Comparison")
    lines.append("")
    lines.append("| Patient | Name | Expected | Actual | Match | Total (s) | Scribe (s) | Coding (s) | Primary ICD-10 | Secondary | CPT | Flags | Human Review |")
    lines.append("|---------|------|----------|--------|-------|-----------|------------|------------|----------------|-----------|-----|-------|--------------|")

    for r in results:
        expected = r["expected_path"]
        actual = r["actual_path"]
        match = "✅" if r["path_matched"] else "❌ **MISMATCH**"
        secondary = "; ".join(r["coding"]["secondary_icd10"]) if r["coding"]["secondary_icd10"] else "—"
        cpt = r["coding"]["cpt_suggestions"][0] if r["coding"]["cpt_suggestions"] else "—"
        flags = len(r["governance"]["quality_flags"])
        human = "⚠️" if r["governance"]["needs_human_review"] else "—"

        lines.append(f"| {r['patient_id']} | {r['patient_name']} | {expected} | {actual} | {match} | "
                     f"{r['total_seconds']:.1f} | {r['timing'].get('scribe_seconds', 0):.1f} | "
                     f"{r['timing'].get('coding_seconds', 0):.1f} | {r['coding']['primary_icd10'] or '—'} | "
                     f"{secondary} | {cpt} | {flags} | {human} |")

    lines.append("")
    lines.append("## Detailed Per-Patient Results")
    lines.append("")

    for r in results:
        lines.append(f"### {r['patient_id']} — {r['patient_name']}")
        lines.append(f"**Expected Path:** {r['expected_path']}  ")
        lines.append(f"**Actual Path:** {r['actual_path']}  ")
        lines.append(f"**Path Match:** {'✅ Yes' if r['path_matched'] else '❌ No'}  ")
        lines.append(f"**Demo Notes:** {r['demo_notes']}  ")
        lines.append("")

        lines.append("#### Timing")
        lines.append(f"- **Total:** {r['total_seconds']:.1f}s  ")
        for k, v in r["timing"].items():
            if v is not None:
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v:.1f}s  ")
        lines.append("")

        lines.append("#### Coding Results")
        lines.append(f"- **Primary ICD-10:** {r['coding']['primary_icd10'] or '—'}  ")
        if r["coding"]["secondary_icd10"]:
            for sec in r["coding"]["secondary_icd10"]:
                lines.append(f"- **Secondary ICD-10:** {sec}  ")
        else:
            lines.append(f"- **Secondary ICD-10:** —  ")
        if r["coding"]["cpt_suggestions"]:
            for cpt in r["coding"]["cpt_suggestions"]:
                lines.append(f"- **CPT:** {cpt}  ")
        else:
            lines.append(f"- **CPT:** —  ")
        lines.append(f"- **CPT Grounding:** {'✅ Present' if r['coding']['has_cpt_grounding'] else '❌ Missing'}  ")
        if r["coding"]["cpt_grounding"]:
            cg = r["coding"]["cpt_grounding"]
            lines.append(f"  - MDM Level: {cg.get('overall_mdm_level', '—')}  ")
            lines.append(f"  - Justification: {cg.get('justification', '—')[:200]}  ")
        lines.append("")

        lines.append("#### Governance & Safety")
        g = r["governance"]
        lines.append(f"- **Retry Count:** {g['retry_count']}  ")
        lines.append(f"- **Empty SOAP Retries:** {g['empty_soap_retries']}  ")
        lines.append(f"- **Needs Human Review:** {'⚠️ Yes' if g['needs_human_review'] else 'No'}  ")
        lines.append(f"- **Final Decision:** {g['final_decision'] or '—'}  ")
        if g["quality_flags"]:
            lines.append(f"- **Quality Flags ({len(g['quality_flags'])}):**")
            for flag in g["quality_flags"]:
                lines.append(f"  - {flag}  ")
        else:
            lines.append(f"- **Quality Flags:** None  ")
        lines.append("")

        lines.append("#### SOAP Preview (truncated)")
        for section in ["subjective", "objective", "assessment", "plan"]:
            text = r["soap_preview"][section]
            if text:
                lines.append(f"**{section.title()}:** {text}...  ")
            else:
                lines.append(f"**{section.title()}:** —  ")
        lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective demo batch evaluation")
    parser.add_argument("--patients", default=",".join(DEFAULT_PATIENTS),
                        help=f"Comma-separated patient IDs (default: {','.join(DEFAULT_PATIENTS)})")
    parser.add_argument("--mode", default="fast", choices=["fast", "quality"], help="LLM mode")
    parser.add_argument("--warmup", action="store_true", help="Run one discarded warmup iteration per patient")
    parser.add_argument("--output", default="results/selective_demo_report.json", help="Output JSON path")
    args = parser.parse_args()

    patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]

    all_patients = load_internist_patients()
    selected = filter_patients(all_patients, patient_ids)

    if not selected:
        print(f"ERROR: No matching patients found for IDs: {patient_ids}")
        sys.exit(1)

    found_ids = [p["patient_id"] for p in selected]
    missing = set(patient_ids) - set(found_ids)
    if missing:
        print(f"WARNING: Patients not found: {missing}")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting selective batch")
    print(f"Patients: {found_ids}")
    print(f"Mode: {args.mode}")
    print(f"Warmup: {args.warmup}")
    print()

    results = []
    for i, patient in enumerate(selected, 1):
        print(f"--- {i}/{len(selected)}: {patient['patient_id']} ({patient['patient_name']}) ---")
        try:
            res = run_single_patient(patient, mode=args.mode, warmup=args.warmup)
            results.append(res)
            status = "✅" if res["path_matched"] else "❌ MISMATCH"
            print(f"  → {res['total_seconds']:.1f}s total, path={res['actual_path']} ({status}), "
                  f"scribe={res['timing'].get('scribe_seconds', 0):.1f}s, "
                  f"flags={len(res['governance']['quality_flags'])}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "error": str(e),
            })

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    report = {
        "benchmark_meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "warmup": args.warmup,
            "patient_ids": found_ids,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved to: {output_path}")

    # Write Markdown report
    md_path = output_path.with_suffix(".md")
    md_content = generate_markdown_report(results, args.mode)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_path}")

    # Print summary
    mismatches = [r for r in results if not r.get("path_matched", True)]
    if mismatches:
        print(f"\n⚠️  PATH MISMATCHES: {len(mismatches)}")
        for m in mismatches:
            print(f"  {m['patient_id']}: expected={m['expected_path']}, actual={m['actual_path']}")
    else:
        print(f"\n✅ All {len(results)} patients matched expected path.")


if __name__ == "__main__":
    main()