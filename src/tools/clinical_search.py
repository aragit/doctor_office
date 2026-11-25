"""
Clinical note search tool - wraps the existing RAG vector store.

Two retrieval paths are combined on purpose:

1. **Vector similarity** over `data/notes` (the existing `rag.simple_rag`
   store) for general clinical reference material.
2. **Patient memory** lookup, so a query can be answered with facts specific
   to the patient when a `patient_id` is supplied.

The memory path is what makes this more than the old inline
`similarity_search` call: the agents no longer need the raw note string passed
in through state, they can ask for what they need.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DESCRIPTION_SEARCH = (
    "Search clinical reference notes (RAG) for a query. Optionally scope the "
    "answer with a patient_id to also include that patient's stored history. "
    "Returns a dict with 'notes' (reference text) and 'patient_context'."
)


def search_clinical_notes(
    query: str,
    patient_id: Optional[str] = None,
    k: int = 1,
) -> Dict[str, Any]:
    """Search the clinical reference vector store.

    Args:
        query: Natural-language search query, e.g. "diabetes management".
        patient_id: Optional patient id. When given, that patient's stored
            summary and recent visits are included alongside the RAG hits.
        k: Number of reference chunks to retrieve.

    Returns:
        Dict with:
        - notes: concatenated reference text (empty string if RAG unavailable)
        - hits: number of reference chunks returned
        - patient_context: compact memory context for `patient_id`
        - patient_found: bool
        - error: present only when retrieval failed
    """
    result: Dict[str, Any] = {
        "query": query,
        "patient_id": patient_id,
        "notes": "",
        "hits": 0,
        "patient_context": "",
        "patient_found": False,
    }

    # -- patient memory context (never fails the tool) --------------------
    if patient_id:
        try:
            from memory.patient_memory import get_memory_store

            store = get_memory_store()
            summary = store.get_summary(patient_id)
            if summary:
                current = summary.get("current", {})
                result["patient_found"] = True

                def _or_none(values: Any) -> str:
                    values = list(values or [])
                    return ", ".join(values) if values else "none documented"

                lines = [
                    f"Patient: {summary.get('name') or patient_id}",
                    f"Conditions: {_or_none(current.get('conditions'))}",
                    f"Medications: {_or_none(current.get('medications'))}",
                    f"Allergies: {_or_none(current.get('allergies'))}",
                ]
                history_block = store.build_context_block(patient_id)
                if history_block:
                    lines.append("")
                    lines.append(history_block)
                result["patient_context"] = "\n".join(lines)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Patient memory lookup failed for %s: %s", patient_id, exc)
            result["error"] = f"memory lookup failed: {exc}"

    # -- vector retrieval ---------------------------------------------------
    if query and query.strip():
        try:
            from rag.simple_rag import get_vectorstore

            docs = get_vectorstore().similarity_search(query, k=k)
            result["notes"] = "\n\n".join(doc.page_content for doc in docs)
            result["hits"] = len(docs)
        except Exception as exc:
            # A missing/empty vector store must not break a visit; the agents
            # already handle empty retrieved notes.
            logger.warning("Clinical note search failed for %r: %s", query, exc)
            result.setdefault("error", f"rag unavailable: {exc}")

    return result


def format_search_result(result: Dict[str, Any], char_limit: int = 300) -> str:
    """Render a search result as a compact prompt block."""
    blocks: List[str] = []
    if result.get("patient_context"):
        blocks.append(result["patient_context"])
    if result.get("notes"):
        notes = result["notes"][:char_limit]
        blocks.append("CLINICAL REFERENCE NOTES:\n" + notes)
    return "\n\n".join(blocks)


__all__ = ["search_clinical_notes", "format_search_result", "DESCRIPTION_SEARCH"]
