"""
Orchestrator - backward-compatible entry point.

The sequential implementation that used to live here now runs on LangGraph.
This module is kept as a thin re-export shim so existing callers
(`from agents.orchestrator import run_visit_pipeline`, Streamlit UI, scripts)
continue to work without modification.

New code should import from `agents.langgraph_orchestrator`.
"""

from agents.langgraph_orchestrator import (
    PIPELINE_VERSION,
    build_graph,
    get_graph,
    graph_mermaid,
    run_demo,
    run_visit_pipeline,
    stream_visit_pipeline,
)

__all__ = [
    "PIPELINE_VERSION",
    "build_graph",
    "get_graph",
    "graph_mermaid",
    "run_demo",
    "run_visit_pipeline",
    "stream_visit_pipeline",
]

if __name__ == "__main__":
    print("[orchestrator] Delegating to the LangGraph orchestrator...\n")
    run_demo()
