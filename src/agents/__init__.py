"""
Agents Package - Medical AI Agents for Clinical Documentation

Available agents:
- ScribeAgent: Generates structured SOAP notes from clinical encounters
- CodingAgent: Refines ICD-10-CM and suggests CPT codes from SOAP notes
- LangGraph orchestrator: Central Executor that runs the graph
  (memory_load -> scribe -> cleanup -> coding -> validation -> supervisor
  -> memory_save | retry | human_review)

The orchestrator owns the order of work; agents never call one another. Each
node runs a scoped toolset from `tools.registry`, and the zero-LLM Supervisor
node routes retries and human-review escalation after validation.
"""

from agents.scribe_agent import (
    SOAPNote,
    generate_soap_note,
    create_scribe_chain,
)

from agents.coding_agent import (
    CodingResult,
    generate_coding,
    create_coding_chain,
)

from agents.graph_state import (
    DETERMINISTIC_STEPS,
    CYCLE_FLAG_BUCKETS,
    PIPELINE_STEPS,
    PIPELINE_VERSION,
    AgentState,
    build_initial_state,
    reset_cycle_flags,
)

from agents.memory_node import memory_load_node, memory_save_node
from agents.scribe_node import build_grounding_block, scribe_node
from agents.cleanup_node import cleanup_node
from agents.coding_node import build_symptom_hint, coding_node
from agents.validation_node import validation_node
from agents.supervisor_node import (
    MAX_RETRIES,
    MAX_REVALIDATIONS,
    determine_route,
    human_review_node,
    supervisor_node,
    route_after_validation,
)

from agents.langgraph_orchestrator import (
    build_graph,
    get_graph,
    graph_mermaid,
    run_demo,
    run_visit_pipeline,
    stream_visit_pipeline,
)

__all__ = [
    # Scribe Agent
    "SOAPNote",
    "generate_soap_note",
    "create_scribe_chain",
    # Coding Agent
    "CodingResult",
    "generate_coding",
    "create_coding_chain",
    # Graph state
    "AgentState",
    "build_initial_state",
    "PIPELINE_STEPS",
    "DETERMINISTIC_STEPS",
    "PIPELINE_VERSION",
    "CYCLE_FLAG_BUCKETS",
    "reset_cycle_flags",
    # Graph nodes
    "memory_load_node",
    "scribe_node",
    "cleanup_node",
    "coding_node",
    "validation_node",
    "supervisor_node",
    "memory_save_node",
    "human_review_node",
    # Supervisor
    "MAX_RETRIES",
    "MAX_REVALIDATIONS",
    "determine_route",
    "route_after_validation",
    # Node helpers
    "build_grounding_block",
    "build_symptom_hint",
    # Orchestrator
    "run_visit_pipeline",
    "stream_visit_pipeline",
    "run_demo",
    "build_graph",
    "get_graph",
    "graph_mermaid",
]
