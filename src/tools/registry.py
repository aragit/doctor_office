"""
Tool registry - the single place that decides who may call what.

Design notes
------------
The Central Executor pattern is preserved: **the graph decides the order of
work and the nodes decide which tools to run**. Agents never choose tools by
themselves, so a 1.5B model cannot talk itself into skipping a grounding step.

Each tool is defined once as a plain function and wrapped in a LangChain
`StructuredTool`. The plain function is what the nodes call; the
`StructuredTool` wrapper is what a future `llm.bind_tools(...)` step would
use, so upgrading to model-driven tool calling later needs no rewrite.

Tools are partitioned per agent. `get_tools_for_agent("scribe_node")` returns a
scoped list; there is no "all agents can do anything" bucket.
"""

import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from langchain_core.tools import StructuredTool

from tools import coding_tools, patient_tools
from tools.clinical_search import DESCRIPTION_SEARCH, search_clinical_notes
from tools.em_calculator import DESCRIPTION_CALCULATE, calculate_em_code, derive_mdm_domains
from tools.icd_validator import DESCRIPTION_VALIDATE_ICD, validate_icd10_code

logger = logging.getLogger(__name__)

SCRIBE_AGENT = "scribe_node"
CODING_AGENT = "coding_node"
MEMORY_AGENT = "memory_node"
SUPERVISOR_AGENT = "supervisor_node"

# Tools that do not require patient context.
GENERAL_TOOL_NAMES = ("search_clinical_notes",)


class RegisteredTool(NamedTuple):
    """A tool plus the metadata needed to present it to a node."""

    name: str
    func: Callable[..., Any]
    description: str
    langchain_tool: StructuredTool
    takes_patient_context: bool


def _register(name: str, func: Callable[..., Any], description: str, takes_patient_context: bool = True) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        func=func,
        description=description,
        langchain_tool=StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
        ),
        takes_patient_context=takes_patient_context,
    )


TOOLS: Dict[str, RegisteredTool] = {
    tool.name: tool
    for tool in (
        _register(
            "get_patient_conditions",
            patient_tools.get_patient_conditions,
            patient_tools.DESCRIPTION_CONDITIONS,
        ),
        _register(
            "get_patient_medications",
            patient_tools.get_patient_medications,
            patient_tools.DESCRIPTION_MEDICATIONS,
        ),
        _register(
            "get_patient_allergies",
            patient_tools.get_patient_allergies,
            patient_tools.DESCRIPTION_ALLERGIES,
        ),
        _register(
            "get_patient_profile",
            patient_tools.get_patient_profile,
            patient_tools.DESCRIPTION_PROFILE,
        ),
        _register(
            "search_clinical_notes",
            search_clinical_notes,
            DESCRIPTION_SEARCH,
            takes_patient_context=False,
        ),
        _register(
            "validate_icd_against_patient",
            coding_tools.validate_icd_against_patient,
            coding_tools.DESCRIPTION_VALIDATE,
        ),
        _register(
            "simple_symptom_to_icd_suggestion",
            coding_tools.simple_symptom_to_icd_suggestion,
            coding_tools.DESCRIPTION_SYMPTOM,
            takes_patient_context=False,
        ),
        _register(
            "calculate_em_code",
            calculate_em_code,
            DESCRIPTION_CALCULATE,
            takes_patient_context=False,
        ),
        _register(
            "derive_mdm_domains",
            derive_mdm_domains,
            "Derive the three MDM element levels (problems/data/risk) from a "
            "cleaned SOAP note deterministically, LLM-free.",
            takes_patient_context=False,
        ),
        _register(
            "validate_icd10_code",
            validate_icd10_code,
            DESCRIPTION_VALIDATE_ICD,
            takes_patient_context=False,
        ),
    )
}

# Scoped tool access per agent. Tools absent from a node's list are not
# reachable from that node.
AGENT_TOOLS: Dict[str, List[str]] = {
    SCRIBE_AGENT: [
        "get_patient_conditions",
        "get_patient_medications",
        "get_patient_allergies",
        "search_clinical_notes",
    ],
    CODING_AGENT: [
        "get_patient_conditions",
        "validate_icd_against_patient",
        "simple_symptom_to_icd_suggestion",
        "calculate_em_code",
        "derive_mdm_domains",
        "validate_icd10_code",
    ],
    MEMORY_AGENT: ["get_patient_profile", "search_clinical_notes"],
    SUPERVISOR_AGENT: [
        "calculate_em_code",
        "validate_icd10_code",
        "simple_symptom_to_icd_suggestion",
    ],
}


def get_tool(name: str) -> RegisteredTool:
    """Look up a registered tool by name."""
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name}. Available: {sorted(TOOLS)}")
    return TOOLS[name]


def get_tools_for_agent(agent: str) -> List[RegisteredTool]:
    """Return the tools an agent is permitted to use."""
    if agent not in AGENT_TOOLS:
        logger.warning("No tool policy for agent %r; returning no tools", agent)
        return []
    return [TOOLS[name] for name in AGENT_TOOLS[agent] if name in TOOLS]


def get_langchain_tools_for_agent(agent: str) -> List[StructuredTool]:
    """Return the LangChain tool objects an agent is permitted to use.

    Intended for a future `llm.bind_tools(...)` step. Nodes currently call the
    plain functions directly for deterministic behaviour.
    """
    return [tool.langchain_tool for tool in get_tools_for_agent(agent)]


def call_tool(
    name: str,
    *,
    agent: Optional[str] = None,
    patient_context: str = "",
    **kwargs: Any,
) -> Any:
    """Invoke a registered tool, enforcing the per-agent policy.

    Args:
        name: Registered tool name.
        agent: Optional calling agent. When given, the tool must be in that
            agent's allowed list, otherwise a PermissionError is raised.
        patient_context: Injected as the first argument for tools that need it.
        **kwargs: Remaining tool arguments.

    Returns:
        The tool's return value.

    Raises:
        PermissionError: The agent is not allowed to call this tool.
    """
    tool = get_tool(name)

    if agent is not None:
        allowed = AGENT_TOOLS.get(agent)
        if allowed is None:
            raise PermissionError(f"Unknown agent: {agent}")
        if name not in allowed:
            raise PermissionError(f"Agent {agent} may not call tool {name}")

    args: Dict[str, Any] = {}
    if tool.takes_patient_context:
        args["patient_context"] = patient_context
    args.update(kwargs)

    try:
        return tool.func(**args)
    except TypeError as exc:
        raise TypeError(f"Bad arguments for tool {name}: {exc}") from exc


def describe_tools_for_agent(agent: str) -> List[Dict[str, str]]:
    """Introspect an agent's toolset (name + description), for prompts/UI."""
    return [{"name": t.name, "description": t.description} for t in get_tools_for_agent(agent)]


def registry_as_dict() -> Dict[str, Dict[str, Any]]:
    """Full registry snapshot (name, description, owning agents)."""
    return {
        name: {
            "description": tool.description,
            "takes_patient_context": tool.takes_patient_context,
            "allowed_for": sorted(
                agent for agent, names in AGENT_TOOLS.items() if name in names
            ),
        }
        for name, tool in TOOLS.items()
    }


__all__ = [
    "AGENT_TOOLS",
    "CODING_AGENT",
    "GENERAL_TOOL_NAMES",
    "MEMORY_AGENT",
    "RegisteredTool",
    "SCRIBE_AGENT",
    "SUPERVISOR_AGENT",
    "TOOLS",
    "call_tool",
    "describe_tools_for_agent",
    "get_langchain_tools_for_agent",
    "get_tool",
    "get_tools_for_agent",
    "registry_as_dict",
]
