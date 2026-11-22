"""
Tools layer for the Doctor Office MVP.

Every tool exists twice:
- as a plain Python function, directly importable and testable
- as a LangChain `StructuredTool` exposed through `tools.registry`

The registry enforces a per-agent policy so the Scribe node and the Coding
node each see only the tools the supervisor intended for them.
"""

from tools.patient_tools import (
    get_patient_allergies,
    get_patient_conditions,
    get_patient_medications,
    get_patient_profile,
)
from tools.clinical_search import format_search_result, search_clinical_notes
from tools.coding_tools import (
    simple_symptom_to_icd_suggestion,
    suggest_codes_for_complaint,
    validate_codes_against_patient,
    validate_icd_against_patient,
)
from tools.em_calculator import (
    MDM_LEVELS,
    calculate_em_code,
    cpt_for_mdm,
    derive_mdm_domains,
)
from tools.icd_validator import validate_icd10_code, validate_icd10_list
from tools.registry import (
    AGENT_TOOLS,
    CODING_AGENT,
    MEMORY_AGENT,
    SCRIBE_AGENT,
    SUPERVISOR_AGENT,
    TOOLS,
    call_tool,
    describe_tools_for_agent,
    get_langchain_tools_for_agent,
    get_tool,
    get_tools_for_agent,
    registry_as_dict,
)

__all__ = [
    # patient tools
    "get_patient_conditions",
    "get_patient_medications",
    "get_patient_allergies",
    "get_patient_profile",
    # search
    "search_clinical_notes",
    "format_search_result",
    # coding
    "validate_icd_against_patient",
    "validate_codes_against_patient",
    "simple_symptom_to_icd_suggestion",
    "suggest_codes_for_complaint",
    # E/M + ICD validation
    "MDM_LEVELS",
    "calculate_em_code",
    "cpt_for_mdm",
    "derive_mdm_domains",
    "validate_icd10_code",
    "validate_icd10_list",
    # registry
    "TOOLS",
    "AGENT_TOOLS",
    "SCRIBE_AGENT",
    "CODING_AGENT",
    "MEMORY_AGENT",
    "SUPERVISOR_AGENT",
    "call_tool",
    "get_tool",
    "get_tools_for_agent",
    "get_langchain_tools_for_agent",
    "describe_tools_for_agent",
    "registry_as_dict",
]
