"""
Coding Agent - Refines ICD-10 and suggests CPT codes from SOAP notes.

This module takes a completed SOAP note and patient context to produce
accurate medical coding (ICD-10-CM and CPT) with medical necessity justification.
"""

import logging
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from utils.llm import get_llm
from utils.config import get_model_name
from utils.llm_payload import build_model, coerce_str, extract_json_object, has_content


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodingResult(BaseModel):
    """Medical coding result with ICD-10-CM, CPT, and medical necessity."""
    primary_icd10: str = Field(
        description="Primary diagnosis ICD-10-CM code in format 'CODE - Description' (e.g., 'E11.9 - Type 2 diabetes mellitus without complications')"
    )
    secondary_icd10: List[str] = Field(
        description="Secondary diagnosis ICD-10-CM codes in format 'CODE - Description'"
    )
    cpt_suggestions: List[str] = Field(
        description="CPT procedure codes in format 'CODE - Description' (e.g., '99213 - Office visit, established patient, 20-29 minutes')"
    )
    medical_necessity_note: str = Field(
        description="Brief justification linking diagnoses to procedures for medical necessity"
    )


# Compact system prompt with strict clinical coding rules
CODING_SYSTEM_PROMPT = """You are a certified medical coder (CPC/CCS). Generate accurate ICD-10-CM and CPT codes as JSON.

CRITICAL RULES:
1. Output ONLY valid JSON matching the schema. No extra text, no markdown.
2. GROUNDING: Only code conditions in patient_context or SOAP assessment. No invented diagnoses.
3. PATIENT STATUS: If patient has last_visit_summary or existing conditions → ESTABLISHED. Use 99211-99215. Only 99202-99205 for true new patients.
4. DEFAULT: Established patients with 2+ chronic conditions (e.g., Diabetes + HTN) → 99213 or 99214.
5. PRIMARY ICD-10: SINGLE code only (one principal diagnosis). Format exactly: "CODE - Description". Pick the main reason for visit.
6. ICD-10 MAPPINGS (use EXACTLY when condition matches patient_context):
   - Type 2 Diabetes without complications → E11.9 - Type 2 diabetes mellitus without complications
   - Essential Hypertension → I10 - Essential (primary) hypertension
   - NEVER use I20.x (angina) without documented cardiac ischemia.
   - NEVER add Asthma (J45.x), Allergies (J30.x), COPD, CHF, CKD unless explicitly in patient_context.
7. CPT: Established → 99211-99215, New → 99202-99205. MDM: Straightforward→99211, Low→99212, Moderate→99213, High→99214.

MEDICAL NECESSITY:
- Format: "CPT 99213 supported by E11.9 (diabetes follow-up) and I10 (HTN management)"
- Justify every code from provided information only. No invented conditions.
- Cite the coded symptoms and conditions only. When the chart lists no cardiac
  diagnosis, never explain the visit as ischemic, anginal or coronary: a chest
  pain presentation is justified as a symptom (R07.x) plus management of the
  documented chronic conditions, never as a cardiac diagnosis.
- Do not recommend tests or investigations for a condition you have not coded.

SYMPTOM CODES:
- The prompt may contain a "SYMPTOM CODE CANDIDATES" block derived from the
  chief complaint and the visit data. When it does, include one candidate per
  documented symptom in the SECONDARY list, in the "CODE - Description" form.
- Those candidates are symptoms only. Never upgrade one into a disease code:
  chest pain stays R07.x and must not become I20/I21, because a symptom code is
  evidence of a symptom, not of a disease.
- Never add a candidate for a symptom the encounter does not document, and
  never add a disease code the chart does not list.
- If you omit a candidate anyway the Supervisor adds it deterministically
  afterwards, so omitting it costs an advisory flag, not correctness."""


def create_coding_chain(mode: str = "fast", model_name: Optional[str] = None):
    """Create the Coding Agent chain with JSON parser."""
    parser = JsonOutputParser(pydantic_object=CodingResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CODING_SYSTEM_PROMPT),
        ("human", """Patient Context:
{patient_context}

SOAP Note:
Subjective: {subjective}
Objective: {objective}
Assessment: {assessment}
Plan: {plan}

Existing ICD-10 from Scribe: {existing_icd10}

{format_instructions}"""),
    ])
    num_predict = 128 if mode == "fast" else 256
    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.05, num_predict=num_predict)
    else:
        llm = get_llm(mode=mode, temperature=0.05, num_predict=num_predict)
    chain = prompt | llm | parser
    return chain


def _repair_json_output(raw_output: str, mode: str = "fast", model_name: Optional[str] = None) -> Optional[CodingResult]:
    """Attempt to repair malformed JSON output using a repair prompt."""
    repair_prompt = f"""Fix this malformed JSON to match the CodingResult schema exactly.
Return ONLY corrected JSON.

Malformed output:
{raw_output}

Required:
- primary_icd10: "CODE - Description"
- secondary_icd10: list of "CODE - Description"
- cpt_suggestions: list of "CODE - Description"
- medical_necessity_note: string"""

    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.0, num_predict=128)
    else:
        llm = get_llm(mode=mode, temperature=0.0, num_predict=128)
    try:
        response = llm.invoke(repair_prompt)
        content = getattr(response, "content", response)
        payload = extract_json_object(coerce_str(content))
        if payload is None:
            logger.warning("Repair pass did not return parsable JSON")
            return None
        return build_model(CodingResult, payload, source="repaired coding output")
    except Exception as exc:
        logger.warning("Repair pass failed: %s", exc)
        return None


def _create_raw_chain(mode: str = "fast", model_name: Optional[str] = None):
    """Create a chain with string output parser for repair."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CODING_SYSTEM_PROMPT),
        ("human", """Patient Context:
{patient_context}

SOAP Note:
Subjective: {subjective}
Objective: {objective}
Assessment: {assessment}
Plan: {plan}

Existing ICD-10 from Scribe: {existing_icd10}

{format_instructions}"""),
    ])
    num_predict = 192 if mode == "fast" else 512
    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.05, num_predict=num_predict)
    else:
        llm = get_llm(mode=mode, temperature=0.05, num_predict=num_predict)
    chain = prompt | llm | StrOutputParser()
    return chain


def generate_coding(
    soap_note: Union[dict, "SOAPNote"],
    patient_context: str,
    mode: str = "fast",
    model_name: Optional[str] = None
) -> CodingResult:
    """
    Generate refined ICD-10 and CPT codes from a SOAP note.
    """
    # Handle both SOAPNote object and dict
    if hasattr(soap_note, 'model_dump'):
        soap_dict = soap_note.model_dump()
    else:
        soap_dict = soap_note

    chain = create_coding_chain(mode=mode, model_name=model_name)
    format_instructions = chain.steps[-1].get_format_instructions() if hasattr(chain.steps[-1], 'get_format_instructions') else ""

    try:
        result = chain.invoke({
            "patient_context": patient_context,
            "subjective": soap_dict.get("subjective", ""),
            "objective": soap_dict.get("objective", ""),
            "assessment": soap_dict.get("assessment", ""),
            "plan": soap_dict.get("plan", ""),
            "existing_icd10": ", ".join(soap_dict.get("icd10_suggestions", [])),
            "format_instructions": format_instructions
        })
        if result is None:
            raise ValueError("JSON parser returned None")
        # Defensive construction: a missing key (e.g. cpt_suggestions) is filled
        # with a safe default instead of raising.
        note = build_model(CodingResult, result, source="coding output")
        if has_content(note):
            return note
        logger.warning("Coding output had no usable content, retrying with raw output...")

    except Exception as e:
        logger.warning(f"Initial coding JSON parse failed ({e}), retrying with raw output...")

    raw_output = ""
    try:
        raw_chain = _create_raw_chain(mode=mode, model_name=model_name)
        raw_output = raw_chain.invoke({
            "patient_context": patient_context,
            "subjective": soap_dict.get("subjective", ""),
            "objective": soap_dict.get("objective", ""),
            "assessment": soap_dict.get("assessment", ""),
            "plan": soap_dict.get("plan", ""),
            "existing_icd10": ", ".join(soap_dict.get("icd10_suggestions", [])),
            "format_instructions": format_instructions
        })
    except Exception as raw_exc:
        logger.warning("Raw coding re-run failed: %s", raw_exc)

    if raw_output:
        payload = extract_json_object(coerce_str(raw_output))
        if payload is not None:
            raw_note = build_model(CodingResult, payload, source="raw coding output")
            if has_content(raw_note):
                return raw_note
        repaired = _repair_json_output(coerce_str(raw_output), mode=mode, model_name=model_name)
        if repaired and has_content(repaired):
            return repaired

    # Nothing salvageable came back. Return an empty, schema-valid result so
    # the pipeline and the Physician Review tab still render.
    logger.error("Coding Agent produced no usable content - returning empty result")
    return build_model(CodingResult, {}, source="empty fallback")


if __name__ == "__main__":
    # Demo with sample SOAP note
    sample_soap = {
        "subjective": "Patient reports increased fatigue and occasional dizziness. History of Type 2 Diabetes and Hypertension. Occasional polyuria and fatigue reported previously.",
        "objective": "BP 150/94. HbA1c 7.8%. BMI 31.2. Medications: Metformin 500mg BID, Lisinopril 10mg daily.",
        "assessment": "Patient presents with persistent hyperglycemia despite Metformin and hypertension. BP uncontrolled. Considering medication intensification for diabetes and further optimization of hypertension management.",
        "plan": "Continue Metformin 500mg BID. Continue Lisinopril 10mg daily. Monitor BP and glucose twice daily. Encourage dietary modifications and regular exercise. Follow-up in 4 weeks for BP and repeat HbA1c. Consider GLP-1 agonist or SGLT2 inhibitor for diabetes intensification.",
        "icd10_suggestions": [
            "E11.9 - Type 2 diabetes mellitus without complications",
            "I10 - Essential (primary) hypertension"
        ]
    }

    patient_context = """
Patient: John Doe, 58yo male
Conditions: Type 2 Diabetes, Hypertension
Medications: Metformin 500mg BID, Lisinopril 10mg daily
Allergies: Penicillin
Last Visit: Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence.
"""

    coding = generate_coding(sample_soap, patient_context, mode="fast")
    print(coding.model_dump_json(indent=2))