"""
Scribe Agent - Generates structured SOAP notes from clinical encounters.

This module provides a clinical scribe agent that produces SOAP notes with
ICD-10 suggestions using local LLMs via llama.cpp server.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from utils.llm import get_llm
from utils.config import get_model_name
from utils.llm_payload import build_model, coerce_str, extract_json_object, has_content
from utils.validation import CHIEF_COMPLAINT_CODE_HINTS
from utils.symptom_codes import phrase_present_not_negated
from agents.visit_context import get_specialty_fields, visit_context_to_prompt_block
from agents.specialty_profiles import get_specialty_profile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SOAP_TEXT_FIELDS = ("subjective", "objective", "assessment", "plan")


class SOAPNote(BaseModel):
    """Structured SOAP note with ICD-10 suggestions."""
    subjective: str = Field(description="Patient's reported symptoms and history")
    objective: str = Field(description="Measurable findings: vitals, exam, labs")
    assessment: str = Field(description="Clinical reasoning: synthesis, differential")
    plan: str = Field(description="Actionable plan: meds, orders, referrals, follow-up")
    icd10_suggestions: List[str] = Field(description="ICD-10 codes as 'CODE - Description'")


def _has_soap_text(note: Optional[SOAPNote]) -> bool:
    """Return True when at least one core SOAP narrative section is non-empty."""
    if note is None or not has_content(note):
        return False
    return any(coerce_str(getattr(note, field, None)).strip() for field in _SOAP_TEXT_FIELDS)


SYSTEM_PROMPT = """You are an expert medical scribe. Output ONLY valid JSON with keys: subjective, objective, assessment, plan, icd10_suggestions. No extra text, no markdown.

=== SECTION RULES ===
- SUBJECTIVE = ONLY patient-reported: complaint, HPI (onset, character, severity, radiation, associated symptoms, pertinent pos/neg, modifying factors). NEVER exam findings, vitals, labs. MANDATORY: The Subjective section must NEVER be empty. If the chief complaint indicates a routine follow-up and no new symptoms are reported, you MUST synthesize a Subjective statement based on the patient's Last Visit Summary and active conditions. Example: 'Patient presents for routine follow-up of Type 2 Diabetes and Hypertension. Denies any new acute symptoms.' Do not output 'No reported symptoms' alone.
- OBJECTIVE = ONLY measured/tested/examined: vitals, exam, labs, imaging, clinician observations. NEVER patient-reported history. NEVER invent findings/values.
- ASSESSMENT = numbered problem list (1. 2. 3.) prioritized by clinical urgency/acuity. Each numbered item states: problem/condition + status (e.g., suboptimal control, not at goal, controlled, or suspected / rule-out when unconfirmed) + supporting findings/pertinent negatives + brief clinical reasoning (WHY). NO CIRCULAR REASONING: never restate condition as its own explanation. For unconfirmed acute conditions (e.g., possible ACS), use 'suspected / rule-out' rather than asserting a confirmed diagnosis.
- PLAN = numbered action list (1. 2. 3. ...) ordered by acuity first then chronic problems. Include concrete items grounded in supplied context:
  - Medications: explicit action (Continue, Start, Adjust/Titrate, Stop/Hold) + drug name + dose/frequency/special instructions ONLY when grounded in Patient Context or visit data.
  - Diagnostics/labs/imaging, referrals, home monitoring/lifestyle, follow-up timing, and return precautions/red flags.
  - NEVER write vague filler like 'possibly adjust' or 'consider referral' or 'continue meds and follow up' when specific grounded actions or follow-up intervals apply. NEVER recommend test/ECG/scan already completed today — act on documented result. NEVER invent med/dose/referral/test without basis in supplied context.

=== DATA DISCIPLINE & PRIORITY ===
- Short honest note beats complete invented one. Never pad. Only assert what data supports. Missing? Say so ("no examination documented today", "duration not specified", "no prior value for comparison"). Never assume duration, severity, laterality, count not in chart. Never carry number from different value type. Never infer symptom patient didn't report and no exam documented.
- If Chief Complaint/HPI names NEW/ACUTE symptom → PRIMARY problem. Assessment/Plan address it FIRST. Chronic stays relevant but must not crowd out acute. Never write chronic-care-only note for new problem. Assessment leads with presenting symptom, chronic second/brief. If purely chronic disease → chronic management leads.

=== CLINICAL INTERPRETATION GUARDRAILS ===
- HbA1c ≥ 6.5% = diabetes; 5.7–6.4% = prediabetes; <5.7% = normal. NEVER interpret elevated HbA1c as hypoglycemia.
- Glucose: fasting ≥ 126 or random ≥ 200 with symptoms = diabetes. Distinguish from hypoglycemia (<70).
- NEVER invent causal link between result and condition; never use result to support mechanism data doesn't state.
- Never convert value types (HbA1c ≠ blood glucose; HR ≠ BP).
- Never call values "normal"/"WNL" unless documented values are actually unremarkable. Report as given.

=== GROUNDING & SAFETY (INVIOLABLE) ===
- Base note ONLY on supplied context. No hallucination. Use ONLY conditions/meds/allergies from Patient Context.
- NO NEW CHRONIC CONDITIONS: never add condition Patient Context lacks.
- NO UNSUPPORTED CARDIAC DX: when Patient Context lacks cardiac dx, NEVER name/imply one. No "ischemic etiology/heart disease", "consistent with ischemia/angina", no word "angina", no MI/ACS/coronary ischemia from presenting symptom. Name as symptom, list risk factors as risk factors without concluding cardiac dx. Plan follow-up, risk-factor control, return precautions, correlate with documented ECG. Applies to ALL sections.
- PLAN GROUNDING: address only documented conditions/reported symptoms. RETRIEVED NOTES: ignore content about conditions not in Patient Context.
- Never state symptom/duration/diagnosis not supplied by patient/chart. Do not import from these instructions.
- Never relabel a result: if clinician wrote "normal", do not call "elevated"/"abnormal"; never treat normal as disease evidence.
- Never order/recommend test/ECG/scan visit data shows was done today — act on documented result.
- If no vitals/exam/tests provided → "Not documented today" for that Objective part.

=== ICD-10 SUGGESTIONS (max 5) ===
- Format: "CODE - Description". Use table ONLY for condition actually in Patient Context. Table is lookup, NOT suggestion: listing ≠ patient has it. If not in Patient Context, DO NOT CODE.
  Type 2 Diabetes without complications -> E11.9
  Essential (primary) Hypertension -> I10
  Seasonal Allergies / Allergic Rhinitis -> J30.9
  Asthma -> J45.909
- Never invent diagnosis to justify code; never attach code to condition it doesn't describe.
- Symptom codes for presenting complaint added downstream by Supervisor; do not add them.
- NEVER use I20.x (angina) or I21.x (MI). ONLY if Patient Context itself lists cardiac dx (angina, ischemic HD, MI). ECG/chest-pain/stress test do NOT authorize cardiac dx code — do not diagnose ischaemia from them.
- Lead with code for leading problem, then chronic conditions addressed. Always return icd10_suggestions, even if empty.

=== ABSTRACT TEMPLATES (shape only) ===
- SUBJECTIVE: "<SYMPTOM> x <DURATION>; <CHARACTER/SEVERITY>; <ASSOCIATED SYMPTOMS>; <PERTINENT NEGATIVES>."
- OBJECTIVE: "<VITALS AS GIVEN>. <EXAM AS GIVEN>. <TESTS AS GIVEN>."
- ASSESSMENT: "1. <PRIMARY PROBLEM> — <STATUS> (<FINDINGS/PERTINENT NEGATIVES>); <REASONING>. 2. <SECONDARY PROBLEM> — <STATUS>; <REASONING>."
- PLAN: "1. Medications: <Continue/Start/Adjust/Stop + grounded drug/dose/instructions>. 2. Diagnostics/Monitoring: <grounded labs/home tracking>. 3. Follow-up & Precautions: <timeframe & red flags>."

NEVER put drug/test/referral/symptom/duration/exam finding in note that appears nowhere in supplied context."""

FAST_SYSTEM_PROMPT = """You are an expert medical scribe. Output ONLY valid JSON with keys: subjective, objective, assessment, plan, icd10_suggestions. No extra text.

SECTION RULES:
- SUBJECTIVE = patient-reported only (complaint, HPI). NEVER exam/vitals/labs. Never empty — synthesize from Last Visit Summary if routine follow-up.
- OBJECTIVE = measured/tested only (vitals, exam, labs, imaging). NEVER patient history. NEVER invent.
- ASSESSMENT = numbered problem list (1. 2. 3.). For chronic disease follow-ups, lead with documented chronic conditions and control status (avoid circular phrasing), then address reported symptoms with supporting findings/pertinent negatives and brief reasoning. Never state unconfirmed acute diagnoses as certain.
- PLAN = numbered action list (1. 2. 3. ...). Use 'Continue <drug>' for medications already active in Patient Context; only use 'Start <drug>' for a new medication not already active. Include grounded diagnostics/monitoring, follow-up interval, and return precautions. Avoid vague filler ('possibly adjust', 'consider referral', 'continue meds and follow up') when concrete grounded steps apply.

DATA DISCIPLINE:
- Short honest note beats complete invented one. Only assert what data supports. Missing? Say so. Never assume duration/severity/laterality/count not in chart.
- NEW/ACUTE symptom in CC/HPI → PRIMARY problem. Assessment/Plan address it FIRST. Chronic = secondary/brief.

CLINICAL GUARDRAILS:
- HbA1c ≥6.5%=DM, 5.7–6.4%=prediabetes, <5.7%=normal. NEVER call elevated HbA1c hypoglycemia.
- Glucose: fasting ≥126 or random ≥200 with symptoms = DM. Distinguish from hypoglycemia (<70).
- NEVER invent causal links. NEVER convert value types (HbA1c≠glucose, HR≠BP).
- Report values as given — never call "normal"/"WNL" unless documented.

GROUNDING & SAFETY (INVIOLABLE):
- Base ONLY on supplied context. No hallucination. Use ONLY conditions/meds/allergies from Patient Context.
- NO NEW CHRONIC CONDITIONS. NO UNSUPPORTED CARDIAC DX when no cardiac dx in context.
- Name symptoms as symptoms; list risk factors without concluding cardiac dx. Correlate with documented ECG.
- NEVER order test/scan already done. If no vitals/exam/tests → "Not documented today".

ICD-10 (max 5): "CODE - Description" format. ONLY for conditions in Patient Context. Table is lookup, NOT suggestion.
  Type 2 Diabetes → E11.9 | HTN → I10 | Allergic Rhinitis → J30.9 | Asthma → J45.909
- NEVER I20.x/I21.x unless Patient Context lists cardiac dx. ECG/chest pain do NOT authorize cardiac dx.
- Lead with leading problem code, then chronic. Always return icd10_suggestions (even empty).

NEVER put drug/test/referral/symptom/duration/exam finding not in supplied context."""

HUMAN_TEMPLATE = """Patient Context:
{patient_context}

{specialty_guidance}
{focus_block}{visit_context_block}
Relevant Clinical Notes:
{retrieved_notes}

Chief Complaint:
{chief_complaint}

{format_instructions}"""

_SYMPTOM_PHRASES = tuple(phrase for phrase, _prefix, _label in CHIEF_COMPLAINT_CODE_HINTS)

_CHRONIC_FOCUS_BLOCK = """VISIT FOCUS: No new or acute symptom was named in the Chief Complaint.
This is a chronic-disease visit: lead the Assessment with a numbered problem list
and current control status, and make the Plan concrete about medication actions,
monitoring, and follow-up intervals."""


def _render_visit_context(visit_context: Optional[dict]) -> str:
    """Render the clinician-provided block, or "" when nothing was provided."""
    block = visit_context_to_prompt_block(visit_context)
    return f"{block}\n\n" if block else ""


def _render_focus_block(chief_complaint: str, visit_context: Optional[dict]) -> str:
    """Name the visit's primary problem so chronic context cannot crowd it out."""
    haystack_parts = [str(chief_complaint or "")]
    if isinstance(visit_context, dict):
        haystack_parts.append(str(visit_context.get("hpi") or ""))
        for key, value in get_specialty_fields(visit_context).items():
            haystack_parts.extend((key.replace("_", " "), value))

    haystack = " ".join(parts for parts in haystack_parts if parts).lower()
    if not haystack.strip():
        return ""

    found = [phrase for phrase in _SYMPTOM_PHRASES
             if phrase_present_not_negated(phrase, haystack)]
    if not found:
        return f"{_CHRONIC_FOCUS_BLOCK}\n\n"

    logger.info("Scribe focus anchor: acute symptom(s) %s", found)
    return (
        "VISIT FOCUS (highest priority): the Chief Complaint / HPI describes the "
        f"following new or acute symptom(s): {', '.join(found)}.\n"
        "Treat this as the PRIMARY problem for this visit. The Assessment must open "
        "with a numbered problem list leading with what these symptoms plus today's exam, "
        "vitals and test findings mean, and the Plan must address them first and specifically "
        "with explicit actions and safety precautions. Address chronic conditions from Patient "
        "Context afterwards, and only briefly.\n\n"
    )


def _render_specialty_guidance(specialty: Optional[str]) -> str:
    """Render specialty-specific guidance block for the Scribe prompt."""
    if not specialty:
        specialty = "general"
    profile = get_specialty_profile(specialty)
    if not profile.prompt_emphasis:
        return ""
    return (
        f"SPECIALTY-SPECIFIC GUIDANCE ({profile.display_name}):\n"
        f"{profile.prompt_emphasis}\n"
    )


NUM_PREDICT = {"fast": 480, "quality": 480}
DEFAULT_NUM_PREDICT = NUM_PREDICT["fast"]


def _num_predict(mode: str) -> int:
    return NUM_PREDICT.get(mode, DEFAULT_NUM_PREDICT)


def _select_system_prompt(mode: str) -> str:
    return FAST_SYSTEM_PROMPT if mode == "fast" else SYSTEM_PROMPT


def create_scribe_chain(mode: str = "fast", model_name: Optional[str] = None):
    """Create the Scribe Agent chain with JSON parser."""
    parser = JsonOutputParser(pydantic_object=SOAPNote)
    system_prompt = _select_system_prompt(mode)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", HUMAN_TEMPLATE),
    ])
    num_predict = _num_predict(mode)
    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.05, num_predict=num_predict)
    else:
        llm = get_llm(mode=mode, temperature=0.05, num_predict=num_predict)
    chain = prompt | llm | parser
    return chain


def _repair_json_output(raw_output: str, mode: str = "fast", model_name: Optional[str] = None) -> Optional[SOAPNote]:
    """Attempt to repair malformed JSON output using a repair prompt."""
    from utils.llm_payload import extract_json_object, build_model

    payload = extract_json_object(coerce_str(raw_output))
    if payload is not None:
        try:
            note = build_model(SOAPNote, payload, source="deterministic repair")
            if _has_soap_text(note):
                logger.info("Deterministic JSON repair succeeded (no LLM call)")
                return note
        except Exception as exc:
            logger.debug(f"Deterministic schema validation failed: {exc}")

    logger.info("Invoking LLM JSON repair (deterministic extraction failed)")
    repair_prompt = f"""Fix this malformed JSON to match the SOAPNote schema exactly.
Return ONLY corrected JSON.

Malformed output:
{raw_output}

Required: subjective, objective, assessment, plan, icd10_suggestions (list of "CODE - Description")"""

    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.0, num_predict=256)
    else:
        llm = get_llm(mode=mode, temperature=0.0, num_predict=256)
    try:
        response = llm.invoke(repair_prompt)
        content = getattr(response, "content", response)
        payload = extract_json_object(coerce_str(content))
        if payload is None:
            logger.warning("Repair pass did not return parsable JSON")
            return None
        return build_model(SOAPNote, payload, source="repaired scribe output")
    except Exception as exc:
        logger.warning("Repair pass failed: %s", exc)
        return None


def _create_raw_chain(mode: str = "fast", model_name: Optional[str] = None):
    """Create a chain with string output parser for repair."""
    system_prompt = _select_system_prompt(mode)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", HUMAN_TEMPLATE),
    ])
    num_predict = _num_predict(mode)
    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.05, num_predict=num_predict)
    else:
        llm = get_llm(mode=mode, temperature=0.05, num_predict=num_predict)
    chain = prompt | llm | StrOutputParser()
    return chain


def _filter_retrieved_notes(retrieved_notes: str, patient_context: str) -> str:
    """Filter retrieved notes to only keep content relevant to patient's conditions."""
    if not retrieved_notes or not retrieved_notes.strip():
        return ""

    context_lower = patient_context.lower()
    condition_keywords = []
    for cond in ["diabetes", "hypertension", "htn", "asthma", "copd", "heart failure", "ckd", "allerg", "rhinitis"]:
        if cond in context_lower:
            condition_keywords.append(cond)

    if not condition_keywords:
        logger.info("No condition keywords found in patient context, filtering out all retrieved notes")
        return ""

    logger.info(f"Patient conditions detected: {condition_keywords}")

    notes = retrieved_notes.split("\n\n")
    filtered = []
    for note in notes:
        note_lower = note.lower()
        if any(keyword in note_lower for keyword in condition_keywords):
            filtered.append(note)
            logger.debug(f"Kept note: {note[:80]}...")
        else:
            logger.debug(f"Filtered out note (irrelevant): {note[:80]}...")

    result = "\n\n".join(filtered[:1])
    if len(result) > 200:
        result = result[:200] + "..."

    logger.info(f"Filtered retrieved notes from {len(retrieved_notes)} to {len(result)} chars ({len(filtered)} notes kept)")
    return result


def generate_soap_note(
    patient_context: str,
    retrieved_notes: str,
    chief_complaint: str,
    mode: str = "fast",
    model_name: Optional[str] = None,
    visit_context: Optional[dict] = None,
    specialty: Optional[str] = None,
) -> SOAPNote:
    """Generate a SOAP note from patient context and chief complaint."""
    filtered_notes = _filter_retrieved_notes(retrieved_notes, patient_context)

    chain = create_scribe_chain(mode=mode, model_name=model_name)

    format_instructions = chain.steps[-1].get_format_instructions() if hasattr(chain.steps[-1], 'get_format_instructions') else ""
    visit_block = _render_visit_context(visit_context)
    specialty_block = _render_specialty_guidance(specialty)
    focus_block = _render_focus_block(chief_complaint, visit_context)
    active_system_prompt = _select_system_prompt(mode)
    prompt_size = (
        len(active_system_prompt)
        + len(patient_context)
        + len(filtered_notes)
        + len(chief_complaint)
        + len(visit_block)
        + len(specialty_block)
        + len(focus_block)
        + len(format_instructions)
    )
    logger.info(
        "Scribe Agent prompt size: ~%d chars, mode=%s, rich_visit_context=%s "
        "(fields=%d, %d chars), specialty=%s, focus_anchor=%s",
        prompt_size,
        mode,
        bool(visit_block),
        len(visit_context or {}),
        len(visit_block),
        specialty,
        "acute" if focus_block and "PRIMARY problem" in focus_block else ("chronic" if focus_block else "none"),
    )

    prompt_inputs = {
        "patient_context": patient_context,
        "specialty_guidance": specialty_block,
        "focus_block": focus_block,
        "visit_context_block": visit_block,
        "retrieved_notes": filtered_notes,
        "chief_complaint": chief_complaint,
        "format_instructions": format_instructions,
    }

    try:
        result = chain.invoke(prompt_inputs)
    except Exception as e:
        logger.warning(f"Initial JSON parse failed ({e}), falling back to raw output...")
        result = None

    note = build_model(SOAPNote, result, source="scribe output")
    if _has_soap_text(note):
        return note

    logger.warning("Scribe output had no usable SOAP content, retrying with raw output...")
    raw_output = ""
    try:
        raw_chain = _create_raw_chain(mode=mode, model_name=model_name)
        raw_output = raw_chain.invoke(prompt_inputs)
    except Exception as e:
        logger.warning(f"Raw scribe re-run failed: {e}")

    if raw_output:
        payload = extract_json_object(coerce_str(raw_output))
        if payload is not None:
            raw_note = build_model(SOAPNote, payload, source="raw scribe output")
            if _has_soap_text(raw_note):
                return raw_note

        repaired = _repair_json_output(coerce_str(raw_output), mode=mode, model_name=model_name)
        if repaired and _has_soap_text(repaired):
            return repaired

    logger.error("Scribe Agent produced no usable SOAP content - returning empty note")
    return build_model(SOAPNote, {}, source="empty fallback")


if __name__ == "__main__":
    from rag.simple_rag import get_vectorstore

    patient = {
        "name": "John Doe",
        "age": 58,
        "gender": "male",
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
        "allergies": ["Penicillin"],
        "last_visit_summary": "Blood pressure 148/92. HbA1c 7.8%. Advised lifestyle changes and medication adherence."
    }

    patient_context = f"""
Patient: {patient['name']}, {patient['age']}yo {patient['gender']}
Conditions: {', '.join(patient['conditions'])}
Medications: {', '.join(patient['medications'])}
Allergies: {', '.join(patient['allergies'])}
Last Visit: {patient['last_visit_summary']}
"""

    vs = get_vectorstore()
    from utils.config import RAG_TOP_K
    retrieved_docs = vs.similarity_search("diabetes hypertension management", k=RAG_TOP_K)
    retrieved_notes = "\n\n".join([doc.page_content for doc in retrieved_docs])

    chief_complaint = "Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."

    soap = generate_soap_note(patient_context, retrieved_notes, chief_complaint, mode="fast")
    print(soap.model_dump_json(indent=2))