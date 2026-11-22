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


class SOAPNote(BaseModel):
    """Structured SOAP note with ICD-10 suggestions."""
    subjective: str = Field(
        description="Patient's reported symptoms, history, and complaints in their own words"
    )
    objective: str = Field(
        description="Measurable clinical findings: vitals, exam results, lab values, observations"
    )
    assessment: str = Field(
        description="Clinical reasoning: synthesis of subjective/objective, differential diagnosis, problem list"
    )
    plan: str = Field(
        description="Actionable treatment plan: medications, orders, referrals, follow-up, patient education"
    )
    icd10_suggestions: List[str] = Field(
        description="ICD-10 codes in format 'CODE - Description' (e.g., 'E11.9 - Type 2 diabetes mellitus without complications')"
    )


# Compact system prompt with strict clinical grounding rules.
#
# Written for a 1.5B model on CPU, so every character is load-bearing:
# - rules are grouped into labelled blocks the model can scan
# - all examples are ABSTRACT templates built from <ANGLE_BRACKET> placeholders.
#   Concrete clinical examples were removed on purpose: with a small model they
#   get copied verbatim into the note, which is a hallucination of the worst kind
#   (it invents exam findings and drugs for a patient who has none).
SYSTEM_PROMPT = """You are an expert medical scribe. Generate a SOAP note as JSON.

=== FORMAT ===
- Output ONLY valid JSON with keys: subjective, objective, assessment, plan, icd10_suggestions.
- No extra text, no markdown.
- Be brief. A note is not better for being longer.

=== SECTION RULES (never move data between sections) ===
- SUBJECTIVE = ONLY what the PATIENT reports: the complaint, the HPI, symptom
  character, duration, context. Never state an examination finding that was not
  provided, and never put vitals, labs or imaging here.
- OBJECTIVE = ONLY what was measured, tested or examined: vitals, exam findings,
  labs, imaging, ECGs, clinician observations. Never put patient-reported
  history here, and never invent an examination finding or a value.

=== ASSESSMENT (a synthesis, not a restatement) ===
- Reason about what today's Subjective + Objective MEAN, in light of the known
  chronic conditions.
- NEVER copy or retell the note: no re-listing vitals, no repeating the HPI
  sentence by sentence, no describing the note itself.
- Never say the same thing twice. State each point once.
- Sentence 1: the leading problem for this visit and what supports it.
- Later sentences: how findings relate, and the main alternative explanation if
  the data genuinely allows one. Do not invent a diagnosis to fill the slot.
- 2-4 sentences total. Never leave assessment empty.

=== PLAN (specific and actionable) ===
- Short numbered steps, one action each. Every step must follow from a finding
  or problem named in ASSESSMENT.
- Each step says WHAT to do and WHY it follows, not just a category:
  GOOD: "<TEST> in <TIMEFRAME> - <FINDING> changed."
  BAD: "Follow up on labs."
- Order the steps: leading problem first, then chronic problems addressed.
- A plan of only "continue current medications and follow up" is NOT acceptable.
  Add the monitoring, testing, education, escalation criteria or follow-up
  interval the documented findings justify.
- PLAN MUST NEVER BE EMPTY. If the evidence is thin, each step becomes a
  concrete GATHER-MISSING-DATA step - name the test, recheck or clarification
  that would resolve the gap - instead of inventing a treatment.
- Never recommend a test, ECG, scan or referral that the visit data or the Chief
  Complaint shows has already been done; act on the result instead.
- Never invent a medication, dose, referral, test or procedure that has no basis
  in Patient Context, the Chief Complaint or the visit data. You MAY recommend
  further evaluation that the documented findings clearly warrant.

=== WHEN THE DATA IS THIN ===
- A short honest note beats a complete invented one. Never pad the note.
- Only assert what the supplied data supports. Where something is missing,
  say so plainly ("no examination documented today", "duration not specified",
  "no prior value for comparison").
- Never assume a duration, severity, laterality or count that the chart does not
  state, and never carry a number over from a different kind of value.
- Never infer a symptom the patient did not report and no exam documented.

=== ACUTE vs CHRONIC PRIORITY ===
- If the Chief Complaint or HPI names a NEW or ACUTE symptom, that symptom is
  the PRIMARY problem for this visit. Assessment and Plan address it FIRST and
  most specifically.
- Chronic conditions stay relevant, but must not crowd out or replace the
  acute presentation.
- Never write a chronic-care-only note when the visit is about a new problem:
  the Assessment leads with the presenting symptom, and the chronic context
  comes second and stays brief.
- If the visit is purely about chronic disease, chronic management leads.

=== GROUNDING AND SAFETY (never relax these) ===
- Base the note ONLY on the supplied context. Do not hallucinate.
- Use ONLY conditions, medications and allergies from Patient Context.
- NO NEW CHRONIC CONDITIONS: never add a condition the Patient Context lacks.
- NO UNSUPPORTED CARDIAC DIAGNOSIS: when Patient Context lists no cardiac
  diagnosis, never name one and never imply one. Do not write "ischemic
  etiology", "ischemic heart disease", "consistent with ischemia/ischaemia",
  "consistent with angina", do not use the word "angina", and do not diagnose or
  imply myocardial infarction, ACS or coronary ischemia from a presenting
  symptom. Name the presentation as a symptom, list risk factors as risk factors
  without concluding a cardiac diagnosis, and plan follow-up, risk-factor
  control, return precautions and correlation with any documented ECG. This
  applies to every section, including the medical plan.
- PLAN GROUNDING: address only documented conditions or reported symptoms.
- RETRIEVED NOTES: ignore retrieved content about conditions not in Patient Context.
- Never state a symptom, duration or diagnosis that the patient or the chart
  did not supply. Do not import a symptom from these instructions.
- Never relabel a result. Carry over the clinician's own descriptor: if a value
  was written as normal, do not call it elevated or abnormal, and never treat a
  normal result as evidence of a disease.
- Never order or recommend a test, ECG or scan that the visit data shows was
  already performed today; instead, act on the result that was documented.
- Never invent a causal link between a result and a condition, and never use a
  result to support a mechanism the data does not state.
- Never convert one kind of value into another (an HbA1c is not a blood glucose;
  a heart rate is not a blood pressure).
- Do not call a set of values "normal" or "within normal limits" unless the
  documented values are in fact unremarkable. Report values as given.
- If no vitals, exam or tests were provided, write "Not documented today" for
  that part of Objective instead of guessing.
- The templates below are ABSTRACT. <ANGLE_BRACKETS> are placeholders, never
  copy them or any example content into the note as clinical fact.

=== ICD-10 SUGGESTIONS (max 5) ===
- Format each entry as "CODE - Description".
- Use this table ONLY for a condition that is actually named in Patient Context.
  The table is a lookup, not a suggestion: listing a condition here does NOT mean
  the patient has it. If it is not in Patient Context, do not code it.
    Type 2 Diabetes without complications -> E11.9
    Essential (primary) Hypertension       -> I10
    Seasonal Allergies / Allergic Rhinitis  -> J30.9
    Asthma                                  -> J45.909
- Never invent a diagnosis in order to justify a code, and never attach a code
  to a condition it does not describe.
- Symptom codes for the presenting complaint are added downstream by the
  Supervisor; do not add them yourself.
- NEVER use I20.x (angina) or I21.x (myocardial infarction). Use them ONLY if
  Patient Context itself already lists a cardiac diagnosis (angina, ischemic
  heart disease, myocardial infarction). ECG findings, chest-pain descriptions
  or a stress test do NOT authorise a cardiac diagnosis code - do not diagnose
  ischaemia from them.
- Lead with the code for the leading problem, then the chronic conditions you
  addressed.
- Never omit the key: always return icd10_suggestions, even if empty.

=== ABSTRACT TEMPLATES (shape only) ===
- SUBJECTIVE: "<SYMPTOM> x <DURATION>; <PERTINENT NEGATIVES>."
- OBJECTIVE: "<VITALS AS GIVEN>. <EXAM AS GIVEN>. <TESTS AS GIVEN>."
- ASSESSMENT: "<FINDING> in a patient with <CHART_CONDITION> is most consistent
  with <INTERPRETATION>. <SUPPORTING FINDING> raises <ALTERNATIVE>."
- PLAN: number your own steps as 1. 2. 3., one per line. Each line names an
  <ACTION> and the <FINDING THAT JUSTIFIES IT>.

NEVER: put a drug, test, referral, symptom, duration or exam finding in the note
that appears nowhere in the supplied context."""

# The visit-data block is inserted between the chart and the retrieved notes so
# the most authoritative, most specific data is closest to the instruction.
HUMAN_TEMPLATE = """Patient Context:
{patient_context}

{specialty_guidance}
{focus_block}{visit_context_block}
Relevant Clinical Notes:
{retrieved_notes}

Chief Complaint:
{chief_complaint}

{format_instructions}"""

# Acute-symptom phrases that make a visit problem-focused rather than a routine
# chronic review. Reused from the validation layer so the prompt and the
# Supervisor agree on what counts as a presenting symptom.
_SYMPTOM_PHRASES = tuple(phrase for phrase, _prefix, _label in CHIEF_COMPLAINT_CODE_HINTS)

# Wording for a chronic-disease follow-up, where no acute symptom was named.
_CHRONIC_FOCUS_BLOCK = """VISIT FOCUS: No new or acute symptom was named in the Chief Complaint.
This is a chronic-disease visit: lead the Assessment with how the documented
conditions and findings stand today, and make the Plan concrete about monitoring,
medication management and follow-up intervals."""


def _render_visit_context(visit_context: Optional[dict]) -> str:
    """Render the clinician-provided block, or "" when nothing was provided."""
    block = visit_context_to_prompt_block(visit_context)
    return f"{block}\n\n" if block else ""


def _render_focus_block(chief_complaint: str, visit_context: Optional[dict]) -> str:
    """Name the visit's primary problem so chronic context cannot crowd it out.

    Purely deterministic and text-based: it only reports symptoms the clinician
    actually wrote in the chief complaint, the HPI, or a specialty-specific
    field, and never asserts a diagnosis. An acute symptom found there means the
    Assessment and Plan must lead with it; otherwise the visit is treated as
    chronic-disease follow-up.

    Specialty field keys and labels are scanned too, because filling in a field
    like "Chest Pain Characteristics" is itself the clinician stating that the
    visit concerns that symptom. That keeps the rule configuration-driven: any
    specialty field whose name names a symptom anchors the visit.

    Returns an empty string when neither the complaint nor the visit context
    carries anything to anchor on, keeping the chief-complaint-only prompt
    unchanged.
    """
    haystack_parts = [str(chief_complaint or "")]
    if isinstance(visit_context, dict):
        haystack_parts.append(str(visit_context.get("hpi") or ""))
        for key, value in get_specialty_fields(visit_context).items():
            # Underscores read as spaces so a key like `chest_pain_characteristics`
            # matches the symptom phrase "chest pain".
            haystack_parts.extend((key.replace("_", " "), value))

    haystack = " ".join(parts for parts in haystack_parts if parts).lower()
    if not haystack.strip():
        return ""

    # Same negation-aware matcher as the symptom-code map, so the Scribe
    # and the Coding/Supervisor layers agree on what counts as a
    # presenting symptom. A denied symptom ("Denies chest pain") is not
    # an acute focus.
    found = [phrase for phrase in _SYMPTOM_PHRASES
             if phrase_present_not_negated(phrase, haystack)]
    if not found:
        return f"{_CHRONIC_FOCUS_BLOCK}\n\n"

    logger.info("Scribe focus anchor: acute symptom(s) %s", found)
    return (
        "VISIT FOCUS (highest priority): the Chief Complaint / HPI describes the "
        f"following new or acute symptom(s): {', '.join(found)}.\n"
        "Treat this as the PRIMARY problem for this visit. The Assessment must open "
        "with what these symptoms plus today's exam, vitals and test findings mean, "
        "and the Plan must address them first and specifically. Address the chronic "
        "conditions from Patient Context afterwards, and only briefly. Do not let "
        "chronic disease management replace the acute problem.\n\n"
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


# Generation budget per mode. A bounded Assessment (2-4 sentences) plus a real
# numbered Plan needs more than the old 512: the previous cap let a verbose
# Assessment consume every token and left Plan empty.
NUM_PREDICT = {"fast": 768, "quality": 1280}
DEFAULT_NUM_PREDICT = NUM_PREDICT["fast"]


def _num_predict(mode: str) -> int:
    return NUM_PREDICT.get(mode, DEFAULT_NUM_PREDICT)


def create_scribe_chain(mode: str = "fast", model_name: Optional[str] = None):
    """Create the Scribe Agent chain with JSON parser."""
    parser = JsonOutputParser(pydantic_object=SOAPNote)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])
    # Fast mode: limit tokens to prevent hanging
    num_predict = _num_predict(mode)
    if model_name:
        llm = get_llm(model_name=model_name, temperature=0.05, num_predict=num_predict)
    else:
        llm = get_llm(mode=mode, temperature=0.05, num_predict=num_predict)
    chain = prompt | llm | parser
    return chain


def _repair_json_output(raw_output: str, mode: str = "fast", model_name: Optional[str] = None) -> Optional[SOAPNote]:
    """Attempt to repair malformed JSON output using a repair prompt."""
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
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
    # Extract condition keywords from patient context
    condition_keywords = []
    for cond in ["diabetes", "hypertension", "htn", "asthma", "copd", "heart failure", "ckd", "allerg", "rhinitis"]:
        if cond in context_lower:
            condition_keywords.append(cond)
    
    # If no specific conditions found, return empty to avoid noise
    if not condition_keywords:
        logger.info("No condition keywords found in patient context, filtering out all retrieved notes")
        return ""
    
    logger.info(f"Patient conditions detected: {condition_keywords}")
    
    # Split notes by clinical note boundaries and filter
    notes = retrieved_notes.split("\n\n")
    filtered = []
    for note in notes:
        note_lower = note.lower()
        # Keep note if it mentions any of the patient's conditions
        if any(keyword in note_lower for keyword in condition_keywords):
            filtered.append(note)
            logger.debug(f"Kept note: {note[:80]}...")
        else:
            logger.debug(f"Filtered out note (irrelevant): {note[:80]}...")
    
    # Limit to max 2 notes, cap total length at 600 chars
    result = "\n\n".join(filtered[:2])
    if len(result) > 600:
        result = result[:600] + "..."
    
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
    """
    Generate a SOAP note from patient context and chief complaint.

    Args:
        visit_context: Optional clinician-provided encounter data - HPI, vitals,
            physical examination, additional findings (see
            `agents.visit_context`). When omitted the prompt is unchanged from
            the chief-complaint-only version.
        specialty: Optional medical specialty ID for specialty-specific guidance.
    """
    # Filter retrieved notes to only include content relevant to patient's conditions
    filtered_notes = _filter_retrieved_notes(retrieved_notes, patient_context)

    chain = create_scribe_chain(mode=mode, model_name=model_name)

    # Log prompt size for debugging
    format_instructions = chain.steps[-1].get_format_instructions() if hasattr(chain.steps[-1], 'get_format_instructions') else ""
    visit_block = _render_visit_context(visit_context)
    specialty_block = _render_specialty_guidance(specialty)
    focus_block = _render_focus_block(chief_complaint, visit_context)
    prompt_size = len(SYSTEM_PROMPT) + len(patient_context) + len(filtered_notes) + len(chief_complaint) + len(visit_block) + len(specialty_block) + len(focus_block) + len(format_instructions)
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

    # Defensive construction: a missing key (e.g. icd10_suggestions) is filled
    # with a safe default instead of raising - the JsonOutputParser hands back a
    # plain dict and the model regularly omits fields, especially on revisions.
    note = build_model(SOAPNote, result, source="scribe output")
    if has_content(note):
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
            if has_content(raw_note):
                return raw_note

        repaired = _repair_json_output(coerce_str(raw_output), mode=mode, model_name=model_name)
        if repaired and has_content(repaired):
            return repaired

    # Nothing salvageable came back. Return an empty, schema-valid note so the
    # pipeline and the Physician Review tab still render instead of crashing.
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
    from ..config import RAG_TOP_K
    retrieved_docs = vs.similarity_search("diabetes hypertension management", k=RAG_TOP_K)
    retrieved_notes = "\n\n".join([doc.page_content for doc in retrieved_docs])

    chief_complaint = "Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."

    soap = generate_soap_note(patient_context, retrieved_notes, chief_complaint, mode="fast")
    print(soap.model_dump_json(indent=2))