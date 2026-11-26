# IMPORTANT: Run this file with:
#   streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
# Do NOT execute with plain "python app/streamlit_app.py" - Streamlit needs its runtime.
# Do NOT run in background/nohup - it will start then immediately stop due to
# import-time atexit conflicts (sentence-transformers → scikit-learn → joblib → loky).

"""
Doctor Office Intelligent System - Multi-Agent Pipeline UI

Week 5: LangGraph + tools + light memory
- Memory load: reads prior *approved* visits for this patient
- Scribe: Generates structured SOAP notes (uses chart grounding tools)
- Cleanup: Grounding pass that removes hallucinated conditions
- Coding: Refines ICD-10-CM and suggests CPT codes (uses grounding + symptom tools)
- Validation: Final quality gate
- Supervisor: Routing / retries / escalation (zero-LLM)
- Memory save: records the run as an unsigned draft
- Orchestrator: LangGraph StateGraph, Central Executor pattern

The orchestrator owns the order of work; agents never call one another.

A pipeline run never becomes patient history on its own: it records a draft, and
only the physician's Approve in Physician Review promotes that draft into the
approved history the Scribe reads on the next visit.
"""

import json
import logging
import re
import time
import streamlit as st
from pathlib import Path
from utils.config import get_model_name, get_available_modes
from utils.review_state import regeneration_progress_message
from agents.specialty_profiles import get_specialty_profile, get_specialty_choices, get_default_specialty_id

logger = logging.getLogger(__name__)


# ─── Clinical CSS Theme ──────────────────────────────────────────────
CLINICAL_CSS = """
<style>
/* ─── Base ─── */
:root {
  --bg: #F8FAFC;
  --card: #FFFFFF;
  --border: #E2E8F0;
  --text: #1E293B;
  --text-muted: #64748B;
  --primary: #0D9488;
  --primary-hover: #0F766E;
  --secondary: #334155;
  --secondary-hover: #1E293B;
  --danger: #DC2626;
  --danger-hover: #B91C1C;
  --success: #16A34A;
  --warning: #D97706;
  --info: #2563EB;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-hover: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
  --radius: 10px;
  --radius-sm: 6px;
}

/* Force app background */
.stApp, .stApp > div, .stApp > div > div, .main, .main > div {
  background: var(--bg) !important;
}

/* ─── Status Banner ─── */
.status-banner {
  display: flex !important;
  align-items: center !important;
  gap: 1rem !important;
  padding: 1rem 1.25rem !important;
  border-radius: var(--radius) !important;
  margin-bottom: 1.25rem !important;
  font-weight: 500 !important;
  font-size: 0.95rem !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
.status-banner--clean {
  background: #ECFDF5 !important;
  border: 1px solid #A7F3D0 !important;
  color: #065F46 !important;
}
.status-banner--flags {
  background: #FFFAEB !important;
  border: 1px solid #FDE68A !important;
  color: #92400E !important;
}
.status-banner--review {
  background: #FEF2F2 !important;
  border: 1px solid #FECACA !important;
  color: #991B1B !important;
}
.status-banner__icon {
  flex-shrink: 0 !important;
  font-size: 1.25rem !important;
}
.status-banner__content {
  flex: 1 !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0.25rem !important;
}
.status-banner__title {
  font-weight: 600 !important;
}
.status-banner__meta {
  display: flex !important;
  gap: 1.5rem !important;
  font-size: 0.8rem !important;
  opacity: 0.85 !important;
}
.status-banner__meta span {
  display: flex !important;
  align-items: center !important;
  gap: 0.35rem !important;
}

/* ─── Patient Context Card ─── */
.patient-card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
  box-shadow: var(--shadow) !important;
}
.patient-card__header {
  display: flex !important;
  align-items: baseline !important;
  gap: 0.75rem !important;
  margin-bottom: 0.75rem !important;
  padding-bottom: 0.75rem !important;
  border-bottom: 1px solid var(--border) !important;
}
.patient-card__name {
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
}
.patient-card__id {
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
  background: var(--bg) !important;
  padding: 0.15rem 0.5rem !important;
  border-radius: 9999px !important;
}
.patient-card__demographics {
  display: flex !important;
  gap: 1rem !important;
  font-size: 0.85rem !important;
  color: var(--text-muted) !important;
  margin-bottom: 0.75rem !important;
}
.patient-card__section {
  margin-bottom: 0.75rem !important;
}
.patient-card__section-title {
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  color: var(--text-muted) !important;
  margin-bottom: 0.4rem !important;
  font-weight: 600 !important;
}
.patient-card__pills {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.35rem !important;
}
.pill {
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.35rem !important;
  padding: 0.25rem 0.6rem !important;
  font-size: 0.75rem !important;
  border-radius: 9999px !important;
  border: 1px solid transparent !important;
}
.pill--condition {
  background: #E0F2FE !important;
  color: #0369A1 !important;
  border-color: #BAE6FD !important;
}
.pill--medication {
  background: #F3E8FF !important;
  color: #7C3AED !important;
  border-color: #E9D5FF !important;
}
.pill--allergy {
  background: #FEF2F2 !important;
  color: #DC2626 !important;
  border-color: #FECACA !important;
}
.pill--value {
  background: #ECFDF5 !important;
  color: #065F46 !important;
  border-color: #A7F3D0 !important;
}
.pill--empty {
  background: var(--bg) !important;
  color: var(--text-muted) !important;
  border-color: var(--border) !important;
  font-style: italic !important;
}

/* ─── SOAP Cards ─── */
.soap-grid {
  display: grid !important;
  grid-template-columns: repeat(2, 1fr) !important;
  gap: 1rem !important;
  margin-bottom: 1rem !important;
  width: 100% !important;
}
@media (max-width: 900px) {
  .soap-grid {
    grid-template-columns: 1fr !important;
  }
}
.soap-card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
  box-shadow: var(--shadow) !important;
  height: 100% !important;
}
.soap-card__header {
  display: flex !important;
  align-items: center !important;
  gap: 0.5rem !important;
  margin-bottom: 0.75rem !important;
  padding-bottom: 0.5rem !important;
  border-bottom: 1px solid var(--border) !important;
}
.soap-card__icon {
  font-size: 1.1rem !important;
}
.soap-card__title {
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  color: var(--text) !important;
}
.soap-card__content {
  line-height: 1.6 !important;
  color: var(--text) !important;
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
}

/* ─── Coding Results ─── */
.coding-section {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.25rem !important;
  margin-bottom: 1rem !important;
  box-shadow: var(--shadow) !important;
}
.coding-section__header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: 1rem !important;
  padding-bottom: 0.75rem !important;
  border-bottom: 1px solid var(--border) !important;
}
.coding-section__title {
  font-weight: 600 !important;
  font-size: 1rem !important;
  color: var(--text) !important;
}
.coding-badges {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.5rem !important;
  margin-bottom: 1rem !important;
}
.badge {
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.4rem !important;
  padding: 0.35rem 0.75rem !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid !important;
}
.badge--primary {
  background: #F0FDFA !important;
  color: #0D9488 !important;
  border-color: #99F6E4 !important;
}
.badge--secondary {
  background: #EFF6FF !important;
  color: #2563EB !important;
  border-color: #BFDBFE !important;
}
.badge--cpt {
  background: #FEF9C3 !important;
  color: #854D0E !important;
  border-color: #FDE047 !important;
}
.badge--fast {
  background: #ECFDF5 !important;
  color: #16A34A !important;
  border-color: #A7F3D0 !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}
.cpt-detail {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1rem !important;
  margin-top: 0.75rem !important;
}
.cpt-detail__row {
  display: flex !important;
  justify-content: space-between !important;
  padding: 0.4rem 0 !important;
  border-bottom: 1px solid var(--border) !important;
}
.cpt-detail__row:last-child {
  border-bottom: none !important;
}
.cpt-detail__label {
  color: var(--text-muted) !important;
  font-size: 0.85rem !important;
}
.cpt-detail__value {
  font-weight: 500 !important;
  color: var(--text) !important;
}
.justification-box {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1rem !important;
  margin-top: 0.75rem !important;
  font-size: 0.85rem !important;
  line-height: 1.6 !important;
  color: var(--text) !important;
}

/* ─── Pipeline Progress ─── */
.pipeline-progress {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
  margin-bottom: 1rem !important;
}
.pipeline-progress__steps {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.5rem !important;
  margin-top: 0.75rem !important;
}
.pipeline-step {
  display: flex !important;
  align-items: center !important;
  gap: 0.4rem !important;
  padding: 0.35rem 0.75rem !important;
  font-size: 0.75rem !important;
  border-radius: 9999px !important;
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
}
.pipeline-step--done {
  background: #ECFDF5 !important;
  border-color: #A7F3D0 !important;
  color: #065F46 !important;
}
.pipeline-step--current {
  background: #EFF6FF !important;
  border-color: #BFDBFE !important;
  color: #1D4ED8 !important;
  font-weight: 500 !important;
}
.pipeline-step--pending {
  opacity: 0.5 !important;
}

/* ─── Override Streamlit defaults ─── */
.stApp header { display: none !important; }
.stApp footer { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }

/* Reduce default spacing */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 2rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem !important;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  padding: 0.625rem 1rem !important;
  font-weight: 500 !important;
  color: var(--text-muted) !important;
  transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
  background: var(--card) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-weight: 500 !important;
}
.streamlit-expanderContent {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  background: var(--bg) !important;
}

/* Button styling overrides */
.stButton > button {
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--secondary) !important;
  border-color: var(--secondary) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--secondary-hover) !important;
  border-color: var(--secondary-hover) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--card) !important;
  border-right: 1px solid var(--border) !important;
}

/* Code blocks */
.stCodeBlock, .stCodeBlock > div {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
}

/* Make sure markdown containers don't add extra spacing */
.stMarkdown {
  margin-bottom: 0 !important;
}
.stMarkdown > div {
  margin-bottom: 0 !important;
}
</style>
"""

def inject_clinical_css():
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)


@st.cache_data
def load_patients():
    path = Path(__file__).parent.parent / "data" / "synthetic" / "patients.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_internist_patients():
    """Load the 10 synthetic Internist patients for demo/testing."""
    path = Path(__file__).parent.parent / "data" / "synthetic" / "internist_patients.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_resource
def get_vectorstore():
    """Load vector store with Streamlit resource caching."""
    from rag.simple_rag import get_vectorstore as _get_vectorstore
    try:
        return _get_vectorstore()
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        st.info("Try running: python -m src.rag.simple_rag to rebuild the database")
        raise


def load_patient_history(patient_id: str):
    """Read the stored long-term memory for a patient (sidebar display).

    Deliberately uncached: the pipeline writes to this store during a run, so a
    cached copy would show a stale visit count until the session restarts.

    Returns approved/draft counts plus both kinds of row. The sidebar is an audit
    view, so it shows unsigned drafts too - labelled - while the headline count
    stays approved-only, because that is the number that means "visits this
    patient's record actually holds".
    """
    from memory.patient_memory import get_memory_store, visit_status
    try:
        store = get_memory_store()
        summary = store.get_summary(patient_id)
        visits = store.get_visit_history(patient_id, limit=5)
        for visit in visits:
            visit["display_status"] = visit_status(visit)
        return {
            "visit_count": int(summary.get("visit_count", 0) or 0),
            "approved_visit_count": int(summary.get("approved_visit_count", 0) or 0),
            "draft_visit_count": int(summary.get("draft_visit_count", 0) or 0),
            "visits": visits,
        }
    except Exception:
        return {
            "visit_count": 0,
            "approved_visit_count": 0,
            "draft_visit_count": 0,
            "visits": [],
        }


def render_specialty_selector() -> tuple[str, dict]:
    """Render specialty dropdown and collect specialty-specific extra fields.

    Returns:
        Tuple of (selected_specialty_id, extra_fields_dict).
    """
    st.header("Specialty")
    st.caption(
        "Select the clinical specialty for this encounter. Specialty-specific "
        "guidance will be provided to the Scribe Agent, and relevant optional "
        "extra fields will appear below."
    )

    specialty_choices = get_specialty_choices()
    default_specialty = get_default_specialty_id()

    # Find default index
    default_idx = next(
        (i for i, (sid, _) in enumerate(specialty_choices) if sid == default_specialty),
        0,
    )

    selected_specialty = st.selectbox(
        "Specialty",
        options=[sid for sid, _ in specialty_choices],
        format_func=lambda sid: next(name for s_id, name in specialty_choices if s_id == sid),
        index=default_idx,
        key="specialty_select",
    )

    # Render specialty-specific extra input fields
    profile = get_specialty_profile(selected_specialty)
    extra_fields = {}

    if profile.extra_input_fields:
        st.subheader(f"{profile.display_name} - Additional Fields")
        st.caption("Optional specialty-specific fields. Leave blank if not applicable.")

        for field_def in profile.extra_input_fields:
            key = f"specialty_{selected_specialty}_{field_def.key}"
            if field_def.field_type == "text_area":
                value = st.text_area(
                    field_def.label,
                    height=100,
                    key=key,
                    placeholder=field_def.placeholder,
                    help=field_def.help_text,
                )
                if value.strip():
                    extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "text_input":
                value = st.text_input(
                    field_def.label,
                    key=key,
                    placeholder=field_def.placeholder,
                    help=field_def.help_text,
                )
                if value.strip():
                    extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "select":
                value = st.selectbox(
                    field_def.label,
                    options=field_def.options,
                    key=key,
                    help=field_def.help_text,
                    index=0,
                )
                if value and value != "Not assessed" and value != "Select":
                    extra_fields[field_def.key] = value

    return selected_specialty, extra_fields


def build_patient_context(patient: dict) -> str:
    """Render the chart block for a patient.

    Single source of truth for the `patient_context` string: the pipeline, the
    memory store and the review screen must all see the same text, so a later
    write cannot disagree with what the note was generated from.
    """
    return f"""
Patient: {patient['name']}, {patient['age']}yo {patient['gender']}
Conditions: {', '.join(patient['conditions'])}
Medications: {', '.join(patient['medications'])}
Allergies: {', '.join(patient['allergies']) or 'None'}
Last Visit: {patient['last_visit_summary']}
"""


def chart_text_for_review(patient_id: str) -> str:
    """Chart text for the visit under review.

    Prefers the exact context the reviewed run was given (kept in session state)
    and falls back to rebuilding it from the selected patient.
    """
    params = st.session_state.pipeline_params or {}
    if params.get("patient_context"):
        return params["patient_context"]
    for candidate in load_patients():
        if str(candidate.get("patient_id")) == str(patient_id):
            return build_patient_context(candidate)
    return f"Patient ID: {patient_id}\n"


def run_pipeline(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str,
    mode: str = "fast",
    patient_id: str = None,
    use_memory: bool = True,
    visit_context: dict = None,
):
    """Blocking call into the LangGraph orchestrator (same signature as before)."""
    from agents.orchestrator import run_visit_pipeline
    from utils.config import is_known_synthetic_patient
    return run_visit_pipeline(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        patient_id=patient_id,
        use_memory=use_memory,
        visit_context=visit_context,
        is_synthetic=is_known_synthetic_patient(patient_id),
    )


def run_pipeline_streaming(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str,
    mode: str = "fast",
    patient_id: str = None,
    use_memory: bool = True,
    visit_context: dict = None,
):
    """Yield a partial result after each LangGraph node completes.

    Same output shape as `run_pipeline`, but delivered progressively so the UI
    can show which agent is currently working.
    """
    from agents.orchestrator import stream_visit_pipeline
    from utils.config import is_known_synthetic_patient
    yield from stream_visit_pipeline(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        patient_id=patient_id,
        use_memory=use_memory,
        visit_context=visit_context,
        is_synthetic=is_known_synthetic_patient(patient_id),
    )


def render_visit_context_inputs(specialty_extra_fields: dict = None) -> dict:
    """Render the optional encounter-data section and collect its values.

    Every field is optional: an empty dict keeps the chief-complaint-only
    behaviour of earlier versions. Returns a normalised dict with only the
    fields the clinician actually filled in.
    """
    from agents.visit_context import normalize_visit_context

    st.header("Encounter Details")
    st.caption(
        "Optional. Anything left blank is treated as *not documented today* — "
        "the Scribe Agent will not invent it."
    )

    raw = {}

    # Include specialty-specific extra fields at the top if provided
    if specialty_extra_fields:
        for key, value in specialty_extra_fields.items():
            raw[key] = value

    hpi = st.text_area(
        "History of Present Illness (HPI)",
        height=120,
        key="vc_hpi",
        placeholder=(
            "Patient-reported story behind the visit, in their own terms. "
            "e.g., 'Fatigue for 3 weeks, worse in the afternoons. Drinking more "
            "water than usual. Denies chest pain or shortness of breath.'"
        ),
    )
    if hpi.strip():
        raw["hpi"] = hpi

    st.subheader("Vitals")
    bp_col, hr_col, temp_col, rr_col, spo2_col = st.columns(5)
    bp = bp_col.text_input("BP (mmHg)", placeholder="148/92", key="vc_bp")
    hr = hr_col.text_input("HR (bpm)", placeholder="78", key="vc_hr")
    temp = temp_col.text_input("Temp (°F)", placeholder="98.6", key="vc_temp")
    rr = rr_col.text_input("RR (/min)", placeholder="16", key="vc_rr")
    spo2 = spo2_col.text_input("SpO₂ (%)", placeholder="98", key="vc_spo2")
    for key, value in (
        ("blood_pressure", bp),
        ("heart_rate", hr),
        ("temperature", temp),
        ("respiratory_rate", rr),
        ("spo2", spo2),
    ):
        if value.strip():
            raw[key] = value

    weight_col, bmi_col = st.columns(2)
    weight = weight_col.text_input("Weight (kg) — optional", placeholder="72.5", key="vc_weight")
    bmi = bmi_col.text_input("BMI — optional", placeholder="26.4", key="vc_bmi")
    if weight.strip():
        raw["weight_kg"] = weight
    if bmi.strip():
        raw["bmi"] = bmi

    vitals_note = st.text_area(
        "Other vitals / observations — optional",
        height=70,
        key="vc_vitals_note",
        placeholder="e.g., 'Seated BP, repeat 148/90', 'Pain 3/10 with movement'",
    )
    if vitals_note.strip():
        raw["vitals_note"] = vitals_note

    exam = st.text_area(
        "Physical Examination",
        height=120,
        key="vc_exam",
        placeholder=(
            "Only what you actually observed. "
            "e.g., 'Dry mucous membranes. No peripheral edema. Foot skin intact, "
            "sensation intact to monofilament.'"
        ),
    )
    if exam.strip():
        raw["physical_exam"] = exam

    findings = st.text_area(
        "Additional Findings / Test Results — optional",
        height=100,
        key="vc_findings",
        placeholder="e.g., 'HbA1c 7.8% (was 7.2)', 'Urine microalbumin 42 mg/g', 'CXR: no infiltrate'",
    )
    if findings.strip():
        raw["additional_findings"] = findings

    return normalize_visit_context(raw)


def render_visit_context_summary(visit_context: dict, specialty_fields: dict = None) -> None:
    """Show the encounter data the note was generated from, in Physician Review.

    The specialty itself is shown by the caller in the patient block, next to
    the chief complaint, because it decides how the whole note is read. This
    function covers the encounter fields and the specialty's own extra inputs,
    which travel in `visit_context["specialty_fields"]` and therefore survive a
    regeneration without a second copy in session state.
    """
    from agents.visit_context import format_vitals

    if not visit_context and not specialty_fields:
        st.caption("No encounter details were entered for this visit — chief complaint only.")
        return

    if specialty_fields:
        st.markdown("**Specialty-Specific Fields:**")
        for key, value in specialty_fields.items():
            # Convert snake_case to Title Case for display
            display_key = key.replace("_", " ").title()
            st.markdown(f"- **{display_key}:** {value}")

    st.markdown("#### 🗒️ Visit Context Used")
    for label, key in (
        ("HPI", "hpi"),
        ("Physical Exam", "physical_exam"),
        ("Additional Findings / Tests", "additional_findings"),
        ("Other Vitals / Observations", "vitals_note"),
    ):
        value = visit_context.get(key)
        if value:
            st.markdown(f"**{label}:** {value}")

    vitals = format_vitals(visit_context)
    if vitals:
        st.markdown(f"**Vitals:** {vitals}")

    st.caption(
        "Only the information above (plus the patient chart) may appear as fact "
        "in the note. Review any measurement you cannot confirm."
    )


def render_symptom_coding_summary(
    metadata: dict,
    coding_result: dict,
    heading: str = "🩺 Symptom coding",
    compact: bool = False,
) -> None:
    """Show which symptom codes were considered and where each one came from.

    Reads `metadata["symptom_coding"]`, which the orchestrator records from what
    the Coding node and the Supervisor already did. Purely a read: this panel
    never changes the code list, it only explains it.

    Args:
        metadata: The run's `metadata` dict.
        coding_result: The run's `coding_result`, used to show which candidates
            actually made it into the final list.
        heading: Section heading.
        compact: One summary line instead of the full per-candidate table, used
            in Physician Review.
    """
    from utils.review_presentation import (
        origin_label,
        source_label,
        symptom_coding_summary,
    )

    summary = symptom_coding_summary(metadata, coding_result)
    if not summary.rows:
        st.caption(
            "No symptom-code candidates were derived for this visit - no symptom "
            "from the chief complaint or the encounter data matched the code map."
        )
        return

    st.markdown(f"**{heading}**")
    parts = []
    if summary.coding_added:
        parts.append(f"{summary.coding_added} added by Coding")
    if summary.supervisor_added:
        parts.append(f"{summary.supervisor_added} added by Supervisor")
    if summary.already_present:
        parts.append(f"{summary.already_present} already coded")
    st.caption(
        f"{summary.considered} candidate(s) considered · "
        + (" · ".join(parts) if parts else "none completed")
    )

    if compact:
        for row in summary.rows:
            st.markdown(
                f"- `{row.code}` {origin_label(row.origin)}"
                + (f" — “{row.phrase}” from {source_label(row.source)}" if row.phrase else "")
            )
        return

    st.table(
        [
            {
                "Code": f"`{row.code}`",
                "Symptom": row.phrase or "—",
                "Found in": source_label(row.source),
                "Outcome": origin_label(row.origin),
            }
            for row in summary.rows
        ]
    )
    if summary.missing:
        st.caption(
            f"{summary.missing} candidate(s) did not reach the final code list - "
            "the physician decides whether to add them."
        )


def render_quality_flags_grouped(quality_flags: list, heading: str = "Quality flags") -> None:
    """Render quality flags in readable groups, keeping the raw text visible.

    Three buckets, in reading order: completed automatically, review advised,
    removed by a safety check. Every entry shows the plain-language label and,
    underneath it, the untouched raw flag for anyone debugging. The Raw JSON tab
    still carries the original list, so nothing is lost by grouping.
    """
    from utils.review_presentation import flag_groups

    groups = flag_groups(quality_flags)
    if not groups:
        st.success("No quality flags and no auto-corrections.")
        return

    st.markdown(f"**{heading}**")
    for group in groups:
        st.markdown(f"**{group['title']}**")
        st.caption(group["caption"])
        for flag in group["flags"]:
            level = getattr(st, group["severity"], st.write)
            level(f"• {flag.label}")
            with st.expander("Technical detail"):
                st.caption(flag.detail)


def render_specialty_fidelity_notes(quality_flags: list) -> None:
    """One plain line per specialty field the note under-reflected."""
    from utils.review_presentation import specialty_fidelity_notes

    notes = specialty_fidelity_notes(quality_flags)
    if not notes:
        return
    st.markdown("**📐 Specialty field fidelity**")
    st.caption(
        "The clinician recorded these today; the note carries them less clearly "
        "than it should."
    )
    for note in notes:
        st.warning(f"• {note}")


def specialty_display_name(specialty: str) -> str:
    """Display name for a specialty id, tolerant of an unrecognised value."""
    if not specialty:
        return ""
    try:
        return get_specialty_profile(specialty).display_name
    except Exception:
        return str(specialty)


def review_feedback_key() -> str:
    """Widget key for the revision-feedback box of the draft on screen.

    Keyed by how many revisions were already submitted, so each round opens an
    empty box instead of re-showing the feedback that was just sent, and a
    draft in the `changes_requested` state always has somewhere to read from.
    """
    rounds = len(st.session_state.get("review_feedback_history") or [])
    return f"review_feedback_input_{rounds}"


def resolve_review_state(result: dict, use_memory: bool):
    """The single answer to "which state is this review in, and what now?".

    Reads only session state the app already keeps, so the banner and the
    decision buttons cannot disagree about the state of the review.

    P2 keeps two facts apart: `draft_recorded` (the pipeline wrote an unsigned
    draft row for this run) and `approved` (this visit is in the patient's
    history, which only an Approve can make true).
    """
    from utils.review_state import review_state

    meta = result.get("metadata", {}) or {}
    memory_meta = meta.get("memory") or {}
    status = st.session_state.get("review_status")
    return review_state(
        status,
        needs_human_review=bool(
            (meta.get("supervisor") or {}).get("needs_human_review")
        ),
        approved=bool(st.session_state.get("review_persisted")),
        draft_recorded=bool(memory_meta.get("saved")),
        memory_enabled=bool(use_memory),
        error=st.session_state.get("review_error"),
        revision_count=len(st.session_state.get("review_feedback_history") or []),
        feedback_text=st.session_state.get(review_feedback_key()) or "",
    )


def render_review_status_banner(state) -> None:
    """The authoritative statement of what this note is and what happens next.

    Deliberately the first thing in Physician Review, and the only place a
    decision is announced: Approve, Reject and Revise all write session state
    and re-run, so their wording comes from here rather than from a toast that
    disappears on the next click.
    """
    level = getattr(st, state.level, st.info)
    level(f"**{state.label}** — {state.detail}")
    if state.next_step:
        st.caption(f"Next: {state.next_step}")


def render_revision_history() -> None:
    """Which feedback produced the draft on screen. Deliberately not a diff.

    A full comparison of two generated notes would claim more certainty than a
    string diff earns, so this only names the feedback that was sent, and says
    plainly that the earlier draft is gone.
    """
    from utils.review_state import revision_summary

    lines = revision_summary(st.session_state.get("review_feedback_history"))
    if not lines:
        return
    st.caption(
        f"{len(lines)} revision round(s) in this session. The earlier draft is "
        "not kept or diffed — only the version below is on screen."
    )
    with st.expander("Feedback that produced this draft"):
        for line in lines:
            st.markdown(f"- {line}")


def render_approval_preview(preview) -> None:
    """Say what an Approve will store, before it is clicked."""
    st.caption(preview.summary)
    if preview.warning:
        st.warning(preview.warning)
    with st.expander("Exactly what gets stored"):
        for line in preview.detail_lines:
            st.markdown(f"- {line}")


def render_physician_review_tab(result: dict, patient: dict, chief_complaint: str, use_memory: bool, specialty: str = None, specialty_extra_fields: dict = None):
    """Render the Physician Review tab: state, content, then decisions.

    Reading order is deliberate — status first so the physician knows what kind
    of note this is before reading it, encounter data before the note it
    produced, provenance and quality flags after the note, and the three
    decision buttons last so the decision follows the reading.

    The state resolved at the top gates every button, so an action that makes no
    sense in the current state is disabled rather than silently re-runnable.
    """
    from utils.review_state import approval_preview

    st.header("👨‍⚕️ Physician Review")
    st.caption(
        "Review the generated clinical note, codes, and quality flags before "
        "finalizing. Your decision determines whether this visit stands as final."
    )

    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result.get("metadata", {}) or {}
    supervisor_meta = meta.get("supervisor", {}) or {}
    quality_flags = result.get("quality_flags", []) or []
    review_flags = [f for f in quality_flags if not f.startswith("Cleanup: ")]

    # The run's own specialty wins over the sidebar selection: it is what this
    # note was actually written with, and it is what a regeneration repeats.
    specialty = meta.get("specialty") or specialty
    specialty_name = specialty_display_name(specialty)

    # The reviewed run's own encounter data, with the sidebar's current copy as
    # a fallback for a result that carries none.
    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    specialty_fields = (
        visit_context_used.get("specialty_fields")
        or specialty_extra_fields
        or {}
    )

    # --- 1. Review state, before anything else ----------------------------
    state = resolve_review_state(result, use_memory)
    render_review_status_banner(state)
    render_revision_history()

    if supervisor_meta.get("needs_human_review"):
        st.error(
            f"⚠️ **ESCALATED FOR HUMAN REVIEW** — The pipeline exhausted retries "
            f"({supervisor_meta.get('retry_count', 0)}) and could not auto-correct all issues. "
            "This run recorded nothing for this patient - not even a draft. Please review carefully."
        )
    elif supervisor_meta.get("applied"):
        st.warning(
            "🔧 **AUTO-CORRECTED BY SUPERVISOR** — The following deterministic fixes were applied "
            f"({supervisor_meta.get('retry_count', 0)} retry pass(es)):"
        )
        for fix in supervisor_meta["applied"]:
            st.caption(f"• {fix}")
    elif review_flags:
        st.info("⚠️ **QUALITY FLAGS PRESENT** — Review flags before signing.")
    else:
        st.success("✅ **CLEAN** — No quality flags, no auto-corrections needed.")

    st.divider()

    # --- 2. Patient, specialty and reason for the visit --------------------
    st.subheader("👤 Patient & Encounter")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Name:** {patient['name']}")
        st.write(f"**Age:** {patient['age']} · **Gender:** {patient['gender']}")
        st.write(f"**Patient ID:** {patient['patient_id']}")
    with col2:
        st.write(f"**Specialty:** {specialty_name or 'General (no specialty selected)'}")
        st.write(f"**Conditions:** {', '.join(patient['conditions'])}")
        st.write(f"**Medications:** {', '.join(patient['medications']) or 'None'}")
    with col3:
        st.write(f"**Allergies:** {', '.join(patient['allergies']) or 'None'}")
        st.write(f"**Prior approved visits read:** {(meta.get('memory') or {}).get('prior_visits', 0)}")
        mem_meta = meta.get("memory") or {}
        st.write(
            "**Recorded by pipeline:** "
            + (f"draft {mem_meta.get('visit_id')}" if mem_meta.get("saved") else "No")
        )
        # `state.in_history` is the single answer, and it knows about an Approve
        # that happened after this run: the run metadata always says
        # approved=False, because a pipeline run cannot approve anything.
        st.write(
            "**In patient history:** Yes (approved)"
            if state.in_history
            else "**In patient history:** No - not history until approved"
        )

    st.markdown("**Chief complaint**")
    st.info(chief_complaint or "_none recorded_")

    # --- 3. Encounter data behind the note (collapsed: supporting detail) ---
    with st.expander("🗒️ Visit context & specialty fields", expanded=False):
        render_visit_context_summary(visit_context_used, specialty_fields)

    st.divider()

    # --- 4. SOAP Note ---
    st.subheader("📋 Generated SOAP Note")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Subjective**")
        st.write(soap["subjective"])
        st.markdown("**Objective**")
        st.write(soap["objective"])
    with col2:
        st.markdown("**Assessment**")
        st.write(soap["assessment"])
        st.markdown("**Plan**")
        st.write(soap["plan"])

    st.divider()

    # --- 5. ICD-10 Codes ---
    st.subheader("🔖 ICD-10-CM Codes")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primary**")
        st.code(coding["primary_icd10"])
    with col2:
        st.markdown("**Secondary**")
        for code in coding["secondary_icd10"]:
            st.code(code)

    st.divider()

    # --- 6. CPT Code ---
    st.subheader("💰 CPT Code (Deterministic E/M)")
    cpt_grounding = coding.get("cpt_grounding")
    if cpt_grounding:
        st.write(f"**Selected CPT:** `{cpt_grounding.get('cpt_code', 'N/A')}`")
        st.write(f"**MDM Level:** {cpt_grounding.get('overall_mdm_level', 'N/A')}")
        with st.expander("MDM Domain Breakdown"):
            domains = cpt_grounding.get("domains", {})
            for domain, level in domains.items():
                st.write(f"- {domain}: **{level}**")
        with st.expander("Justification"):
            st.write(cpt_grounding.get("justification", ""))
    else:
        st.write("**CPT Suggestions (from model):**")
        for code in coding["cpt_suggestions"]:
            st.code(code)
        st.warning("Deterministic calculator output not available; showing model suggestions.")

    st.divider()

    # --- 7. Symptom coding provenance (Phase A4) ---
    render_symptom_coding_summary(meta, coding, compact=True)

    st.divider()

    # --- 8. Quality flags & Supervisor Actions (Phase A4) ---
    st.subheader("🚩 Quality Flags & Supervisor Actions")

    if supervisor_meta.get("applied"):
        st.markdown("**🔧 Supervisor Auto-Applied Fixes**")
        for fix in supervisor_meta["applied"]:
            st.info(f"• {fix}")

    # Grouped, plain-language view. The raw flags remain in the Raw JSON tab.
    render_specialty_fidelity_notes(quality_flags)
    render_quality_flags_grouped(quality_flags)

    if supervisor_meta.get("feedback"):
        with st.expander("💬 Supervisor Feedback (passed to retry passes)"):
            for item in supervisor_meta["feedback"]:
                st.markdown(f"- {item}")

    st.divider()

    # --- 9. Decision, gated by the state resolved above --------------------
    st.subheader("✍️ Physician Decision")

    draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not state.in_history
    if state.key == "changes_requested":
        st.caption(
            "Approve, Request Changes and Reject are closed on this draft until "
            "the revision below is submitted. A regeneration is a new pipeline "
            "run, not an edit to this note - the earlier run stays in long-term "
            "memory as its own unsigned draft, and approving the revision "
            "approves only the new one."
        )
    elif state.allow_approve or state.allow_request_changes:
        # Only meaningful while a decision is still open: after one is taken the
        # banner is the record of what happened, and re-describing the write
        # next to a disabled Approve button would only invite the wrong click.
        render_approval_preview(
            approval_preview(
                result,
                patient_name=patient.get("name", ""),
                patient_id=patient.get("patient_id"),
                chief_complaint=chief_complaint or params.get("chief_complaint", ""),
                specialty_name=specialty_name or None,
                memory_enabled=bool(use_memory),
                draft_recorded=draft_recorded,
            )
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve & Save", type="primary", use_container_width=True,
                     disabled=not state.allow_approve):
            handle_approve(result, patient["patient_id"], use_memory)

    with col2:
        if st.button("✏️ Request Changes", use_container_width=True,
                     disabled=not state.allow_request_changes):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()

    with col3:
        if st.button("🗑️ Reject", use_container_width=True,
                     disabled=not state.allow_reject):
            handle_reject()

    # The way back into reviewing a note the physician has already decided on.
    if state.allow_revise:
        if st.button("✏️ Revise this note (new draft)", use_container_width=True):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()
        st.caption(
            "Revising regenerates the note with your feedback and supersedes this "
            "draft for review: the new run becomes the version on screen and the "
            "one Approve would sign. Each run keeps its own row in long-term "
            "memory — earlier versions stay as unsigned drafts, so nothing that "
            "was superseded can silently become history."
        )

    # Feedback box, only while a revision is in progress.
    if state.key == "changes_requested":
        st.markdown("**Revision feedback**")
        st.caption(
            "This text is sent into the *next pipeline run*: the Scribe and the "
            "Coding agent both receive it as a revision instruction. It is not "
            "stored with the visit, and nothing on this screen is edited — the "
            "run produces a new draft to review."
        )
        feedback_key = review_feedback_key()
        st.text_area(
            "Feedback for the next run (Scribe + Coding)",
            key=feedback_key,
            height=110,
            placeholder=(
                "e.g., 'Add ICD-10 for chest pain (R07.9)', 'Change CPT to "
                "99214 - moderate MDM', 'The objective section must carry the "
                "ECG findings I entered.'"
            ),
        )
        if st.button(
            "🔄 Submit Feedback & Regenerate",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.get(feedback_key) or "").strip(),
        ):
            submit_revision(
                (st.session_state.get(feedback_key) or "").strip(),
                result,
                patient,
                chief_complaint,
                specialty,
            )


def submit_revision(feedback: str, result: dict, patient: dict, chief_complaint: str, specialty: str) -> None:
    """Queue a feedback-driven regeneration of the reviewed run.

    Everything except the feedback is replayed unchanged — the same chart, chief
    complaint, mode, encounter data and specialty — so the revision is judged
    against the same clinical facts as the draft it replaces. `should_run_pipeline`
    then drives the ordinary streaming block, so a revision is a real pipeline
    run (Scribe + Coding see the feedback) rather than a text substitution.
    """
    params = st.session_state.pipeline_params or {}
    meta = result.get("metadata", {}) or {}

    st.session_state.review_feedback_history = list(
        st.session_state.get("review_feedback_history") or []
    ) + [feedback]
    st.session_state.review_feedback = feedback
    # Kept as `changes_requested` while the run is in flight, so a failed
    # regeneration leaves the feedback on screen instead of pretending the
    # draft is still awaiting a decision.
    st.session_state.review_status = "changes_requested"
    st.session_state.review_error = None
    st.session_state.review_persisted = None

    st.session_state.pipeline_params = {
        "patient_context": build_patient_context(patient),
        "chief_complaint": params.get("chief_complaint") or chief_complaint,
        "mode": params.get("mode") or selected_mode,
        "patient_id": patient["patient_id"],
        "use_memory": params.get("use_memory", True),
        "physician_feedback": feedback,
        # Reuse the encounter data the reviewed note was built from, so a
        # revision is judged against the same clinical facts.
        "visit_context": (meta.get("visit_context") or params.get("visit_context") or {}),
        "specialty": specialty,
        "specialty_extra_fields": params.get("specialty_extra_fields"),
    }
    st.session_state.should_run_pipeline = True
    st.rerun()


def handle_approve(result: dict, patient_id: str, use_memory: bool):
    """Handle the Approve & Save decision - the only path into patient history.

    The pipeline's memory-save step records the run as a **draft**; it can never
    approve anything. An Approve therefore always goes through
    `PatientMemory.approve_visit`, which promotes the draft row this run wrote
    (the row id comes back in run metadata, so a revision run approves its own
    draft and not the one it replaced) or writes an approved row when there is no
    draft to promote - an escalated visit, a run with memory off, or a draft that
    was trimmed away. Idempotent, so a double click cannot double-count a visit.

    `approve_visit` takes keyword arguments - `patient_context` is the chart
    *text* the run was given, not a visit dict - so every field is passed
    explicitly. A failure is logged and surfaced through the status banner, and
    the state machine keeps Approve enabled so the physician can retry instead
    of losing the visit silently.

    No wording is chosen here: the banner in `render_review_status_banner` is
    the single place a decision is announced, so Approve, Reject and Revise
    cannot drift apart.
    """
    if not use_memory:
        st.session_state.review_status = "approved"
        st.session_state.review_persisted = False
        st.session_state.review_error = None
        st.rerun()
        return

    meta = result.get("metadata", {}) or {}
    mem_meta = meta.get("memory") or {}
    params = st.session_state.pipeline_params or {}
    patient_context = chart_text_for_review(patient_id)
    chief_complaint = params.get("chief_complaint", "") or ""

    from memory.patient_memory import get_memory_store

    try:
        get_memory_store().approve_visit(
            patient_id=patient_id,
            visit_id=mem_meta.get("visit_id") or None,
            patient_context=patient_context,
            chief_complaint=chief_complaint,
            coding_result=result.get("coding_result") or {},
            soap_note=result.get("soap_note") or {},
            quality_flags=list(result.get("quality_flags") or []),
        )
    except Exception as exc:
        logger.exception("Approve & Save failed for patient %s", patient_id)
        st.session_state.review_error = str(exc)
        st.session_state.review_status = None  # keep Approve clickable to retry
        st.rerun()
        return

    logger.info(
        "Visit approved into patient history for %s (promoted pipeline draft=%s)",
        patient_id, bool(mem_meta.get("visit_id")),
    )
    st.session_state.review_status = "approved"
    st.session_state.review_persisted = True
    st.session_state.review_error = None
    st.rerun()


def handle_reject():
    """Handle the Reject decision.

    A rejection is a statement about the note, not an edit - and explicitly not
    an approval: nothing becomes part of this patient's history. `reject` never
    writes, and there is no un-store operation either, so the unsigned draft the
    pipeline recorded stays in the store for audit: uncounted, and unread by the
    Scribe. The banner says exactly that rather than implying the record was
    removed.

    The review state stays terminal but not stuck: the Revise button and Run
    Pipeline both start again from a clean state.
    """
    st.session_state.review_status = "rejected"
    st.session_state.review_persisted = False
    st.session_state.review_error = None
    st.rerun()


# ─── Clinical UI Rendering Functions (Sprint 1) ──────────────────────

def render_status_banner(result: dict, meta: dict, total_time: float = None):
    """Render the prominent status banner after pipeline completion."""
    supervisor_meta = meta.get("supervisor", {}) or {}
    quality_flags = result.get("quality_flags", []) or []
    coding_path = meta.get("coding_path", "slow")
    visit_id = meta.get("memory", {}).get("visit_id", "—")
    flag_count = len([f for f in quality_flags if not f.startswith("Cleanup:")])
    
    # Determine status
    if supervisor_meta.get("needs_human_review"):
        status_class = "status-banner--review"
        icon = "🚨"
        title = "Needs Human Review"
        detail = f"Pipeline escalated after {supervisor_meta.get('retry_count', 0)} retries — review required before approval"
    elif flag_count > 0:
        status_class = "status-banner--flags"
        icon = "⚠️"
        title = "Completed with Quality Flags"
        detail = f"{flag_count} quality flag(s) present — review recommended before approval"
    else:
        status_class = "status-banner--clean"
        icon = "✅"
        title = "Clean Run"
        detail = "No critical issues detected"
    
    meta_items = []
    if total_time is not None:
        meta_items.append(f"⏱️ {total_time:.1f}s total")
    meta_items.append(f"🔧 Coding: {coding_path.upper()}")
    meta_items.append(f"📄 Visit: {visit_id}")
    meta_items.append(f"🚩 Flags: {flag_count}")
    
    meta_html = "".join(f'<span>{item}</span>' for item in meta_items)
    
    st.markdown(f"""
    <div class="status-banner {status_class}">
        <span class="status-banner__icon">{icon}</span>
        <div class="status-banner__content">
            <span class="status-banner__title">{title}</span>
            <div class="status-banner__meta">{meta_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_patient_card(patient: dict, meta: dict = None, visit_context: dict = None):
    """Render the compact patient context card."""
    # Extract vitals from visit_context if available
    bp = ""
    hba1c = ""
    if visit_context:
        bp = visit_context.get("blood_pressure", "")
        # Try to find HbA1c in additional_findings
        findings = visit_context.get("additional_findings", "")
        if findings and "hba1c" in findings.lower():
            import re
            match = re.search(r'HbA1c\s+([\d.]+%)', findings, re.IGNORECASE)
            if match:
                hba1c = match.group(1)
    
    # Conditions as pills
    conditions = patient.get("conditions", [])
    conditions_html = "".join(
        f'<span class="pill pill--condition">{c}</span>' for c in conditions
    ) or '<span class="pill pill--empty">No conditions listed</span>'
    
    # Medications (top 3)
    meds = patient.get("medications", [])
    meds_html = "".join(
        f'<span class="pill pill--medication">{m}</span>' for m in meds[:3]
    ) or '<span class="pill pill--empty">No medications</span>'
    if len(meds) > 3:
        meds_html += f'<span class="pill pill--empty">+{len(meds) - 3} more</span>'
    
    # Allergies
    allergies = patient.get("allergies", [])
    allergies_html = "".join(
        f'<span class="pill pill--allergy">{a}</span>' for a in allergies
    ) or '<span class="pill pill--allergy">NKDA</span>'
    
    # Key values
    values = []
    if bp:
        values.append(f'<span class="pill pill--value">BP: {bp}</span>')
    if hba1c:
        values.append(f'<span class="pill pill--value">HbA1c: {hba1c}</span>')
    values_html = "".join(values) if values else '<span class="pill pill--empty">No recent values</span>'
    
    st.markdown(f"""
    <div class="patient-card">
        <div class="patient-card__header">
            <span class="patient-card__name">{patient.get('name', 'Unknown')}</span>
            <span class="patient-card__id">{patient.get('patient_id', '—')}</span>
        </div>
        <div class="patient-card__demographics">
            <span>{patient.get('age', '?')}y</span>
            <span>·</span>
            <span>{patient.get('gender', '?')}</span>
        </div>
        <div class="patient-card__section">
            <div class="patient-card__section-title">Conditions</div>
            <div class="patient-card__pills">{conditions_html}</div>
        </div>
        <div class="patient-card__section">
            <div class="patient-card__section-title">Medications</div>
            <div class="patient-card__pills">{meds_html}</div>
        </div>
        <div class="patient-card__section">
            <div class="patient-card__section-title">Allergies</div>
            <div class="patient-card__pills">{allergies_html}</div>
        </div>
        <div class="patient-card__section">
            <div class="patient-card__section-title">Key Values</div>
            <div class="patient-card__pills">{values_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_soap_cards(soap: dict):
    """Render SOAP note as four distinct cards in a 2x2 grid."""
    sections = [
        ("Subjective", "💬", soap.get("subjective", "")),
        ("Objective", "🔍", soap.get("objective", "")),
        ("Assessment", "🧠", soap.get("assessment", "")),
        ("Plan", "📋", soap.get("plan", "")),
    ]
    
    cards_html = ""
    for title, icon, content in sections:
        safe_content = (content or "_No content_").replace("\n", "<br>")
        cards_html += f"""
        <div class="soap-card">
            <div class="soap-card__header">
                <span class="soap-card__icon">{icon}</span>
                <span class="soap-card__title">{title}</span>
            </div>
            <div class="soap-card__content">{safe_content}</div>
        </div>
        """
    
    st.markdown(f'<div class="soap-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_coding_results(coding: dict, meta: dict = None):
    """Render coding results with badges and structured CPT display."""
    primary = coding.get("primary_icd10", "—")
    secondary = coding.get("secondary_icd10", [])
    cpt_suggestions = coding.get("cpt_suggestions", [])
    cpt_grounding = coding.get("cpt_grounding")
    medical_necessity = coding.get("medical_necessity_note", "")
    coding_path = meta.get("coding_path", "slow") if meta else "slow"
    
    # Extract CPT details
    cpt_code = "—"
    mdm_level = "—"
    domains = {}
    justification = ""
    
    if cpt_grounding:
        cpt_code = cpt_grounding.get("cpt_code", "—")
        mdm_level = cpt_grounding.get("overall_mdm_level", "—")
        domains = cpt_grounding.get("domains", {})
        justification = cpt_grounding.get("justification", "")
    elif cpt_suggestions:
        # Parse first suggestion
        first = cpt_suggestions[0]
        if " - " in first:
            cpt_code, _ = first.split(" - ", 1)
        else:
            cpt_code = first
    
    # Primary ICD badge
    primary_html = f'<span class="badge badge--primary">{primary}</span>'
    
    # Secondary ICD badges
    secondary_html = ""
    if secondary:
        for code in secondary:
            secondary_html += f'<span class="badge badge--secondary">{code}</span>'
    else:
        secondary_html = '<span class="badge badge--secondary">None</span>'
    
    # CPT badge
    cpt_badge = f'<span class="badge badge--cpt">CPT {cpt_code} · MDM: {mdm_level}</span>'
    
    # Fast Path badge
    fast_badge = f'<span class="badge badge--fast">Fast Path</span>' if coding_path == "fast" else ""
    
    # Domain breakdown
    domains_html = ""
    if domains:
        for domain, level in domains.items():
            domains_html += f"""
            <div class="cpt-detail__row">
                <span class="cpt-detail__label">{domain.title()}</span>
                <span class="cpt-detail__value">{level}</span>
            </div>
            """
    
    st.markdown(f"""
    <div class="coding-section">
        <div class="coding-section__header">
            <span class="coding-section__title">Coding Results</span>
            <div>{fast_badge}</div>
        </div>
        
        <div class="coding-badges">
            {primary_html}
            {cpt_badge}
        </div>
        
        <div class="coding-section__header">
            <span class="coding-section__title">Secondary ICD-10</span>
        </div>
        <div class="coding-badges">
            {secondary_html}
        </div>
        
        <div class="cpt-detail">
            {domains_html}
        </div>
        
        <div class="justification-box">
            <strong>Medical Necessity:</strong><br>
            {medical_necessity or "_No justification provided_"}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_physician_decision_bar(state, result: dict, patient: dict, use_memory: bool, 
                                   chief_complaint: str, specialty: str = None, 
                                   specialty_extra_fields: dict = None):
    """Render the sticky physician decision bar with Approve/Request Changes/Reject."""
    from utils.review_state import approval_preview
    
    meta = result.get("metadata", {}) or {}
    draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not state.in_history
    
    # Show preview if decision is open
    if state.allow_approve or state.allow_request_changes:
        render_approval_preview(
            approval_preview(
                result,
                patient_name=patient.get("name", ""),
                patient_id=patient.get("patient_id"),
                chief_complaint=chief_complaint or meta.get("chief_complaint", ""),
                specialty_name=specialty_display_name(specialty) or None,
                memory_enabled=bool(use_memory),
                draft_recorded=draft_recorded,
            )
        )
    
    # Decision buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve & Promote to History", type="primary", 
                     use_container_width=True, disabled=not state.allow_approve,
                     key="btn_approve_main"):
            handle_approve(result, patient["patient_id"], use_memory)
    
    with col2:
        if st.button("✏️ Request Changes", use_container_width=True,
                     disabled=not state.allow_request_changes, key="btn_changes_main"):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()
    
    with col3:
        if st.button("🗑️ Reject", use_container_width=True,
                     disabled=not state.allow_reject, key="btn_reject_main"):
            handle_reject()
    
    # Revise option
    if state.allow_revise:
        st.markdown("---")
        if st.button("✏️ Revise this note (new draft)", use_container_width=True, key="btn_revise_main"):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()
        st.caption(
            "Revising regenerates the note with your feedback and supersedes this "
            "draft for review: the new run becomes the version on screen and the "
            "one Approve would sign. Each run keeps its own row in long-term "
            "memory — earlier versions stay as unsigned drafts, so nothing that "
            "was superseded can silently become history."
        )
    
    # Feedback box for revision
    if state.key == "changes_requested":
        st.markdown("**Revision feedback**")
        st.caption(
            "This text is sent into the *next pipeline run*: the Scribe and the "
            "Coding agent both receive it as a revision instruction. It is not "
            "stored with the visit, and nothing on this screen is edited — the "
            "run produces a new draft to review."
        )
        feedback_key = review_feedback_key()
        st.text_area(
            "Feedback for the next run (Scribe + Coding)",
            key=feedback_key,
            height=110,
            placeholder=(
                "e.g., 'Add ICD-10 for chest pain (R07.9)', 'Change CPT to "
                "99214 - moderate MDM', 'The objective section must carry the "
                "ECG findings I entered.'"
            ),
        )
        if st.button(
            "🔄 Submit Feedback & Regenerate",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.get(feedback_key) or "").strip(),
            key="btn_submit_feedback_main"
        ):
            submit_revision(
                (st.session_state.get(feedback_key) or "").strip(),
                result,
                patient,
                chief_complaint,
                specialty,
            )


def render_pipeline_progress(meta: dict, total_time: float = None):
    """Render the pipeline progress indicator."""
    node_order = [
        "memory_load", "scribe", "cleanup", "coding",
        "validation", "supervisor", "memory_save", "human_review"
    ]
    node_labels = {
        "memory_load": "🧠 Memory",
        "scribe": "📝 Scribe",
        "cleanup": "🧹 Cleanup",
        "coding": "🔖 Coding",
        "validation": "✅ Validation",
        "supervisor": "⚖️ Supervisor",
        "memory_save": "💾 Memory",
        "human_review": "👤 Review",
    }
    
    # Check which steps completed
    tool_meta = meta.get("tools", {}) or {}
    tools_used = tool_meta.get("tools_used", [])
    supervisor_meta = meta.get("supervisor", {}) or {}
    
    # Determine progress based on what's in metadata
    completed = set()
    if "memory_load" in str(meta.get("tools", {})).lower() or meta.get("memory"):
        completed.add("memory_load")
    if "scribe" in str(meta.get("tools", {})).lower() or meta.get("soap_note"):
        completed.add("scribe")
    if "cleanup" in str(meta.get("tools", {})).lower() or meta.get("cleaned_soap_note"):
        completed.add("cleanup")
    if meta.get("coding_result"):
        completed.add("coding")
    if meta.get("validation_node"):
        completed.add("validation")
    if supervisor_meta:
        completed.add("supervisor")
    if meta.get("memory", {}).get("saved"):
        completed.add("memory_save")
    if supervisor_meta.get("needs_human_review"):
        completed.add("human_review")
    
    # Current step
    current = None
    for step in node_order:
        if step not in completed:
            current = step
            break
    if not current and completed:
        current = node_order[-1]
    
    steps_html = ""
    for step in node_order:
        label = node_labels[step]
        if step in completed:
            steps_html += f'<span class="pipeline-step pipeline-step--done">{label} ✓</span>'
        elif step == current:
            steps_html += f'<span class="pipeline-step pipeline-step--current">{label} ⏳</span>'
        else:
            steps_html += f'<span class="pipeline-step pipeline-step--pending">{label}</span>'
    
    st.markdown(f"""
    <div class="pipeline-progress">
        <strong>Pipeline Progress</strong>
        <div class="pipeline-progress__steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)

import time

st.set_page_config(page_title="Doctor Office Intelligent System", layout="wide")

# Inject clinical CSS theme
inject_clinical_css()

# Initialize session state for HITL review
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "review_status" not in st.session_state:
    st.session_state.review_status = None  # None, "approved", "rejected", "changes_requested"
if "review_feedback" not in st.session_state:
    st.session_state.review_feedback = ""
# Phase A5: what the review banner needs to be truthful about one decision.
if "review_feedback_history" not in st.session_state:
    st.session_state.review_feedback_history = []  # one entry per submitted revision
if "review_persisted" not in st.session_state:
    st.session_state.review_persisted = None  # True once an Approve put the visit in history
if "review_error" not in st.session_state:
    st.session_state.review_error = None  # a failed Approve write, if any
if "should_run_pipeline" not in st.session_state:
    st.session_state.should_run_pipeline = False
if "pipeline_params" not in st.session_state:
    st.session_state.pipeline_params = {}

st.title("🏥 Doctor Office Intelligent System - Multi-Agent Pipeline")
st.caption(
    "LangGraph: Memory Load → Scribe → Cleanup → Coding → Validation → Supervisor → Memory Save"
)

patients = load_patients()

with st.sidebar:
    st.header("Patient Selection")
    patient_names = [f"{p['name']} ({p['patient_id']})" for p in patients]
    selected_idx = st.selectbox("Select Patient", range(len(patients)), format_func=lambda i: patient_names[i])
    patient = patients[selected_idx]

    st.divider()
    with st.container(border=True):
        st.subheader("Patient Details")
        st.write(f"**Age:** {patient['age']}")
        st.write(f"**Gender:** {patient['gender']}")
        st.write(f"**Conditions:** {', '.join(patient['conditions'])}")
        st.write(f"**Medications:** {', '.join(patient['medications'])}")
        st.write(f"**Allergies:** {', '.join(patient['allergies']) or 'None'}")
        st.write(f"**Last Visit:** {patient['last_visit_summary']}")

    # --- Synthetic Internist Patients (Demo) ---
    st.divider()
    st.subheader("Load Synthetic Patient (Demo)")
    internist_patients = load_internist_patients()
    if internist_patients is None:
        st.warning("Internist patient data not found. Run from project root.")
    else:
        # Filter options
        filter_option = st.selectbox(
            "Filter by expected path",
            ["All", "Fast Path", "Slow Path"],
            key="internist_filter",
        )
        if filter_option == "Fast Path":
            filtered = [p for p in internist_patients if p["expected_path"] == "fast"]
        elif filter_option == "Slow Path":
            filtered = [p for p in internist_patients if p["expected_path"] == "slow"]
        else:
            filtered = internist_patients

        patient_options = [
            f"{p['patient_id']} | {p['patient_name']} | {p['expected_path'].upper()}"
            for p in filtered
        ]
        selected_internist_idx = st.selectbox(
            "Select synthetic patient",
            range(len(filtered)),
            format_func=lambda i: patient_options[i],
            key="internist_select",
        )
        selected_internist = filtered[selected_internist_idx]

        # Show demo notes
        st.caption(f"📋 {selected_internist['demo_notes']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Patient", type="primary", use_container_width=True, key="load_internist"):
                # Store in session state for form pre-filling
                st.session_state.loaded_chief_complaint = selected_internist["chief_complaint"]
                st.session_state.loaded_patient_context = selected_internist["patient_context"]
                st.session_state.loaded_visit_context = selected_internist["visit_context"]
                st.session_state.loaded_patient_id = selected_internist["patient_id"]
                st.session_state.loaded_patient_name = selected_internist["patient_name"]
                st.session_state.loaded_specialty = selected_internist.get("specialty", "general_internal_medicine")
                st.success(f"Loaded {selected_internist['patient_name']} ({selected_internist['patient_id']})")
                st.rerun()
        with col2:
            if st.button("Clear", use_container_width=True, key="clear_internist"):
                for key in [
                    "loaded_chief_complaint", "loaded_patient_context", "loaded_visit_context",
                    "loaded_patient_id", "loaded_patient_name", "loaded_specialty",
                    "vc_hpi", "vc_bp", "vc_hr", "vc_temp", "vc_rr", "vc_spo2",
                    "vc_weight", "vc_bmi", "vc_exam", "vc_findings", "vc_vitals_note"
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

    st.divider()
    st.subheader("Long-Term Memory")
    use_memory = st.toggle(
        "Remember visits between sessions",
        value=True,
        help="Reads prior visits for this patient before the note is written, "
             "and records the finished run as a draft afterwards. A draft becomes "
             "this patient's history only when you approve it in Physician "
             "Review. Stored locally in data/memory/.",
    )
    history = load_patient_history(patient["patient_id"]) if use_memory else None
    if use_memory and history and history["visits"]:
        approved = history.get("approved_visit_count", 0)
        drafts = history.get("draft_visit_count", 0)
        st.caption(
            f"{history['visit_count']} approved visit(s) on record"
            + (f" · {drafts} unsigned draft(s)" if drafts else "")
        )
        st.caption(
            "Only approved visits are read as this patient's history and fed to "
            "the Scribe on the next run."
        )
        with st.expander("Prior visits for this patient", expanded=False):
            for visit in history["visits"]:
                codes = ", ".join(visit.get("icd10", [])) or "no codes"
                badge = (
                    "✅ approved"
                    if visit.get("display_status") == "approved"
                    else "📝 draft (not history)"
                )
                st.markdown(
                    f"**{visit.get('timestamp', '')[:16].replace('T', ' ')}**"
                    f" — {visit.get('chief_complaint', '')} — _{badge}_\n\n"
                    f"_{codes}_"
                )
    elif use_memory:
        st.caption(
            "No stored visits yet for this patient - nothing a pipeline run "
            "recorded counts as history until it is approved."
        )

    st.divider()
    st.subheader("Model Settings")

    mode_options = get_available_modes()
    mode_labels = {
        "fast": f"⚡ Fast ({get_model_name('fast')})",
        "quality": f"🎯 Quality ({get_model_name('quality')})"
    }
    selected_mode = st.radio(
        "Mode",
        options=mode_options,
        format_func=lambda m: mode_labels.get(m, m),
        index=0,
        horizontal=True
    )

    # Show estimated time
    if selected_mode == "fast":
        st.caption("⏱️ Estimated time: ~1–3 minutes total")
    else:
        st.caption("⏱️ Estimated time: ~6–10 minutes total")

st.header("Chief Complaint")
# Use loaded value if available, otherwise empty
default_chief_complaint = st.session_state.get("loaded_chief_complaint", "")
chief_complaint = st.text_area(
    "Enter the patient's chief complaint / reason for visit:",
    height=100,
    value=default_chief_complaint,
    placeholder="e.g., Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
)

# Specialty selector (Phase 2)
selected_specialty, specialty_extra_fields = render_specialty_selector()

st.divider()

# Pre-fill visit context from loaded synthetic patient
loaded_vc = st.session_state.get("loaded_visit_context")
if loaded_vc:
    # Set widget keys before rendering so Streamlit uses them as defaults
    if "hpi" in loaded_vc:
        st.session_state.vc_hpi = loaded_vc["hpi"]
    if "blood_pressure" in loaded_vc:
        st.session_state.vc_bp = loaded_vc["blood_pressure"]
    if "heart_rate" in loaded_vc:
        st.session_state.vc_hr = str(loaded_vc["heart_rate"])
    if "temperature" in loaded_vc:
        st.session_state.vc_temp = str(loaded_vc["temperature"])
    if "respiratory_rate" in loaded_vc:
        st.session_state.vc_rr = str(loaded_vc["respiratory_rate"])
    if "spo2" in loaded_vc:
        st.session_state.vc_spo2 = str(loaded_vc["spo2"])
    if "weight_kg" in loaded_vc:
        st.session_state.vc_weight = str(loaded_vc["weight_kg"])
    if "bmi" in loaded_vc:
        st.session_state.vc_bmi = str(loaded_vc["bmi"])
    if "physical_exam" in loaded_vc:
        st.session_state.vc_exam = loaded_vc["physical_exam"]
    if "additional_findings" in loaded_vc:
        st.session_state.vc_findings = loaded_vc["additional_findings"]
    if "vitals_note" in loaded_vc:
        st.session_state.vc_vitals_note = loaded_vc["vitals_note"]

visit_context = render_visit_context_inputs(specialty_extra_fields)

def run_pipeline_with_feedback(
    patient_context: str,
    chief_complaint: str,
    retrieved_notes: str,
    mode: str,
    patient_id: str,
    use_memory: bool,
    physician_feedback: str = None,
    visit_context: dict = None,
    specialty: str = None,
):
    """Run the pipeline with optional physician feedback for revision."""
    from agents.orchestrator import stream_visit_pipeline
    from utils.config import is_known_synthetic_patient
    yield from stream_visit_pipeline(
        patient_context=patient_context,
        chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes,
        mode=mode,
        patient_id=patient_id,
        use_memory=use_memory,
        physician_feedback=physician_feedback,
        visit_context=visit_context,
        specialty=specialty,
        is_synthetic=is_known_synthetic_patient(patient_id),
    )


if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
    if not chief_complaint.strip():
        st.error("Please enter a chief complaint.")
    else:
        # Use loaded synthetic patient context if available, otherwise build from sidebar patient
        loaded_context = st.session_state.get("loaded_patient_context")
        loaded_pid = st.session_state.get("loaded_patient_id")
        loaded_name = st.session_state.get("loaded_patient_name")
        
        if loaded_context and loaded_pid:
            patient_context = loaded_context
            patient_id = loaded_pid
            # Create a complete patient dict for display purposes
            # Parse patient_context to extract age, gender, conditions, medications, allergies
            # The patient_context format is:
            # Patient: John Doe, 58-year-old male
            # Conditions: Type 2 Diabetes Mellitus, Essential Hypertension
            # Medications: Metformin 500mg BID, Lisinopril 10mg daily, Atorvastatin 20mg nightly
            # Allergies: Penicillin (rash)
            # Last Visit: ...
            import re
            ctx = loaded_context
            age_match = re.search(r'(\d+)-year-old', ctx)
            gender_match = re.search(r'-year-old (\w+)', ctx)
            conditions_match = re.search(r'Conditions: ([^\n]+)', ctx)
            medications_match = re.search(r'Medications: ([^\n]+)', ctx)
            allergies_match = re.search(r'Allergies: ([^\n]+)', ctx)
            
            display_patient = {
                "name": loaded_name or "Synthetic Patient",
                "patient_id": loaded_pid,
                "age": int(age_match.group(1)) if age_match else 0,
                "gender": gender_match.group(1).capitalize() if gender_match else "Unknown",
                "conditions": [c.strip() for c in conditions_match.group(1).split(',')] if conditions_match else [],
                "medications": [m.strip() for m in medications_match.group(1).split(',')] if medications_match else [],
                "allergies": [a.strip() for a in allergies_match.group(1).split(',')] if allergies_match else [],
            }
        else:
            patient_context = build_patient_context(patient)
            patient_id = patient["patient_id"]
            display_patient = patient

        # Store params and trigger pipeline run
        st.session_state.review_status = None
        st.session_state.review_feedback = ""
        # A new visit starts a new review: previous revisions belong to the
        # previous note, so the history is cleared rather than inherited.
        st.session_state.review_feedback_history = []
        st.session_state.review_persisted = None
        st.session_state.review_error = None
        st.session_state.pipeline_params = {
            "patient_context": patient_context,
            "chief_complaint": chief_complaint,
            "visit_context": visit_context,
            "mode": selected_mode,
            "patient_id": patient_id,
            "use_memory": use_memory,
            "physician_feedback": None,
            "specialty": selected_specialty,
            "specialty_extra_fields": specialty_extra_fields,
        }
        # Store display patient for review tab
        st.session_state.display_patient = display_patient
        st.session_state.should_run_pipeline = True
        st.rerun()

# Execute pipeline if flag is set (runs on page load after button click or regeneration)
if st.session_state.should_run_pipeline:
    st.session_state.should_run_pipeline = False  # Clear flag immediately
    
    params = st.session_state.pipeline_params
    if params:
        status_text = st.empty()
        
        try:
            vs = get_vectorstore()

            retrieved_docs = vs.similarity_search(params["chief_complaint"], k=1)
            retrieved_notes = "\n\n".join([doc.page_content for doc in retrieved_docs])[:300]

            # Run pipeline with progress updates per LangGraph node
            node_order = [
                "memory_load", "scribe", "cleanup", "coding",
                "validation", "supervisor", "memory_save", "human_review"
            ]
            node_labels = {
                "memory_load": "🧠 Memory Load - reading prior visits",
                "scribe": "📝 Scribe Agent - generating SOAP note",
                "cleanup": "🧹 Cleanup - grounding pass",
                "coding": "🔖 Coding Agent - ICD-10 + CPT (deterministic E/M)",
                "validation": "✅ Validation - quality checks",
                "supervisor": "🧑‍⚖️ Supervisor - routing / retries",
                "memory_save": "💾 Memory Save - storing this visit",
                "human_review": "⚠️ Human Review - retries exhausted",
            }

            start_time = time.time()
            progress_message = regeneration_progress_message(
                params.get("physician_feedback")
            )
            if progress_message:
                status_text.info(progress_message)
            else:
                status_text.info(f"⏳ {node_labels[node_order[0]]}...")

            result = None
            step_index = 0
            for partial in run_pipeline_with_feedback(
                patient_context=params["patient_context"],
                chief_complaint=params["chief_complaint"],
                retrieved_notes=retrieved_notes,
                mode=params["mode"],
                patient_id=params["patient_id"],
                use_memory=params["use_memory"],
                physician_feedback=params.get("physician_feedback"),
                visit_context=params.get("visit_context"),
                specialty=params.get("specialty"),
            ):
                result = partial
                graph_step = partial["metadata"].get("graph")
                if graph_step in node_order:
                    step_index = node_order.index(graph_step) + 1
                timing = partial["metadata"].get("timing", {})
                elapsed = timing.get("total_seconds", round(time.time() - start_time, 1))
                if step_index < len(node_order):
                    status_text.info(
                        f"✅ {node_labels[node_order[step_index - 1]]} done ({elapsed}s)\n\n"
                        f"⏳ {node_labels[node_order[step_index]]}..."
                    )
                else:
                    status_text.success(
                        f"✅ LangGraph pipeline completed in {elapsed}s "
                        f"({step_index}/{len(node_order)} nodes)"
                    )

            total_time = time.time() - start_time
            status_text.empty()

            # Store result in session state for review
            st.session_state.pipeline_result = result

            # A finished run is a new draft, whatever produced it: the previous
            # decision does not carry over, so the physician reviews this note
            # from scratch instead of a stale "approved" banner. A
            # feedback-driven run stays in `changes_requested` only if it
            # failed, which keeps the feedback on screen.
            st.session_state.review_status = None
            st.session_state.review_persisted = None
            st.session_state.review_error = None

            supervisor_meta = (result.get("metadata", {}).get("supervisor") or {})
            if supervisor_meta.get("needs_human_review"):
                st.error(
                    f"⚠️ Pipeline finished but the visit was ESCALATED FOR HUMAN REVIEW "
                    f"after {supervisor_meta.get('retry_count')} retries. "
                    f"Nothing was recorded for this patient - not even a draft."
                )
            else:
                if params.get("physician_feedback"):
                    rounds = len(st.session_state.get("review_feedback_history") or [])
                    st.success(
                        f"✅ Revision completed in {total_time:.1f} seconds! "
                        f"This is a new draft written from your feedback "
                        f"(revision {rounds}); the previous version is not on screen."
                    )
                else:
                    st.success(f"Pipeline Completed Successfully in {total_time:.1f} seconds!")
                # The run finished, but finishing is not approving: say so here so
                # "Completed" is never read as "this visit is in the record".
                memory_after = (result.get("metadata", {}) or {}).get("memory") or {}
                if memory_after.get("saved"):
                    st.info(
                        f"📝 Recorded as an unsigned draft (`{memory_after.get('visit_id') or 'no id'}`). "
                        "It is not part of this patient's history until you approve "
                        "it in Physician Review."
                    )
                if supervisor_meta.get("retry_count"):
                    st.info(
                        f"🧑‍⚖️ Supervisor triggered {supervisor_meta.get('retry_count')} retry run(s) "
                        f"with feedback before completing."
                    )
                if supervisor_meta.get("applied"):
                    for fix in supervisor_meta["applied"]:
                        st.info(f"🧑‍⚖️ Supervisor auto-applied: {fix}")

            # Quality flags in the same grouped, plain-language form the
            # Coding tab and Physician Review use (Phase A4). Raw strings stay
            # in the Raw JSON tab.
            render_quality_flags_grouped(
                result.get("quality_flags", []), heading="Quality flags"
            )

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            st.exception(e)
            if params.get("physician_feedback"):
                # The revision failed: keep the feedback and the
                # changes-requested state so it can be corrected and resubmitted.
                st.warning(
                    "The revision did not run. Your feedback is still below — "
                    "edit it and submit again."
                )

# ─── New Clinical Review Tab ───
def render_clinical_review_tab(result: dict, patient: dict, chief_complaint: str, use_memory: bool, 
                                specialty: str = None, specialty_extra_fields: dict = None):
    """Render the Physician Review tab with new clinical UI components."""
    from utils.review_state import resolve_review_state, approval_preview
    
    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result.get("metadata", {}) or {}
    supervisor_meta = meta.get("supervisor", {}) or {}
    quality_flags = result.get("quality_flags", []) or []
    
    specialty = meta.get("specialty") or specialty
    specialty_name = specialty_display_name(specialty)
    
    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    specialty_fields = (
        visit_context_used.get("specialty_fields")
        or specialty_extra_fields
        or {}
    )
    
    # Review state
    state = resolve_review_state(result, use_memory)
    
    # ─── Status Banner (inside tab) ───
    render_status_banner(result, meta, meta.get("timing", {}).get("total_seconds"))
    
    # ─── Patient Context Card ───
    render_patient_card(patient, meta, visit_context_used)
    
    st.divider()
    
    # ─── SOAP Note (four cards) ───
    st.subheader("📋 SOAP Note for Review")
    render_soap_cards(soap)
    
    st.divider()
    
    # ─── Coding Results (badges) ───
    st.subheader("🔖 Coding Results")
    render_coding_results(coding, meta)
    
    st.divider()
    
    # ─── Quality Flags ───
    st.subheader("🚩 Quality Flags & Supervisor Actions")
    
    if supervisor_meta.get("applied"):
        st.markdown("**🔧 Supervisor Auto-Applied Fixes**")
        for fix in supervisor_meta["applied"]:
            st.info(f"• {fix}")
    
    render_specialty_fidelity_notes(quality_flags)
    render_quality_flags_grouped(quality_flags)
    
    if supervisor_meta.get("feedback"):
        st.markdown("**💬 Supervisor Feedback (passed to retry passes)**")
        for item in supervisor_meta["feedback"]:
            st.markdown(f"- {item}")
    
    st.divider()
    
    # ─── Encounter Data (collapsed) ───
    with st.expander("🗒️ Visit Context & Specialty Fields", expanded=False):
        render_visit_context_summary(visit_context_used, specialty_fields)
    
    st.divider()
    
    # ─── Decision Preview ───
    draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not state.in_history
    if state.allow_approve or state.allow_request_changes:
        render_approval_preview(
            approval_preview(
                result,
                patient_name=patient.get("name", ""),
                patient_id=patient.get("patient_id"),
                chief_complaint=chief_complaint or params.get("chief_complaint", ""),
                specialty_name=specialty_name or None,
                memory_enabled=bool(use_memory),
                draft_recorded=draft_recorded,
            )
        )
    
    # ─── Decision Buttons ───
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve & Promote to History", type="primary", use_container_width=True,
                     disabled=not state.allow_approve, key="btn_approve_tab3"):
            handle_approve(result, patient["patient_id"], use_memory)
    with col2:
        if st.button("✏️ Request Changes", use_container_width=True,
                     disabled=not state.allow_request_changes, key="btn_changes_tab3"):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()
    with col3:
        if st.button("🗑️ Reject", use_container_width=True,
                     disabled=not state.allow_reject, key="btn_reject_tab3"):
            handle_reject()
    
    # Revise option
    if state.allow_revise:
        st.markdown("---")
        if st.button("✏️ Revise this note (new draft)", use_container_width=True, key="btn_revise_tab3"):
            st.session_state.review_status = "changes_requested"
            st.session_state.review_error = None
            st.rerun()
        st.caption(
            "Revising regenerates the note with your feedback and supersedes this "
            "draft for review: the new run becomes the version on screen and the "
            "one Approve would sign. Each run keeps its own row in long-term "
            "memory — earlier versions stay as unsigned drafts, so nothing that "
            "was superseded can silently become history."
        )
    
    # Feedback box
    if state.key == "changes_requested":
        st.markdown("**Revision feedback**")
        st.caption(
            "This text is sent into the *next pipeline run*: the Scribe and the "
            "Coding agent both receive it as a revision instruction. It is not "
            "stored with the visit, and nothing on this screen is edited — the "
            "run produces a new draft to review."
        )
        feedback_key = review_feedback_key()
        st.text_area(
            "Feedback for the next run (Scribe + Coding)",
            key=feedback_key,
            height=110,
            placeholder=(
                "e.g., 'Add ICD-10 for chest pain (R07.9)', 'Change CPT to "
                "99214 - moderate MDM', 'The objective section must carry the "
                "ECG findings I entered.'"
            ),
        )
        if st.button(
            "🔄 Submit Feedback & Regenerate",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.get(feedback_key) or "").strip(),
            key="btn_submit_feedback_tab3"
        ):
            submit_revision(
                (st.session_state.get(feedback_key) or "").strip(),
                result,
                patient,
                chief_complaint,
                specialty,
            )


# Display results with new clinical UI if we have a pipeline result
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result.get("metadata", {})
    total_time = meta.get("timing", {}).get("total_seconds", None)
    
    # Get display patient (for synthetic patients)
    display_patient = st.session_state.get("display_patient", patient)
    
    # Get visit context for patient card
    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    
    # Resolve review state for decision bar
    from utils.review_state import resolve_review_state
    review_state = resolve_review_state(result, use_memory)
    
    # ─── Status Banner (highest priority) ───
    render_status_banner(result, meta, total_time)
    
    # ─── Patient Context Card + Pipeline Progress ───
    col_patient, col_progress = st.columns([2, 1], gap="large")
    with col_patient:
        render_patient_card(display_patient, meta, visit_context_used)
    with col_progress:
        render_pipeline_progress(meta, total_time)
    
    # ─── Main Content Tabs ───
    tab1, tab2, tab3 = st.tabs(
        ["📋 SOAP Note", "🔖 Coding Result", "👨‍⚕️ Physician Review"]
    )
    
    with tab1:
        render_soap_cards(soap)
        
        st.divider()
        st.subheader("ICD-10 Suggestions (from Scribe)")
        for code in soap["icd10_suggestions"]:
            st.code(code)
    
    with tab2:
        render_coding_results(coding, meta)
        
        st.divider()
        # Phase A4: which symptom codes were considered
        render_symptom_coding_summary(meta, coding)
    
    with tab3:
        # Use new clinical review tab (matching tab1/tab2 style)
        from utils.review_state import resolve_review_state
        review_state = resolve_review_state(result, use_memory)
        render_clinical_review_tab(
            result, display_patient, chief_complaint, use_memory,
            specialty=st.session_state.pipeline_params.get("specialty"),
            specialty_extra_fields=st.session_state.pipeline_params.get("specialty_extra_fields"),
        )
    
    # ─── Physician Decision Section (always visible, not in a tab) ───
    render_physician_decision_bar(
        review_state, result, display_patient, use_memory,
        chief_complaint=params.get("chief_complaint", chief_complaint),
        specialty=st.session_state.pipeline_params.get("specialty"),
        specialty_extra_fields=st.session_state.pipeline_params.get("specialty_extra_fields"),
    )
    
    # Technical Details (Advanced) - collapsed by default
    with st.expander("🔧 Technical Details (Advanced)", expanded=False):
        tech_tab1, tech_tab2 = st.tabs(["🧠 Memory & Tools", "📄 Raw JSON"])

        with tech_tab1:
            st.subheader("Long-Term Memory")
            mem_meta = meta.get("memory", {})
            if use_memory:
                st.write(f"**Patient ID:** {mem_meta.get('patient_id')}")
                st.write(
                    f"**Approved visits on record:** {mem_meta.get('prior_visits', 0)} "
                    "(the only visits read as this patient's history)"
                )
                st.write(
                    "**History used in this note:** "
                    + ("Yes" if mem_meta.get("memory_context_used") else "No")
                )
                st.write(
                    "**This run recorded:** "
                    + (
                        f"draft `{mem_meta.get('visit_id')}` (unsigned - not history "
                        f"until a physician approves it)"
                        if mem_meta.get("saved")
                        else "nothing"
                    )
                )
                st.write(
                    f"**Drafts / approved on file:** "
                    f"{mem_meta.get('draft_visit_count', 0)} / "
                    f"{mem_meta.get('approved_visit_count', 0)}"
                )
                if mem_meta.get("memory_context"):
                    with st.expander("History text supplied to the agents"):
                        st.text(mem_meta["memory_context"])
            else:
                st.info("Memory was disabled for this run; nothing was read or written.")

            st.divider()
            st.subheader("Supervisor (routing / retries)")
            sup_meta = meta.get("supervisor", {})
            st.write(f"**Retries:** {sup_meta.get('retry_count', 0)}")
            st.write(
                "**Status:** "
                + (
                    "Escalated for human review" if sup_meta.get("needs_human_review")
                    else "Completed"
                )
            )
            if sup_meta.get("applied"):
                st.write("**Auto-applied fixes:**")
                for fix in sup_meta["applied"]:
                    st.caption(f"• {fix}")
            if sup_meta.get("feedback"):
                st.markdown("**Feedback passed to retry passes:**")
                for item in sup_meta["feedback"]:
                    st.markdown(f"- {item}")

            st.divider()
            st.subheader("Tools Used This Visit")
            tool_meta = meta.get("tools", {})
            st.caption(
                "The orchestrator grants each agent only the tools it needs; "
                "the Scribe never sees the coding validation tool."
            )
            st.write("**Tools called:** " + (
                ", ".join(f"`{t}`" for t in tool_meta.get("tools_used", []))
                or "None"
            ))
            for node, trace in (tool_meta.get("tool_results") or {}).items():
                if not trace:
                    continue
                st.markdown(f"**`{node}`**")
                for tool_name, payload in trace.items():
                    st.markdown(f"**🔧 Tool Call: {tool_name}**")
                    st.json(payload)

        with tech_tab2:
            st.subheader("Full Pipeline Output")
            st.json(result)