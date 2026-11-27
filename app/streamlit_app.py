# IMPORTANT: Run this file with:
#   streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
# Do NOT execute with plain "python app/streamlit_app.py" - Streamlit needs its runtime.
# Do NOT run in background/nohup - it will start then immediately stop due to
# import-time atexit conflicts (sentence-transformers → scikit-learn → joblib → loky).

# ─── Compatibility patch: suppress Streamlit/PyTorch & LangGraph startup noise ───
import warnings

# Suppress LangChainPendingDeprecationWarning about allowed_objects
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change in a future version.*",
        category=LangChainPendingDeprecationWarning,
    )
except Exception:
    # Fallback if langchain_core is not yet importable
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change in a future version.*",
    )

# Workaround for Streamlit + PyTorch: torch.classes is a namespace package
# without a normal __path__, which Streamlit tries to introspect.
try:
    import torch
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    # torch may be optional in some environments; do not fail app startup.
    pass

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

# ─── Page config (MUST be first Streamlit command) ───
# Guard against hot-reload calling set_page_config multiple times
try:
    st.set_page_config(page_title="Doctor Office Intelligent System", layout="wide")
except st.errors.StreamlitAPIException:
    # Already set (e.g., on hot-reload); ignore
    pass

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
  color: var(--text) !important;
}

/* Ensure all Streamlit widget text is dark */
.stApp, .stApp * {
  color: var(--text) !important;
}

/* Override specific Streamlit widget text colors */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label,
.stDateInput label,
.stTimeInput label,
.stFileUploader label,
.stSlider label,
.stRadio > label,
.stCheckbox > label,
.stMarkdown,
.stMarkdown p,
.stMarkdown div,
.stMarkdown span,
.stCaption,
.stText,
.stHeader,
.stSubheader,
.stTitle,
h1, h2, h3, h4, h5, h6,
p, span, div, label {
  color: var(--text) !important;
}

/* Exception: buttons with colored backgrounds keep white text */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="secondary"],
.stButton > button[kind="secondary"]:hover {
  color: white !important;
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

# ─── Clinical CSS Theme injection ───
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

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

# ─── Helper: load_patient_history (must be defined before sidebar uses it) ───
def load_patient_history(patient_id: str):
    """Read the stored long-term memory for a patient (sidebar display)."""
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


st.title("🏥 Doctor Office Intelligent System - Multi-Agent Pipeline")
st.caption(
    "LangGraph: Memory Load → Scribe → Cleanup → Coding → Validation → Supervisor → Memory Save"
)

# ─── Sidebar: Patient Selection ───
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
    hist = load_patient_history(patient["patient_id"])
    st.write(f"**Approved visits:** {hist['approved_visit_count']}")
    st.write(f"**Drafts on file:** {hist['draft_visit_count']}")
    for v in hist["visits"]:
        st.caption(f"{v['display_status']} — {v['timestamp'][:16].replace('T', ' ')} — {v.get('visit_id', '?')}")

# ─── Main Input Form ───
st.header("Chief Complaint")
default_chief_complaint = st.session_state.get("loaded_chief_complaint", "")
chief_complaint = st.text_area(
    "Enter the patient's chief complaint / reason for visit:",
    height=100,
    value=default_chief_complaint,
    placeholder="e.g., Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
)

# Specialty selector (Phase 2)
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


def render_visit_context_inputs(specialty_extra_fields: dict):
    """Render the structured visit context input form."""
    loaded_vc = st.session_state.get("loaded_visit_context")
    
    # Pre-fill from loaded synthetic patient
    if loaded_vc:
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

    st.header("Visit Context")
    st.caption("Structured encounter data for the Scribe agent.")

    hpi = st.text_area(
        "History of Present Illness (HPI)",
        height=150,
        key="vc_hpi",
        placeholder="Patient reports increased fatigue for 2 weeks..."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        bp = st.text_input("Blood Pressure", key="vc_bp", placeholder="150/94")
        temp = st.text_input("Temperature (°F)", key="vc_temp", placeholder="98.6")
    with col2:
        hr = st.text_input("Heart Rate (bpm)", key="vc_hr", placeholder="78")
        rr = st.text_input("Respiratory Rate", key="vc_rr", placeholder="14")
    with col3:
        spo2 = st.text_input("SpO₂ (%)", key="vc_spo2", placeholder="98")
        weight = st.text_input("Weight (kg)", key="vc_weight", placeholder="95.3")

    bmi = st.text_input("BMI", key="vc_bmi", placeholder="31.2")

    exam = st.text_area(
        "Physical Exam",
        height=150,
        key="vc_exam",
        placeholder="General: Alert, oriented x3..."
    )

    findings = st.text_area(
        "Additional Findings (Labs, Imaging, etc.)",
        height=150,
        key="vc_findings",
        placeholder="HbA1c 7.8%. BMP: Na 139..."
    )

    vitals_note = st.text_area(
        "Vitals Note",
        height=80,
        key="vc_vitals_note",
        placeholder="BP elevated at 150/94..."
    )

    visit_context = {
        "hpi": hpi,
        "blood_pressure": bp,
        "heart_rate": int(hr) if hr.isdigit() else None,
        "temperature": float(temp) if temp.replace('.','').isdigit() else None,
        "respiratory_rate": int(rr) if rr.isdigit() else None,
        "spo2": int(spo2) if spo2.isdigit() else None,
        "weight_kg": float(weight) if weight.replace('.','').isdigit() else None,
        "bmi": float(bmi) if bmi.replace('.','').isdigit() else None,
        "physical_exam": exam,
        "additional_findings": findings,
        "vitals_note": vitals_note,
        "specialty_fields": specialty_extra_fields,
    }
    return visit_context


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


# ─── Run Pipeline Button ───
selected_specialty, specialty_extra_fields = render_specialty_selector()

st.divider()

# Pre-fill visit context from loaded synthetic patient
loaded_vc = st.session_state.get("loaded_visit_context")
if loaded_vc:
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

st.divider()

col_mode, col_mem = st.columns([2, 1])
with col_mode:
    selected_mode = st.selectbox(
        "Mode",
        ["fast"],
        index=0,
        key="mode_select",
        help="Model mode (currently only fast is available)"
    )
with col_mem:
    use_memory = st.checkbox("Use Long-Term Memory", value=True, key="use_memory")

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
            # Create a minimal patient dict for display purposes
            display_patient = {"name": loaded_name or "Synthetic Patient", "patient_id": loaded_pid}
        else:
            patient_context = build_patient_context(patient)
            patient_id = patient["patient_id"]
            display_patient = patient

        # Store params and trigger pipeline run
        st.session_state.review_status = None
        st.session_state.review_feedback = ""
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

# ─── Pipeline Execution ───
if st.session_state.should_run_pipeline:
    st.session_state.should_run_pipeline = False
    params = st.session_state.pipeline_params
    start_time = time.time()
    try:
        result = run_pipeline(
            patient_context=params["patient_context"],
            chief_complaint=params["chief_complaint"],
            retrieved_notes="",
            mode=params["mode"],
            patient_id=params["patient_id"],
            use_memory=params["use_memory"],
            visit_context=params["visit_context"],
        )
        total_time = time.time() - start_time
        
        st.session_state.pipeline_result = result
        st.session_state.pipeline_total_time = total_time
        st.rerun()
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        st.exception(e)

# ─── Display Results ───
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result.get("metadata", {})
    total_time = st.session_state.get("pipeline_total_time", None)
    
    # Get display patient (for synthetic patients)
    display_patient = st.session_state.get("display_patient", patient)
    
    # Get visit context for patient card
    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    
    # Resolve review state for decision bar
    from utils.review_state import resolve_review_state
    review_state = resolve_review_state(result, use_memory)
    
    # ─── Status Banner (highest priority) ───
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
        elif flag_count > 0:
            status_class = "status-banner--flags"
            icon = "⚠️"
            title = "Completed with Quality Flags"
        else:
            status_class = "status-banner--clean"
            icon = "✅"
            title = "Clean Run"
        
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
    
    # ─── Patient Context Card ───
    def render_patient_card(patient: dict, meta: dict = None, visit_context: dict = None):
        """Render the compact patient context card."""
        bp = ""
        hba1c = ""
        if visit_context:
            bp = visit_context.get("blood_pressure", "")
            findings = visit_context.get("additional_findings", "")
            if findings and "hba1c" in findings.lower():
                import re
                match = re.search(r'HbA1c\s+([\d.]+%)', findings, re.IGNORECASE)
                if match:
                    hba1c = match.group(1)
        
        conditions = patient.get("conditions", [])
        conditions_html = "".join(
            f'<span class="pill pill--condition">{c}</span>' for c in conditions
        ) or '<span class="pill pill--empty">No conditions listed</span>'
        
        meds = patient.get("medications", [])
        meds_html = "".join(
            f'<span class="pill pill--medication">{m}</span>' for m in meds[:3]
        ) or '<span class="pill pill--empty">No medications</span>'
        if len(meds) > 3:
            meds_html += f'<span class="pill pill--empty">+{len(meds) - 3} more</span>'
        
        allergies = patient.get("allergies", [])
        allergies_html = "".join(
            f'<span class="pill pill--allergy">{a}</span>' for a in allergies
        ) or '<span class="pill pill--allergy">NKDA</span>'
        
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
    
    # ─── SOAP Cards ───
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
    
    # ─── Coding Results ───
    def render_coding_results(coding: dict, meta: dict = None):
        """Render coding results with badges and structured CPT display."""
        primary = coding.get("primary_icd10", "—")
        secondary = coding.get("secondary_icd10", [])
        cpt_suggestions = coding.get("cpt_suggestions", [])
        cpt_grounding = coding.get("cpt_grounding")
        medical_necessity = coding.get("medical_necessity_note", "")
        coding_path = meta.get("coding_path", "slow") if meta else "slow"
        
        cpt_code = "—"
        mdm_level = "—"
        domains = {}
        
        if cpt_grounding:
            cpt_code = cpt_grounding.get("cpt_code", "—")
            mdm_level = cpt_grounding.get("overall_mdm_level", "—")
            domains = cpt_grounding.get("domains", {})
        elif cpt_suggestions:
            first = cpt_suggestions[0]
            if " - " in first:
                cpt_code, _ = first.split(" - ", 1)
            else:
                cpt_code = first
        
        primary_html = f'<span class="badge badge--primary">{primary}</span>'
        
        secondary_html = ""
        if secondary:
            for code in secondary:
                secondary_html += f'<span class="badge badge--secondary">{code}</span>'
        else:
            secondary_html = '<span class="badge badge--secondary">None</span>'
        
        cpt_badge = f'<span class="badge badge--cpt">CPT {cpt_code} · MDM: {mdm_level}</span>'
        fast_badge = f'<span class="badge badge--fast">Fast Path</span>' if coding_path == "fast" else ""
        
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
    
    # ─── Pipeline Progress ───
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
        
        completed = set()
        if meta.get("memory"):
            completed.add("memory_load")
        if meta.get("soap_note"):
            completed.add("scribe")
        if meta.get("cleaned_soap_note"):
            completed.add("cleanup")
        if meta.get("coding_result"):
            completed.add("coding")
        if meta.get("validation_node"):
            completed.add("validation")
        if meta.get("supervisor"):
            completed.add("supervisor")
        if meta.get("memory", {}).get("saved"):
            completed.add("memory_save")
        if meta.get("supervisor", {}).get("needs_human_review"):
            completed.add("human_review")
        
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
    
    # ─── Render Main UI ───
    render_status_banner(result, meta, total_time)
    
    col_patient, col_progress = st.columns([2, 1], gap="large")
    with col_patient:
        render_patient_card(display_patient, meta, visit_context_used)
    with col_progress:
        render_pipeline_progress(meta, total_time)
    
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
        from agents.orchestrator import render_symptom_coding_summary
        render_symptom_coding_summary(meta, coding)
    
    with tab3:
        # Use new clinical review tab
        from utils.review_state import resolve_review_state
        review_state = resolve_review_state(result, use_memory)
        
        # Status banner inside tab
        from utils.review_state import render_review_status_banner, render_revision_history
        render_review_status_banner(review_state)
        render_revision_history()
        
        supervisor_meta = meta.get("supervisor", {}) or {}
        quality_flags = result.get("quality_flags", []) or []
        
        if supervisor_meta.get("needs_human_review"):
            st.error(
                f"🚨 **ESCALATED FOR HUMAN REVIEW** — The pipeline exhausted retries "
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
        elif quality_flags:
            st.info("⚠️ **QUALITY FLAGS PRESENT** — Review flags before signing.")
        else:
            st.success("✅ **CLEAN** — No quality flags, no auto-corrections needed.")
        
        st.divider()
        
        # Patient & Encounter summary
        st.subheader("👤 Patient & Encounter")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Name:** {display_patient['name']}")
            st.write(f"**Age:** {display_patient['age']} · **Gender:** {display_patient['gender']}")
            st.write(f"**Patient ID:** {display_patient['patient_id']}")
        with col2:
            specialty_name = display_patient.get('specialty', 'General')
            st.write(f"**Specialty:** {specialty_name}")
            st.write(f"**Conditions:** {', '.join(display_patient['conditions'])}")
            st.write(f"**Medications:** {', '.join(display_patient['medications']) or 'None'}")
        with col3:
            st.write(f"**Allergies:** {', '.join(display_patient['allergies']) or 'None'}")
            st.write(f"**Prior approved visits read:** {(meta.get('memory') or {}).get('prior_visits', 0)}")
            mem_meta = meta.get("memory") or {}
            st.write(
                "**Recorded by pipeline:** "
                + (f"draft {mem_meta.get('visit_id')}" if mem_meta.get("saved") else "No")
            )
            from utils.review_state import resolve_review_state
            state = resolve_review_state(result, use_memory)
            st.write(
                "**In patient history:** Yes (approved)"
                if state.in_history
                else "**In patient history:** No - not history until approved"
            )
        
        st.markdown("**Chief complaint**")
        st.info(chief_complaint or "_none recorded_")
        
        # Encounter data (collapsed)
        from agents.orchestrator import render_visit_context_summary
        with st.expander("🗒️ Visit context & specialty fields", expanded=False):
            render_visit_context_summary(visit_context_used, specialty_extra_fields)
        
        st.divider()
        
        # Quality flags
        st.subheader("🚩 Quality Flags & Supervisor Actions")
        
        if supervisor_meta.get("applied"):
            st.markdown("**🔧 Supervisor Auto-Applied Fixes**")
            for fix in supervisor_meta["applied"]:
                st.info(f"• {fix}")
        
        from agents.orchestrator import render_specialty_fidelity_notes, render_quality_flags_grouped
        render_specialty_fidelity_notes(quality_flags)
        render_quality_flags_grouped(quality_flags)
        
        if supervisor_meta.get("feedback"):
            st.markdown("**💬 Supervisor Feedback (passed to retry passes)**")
            for item in supervisor_meta["feedback"]:
                st.markdown(f"- {item}")
        
        st.divider()
        
        # Decision preview
        from utils.review_state import approval_preview, render_approval_preview
        draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not review_state.in_history
        if review_state.allow_approve or review_state.allow_request_changes:
            render_approval_preview(
                approval_preview(
                    result,
                    patient_name=display_patient.get("name", ""),
                    patient_id=display_patient.get("patient_id"),
                    chief_complaint=chief_complaint or params.get("chief_complaint", ""),
                    specialty_name=display_patient.get('specialty', 'General'),
                    memory_enabled=bool(use_memory),
                    draft_recorded=draft_recorded,
                )
            )
        
        # Decision buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve & Promote to History", type="primary", use_container_width=True,
                         disabled=not review_state.allow_approve, key="btn_approve_tab3"):
                from utils.review_state import handle_approve
                handle_approve(result, display_patient["patient_id"], use_memory)
        with col2:
            if st.button("✏️ Request Changes", use_container_width=True,
                         disabled=not review_state.allow_request_changes, key="btn_changes_tab3"):
                st.session_state.review_status = "changes_requested"
                st.session_state.review_error = None
                st.rerun()
        with col3:
            if st.button("🗑️ Reject", use_container_width=True,
                         disabled=not review_state.allow_reject, key="btn_reject_tab3"):
                from utils.review_state import handle_reject
                handle_reject()
        
        # Revise option
        if review_state.allow_revise:
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
        if review_state.key == "changes_requested":
            st.markdown("**Revision feedback**")
            st.caption(
                "This text is sent into the *next pipeline run*: the Scribe and the "
                "Coding agent both receive it as a revision instruction. It is not "
                "stored with the visit, and nothing on this screen is edited — the "
                "run produces a new draft to review."
            )
            from utils.review_state import review_feedback_key, submit_revision
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
                    display_patient,
                    chief_complaint,
                    selected_specialty,
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