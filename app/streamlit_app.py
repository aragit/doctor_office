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
    pass

import json
import logging
import re
import time
import streamlit as st

# ─── Page config (MUST be first Streamlit command) ───
try:
    st.set_page_config(page_title="Doctor Office — Clinical AI Review Console", layout="wide")
except st.errors.StreamlitAPIException:
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
  --text: #0F172A;
  --text-muted: #64748B;
  --primary: #0E7490;
  --primary-hover: #0F766E;
  --secondary: #334155;
  --secondary-hover: #1E293B;
  --danger: #DC2626;
  --danger-hover: #B91C1C;
  --success: #16A34A;
  --warning: #D97706;
  --info: #2563EB;
  --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-hover: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  --radius: 12px;
  --radius-sm: 8px;
}

/* Force app background + dark text */
.stApp, .stApp > div, .stApp > div > div, .main, .main > div {
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* All widget text dark */
.stApp, .stApp * { color: var(--text) !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
  background: white !important; color: var(--text) !important; border: 1px solid var(--border) !important;
}
.stSelectbox label, .stMultiSelect label, .stNumberInput label, .stDateInput label,
.stTimeInput label, .stFileUploader label, .stSlider label, .stRadio > label,
.stCheckbox > label, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
.stCaption, .stText, .stHeader, .stSubheader, .stTitle, h1, h2, h3, h4, h5, h6, p, span, div, label {
  color: var(--text) !important;
}

/* Buttons with colored backgrounds keep white text */
.stButton > button[kind="primary"], .stButton > button[kind="primary"]:hover,
.stButton > button[kind="secondary"], .stButton > button[kind="secondary"]:hover {
  color: white !important;
}

/* ─── Top Bar ─── */
.top-bar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  padding: 0.75rem 1rem !important;
  background: var(--card) !important;
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 1rem !important;
}
.top-bar__brand { font-weight: 700 !important; font-size: 1.1rem !important; color: var(--primary) !important; }
.top-bar__actions { display: flex !important; gap: 0.75rem !important; align-items: center !important; }

/* ─── Status Banner ─── */
.status-banner {
  display: flex !important; align-items: center !important; gap: 1rem !important;
  padding: 1rem 1.25rem !important; border-radius: var(--radius) !important;
  margin-bottom: 1rem !important; font-weight: 500 !important; font-size: 0.95rem !important;
  width: 100% !important; box-sizing: border-box !important;
}
.status-banner--clean { background: #ECFDF5 !important; border: 1px solid #A7F3D0 !important; color: #065F46 !important; }
.status-banner--flags { background: #FFFAEB !important; border: 1px solid #FDE68A !important; color: #92400E !important; }
.status-banner--review { background: #FEF2F2 !important; border: 1px solid #FECACA !important; color: #991B1B !important; }
.status-banner__icon { flex-shrink: 0 !important; font-size: 1.25rem !important; }
.status-banner__content { flex: 1 !important; display: flex !important; flex-direction: column !important; gap: 0.25rem !important; }
.status-banner__title { font-weight: 600 !important; }
.status-banner__meta { display: flex !important; gap: 1.5rem !important; font-size: 0.8rem !important; opacity: 0.85 !important; }
.status-banner__meta span { display: flex !important; align-items: center !important; gap: 0.35rem !important; }

/* ─── Patient Card ─── */
.patient-card { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1rem !important; box-shadow: var(--shadow) !important; }
.patient-card__header { display: flex !important; align-items: baseline !important; gap: 0.75rem !important; margin-bottom: 0.75rem !important; padding-bottom: 0.75rem !important; border-bottom: 1px solid var(--border) !important; }
.patient-card__name { font-size: 1.1rem !important; font-weight: 600 !important; color: var(--text) !important; }
.patient-card__id { font-size: 0.8rem !important; color: var(--text-muted) !important; background: var(--bg) !important; padding: 0.15rem 0.5rem !important; border-radius: 9999px !important; }
.patient-card__demographics { display: flex !important; gap: 1rem !important; font-size: 0.85rem !important; color: var(--text-muted) !important; margin-bottom: 0.75rem !important; }
.patient-card__section { margin-bottom: 0.75rem !important; }
.patient-card__section-title { font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; color: var(--text-muted) !important; margin-bottom: 0.4rem !important; font-weight: 600 !important; }
.patient-card__pills { display: flex !important; flex-wrap: wrap !important; gap: 0.35rem !important; }
.pill { display: inline-flex !important; align-items: center !important; gap: 0.35rem !important; padding: 0.25rem 0.6rem !important; font-size: 0.75rem !important; border-radius: 9999px !important; border: 1px solid transparent !important; }
.pill--condition { background: #E0F2FE !important; color: #0369A1 !important; border-color: #BAE6FD !important; }
.pill--medication { background: #F3E8FF !important; color: #7C3AED !important; border-color: #E9D5FF !important; }
.pill--allergy { background: #FEF2F2 !important; color: #DC2626 !important; border-color: #FECACA !important; }
.pill--value { background: #ECFDF5 !important; color: #065F46 !important; border-color: #A7F3D0 !important; }
.pill--empty { background: var(--bg) !important; color: var(--text-muted) !important; border-color: var(--border) !important; font-style: italic !important; }

/* ─── SOAP Cards ─── */
.soap-grid { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 1rem !important; margin-bottom: 1rem !important; width: 100% !important; }
@media (max-width: 900px) { .soap-grid { grid-template-columns: 1fr !important; } }
.soap-card { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1rem !important; box-shadow: var(--shadow) !important; height: 100% !important; }
.soap-card__header { display: flex !important; align-items: center !important; gap: 0.5rem !important; margin-bottom: 0.75rem !important; padding-bottom: 0.5rem !important; border-bottom: 1px solid var(--border) !important; }
.soap-card__icon { font-size: 1.1rem !important; }
.soap-card__title { font-weight: 600 !important; font-size: 0.95rem !important; color: var(--text) !important; }
.soap-card__content { line-height: 1.6 !important; color: var(--text) !important; white-space: pre-wrap !important; word-wrap: break-word !important; min-height: 120px; }

/* ─── Coding Results ─── */
.coding-section { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1.25rem !important; margin-bottom: 1rem !important; box-shadow: var(--shadow) !important; }
.coding-section__header { display: flex !important; align-items: center !important; justify-content: space-between !important; margin-bottom: 1rem !important; padding-bottom: 0.75rem !important; border-bottom: 1px solid var(--border) !important; }
.coding-section__title { font-weight: 600 !important; font-size: 1rem !important; color: var(--text) !important; }
.coding-badges { display: flex !important; flex-wrap: wrap !important; gap: 0.5rem !important; margin-bottom: 1rem !important; }
.badge { display: inline-flex !important; align-items: center !important; gap: 0.4rem !important; padding: 0.35rem 0.75rem !important; font-size: 0.8rem !important; font-weight: 500 !important; border-radius: var(--radius-sm) !important; border: 1px solid !important; }
.badge--primary { background: #F0FDFA !important; color: #0E7490 !important; border-color: #99F6E4 !important; }
.badge--secondary { background: #EFF6FF !important; color: #2563EB !important; border-color: #BFDBFE !important; }
.badge--cpt { background: #FEF9C3 !important; color: #854D0E !important; border-color: #FDE047 !important; }
.badge--fast { background: #ECFDF5 !important; color: #16A34A !important; border-color: #A7F3D0 !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
.cpt-detail { background: var(--bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; padding: 1rem !important; margin-top: 0.75rem !important; }
.cpt-detail__row { display: flex !important; justify-content: space-between !important; padding: 0.4rem 0 !important; border-bottom: 1px solid var(--border) !important; }
.cpt-detail__row:last-child { border-bottom: none !important; }
.cpt-detail__label { color: var(--text-muted) !important; font-size: 0.85rem !important; }
.cpt-detail__value { font-weight: 500 !important; color: var(--text) !important; }
.justification-box { background: var(--bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; padding: 1rem !important; margin-top: 0.75rem !important; font-size: 0.85rem !important; line-height: 1.6 !important; color: var(--text) !important; }

/* ─── Decision Bar ─── */
.decision-bar { display: flex !important; gap: 0.75rem !important; margin-top: 1rem !important; padding-top: 1rem !important; border-top: 1px solid var(--border) !important; flex-wrap: wrap !important; }
.decision-bar .stButton > button { flex: 1 !important; min-width: 180px !important; }

/* ─── Override Streamlit defaults ─── */
.stApp header { display: none !important; }
.stApp footer { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem !important; background: transparent !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important; padding: 0.625rem 1rem !important; font-weight: 500 !important; color: var(--text-muted) !important; transition: all 0.15s ease !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; background: var(--card) !important; }

/* Expanders */
.streamlit-expanderHeader { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; font-weight: 500 !important; }
.streamlit-expanderContent { border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 var(--radius) var(--radius) !important; background: var(--bg) !important; }

/* Button styling */
.stButton > button { border-radius: var(--radius-sm) !important; font-weight: 500 !important; transition: all 0.15s ease !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; border-color: var(--primary) !important; }
.stButton > button[kind="primary"]:hover { background: var(--primary-hover) !important; border-color: var(--primary-hover) !important; }
.stButton > button[kind="secondary"] { background: var(--secondary) !important; border-color: var(--secondary) !important; }
.stButton > button[kind="secondary"]:hover { background: var(--secondary-hover) !important; border-color: var(--secondary-hover) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border) !important; }

/* Code blocks */
.stCodeBlock, .stCodeBlock > div { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; }

/* Input widgets - light backgrounds */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
  background: white !important; color: var(--text) !important;
}
.stSelectbox > div > div { background: white !important; }

/* Form labels */
.stForm label, .stWidgetLabel { color: var(--text) !important; font-weight: 500 !important; }
</style>
"""

st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

# Initialize session state
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "review_status" not in st.session_state:
    st.session_state.review_status = None
if "review_feedback" not in st.session_state:
    st.session_state.review_feedback = ""
if "review_feedback_history" not in st.session_state:
    st.session_state.review_feedback_history = []
if "review_persisted" not in st.session_state:
    st.session_state.review_persisted = None
if "review_error" not in st.session_state:
    st.session_state.review_error = None
if "should_run_pipeline" not in st.session_state:
    st.session_state.should_run_pipeline = False
if "pipeline_params" not in st.session_state:
    st.session_state.pipeline_params = {}

# ─── Helper: load_patient_history ───
def load_patient_history(patient_id: str):
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
        return {"visit_count": 0, "approved_visit_count": 0, "draft_visit_count": 0, "visits": []}

# ─── Helper: build_patient_context ───
def build_patient_context(patient: dict) -> str:
    return f"""
Patient: {patient['name']}, {patient['age']}yo {patient['gender']}
Conditions: {', '.join(patient['conditions'])}
Medications: {', '.join(patient['medications'])}
Allergies: {', '.join(patient['allergies']) or 'None'}
Last Visit: {patient['last_visit_summary']}
"""

# ─── Top Bar ───
st.markdown("""
<div class="top-bar">
  <div class="top-bar__brand">🏥 Doctor Office</div>
  <div class="top-bar__actions">
    <span style="font-size:0.85rem;color:var(--text-muted);">Clinical AI Review Console</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───
@st.cache_data
def load_patients():
    path = Path(__file__).parent.parent / "data" / "synthetic" / "patients.json"
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_internist_patients():
    path = Path(__file__).parent.parent / "data" / "synthetic" / "internist_patients.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

patients = load_patients()

with st.sidebar:
    st.header("Patient")
    patient_names = [f"{p['name']} ({p['patient_id']})" for p in patients]
    selected_idx = st.selectbox("Select Patient", range(len(patients)), format_func=lambda i: patient_names[i], label_visibility="collapsed")
    patient = patients[selected_idx]

    st.markdown("---")
    st.subheader("Patient Context")
    st.write(f"**{patient['name']}**  \n{patient['age']}yo {patient['gender']}  \nID: `{patient['patient_id']}`")
    
    conditions = patient.get("conditions", [])
    if conditions:
        st.markdown("**Conditions**")
        for c in conditions:
            st.markdown(f'<span class="pill pill--condition">{c}</span>', unsafe_allow_html=True)
    
    meds = patient.get("medications", [])
    if meds:
        st.markdown("**Medications**")
        for m in meds[:3]:
            st.markdown(f'<span class="pill pill--medication">{m}</span>', unsafe_allow_html=True)
        if len(meds) > 3:
            st.caption(f"+{len(meds) - 3} more")
    
    allergies = patient.get("allergies", [])
    if allergies:
        st.markdown("**Allergies**")
        for a in allergies:
            st.markdown(f'<span class="pill pill--allergy">{a}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill--empty">NKDA</span>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Synthetic patient loader
    st.subheader("Load Demo Patient")
    internist_patients = load_internist_patients()
    if internist_patients:
        filter_option = st.selectbox("Filter", ["All", "Fast Path", "Slow Path"], key="internist_filter")
        if filter_option == "Fast Path":
            filtered = [p for p in internist_patients if p["expected_path"] == "fast"]
        elif filter_option == "Slow Path":
            filtered = [p for p in internist_patients if p["expected_path"] == "slow"]
        else:
            filtered = internist_patients
        
        patient_options = [f"{p['patient_id']} | {p['patient_name']} | {p['expected_path'].upper()}" for p in filtered]
        selected_idx = st.selectbox("Select", range(len(filtered)), format_func=lambda i: patient_options[i], key="internist_select")
        selected_internist = filtered[selected_idx]
        st.caption(selected_internist['demo_notes'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Patient", type="primary", use_container_width=True, key="load_internist"):
                st.session_state.loaded_chief_complaint = selected_internist["chief_complaint"]
                st.session_state.loaded_patient_context = selected_internist["patient_context"]
                st.session_state.loaded_visit_context = selected_internist["visit_context"]
                st.session_state.loaded_patient_id = selected_internist["patient_id"]
                st.session_state.loaded_patient_name = selected_internist["patient_name"]
                st.session_state.loaded_specialty = selected_internist.get("specialty", "general_internal_medicine")
                st.success(f"Loaded {selected_internist['patient_name']}")
                st.rerun()
        with col2:
            if st.button("Clear", use_container_width=True, key="clear_internist"):
                for key in ["loaded_chief_complaint", "loaded_patient_context", "loaded_visit_context",
                           "loaded_patient_id", "loaded_patient_name", "loaded_specialty",
                           "vc_hpi", "vc_bp", "vc_hr", "vc_temp", "vc_rr", "vc_spo2",
                           "vc_weight", "vc_bmi", "vc_exam", "vc_findings", "vc_vitals_note"]:
                    st.session_state.pop(key, None)
                st.rerun()
    
    st.markdown("---")
    
    # Long-term memory
    st.subheader("Long-Term Memory")
    hist = load_patient_history(patient["patient_id"])
    st.write(f"**Approved visits:** {hist['approved_visit_count']}")
    st.write(f"**Drafts on file:** {hist['draft_visit_count']}")
    for v in hist["visits"]:
        st.caption(f"{v['display_status']} — {v['timestamp'][:16].replace('T', ' ')} — {v.get('visit_id', '?')}")

    st.markdown("---")
    
    # Run settings
    st.subheader("Run Settings")
    selected_mode = st.selectbox("Mode", ["fast"], index=0, key="mode_select", help="Model mode")
    use_memory = st.checkbox("Use Long-Term Memory", value=True, key="use_memory")

# ─── Main Input Form ───
default_chief_complaint = st.session_state.get("loaded_chief_complaint", "")
chief_complaint = st.text_area(
    "Chief Complaint / Reason for Visit",
    height=100,
    value=default_chief_complaint,
    placeholder="e.g., Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
)

# Specialty selector
def render_specialty_selector() -> tuple[str, dict]:
    st.subheader("Specialty")
    st.caption("Select clinical specialty for this encounter.")
    specialty_choices = get_specialty_choices()
    default_specialty = get_default_specialty_id()
    default_idx = next((i for i, (sid, _) in enumerate(specialty_choices) if sid == default_specialty), 0)
    selected_specialty = st.selectbox(
        "Specialty", options=[sid for sid, _ in specialty_choices],
        format_func=lambda sid: next(name for s_id, name in specialty_choices if s_id == sid),
        index=default_idx, key="specialty_select"
    )
    profile = get_specialty_profile(selected_specialty)
    extra_fields = {}
    if profile.extra_input_fields:
        st.caption("Optional specialty-specific fields.")
        for field_def in profile.extra_input_fields:
            key = f"specialty_{selected_specialty}_{field_def.key}"
            if field_def.field_type == "text_area":
                value = st.text_area(field_def.label, height=100, key=key, placeholder=field_def.placeholder, help=field_def.help_text)
                if value.strip(): extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "text_input":
                value = st.text_input(field_def.label, key=key, placeholder=field_def.placeholder, help=field_def.help_text)
                if value.strip(): extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "select":
                value = st.selectbox(field_def.label, options=field_def.options, key=key, help=field_def.help_text, index=0)
                if value and value not in ("Not assessed", "Select"): extra_fields[field_def.key] = value
    return selected_specialty, extra_fields

selected_specialty, specialty_extra_fields = render_specialty_selector()

st.markdown("---")

# Visit context inputs
def render_visit_context_inputs(specialty_extra_fields: dict):
    loaded_vc = st.session_state.get("loaded_visit_context")
    if loaded_vc:
        for k, vc_key in [("hpi", "vc_hpi"), ("blood_pressure", "vc_bp"), ("heart_rate", "vc_hr"),
                         ("temperature", "vc_temp"), ("respiratory_rate", "vc_rr"), ("spo2", "vc_spo2"),
                         ("weight_kg", "vc_weight"), ("bmi", "vc_bmi"), ("physical_exam", "vc_exam"),
                         ("additional_findings", "vc_findings"), ("vitals_note", "vc_vitals_note")]:
            if k in loaded_vc:
                st.session_state[vc_key] = str(loaded_vc[k]) if not isinstance(loaded_vc[k], str) else loaded_vc[k]

    st.subheader("Visit Context")
    st.caption("Structured encounter data for the Scribe agent.")
    hpi = st.text_area("History of Present Illness (HPI)", height=150, key="vc_hpi", placeholder="Patient reports increased fatigue for 2 weeks...")
    
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
    exam = st.text_area("Physical Exam", height=150, key="vc_exam", placeholder="General: Alert, oriented x3...")
    findings = st.text_area("Additional Findings (Labs, Imaging, etc.)", height=150, key="vc_findings", placeholder="HbA1c 7.8%. BMP: Na 139...")
    vitals_note = st.text_area("Vitals Note", height=80, key="vc_vitals_note", placeholder="BP elevated at 150/94...")

    visit_context = {
        "hpi": hpi, "blood_pressure": bp, "heart_rate": int(hr) if hr.isdigit() else None,
        "temperature": float(temp) if temp.replace('.','').isdigit() else None,
        "respiratory_rate": int(rr) if rr.isdigit() else None, "spo2": int(spo2) if spo2.isdigit() else None,
        "weight_kg": float(weight) if weight.replace('.','').isdigit() else None,
        "bmi": float(bmi) if bmi.replace('.','').isdigit() else None,
        "physical_exam": exam, "additional_findings": findings, "vitals_note": vitals_note,
        "specialty_fields": specialty_extra_fields,
    }
    return visit_context

visit_context = render_visit_context_inputs(specialty_extra_fields)

st.markdown("---")

# Run Pipeline button
col_run, col_mem = st.columns([3, 1])
with col_run:
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not chief_complaint.strip():
            st.error("Please enter a chief complaint.")
        else:
            loaded_context = st.session_state.get("loaded_patient_context")
            loaded_pid = st.session_state.get("loaded_patient_id")
            loaded_name = st.session_state.get("loaded_patient_name")
            
            if loaded_context and loaded_pid:
                patient_context = loaded_context
                patient_id = loaded_pid
                display_patient = {"name": loaded_name or "Synthetic Patient", "patient_id": loaded_pid}
            else:
                patient_context = build_patient_context(patient)
                patient_id = patient["patient_id"]
                display_patient = patient
            
            st.session_state.review_status = None
            st.session_state.review_feedback = ""
            st.session_state.review_feedback_history = []
            st.session_state.review_persisted = None
            st.session_state.review_error = None
            st.session_state.pipeline_params = {
                "patient_context": patient_context, "chief_complaint": chief_complaint,
                "visit_context": visit_context, "mode": selected_mode,
                "patient_id": patient_id, "use_memory": use_memory,
                "physician_feedback": None, "specialty": selected_specialty,
                "specialty_extra_fields": specialty_extra_fields,
            }
            st.session_state.display_patient = display_patient
            st.session_state.should_run_pipeline = True
            st.rerun()
with col_mem:
    st.write("")  # spacer for alignment

# ─── Pipeline Execution ───
def run_pipeline(patient_context: str, chief_complaint: str, retrieved_notes: str,
                 mode: str = "fast", patient_id: str = None, use_memory: bool = True, visit_context: dict = None):
    from agents.orchestrator import run_visit_pipeline
    from utils.config import is_known_synthetic_patient
    return run_visit_pipeline(
        patient_context=patient_context, chief_complaint=chief_complaint,
        retrieved_notes=retrieved_notes, mode=mode, patient_id=patient_id,
        use_memory=use_memory, visit_context=visit_context,
        is_synthetic=is_known_synthetic_patient(patient_id),
    )

if st.session_state.should_run_pipeline:
    st.session_state.should_run_pipeline = False
    params = st.session_state.pipeline_params
    start_time = time.time()
    try:
        result = run_pipeline(
            patient_context=params["patient_context"], chief_complaint=params["chief_complaint"],
            retrieved_notes="", mode=params["mode"], patient_id=params["patient_id"],
            use_memory=params["use_memory"], visit_context=params["visit_context"],
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
    display_patient = st.session_state.get("display_patient", patient)
    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    use_memory = params.get("use_memory", True)
    chief_complaint = params.get("chief_complaint", chief_complaint)
    selected_specialty = params.get("specialty", selected_specialty)
    specialty_extra_fields = params.get("specialty_extra_fields", specialty_extra_fields)

    # Resolve review state
    from utils.review_state import review_state as get_review_state
    review_state = get_review_state(
        status=st.session_state.get("review_status"),
        needs_human_review=meta.get("supervisor", {}).get("needs_human_review", False),
        approved=st.session_state.get("review_persisted", False),
        draft_recorded=bool((meta.get("memory") or {}).get("saved")),
        memory_enabled=use_memory,
        error=st.session_state.get("review_error"),
        revision_count=len(st.session_state.get("review_feedback_history", [])),
        feedback_text=st.session_state.get("review_feedback", ""),
    )

    # ─── Render Functions ───
    def render_status_banner(result: dict, meta: dict, total_time: float = None):
        supervisor_meta = meta.get("supervisor", {}) or {}
        quality_flags = result.get("quality_flags", []) or []
        coding_path = meta.get("coding_path", "slow")
        visit_id = meta.get("memory", {}).get("visit_id", "—")
        flag_count = len([f for f in quality_flags if not f.startswith("Cleanup:")])
        
        if supervisor_meta.get("needs_human_review"):
            status_class = "status-banner--review"; icon = "🚨"; title = "Needs Human Review"
        elif flag_count > 0:
            status_class = "status-banner--flags"; icon = "⚠️"; title = "Completed with Quality Flags"
        else:
            status_class = "status-banner--clean"; icon = "✅"; title = "Clean Run"
        
        meta_items = []
        if total_time is not None: meta_items.append(f"⏱️ {total_time:.1f}s total")
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
        bp = ""; hba1c = ""
        if visit_context:
            bp = visit_context.get("blood_pressure", "")
            findings = visit_context.get("additional_findings", "")
            if findings and "hba1c" in findings.lower():
                match = re.search(r'HbA1c\s+([\d.]+%)', findings, re.IGNORECASE)
                if match: hba1c = match.group(1)
        
        conditions = patient.get("conditions", [])
        conditions_html = "".join(f'<span class="pill pill--condition">{c}</span>' for c in conditions) or '<span class="pill pill--empty">No conditions</span>'
        meds = patient.get("medications", [])
        meds_html = "".join(f'<span class="pill pill--medication">{m}</span>' for m in meds[:3]) or '<span class="pill pill--empty">No medications</span>'
        if len(meds) > 3: meds_html += f'<span class="pill pill--empty">+{len(meds) - 3} more</span>'
        allergies = patient.get("allergies", [])
        allergies_html = "".join(f'<span class="pill pill--allergy">{a}</span>' for a in allergies) or '<span class="pill pill--allergy">NKDA</span>'
        values = []
        if bp: values.append(f'<span class="pill pill--value">BP: {bp}</span>')
        if hba1c: values.append(f'<span class="pill pill--value">HbA1c: {hba1c}</span>')
        values_html = "".join(values) if values else '<span class="pill pill--empty">No recent values</span>'
        
        st.markdown(f"""
        <div class="patient-card">
            <div class="patient-card__header">
                <span class="patient-card__name">{patient.get('name', 'Unknown')}</span>
                <span class="patient-card__id">{patient.get('patient_id', '—')}</span>
            </div>
            <div class="patient-card__demographics"><span>{patient.get('age', '?')}y</span><span>·</span><span>{patient.get('gender', '?')}</span></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Conditions</div><div class="patient-card__pills">{conditions_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Medications</div><div class="patient-card__pills">{meds_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Allergies</div><div class="patient-card__pills">{allergies_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Key Values</div><div class="patient-card__pills">{values_html}</div></div>
        </div>
        """, unsafe_allow_html=True)

    def render_soap_cards(soap: dict):
        sections = [("Subjective", "💬", soap.get("subjective", "")), ("Objective", "🔍", soap.get("objective", "")),
                    ("Assessment", "🧠", soap.get("assessment", "")), ("Plan", "📋", soap.get("plan", ""))]
        cards_html = ""
        for title, icon, content in sections:
            safe_content = (content or "_No content_").replace("\n", "<br>")
            cards_html += f'<div class="soap-card"><div class="soap-card__header"><span class="soap-card__icon">{icon}</span><span class="soap-card__title">{title}</span></div><div class="soap-card__content">{safe_content}</div></div>'
        st.markdown(f'<div class="soap-grid">{cards_html}</div>', unsafe_allow_html=True)

    def render_coding_results(coding: dict, meta: dict = None):
        primary = coding.get("primary_icd10", "—")
        secondary = coding.get("secondary_icd10", [])
        cpt_suggestions = coding.get("cpt_suggestions", [])
        cpt_grounding = coding.get("cpt_grounding")
        medical_necessity = coding.get("medical_necessity_note", "")
        coding_path = meta.get("coding_path", "slow") if meta else "slow"
        
        cpt_code = "—"; mdm_level = "—"; domains = {}
        if cpt_grounding:
            cpt_code = cpt_grounding.get("cpt_code", "—")
            mdm_level = cpt_grounding.get("overall_mdm_level", "—")
            domains = cpt_grounding.get("domains", {})
        elif cpt_suggestions:
            first = cpt_suggestions[0]
            if " - " in first: cpt_code, _ = first.split(" - ", 1)
            else: cpt_code = first
        
        primary_html = f'<span class="badge badge--primary">{primary}</span>'
        secondary_html = ""
        if secondary:
            for code in secondary: secondary_html += f'<span class="badge badge--secondary">{code}</span>'
        else: secondary_html = '<span class="badge badge--secondary">None</span>'
        cpt_badge = f'<span class="badge badge--cpt">CPT {cpt_code} · MDM: {mdm_level}</span>'
        fast_badge = f'<span class="badge badge--fast">Fast Path</span>' if coding_path == "fast" else ""
        
        domains_html = ""
        if domains:
            for domain, level in domains.items():
                domains_html += f'<div class="cpt-detail__row"><span class="cpt-detail__label">{domain.title()}</span><span class="cpt-detail__value">{level}</span></div>'
        
        st.markdown(f"""
        <div class="coding-section">
            <div class="coding-section__header"><span class="coding-section__title">Coding Results</span><div>{fast_badge}</div></div>
            <div class="coding-badges">{primary_html}{cpt_badge}</div>
            <div class="coding-section__header"><span class="coding-section__title">Secondary ICD-10</span></div>
            <div class="coding-badges">{secondary_html}</div>
            <div class="cpt-detail">{domains_html}</div>
            <div class="justification-box"><strong>Medical Necessity:</strong><br>{medical_necessity or "_No justification provided_"}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Main Layout ───
    render_status_banner(result, meta, total_time)
    
    # Patient Card + Main Content
    col_left, col_main = st.columns([1, 3], gap="large")
    with col_left:
        render_patient_card(display_patient, meta, visit_context_used)
    
    with col_main:
        # Clinical Note - SOAP
        st.subheader("📋 Clinical Note")
        render_soap_cards(soap)
        
        st.markdown("---")
        
        # Coding Results
        st.subheader("🔖 Coding Results")
        render_coding_results(coding, meta)
        
        # Phase A4: symptom coding summary
        try:
            from agents.orchestrator import render_symptom_coding_summary
            render_symptom_coding_summary(meta, coding)
        except Exception:
            pass
        
        st.markdown("---")
        
        # Physician Decision Bar
        st.subheader("👨‍⚕️ Physician Decision")
        
        # Decision preview
        from utils.review_state import approval_preview
        draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not review_state.in_history
        if review_state.allow_approve or review_state.allow_request_changes:
            preview = approval_preview(
                result, patient_name=display_patient.get("name", ""), patient_id=display_patient.get("patient_id"),
                chief_complaint=chief_complaint, specialty_name=display_patient.get('specialty', 'General'),
                memory_enabled=bool(use_memory), draft_recorded=draft_recorded,
            )
            with st.expander("📋 What Approve will do", expanded=True):
                st.markdown(f"**{preview.summary}**")
                if preview.warning: st.warning(preview.warning)
                for line in preview.detail_lines: st.markdown(f"- {line}")
        
        # Decision buttons
        st.markdown('<div class="decision-bar">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve & Promote to History", type="primary", use_container_width=True,
                         disabled=not review_state.allow_approve, key="btn_approve"):
                from memory.patient_memory import get_memory_store
                store = get_memory_store()
                store.approve_visit(
                    patient_id=display_patient["patient_id"],
                    visit_id=(meta.get("memory") or {}).get("visit_id"),
                    patient_context=params.get("patient_context", ""),
                    chief_complaint=params.get("chief_complaint", ""),
                    coding_result=result.get("coding_result"),
                    soap_note=result.get("soap_note"),
                    quality_flags=result.get("quality_flags"),
                )
                st.session_state.review_status = "approved"
                st.session_state.review_persisted = True
                st.session_state.review_error = None
                st.success("Approved and promoted to patient history!")
                st.rerun()
        with col2:
            if st.button("✏️ Request Changes", use_container_width=True,
                         disabled=not review_state.allow_request_changes, key="btn_changes"):
                st.session_state.review_status = "changes_requested"
                st.session_state.review_error = None
                st.rerun()
        with col3:
            if st.button("🗑️ Reject", use_container_width=True,
                         disabled=not review_state.allow_reject, key="btn_reject"):
                st.session_state.review_status = "rejected"
                st.session_state.review_error = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Revise option
        if review_state.allow_revise:
            st.markdown("---")
            if st.button("✏️ Revise this note (new draft)", use_container_width=True, key="btn_revise"):
                st.session_state.review_status = "changes_requested"
                st.session_state.review_error = None
                st.rerun()
            st.caption("Revising regenerates the note with your feedback and supersedes this draft for review.")
        
        # Feedback box
        if review_state.key == "changes_requested":
            st.markdown("**Revision Feedback**")
            st.caption("This text is sent into the next pipeline run (Scribe + Coding) as a revision instruction.")
            feedback_key = "review_feedback_input"
            st.text_area("Feedback for next run (Scribe + Coding)", key=feedback_key, height=110,
                placeholder="e.g., 'Add ICD-10 for chest pain (R07.9)', 'Change CPT to 99214 - moderate MDM'")
            if st.button("🔄 Submit Feedback & Regenerate", type="primary", use_container_width=True,
                         disabled=not (st.session_state.get(feedback_key) or "").strip(), key="btn_submit_feedback"):
                feedback_text = (st.session_state.get(feedback_key) or "").strip()
                st.session_state.review_feedback = feedback_text
                st.session_state.review_feedback_history.append(feedback_text)
                st.session_state.review_status = None
                st.session_state.review_error = None
                st.session_state.pipeline_params["physician_feedback"] = feedback_text
                st.session_state.should_run_pipeline = True
                st.rerun()

    # Technical Details - collapsed
    with st.expander("🔧 Technical Details (Advanced)", expanded=False):
        tech_tab1, tech_tab2 = st.tabs(["🧠 Memory & Tools", "📄 Raw JSON"])
        with tech_tab1:
            st.subheader("Long-Term Memory")
            mem_meta = meta.get("memory", {})
            if use_memory:
                st.write(f"**Patient ID:** {mem_meta.get('patient_id')}")
                st.write(f"**Approved visits on record:** {mem_meta.get('prior_visits', 0)} (only visits read as history)")
                st.write(f"**Draft recorded:** {'Yes' if mem_meta.get('saved') else 'No'} ({mem_meta.get('visit_id', '—')})")
            else:
                st.info("Long-term memory is disabled for this session.")
            st.subheader("Pipeline Metadata")
            st.json({k: v for k, v in meta.items() if k not in ("visit_context", "memory")})
        with tech_tab2:
            st.subheader("Full Pipeline Result")
            st.json(result)