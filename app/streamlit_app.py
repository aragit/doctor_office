# IMPORTANT: Run this file with:
#   streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
# Do NOT execute with plain "python app/streamlit_app.py" - Streamlit needs its runtime.
# Do NOT run in background/nohup - it will start then immediately stop due to
# import-time atexit conflicts (sentence-transformers → scikit-learn → joblib → loky).

# ─── Compatibility patch: suppress Streamlit/PyTorch & LangGraph startup noise ───
import warnings
import threading

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
from pathlib import Path
from typing import List
import streamlit as st

# ─── Page config (MUST be first Streamlit command) ───
try:
    st.set_page_config(page_title="Doctor Office — Clinical AI Review Console", layout="wide")
except st.errors.StreamlitAPIException:
    pass

from utils.config import get_model_name, get_available_modes
from utils.review_state import regeneration_progress_message
from agents.specialty_profiles import get_specialty_profile, get_specialty_choices, get_default_specialty_id

logger = logging.getLogger(__name__)

# ─── Clinical CSS Theme ──────────────────────────────────────────────
CLINICAL_CSS = """
<style>
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

.stApp, .stApp > div, .stApp > div > div, .main, .main > div {
  background: var(--bg) !important;
  color: var(--text) !important;
}

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

.stButton > button[kind="primary"], .stButton > button[kind="primary"]:hover,
.stButton > button[kind="secondary"], .stButton > button[kind="secondary"]:hover {
  color: white !important;
}

.decision-bar .stButton > button { color: white !important; }

.top-bar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  padding: 0.75rem 1.1rem !important;
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  margin-bottom: 1rem !important;
}
.top-bar__brand { font-weight: 700 !important; font-size: 1.1rem !important; color: var(--primary) !important; }
.top-bar__actions { display: flex !important; gap: 0.75rem !important; align-items: center !important; }

.status-banner {
  display: flex !important; align-items: center !important; gap: 1rem !important;
  padding: 1rem 1.25rem !important; border-radius: var(--radius) !important;
  margin-bottom: 0.75rem !important; font-weight: 500 !important; font-size: 0.95rem !important;
  width: 100% !important; box-sizing: border-box !important;
}
.status-banner--clean { background: #ECFDF5 !important; border: 1px solid #A7F3D0 !important; color: #065F46 !important; }
.status-banner--flags { background: #FFFAEB !important; border: 1px solid #FDE68A !important; color: #92400E !important; }
.status-banner--review { background: #FEF2F2 !important; border: 1px solid #FECACA !important; color: #991B1B !important; }
.status-banner__icon { flex-shrink: 0 !important; font-size: 1.25rem !important; }
.status-banner__content { flex: 1 !important; display: flex !important; flex-direction: column !important; gap: 0.25rem !important; }
.status-banner__title { font-weight: 600 !important; }
.status-banner__meta { display: flex !important; gap: 1.5rem !important; font-size: 0.8rem !important; opacity: 0.88 !important; flex-wrap: wrap !important; }
.status-banner__meta span { display: flex !important; align-items: center !important; gap: 0.35rem !important; }

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

.soap-grid { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 1rem !important; margin-bottom: 1rem !important; width: 100% !important; }
@media (max-width: 900px) { .soap-grid { grid-template-columns: 1fr !important; } }
.soap-card { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1rem !important; box-shadow: var(--shadow) !important; height: 100% !important; }
.soap-card__header { display: flex !important; align-items: center !important; gap: 0.5rem !important; margin-bottom: 0.75rem !important; padding-bottom: 0.5rem !important; border-bottom: 1px solid var(--border) !important; }
.soap-card__icon { font-size: 1.1rem !important; }
.soap-card__title { font-weight: 600 !important; font-size: 0.95rem !important; color: var(--text) !important; }
.soap-card__content { line-height: 1.6 !important; color: var(--text) !important; white-space: pre-wrap !important; word-wrap: break-word !important; min-height: 120px; }

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

.decision-bar { display: flex !important; gap: 0.75rem !important; margin-top: 1rem !important; padding-top: 1rem !important; border-top: 1px solid var(--border) !important; flex-wrap: wrap !important; }
.decision-bar .stButton > button { flex: 1 !important; min-width: 180px !important; }

.stApp header { display: none !important; }
.stApp footer { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0.25rem !important; background: transparent !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important; padding: 0.625rem 1rem !important; font-weight: 500 !important; color: var(--text-muted) !important; transition: all 0.15s ease !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; background: var(--card) !important; }

.streamlit-expanderHeader, .stExpander .streamlit-expanderContent,
.streamlit-expanderContent [data-testid="stExpanderDetails"],
.approve-preview-content { background: var(--bg) !important; color: var(--text) !important; }
.streamlit-expanderContent *, .stExpander .streamlit-expanderContent *, .approve-preview-content * { color: var(--text) !important; }

.stAlert { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; }
.stAlert * { color: var(--text) !important; }

.stButton > button { border-radius: var(--radius-sm) !important; font-weight: 500 !important; transition: all 0.15s ease !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; border-color: var(--primary) !important; }
.stButton > button[kind="primary"]:hover { background: var(--primary-hover) !important; border-color: var(--primary-hover) !important; }
.stButton > button[kind="secondary"] { background: var(--secondary) !important; border-color: var(--secondary) !important; }
.stButton > button[kind="secondary"]:hover { background: var(--secondary-hover) !important; border-color: var(--secondary-hover) !important; }

section[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border) !important; }
.stCodeBlock, .stCodeBlock > div { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; }

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
  background: white !important; color: var(--text) !important;
}
.stSelectbox > div > div { background: white !important; }

[data-baseweb="popover"] {
  background: white !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; box-shadow: var(--shadow-hover) !important;
}
[data-baseweb="popover"] * { color: var(--text) !important; }
[data-baseweb="popover"] [role="option"] { background: white !important; color: var(--text) !important; padding: 0.5rem 1rem !important; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] { background: var(--bg) !important; color: var(--primary) !important; }
[data-baseweb="popover"] [role="option"]:focus { outline: none !important; box-shadow: inset 2px 0 0 var(--primary) !important; }

.stForm label, .stWidgetLabel { color: var(--text) !important; font-weight: 500 !important; }
</style>
"""

st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

# Initialize session state
for key, default in [
    ("pipeline_result", None),
    ("review_status", None),
    ("review_feedback", ""),
    ("review_feedback_history", []),
    ("review_persisted", None),
    ("review_error", None),
    ("should_run_pipeline", False),
    ("pipeline_params", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


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


def build_patient_context(patient: dict) -> str:
    return f"""
Patient: {patient['name']}, {patient['age']}yo {patient['gender']}
Conditions: {', '.join(patient['conditions'])}
Medications: {', '.join(patient['medications'])}
Allergies: {', '.join(patient['allergies']) or 'None'}
Last Visit: {patient['last_visit_summary']}
"""


st.markdown("""
<div class="top-bar">
  <div class="top-bar__brand">🏥 Doctor Office</div>
  <div class="top-bar__actions">
    <span style="font-size:0.85rem;color:var(--text-muted);">Neuro-Symbolic Clinical AI Review Console</span>
  </div>
</div>
""", unsafe_allow_html=True)


def parse_patient_context(patient_context: str, patient_name: str, patient_id: str) -> dict:
    """Parse the multi-line patient_context string into structured data."""
    result = {
        "name": patient_name,
        "patient_id": patient_id,
        "age": "?",
        "gender": "?",
        "conditions": [],
        "medications": [],
        "allergies": [],
    }

    patient_match = re.search(r'Patient:\s*([^,\n]+),\s*(\d+)-year-old\s+(\w+)', patient_context)
    if patient_match:
        result["name"] = patient_match.group(1).strip()
        result["age"] = patient_match.group(2)
        result["gender"] = patient_match.group(3).lower()
    elif patient_name:
        result["name"] = patient_name

    conditions_match = re.search(r'Conditions:\s*([^\n]+)', patient_context)
    if conditions_match:
        conditions_text = conditions_match.group(1).strip()
        if conditions_text.lower() not in ("none", "n/a", ""):
            result["conditions"] = [c.strip() for c in conditions_text.split(",")]

    medications_match = re.search(r'Medications:\s*([^\n]+)', patient_context)
    if medications_match:
        medications_text = medications_match.group(1).strip()
        if medications_text.lower() not in ("none", "n/a", ""):
            result["medications"] = [m.strip() for m in medications_text.split(",")]

    allergies_match = re.search(r'Allergies:\s*([^\n]+)', patient_context)
    if allergies_match:
        allergies_text = allergies_match.group(1).strip()
        if allergies_text.lower() not in ("none", "n/a", "nkda", ""):
            result["allergies"] = [a.strip() for a in allergies_text.split(",")]

    return result


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
    st.header("Load Demo Patient")
    st.caption("Local Ollama mode prioritizes clinical quality and safety guards.")
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
                st.session_state.loaded_patient_data = parse_patient_context(
                    selected_internist["patient_context"],
                    selected_internist["patient_name"],
                    selected_internist["patient_id"],
                )
                st.success(f"Loaded {selected_internist['patient_name']}")
                st.rerun()
        with col2:
            if st.button("Clear", use_container_width=True, key="clear_internist"):
                for key in ["loaded_chief_complaint", "loaded_patient_context", "loaded_visit_context",
                           "loaded_patient_id", "loaded_patient_name", "loaded_specialty",
                           "loaded_patient_data",
                           "vc_hpi", "vc_bp", "vc_hr", "vc_temp", "vc_rr", "vc_spo2",
                           "vc_weight", "vc_bmi", "vc_exam", "vc_findings", "vc_vitals_note"]:
                    st.session_state.pop(key, None)
                st.rerun()

    st.markdown("---")

    loaded_patient_data = st.session_state.get("loaded_patient_data")
    if loaded_patient_data:
        st.header("Patient Context")
        name = loaded_patient_data.get("name", "Unknown")
        patient_id = loaded_patient_data.get("patient_id", "—")
        age = loaded_patient_data.get("age", "?")
        gender = loaded_patient_data.get("gender", "?")
        conditions = loaded_patient_data.get("conditions", [])
        medications = loaded_patient_data.get("medications", [])
        allergies = loaded_patient_data.get("allergies", [])

        st.markdown(f"**{name}**  \n{age}yo {gender.capitalize() if isinstance(gender, str) else '?'}  \nID: `{patient_id}`")

        if conditions:
            st.markdown("**Conditions**")
            for c in conditions:
                st.markdown(f'<span class="pill pill--condition">{c}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill--empty">No conditions</span>', unsafe_allow_html=True)

        if medications:
            st.markdown("**Active Medications**")
            for m in medications:
                st.markdown(f'<span class="pill pill--medication">{m}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill--empty">No medications</span>', unsafe_allow_html=True)

        if allergies:
            st.markdown("**Allergies**")
            for a in allergies:
                st.markdown(f'<span class="pill pill--allergy">{a}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill--empty">NKDA</span>', unsafe_allow_html=True)

        st.markdown("---")

    memory_patient_id = loaded_patient_data.get("patient_id") if loaded_patient_data else patients[0]["patient_id"]
    st.header("Long-Term Memory")
    hist = load_patient_history(memory_patient_id)
    st.write(f"**Approved visits:** {hist['approved_visit_count']}")
    st.write(f"**Drafts on file:** {hist['draft_visit_count']}")
    for v in hist["visits"]:
        st.caption(f"{v['display_status']} — {v['timestamp'][:16].replace('T', ' ')} — {v.get('visit_id', '?')}")

    st.markdown("---")

    st.header("Run Settings")
    selected_mode = st.selectbox("Mode", ["fast"], index=0, key="mode_select", help="Model mode")
    use_memory = st.checkbox("Use Long-Term Memory", value=True, key="use_memory")

    st.markdown("---")

    with st.expander("Demo Batch Runner (Background)", expanded=False):
        st.caption("Run multiple demo patients sequentially in background. UI stays responsive.")

        internist_patients_batch = load_internist_patients()
        if internist_patients_batch:
            batch_options = {p["patient_id"]: f"{p['patient_id']} | {p['patient_name']} | {p['expected_path'].upper()}" for p in internist_patients_batch}
            default_batch = ["P001", "P002", "P007", "P005"]
            selected_batch = st.multiselect(
                "Select Patients",
                options=list(batch_options.keys()),
                default=default_batch,
                format_func=lambda pid: batch_options[pid],
                key="batch_patient_select"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Selected Batch", type="primary", use_container_width=True, key="run_batch", disabled=st.session_state.get("batch_running", False)):
                    if selected_batch:
                        st.session_state.batch_running = True
                        st.session_state.batch_status = {
                            "status": "running",
                            "selected": selected_batch,
                            "current_index": 0,
                            "completed": [],
                            "results": [],
                            "start_time": time.time(),
                            "error": None,
                        }
                        st.rerun()
                    else:
                        st.warning("Select at least one patient")
            with col2:
                if st.button("Refresh Status", use_container_width=True, key="refresh_batch"):
                    st.rerun()

            batch_status = st.session_state.get("batch_status")
            if batch_status:
                if batch_status["status"] == "running":
                    current_idx = batch_status["current_index"]
                    total = len(batch_status["selected"])
                    if current_idx < total:
                        current_pid = batch_status["selected"][current_idx]
                        st.info(f"Running: {current_pid} ({current_idx + 1}/{total})")
                    else:
                        st.info("Finalizing results...")
                    elapsed = time.time() - batch_status.get("start_time", time.time())
                    st.caption(f"Elapsed: {elapsed:.0f}s")

                if batch_status["completed"]:
                    st.success(f"{len(batch_status['completed'])}/{len(batch_status['selected'])} completed")
                    for res in batch_status.get("results", []):
                        pid = res.get("patient_id", "?")
                        actual = res.get("actual_path", "?")
                        expected = res.get("expected_path", "?")
                        match = "OK" if res.get("path_matched") else "MISMATCH"
                        total_t = res.get("total_seconds", 0)
                        scribe_t = res.get("timing", {}).get("scribe_seconds", 0)
                        flags = len(res.get("governance", {}).get("quality_flags", []))
                        hr = "YES" if res.get("governance", {}).get("needs_human_review") else "—"
                        primary = res.get("coding", {}).get("primary_icd10", "—")
                        secondary = "; ".join(res.get("coding", {}).get("secondary_icd10", [])) or "—"
                        cpt = res.get("coding", {}).get("cpt_suggestions", ["—"])[0]

                        with st.container():
                            st.markdown(f"""
                            **{pid}** | Expected: `{expected}` | Actual: `{actual}` ({match}) | {total_t:.1f}s (Scribe: {scribe_t:.1f}s) | ICD: {primary} | Sec: {secondary} | CPT: {cpt} | Flags: {flags} | Review: {hr}
                            """)

                if batch_status.get("error"):
                    st.error(f"Batch error: {batch_status['error']}")
                    st.session_state.batch_running = False

# ─── Main Input Form ───
default_chief_complaint = st.session_state.get("loaded_chief_complaint", "")
chief_complaint = st.text_area(
    "Chief Complaint / Reason for Visit",
    height=100,
    value=default_chief_complaint,
    placeholder="e.g., Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness."
)


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
                if value.strip():
                    extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "text_input":
                value = st.text_input(field_def.label, key=key, placeholder=field_def.placeholder, help=field_def.help_text)
                if value.strip():
                    extra_fields[field_def.key] = value.strip()
            elif field_def.field_type == "select":
                value = st.selectbox(field_def.label, options=field_def.options, key=key, help=field_def.help_text, index=0)
                if value and value not in ("Not assessed", "Select"):
                    extra_fields[field_def.key] = value
    return selected_specialty, extra_fields


selected_specialty, specialty_extra_fields = render_specialty_selector()

st.markdown("---")


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
        "hpi": hpi,
        "blood_pressure": bp,
        "heart_rate": int(hr) if hr.isdigit() else None,
        "temperature": float(temp) if temp.replace('.', '').isdigit() else None,
        "respiratory_rate": int(rr) if rr.isdigit() else None,
        "spo2": int(spo2) if spo2.isdigit() else None,
        "weight_kg": float(weight) if weight.replace('.', '').isdigit() else None,
        "bmi": float(bmi) if bmi.replace('.', '').isdigit() else None,
        "physical_exam": exam,
        "additional_findings": findings,
        "vitals_note": vitals_note,
        "specialty_fields": specialty_extra_fields,
    }
    return visit_context


visit_context = render_visit_context_inputs(specialty_extra_fields)

st.markdown("---")

col_run, col_mem = st.columns([3, 1])
with col_run:
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        if not chief_complaint.strip():
            st.error("Please enter a chief complaint.")
        else:
            loaded_context = st.session_state.get("loaded_patient_context")
            loaded_pid = st.session_state.get("loaded_patient_id")
            loaded_name = st.session_state.get("loaded_patient_name")
            loaded_patient_data = st.session_state.get("loaded_patient_data")

            if loaded_context and loaded_pid:
                patient_context = loaded_context
                patient_id = loaded_pid
                display_patient = loaded_patient_data or {"name": loaded_name or "Synthetic Patient", "patient_id": loaded_pid}
            else:
                patient_context = build_patient_context(patients[0])
                patient_id = patients[0]["patient_id"]
                display_patient = patients[0]

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
            st.session_state.display_patient = display_patient
            st.session_state.should_run_pipeline = True
            st.rerun()
with col_mem:
    st.caption("Neuro-symbolic guardrails active")


def run_pipeline(patient_context: str, chief_complaint: str, retrieved_notes: str,
                 mode: str = "fast", patient_id: str = None, use_memory: bool = True, visit_context: dict = None):
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


def run_batch_worker(selected_patients: List[str], mode: str):
    """Background thread worker to run batch sequentially."""
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))

    from agents.orchestrator import run_visit_pipeline

    all_patients = load_internist_patients()
    patients_dict = {p["patient_id"]: p for p in all_patients}

    status = st.session_state.batch_status

    for i, pid in enumerate(selected_patients):
        status["current_index"] = i
        patient = patients_dict.get(pid)
        if not patient:
            continue

        try:
            result = run_visit_pipeline(
                patient_context=patient["patient_context"],
                chief_complaint=patient["chief_complaint"],
                retrieved_notes="",
                mode=mode,
                patient_id=patient["patient_id"],
                use_memory=True,
                visit_context=patient["visit_context"],
                specialty=patient.get("specialty", "general_internal_medicine"),
                is_synthetic=True,
            )

            metadata = result.get("metadata") or {}
            timing = metadata.get("timing") or {}
            supervisor = metadata.get("supervisor") or {}
            control_flow = metadata.get("control_flow") or {}
            coding = result.get("coding_result") or {}
            soap = result.get("soap_note") or {}

            res = {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "expected_path": patient["expected_path"],
                "actual_path": metadata.get("coding_path", "unknown"),
                "path_matched": metadata.get("coding_path") == patient["expected_path"],
                "total_seconds": round(time.time() - status.get("start_time", time.time()), 2),
                "timing": {
                    "memory_load_seconds": timing.get("memory_load_seconds"),
                    "scribe_seconds": timing.get("scribe_seconds"),
                    "cleanup_seconds": timing.get("cleanup_seconds"),
                    "coding_seconds": timing.get("coding_seconds"),
                    "validation_seconds": timing.get("validation_seconds"),
                    "supervisor_seconds": timing.get("supervisor_seconds"),
                    "memory_save_seconds": timing.get("memory_save_seconds"),
                },
                "coding": {
                    "primary_icd10": coding.get("primary_icd10"),
                    "secondary_icd10": coding.get("secondary_icd10", []),
                    "cpt_suggestions": coding.get("cpt_suggestions", []),
                    "has_cpt_grounding": bool(coding.get("cpt_grounding")),
                    "cpt_grounding": coding.get("cpt_grounding"),
                },
                "governance": {
                    "retry_count": supervisor.get("retry_count", 0),
                    "empty_soap_retries": metadata.get("empty_soap_retries", 0),
                    "needs_human_review": supervisor.get("needs_human_review", False),
                    "final_decision": control_flow.get("final_decision"),
                    "quality_flags": result.get("quality_flags", []),
                },
                "soap_preview": {
                    "subjective": soap.get("subjective", "")[:300],
                    "objective": soap.get("objective", "")[:300],
                    "assessment": soap.get("assessment", "")[:300],
                    "plan": soap.get("plan", "")[:300],
                },
            }

            status["completed"].append(pid)
            status["results"].append(res)

        except Exception as e:
            status["error"] = f"Patient {pid}: {e}"
            status["status"] = "error"
            return

    status["status"] = "completed"
    status["current_index"] = len(selected_patients)


if st.session_state.should_run_pipeline:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center;">
        <div style="font-size:2.5rem;margin-bottom:0.75rem;">⏳</div>
        <div style="font-size:1.25rem;font-weight:600;color:var(--text);margin-bottom:0.5rem;">Running clinical pipeline…</div>
        <div style="font-size:0.9rem;color:var(--text-muted);">Memory Load → Scribe → Cleanup → Coding → Validation → Supervisor → Memory Save</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running clinical pipeline…"):
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

if st.session_state.get("batch_running") and st.session_state.get("batch_status", {}).get("status") == "running":
    batch_status = st.session_state.batch_status
    if batch_status.get("current_index", 0) < len(batch_status.get("selected", [])):
        if not st.session_state.get("batch_thread_started"):
            thread = threading.Thread(
                target=run_batch_worker,
                args=(batch_status["selected"], st.session_state.get("mode_select", "fast")),
                daemon=True,
            )
            thread.start()
            st.session_state.batch_thread_started = True
            st.session_state.batch_thread = thread
        time.sleep(1)
        st.rerun()
    else:
        st.session_state.batch_running = False
        st.session_state.batch_thread_started = False
        st.session_state.batch_status["status"] = "completed"
        st.rerun()

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    soap = result["soap_note"]
    coding = result["coding_result"]
    meta = result.get("metadata", {})
    total_time = st.session_state.get("pipeline_total_time", None)

    loaded_patient_data = st.session_state.get("loaded_patient_data")
    if loaded_patient_data:
        display_patient = {
            "name": loaded_patient_data.get("name", "Unknown"),
            "patient_id": loaded_patient_data.get("patient_id", "UNK"),
            "conditions": loaded_patient_data.get("conditions", []),
            "medications": loaded_patient_data.get("medications", []),
            "allergies": loaded_patient_data.get("allergies", []),
            "age": loaded_patient_data.get("age") or "?",
            "gender": loaded_patient_data.get("gender") or "?",
        }
    else:
        display_patient = st.session_state.get("display_patient", patients[0] if patients else {"name": "Unknown", "patient_id": "UNK"})

    display_patient.setdefault("conditions", [])
    display_patient.setdefault("medications", [])
    display_patient.setdefault("allergies", [])
    display_patient.setdefault("age", "?")
    display_patient.setdefault("gender", "?")

    params = st.session_state.pipeline_params or {}
    visit_context_used = meta.get("visit_context") or params.get("visit_context") or {}
    use_memory = params.get("use_memory", True)
    chief_complaint = params.get("chief_complaint", chief_complaint)
    selected_specialty = params.get("specialty", selected_specialty)
    specialty_extra_fields = params.get("specialty_extra_fields", specialty_extra_fields)

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

    def render_status_banner(result: dict, meta: dict, total_time: float = None):
        supervisor_meta = meta.get("supervisor", {}) or {}
        quality_flags = result.get("quality_flags", []) or []
        coding_path = meta.get("coding_path", "slow")
        visit_id = meta.get("memory", {}).get("visit_id", "—")
        visible_flags = [f for f in quality_flags if not f.startswith("Cleanup:")]
        flag_count = len(visible_flags)

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

        if visible_flags:
            with st.expander(f"Quality & Safety Flags ({flag_count})", expanded=flag_count > 0 and bool(supervisor_meta.get("needs_human_review"))):
                for flag in visible_flags:
                    st.warning(flag)

    def render_patient_card(patient: dict, meta: dict = None, visit_context: dict = None):
        bp = ""
        hba1c = ""
        if visit_context:
            bp = visit_context.get("blood_pressure", "")
            findings = visit_context.get("additional_findings", "")
            if findings and "hba1c" in findings.lower():
                match = re.search(r'HbA1c\s+([\d.]+%)', findings, re.IGNORECASE)
                if match:
                    hba1c = match.group(1)

        conditions = patient.get("conditions", [])
        conditions_html = "".join(f'<span class="pill pill--condition">{c}</span>' for c in conditions) or '<span class="pill pill--empty">No conditions</span>'
        meds = patient.get("medications", [])
        meds_html = "".join(f'<span class="pill pill--medication">{m}</span>' for m in meds) or '<span class="pill pill--empty">No medications</span>'
        allergies = patient.get("allergies", [])
        allergies_html = "".join(f'<span class="pill pill--allergy">{a}</span>' for a in allergies) or '<span class="pill pill--allergy">NKDA</span>'
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
            <div class="patient-card__demographics"><span>{patient.get('age', '?')}y</span><span>·</span><span>{patient.get('gender', '?')}</span></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Conditions</div><div class="patient-card__pills">{conditions_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Active Medications</div><div class="patient-card__pills">{meds_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Allergies</div><div class="patient-card__pills">{allergies_html}</div></div>
            <div class="patient-card__section"><div class="patient-card__section-title">Key Values</div><div class="patient-card__pills">{values_html}</div></div>
        </div>
        """, unsafe_allow_html=True)

    def render_soap_cards(soap: dict):
        sections = [
            ("Subjective", "💬", soap.get("subjective", "")),
            ("Objective", "🔍", soap.get("objective", "")),
            ("Assessment", "🧠", soap.get("assessment", "")),
            ("Plan", "📋", soap.get("plan", "")),
        ]
        cards_html = ""
        for title, icon, content in sections:
            safe_content = (content or "_No content_").replace("\n", "<br>")
            cards_html += (
                f'<div class="soap-card">'
                f'<div class="soap-card__header"><span class="soap-card__icon">{icon}</span><span class="soap-card__title">{title}</span></div>'
                f'<div class="soap-card__content">{safe_content}</div>'
                f'</div>'
            )
        st.markdown(f'<div class="soap-grid">{cards_html}</div>', unsafe_allow_html=True)

    def render_coding_results(coding: dict, meta: dict = None):
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
            cpt_code = first.split(" - ", 1)[0] if " - " in first else first

        primary_html = f'<span class="badge badge--primary">{primary}</span>'
        secondary_html = "".join(f'<span class="badge badge--secondary">{code}</span>' for code in secondary) if secondary else '<span class="badge badge--secondary">None</span>'
        cpt_badge = f'<span class="badge badge--cpt">CPT {cpt_code} · MDM: {mdm_level}</span>'
        fast_badge = '<span class="badge badge--fast">Fast Path</span>' if coding_path == "fast" else ""

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

    render_status_banner(result, meta, total_time)

    col_left, col_main = st.columns([1, 3], gap="large")
    with col_left:
        render_patient_card(display_patient, meta, visit_context_used)

    with col_main:
        st.subheader("📋 Clinical Note")
        st.caption("Verify numbered Assessment problems, medication actions, and acute precautions against chart context.")
        render_soap_cards(soap)

        st.markdown("---")

        st.subheader("🔖 Coding Results")
        render_coding_results(coding, meta)

        try:
            from agents.orchestrator import render_symptom_coding_summary
            render_symptom_coding_summary(meta, coding)
        except Exception:
            pass

        st.markdown("---")

        st.subheader("👨‍⚕️ Physician Decision")

        from utils.review_state import approval_preview
        draft_recorded = bool((meta.get("memory") or {}).get("saved")) and not review_state.in_history
        if review_state.allow_approve or review_state.allow_request_changes:
            preview = approval_preview(
                result,
                patient_name=display_patient.get("name", ""),
                patient_id=display_patient.get("patient_id"),
                chief_complaint=chief_complaint,
                specialty_name=display_patient.get('specialty', 'General'),
                memory_enabled=bool(use_memory),
                draft_recorded=draft_recorded,
            )
            with st.expander("📋 What Approve will do", expanded=True):
                st.markdown('<div class="approve-preview-content">', unsafe_allow_html=True)
                st.markdown(f"**{preview.summary}**")
                if preview.warning:
                    st.warning(preview.warning)
                for line in preview.detail_lines:
                    st.markdown(f"- {line}")
                st.markdown('</div>', unsafe_allow_html=True)

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

        if review_state.allow_revise:
            st.markdown("---")
            if st.button("✏️ Revise this note (new draft)", use_container_width=True, key="btn_revise"):
                st.session_state.review_status = "changes_requested"
                st.session_state.review_error = None
                st.rerun()
            st.caption("Revising regenerates the note with your feedback and supersedes this draft for review.")

        if review_state.key == "changes_requested":
            st.markdown("**Revision Feedback**")
            st.caption("This text is sent into the next pipeline run (Scribe + Coding) as a revision instruction.")
            feedback_key = "review_feedback_input"
            st.text_area(
                "Feedback for next run (Scribe + Coding)",
                key=feedback_key,
                height=110,
                placeholder="e.g., 'Specify Start/Continue/Adjust for medications', 'Frame suspected ACS as rule-out'"
            )
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