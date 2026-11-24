"""
Specialty Profiles - Configuration-driven specialty definitions.

Each specialty profile defines:
- id: unique identifier
- display_name: human-readable name for UI
- description: brief description of the specialty focus
- prompt_emphasis: guidance injected into the Scribe Agent prompt
- extra_input_fields: optional specialty-specific input fields for the UI,
  each optionally tagged with the SOAP section its content belongs in
  (`ExtraInputField.prompt_section`)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# SOAP sections a specialty field can be tagged with. A tag is guidance for the
# prompt ("put this finding in that section") and a hint for the coverage guard;
# it is not a requirement that the model place the finding in exactly that
# section, so tagging is opt-in per field and defaults to untagged.
PROMPT_SECTIONS: Tuple[str, ...] = ("subjective", "objective", "assessment")


@dataclass
class ExtraInputField:
    """Definition of a specialty-specific extra input field.

    Attributes:
        prompt_section: SOAP section this field's content belongs in
            ("subjective" | "objective" | "assessment"). Empty means untagged:
            the value is still shown to the Scribe, just without a section hint.
    """
    key: str
    label: str
    field_type: str = "text_area"  # "text_area", "text_input", "select"
    placeholder: str = ""
    help_text: str = ""
    options: List[str] = field(default_factory=list)  # for select type
    prompt_section: str = ""


@dataclass
class SpecialtyProfile:
    """Configuration for a medical specialty."""
    id: str
    display_name: str
    description: str
    prompt_emphasis: str
    extra_input_fields: List[ExtraInputField] = field(default_factory=list)


# Specialty profiles - extendable via configuration
SPECIALTY_PROFILES: Dict[str, SpecialtyProfile] = {
    "general": SpecialtyProfile(
        id="general",
        display_name="General / Internal Medicine",
        description="General internal medicine and primary care encounters",
        prompt_emphasis=(
            "General Internal Medicine: comprehensive assessment of presenting complaint "
            "in context of chronic conditions. Address medication management and care "
            "coordination. Document all chronic conditions from patient context; ensure "
            "plan addresses each active problem."
        ),
        extra_input_fields=[],
    ),
    "cardiology": SpecialtyProfile(
        id="cardiology",
        display_name="Cardiology",
        description="Cardiovascular disease evaluation and management",
        prompt_emphasis=(
            "CARDIOLOGY — cardiac problem first. USE DOCUMENTED DATA (tagged by SOAP section):\n"
            "- Chest-pain (SUBJ): quality, location, radiation, severity, triggers, relief.\n"
            "- ECG/rhythm (OBJ): rhythm, rate, ST-T findings as documented.\n"
            "- NYHA class (ASSESS): use supplied class. If tagged field unmentioned → note incomplete.\n\n"
            "ASSESS: Lead with cardiac presentation and documented findings (exertional character, ECG finding, NYHA class). State what data shows AND what it does not establish (ischemia unconfirmed).\n\n"
            "PLAN: Act on results already documented today. If ECG/lab recorded, do not re-order — act on result or say nondiagnostic + next step. For NO charted cardiac dx: plan = symptom follow-up, risk-factor control, return precautions. Naming cardiac condition as reason is NOT allowed.\n\n"
            "BANNED unless SAME dx in Patient Context:\n"
            "- 'ischemic etiology/heart disease', 'consistent with ischemia/angina', word 'angina', MI/ACS/coronary ischemia from presenting symptom.\n"
            "- NEVER invent murmur, gallop, irregular rhythm, ST-T change, troponin. Use only clinician-supplied or chart-stated data.\n"
            "- NEVER diagnose angina/ischemia/MI from chest pain alone. I20.x/I21.x require cardiac dx in Patient Context."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="chest_pain_characteristics",
                label="Chest Pain Characteristics",
                field_type="text_area",
                placeholder="e.g., 'Substernal pressure, 30 min, radiates to left arm, 8/10, exertion, relieved by rest/NTG'",
                help_text="Onset, duration, quality, location, radiation, severity (1-10), precipitating/alleviating factors",
                prompt_section="subjective",
            ),
            ExtraInputField(
                key="ecg_rhythm_notes",
                label="ECG / Rhythm Notes",
                field_type="text_area",
                placeholder="e.g., 'Sinus rhythm 78 bpm, nonspecific ST-T changes V4-V6, no acute ischemic changes'",
                help_text="Rhythm, rate, intervals, axis, ST/T changes, prior ECG comparison",
                prompt_section="objective",
            ),
            ExtraInputField(
                key="functional_status_nyha",
                label="Functional Status / NYHA Class",
                field_type="select",
                placeholder="Select NYHA class",
                help_text="New York Heart Association functional classification",
                options=["Class I - No limitation", "Class II - Slight limitation", "Class III - Marked limitation", "Class IV - Symptoms at rest", "Not assessed"],
                prompt_section="assessment",
            ),
        ],
    ),
    "neurology": SpecialtyProfile(
        id="neurology",
        display_name="Neurology",
        description="Neurological conditions evaluation and management",
        prompt_emphasis=(
            "Neurology: detailed neuro history (onset, progression, distribution, "
            "triggers, associated features). Document full neuro exam: mental status, "
            "cranial nerves, motor, sensory, reflexes, coordination, gait. Assessment: "
            "localize lesion (central/peripheral, cortical/subcortical, specific "
            "nerve/plexus/nerve), prioritized differential. Plan: diagnostic workup "
            "(imaging, EMG/NCS, labs), disease-modifying therapy if applicable, "
            "symptomatic management."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="neurological_symptoms_detail",
                label="Neurological Symptoms Detail",
                field_type="text_area",
                placeholder="e.g., 'Right-sided weakness onset 2h ago, face/arm > leg, no speech difficulty, no visual changes'",
                help_text="Onset, progression, distribution, laterality, associated symptoms",
            ),
            ExtraInputField(
                key="neuro_exam_findings",
                label="Neurological Exam Findings",
                field_type="text_area",
                placeholder="e.g., 'CN II-XII intact. Motor: R UE 3/5, R LE 4/5. Sensory: decreased pinprick R face/arm. Reflexes: 3+ R UE/LE, Babinski R'",
                help_text="Mental status, cranial nerves, motor, sensory, reflexes, coordination, gait",
            ),
        ],
    ),
    "dermatology": SpecialtyProfile(
        id="dermatology",
        display_name="Dermatology",
        description="Skin conditions evaluation and management",
        prompt_emphasis=(
            "Dermatology: lesion morphology (primary type: macule/papule/plaque/"
            "nodule/vesicle/bulla/pustule/wheal), color, border, configuration, "
            "distribution, arrangement. Evolution, symptoms (pruritus/pain/burning), "
            "triggers, prior treatments. Exam: dermatologic terminology, distribution "
            "pattern, systemic signs. Assessment: morphologic dx + differential. "
            "Plan: biopsy if needed, topical/systemic therapy, follow-up for response."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="lesion_description",
                label="Lesion Description",
                field_type="text_area",
                placeholder="e.g., '3 cm well-demarcated erythematous plaque with silvery scale on R elbow, symmetric on knees'",
                help_text="Morphology, size, color, border, configuration, distribution, evolution",
            ),
            ExtraInputField(
                key="dermoscopy_notes",
                label="Dermoscopy Findings (if performed)",
                field_type="text_area",
                placeholder="e.g., 'Regular pigment network, no atypical vessels, no blue-white veil'",
                help_text="Dermoscopic structures, patterns, colors noted",
            ),
        ],
    ),
    "oncology": SpecialtyProfile(
        id="oncology",
        display_name="Oncology",
        description="Cancer evaluation, treatment, and surveillance",
        prompt_emphasis=(
            "Oncology: cancer history (primary site, histology, stage, prior "
            "treatments with agents/doses/dates/response, current line). "
            "Performance status (ECOG), treatment toxicities (CTCAE), disease "
            "symptoms. Exam: primary site, nodal basins, metastatic sites, toxicity. "
            "Assessment: disease status (response per RECIST/iRECIST, progression, "
            "remission), treatment plan rationale, goals of care. Plan: next cycle, "
            "dose modifications, supportive care, surveillance schedule, trials."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="cancer_history_detail",
                label="Cancer History Detail",
                field_type="text_area",
                placeholder="e.g., 'Stage IIIC Breast Cancer (ER+/PR+/HER2-), dx 2022. ddAC-T, bilateral mastectomy, RT chest wall. Adjuvant letrozole x 18mo. No recurrence.'",
                help_text="Primary site, histology, stage, prior treatments with dates/response, current line",
            ),
            ExtraInputField(
                key="performance_status",
                label="Performance Status (ECOG)",
                field_type="select",
                placeholder="Select ECOG status",
                help_text="Eastern Cooperative Oncology Group performance status",
                options=["0 - Fully active", "1 - Restricted in strenuous activity", "2 - Ambulatory, up >50% waking hours", "3 - Limited self-care, bed >50%", "4 - Completely disabled", "Not assessed"],
            ),
            ExtraInputField(
                key="treatment_toxicities",
                label="Current Treatment Toxicities (CTCAE)",
                field_type="text_area",
                placeholder="e.g., 'Grade 2 peripheral neuropathy (hands/feet), Grade 1 fatigue, Grade 0 nausea'",
                help_text="Treatment-related adverse events with CTCAE grades",
            ),
        ],
    ),
}


def get_specialty_profile(specialty_id: str) -> SpecialtyProfile:
    """Get a specialty profile by ID, defaulting to general."""
    return SPECIALTY_PROFILES.get(specialty_id, SPECIALTY_PROFILES["general"])


def get_specialty_choices() -> List[tuple]:
    """Get list of (id, display_name) tuples for UI dropdown."""
    return [(profile.id, profile.display_name) for profile in SPECIALTY_PROFILES.values()]


def get_default_specialty_id() -> str:
    """Get the default specialty ID."""
    return "general"


def validate_specialty_id(specialty_id: str) -> bool:
    """Check if a specialty ID is valid."""
    return specialty_id in SPECIALTY_PROFILES


__all__ = [
    "ExtraInputField",
    "SpecialtyProfile",
    "SPECIALTY_PROFILES",
    "get_specialty_profile",
    "get_specialty_choices",
    "get_default_specialty_id",
    "validate_specialty_id",
]