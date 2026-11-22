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
            "This is a General Internal Medicine encounter. "
            "Focus on comprehensive assessment of the patient's presenting complaint "
            "in the context of their chronic conditions. Address preventive care, "
            "medication management, and coordination of care as appropriate. "
            "Document all chronic conditions from the patient context and ensure "
            "the plan addresses each active problem."
        ),
        extra_input_fields=[],
    ),
    "cardiology": SpecialtyProfile(
        id="cardiology",
        display_name="Cardiology",
        description="Cardiovascular disease evaluation and management",
        prompt_emphasis=(
            "This is a Cardiology encounter. Work the cardiac problem first.\n\n"
            "USE THE DOCUMENTED CARDIAC DATA - do not ignore it:\n"
            "- When the clinician supplied chest-pain characteristics, ECG/rhythm notes "
            "or functional status, the note must reflect them. Each item in the visit "
            "block is tagged with the SOAP section it belongs in: chest-pain character "
            "is SUBJECTIVE, ECG and rhythm findings are OBJECTIVE, functional class is "
            "ASSESSMENT. Place each item where it is tagged, in the clinician's own "
            "words where you can.\n"
            "- SUBJECTIVE must carry the documented chest-pain descriptors (quality, "
            "location, radiation, severity, what brings it on, what relieves it). "
            "'Chest pain x 1 day' is not an acceptable substitute for what the "
            "clinician recorded.\n"
            "- OBJECTIVE must record the supplied ECG/rhythm findings as findings "
            "(rhythm, rate, ST-T findings). Vitals alone do not document the ECG.\n"
            "- ASSESSMENT must use the documented functional class.\n"
            "- If a supplied field goes unmentioned the note is incomplete: the "
            "clinician recorded that data for this visit, so use it.\n\n"
            "ASSESSMENT:\n"
            "- Lead with the cardiac presentation and what the documented evidence "
            "supports, naming the specific findings you were given (exertional "
            "character, a documented ECG finding, the documented functional class).\n"
            "- State what the data shows AND what it does not yet establish. Say that "
            "ischaemia is unconfirmed when it is unconfirmed.\n\n"
            "PLAN:\n"
            "- Act on results already documented today. If an ECG or lab is already "
            "recorded, do not re-order it - act on that result, or say it was "
            "nondiagnostic and name the next step the findings justify.\n"
            "- For a presentation with no charted cardiac diagnosis, the plan is "
            "symptom follow-up, risk-factor control and return precautions. Naming a "
            "cardiac condition as the reason for those steps is not allowed.\n"
            "- Include the risk-factor management and follow-up interval the documented "
            "findings support.\n\n"
            "BANNED unless the SAME diagnosis is already listed in Patient Context:\n"
            "- Do not write 'ischemic etiology', 'ischemic heart disease', "
            "'consistent with ischemia/ischaemia', 'consistent with angina', or name "
            "angina at all.\n"
            "- Do not diagnose or imply myocardial infarction, ACS or coronary "
            "ischaemia from the presenting symptom.\n"
            "- ALLOWED: describe the presentation as a symptom; name risk factors "
            "(diabetes, hypertension) as risk factors without concluding a cardiac "
            "diagnosis; plan follow-up, risk-factor control, return precautions, and "
            "correlate with the documented ECG.\n\n"
            "NEVER:\n"
            "- Never invent a murmur, gallop, irregular rhythm, ST-T change, troponin "
            "value or any other cardiac finding. Use only what the clinician supplied "
            "or the chart states.\n"
            "- Never diagnose angina, ischaemia or myocardial infarction from chest "
            "pain alone. I20.x/I21.x require a cardiac diagnosis already documented in "
            "Patient Context."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="chest_pain_characteristics",
                label="Chest Pain Characteristics",
                field_type="text_area",
                placeholder="e.g., 'Substernal pressure, 30 min duration, radiates to left arm, 8/10 severity, precipitated by exertion, relieved by rest and NTG'",
                help_text="Onset, duration, quality, location, radiation, severity (1-10), precipitating/alleviating factors",
                prompt_section="subjective",
            ),
            ExtraInputField(
                key="ecg_rhythm_notes",
                label="ECG / Rhythm Notes",
                field_type="text_area",
                placeholder="e.g., 'Sinus rhythm 78 bpm, nonspecific ST-T changes in V4-V6, no acute ischemic changes'",
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
            "This is a Neurology encounter. "
            "Emphasize detailed neurological history: onset, progression, "
            "distribution of symptoms, triggers, associated features. "
            "Document neurological exam thoroughly: mental status, cranial nerves, "
            "motor (strength, tone, bulk), sensory (pinprick, vibration, position), "
            "reflexes, coordination, gait, station. "
            "Assessment should localize the lesion (central vs peripheral, "
            "cortical vs subcortical, specific nerve root/plexus/nerve) and "
            "provide a prioritized differential. "
            "Plan should address diagnostic workup (imaging, EMG/NCS, labs), "
            "disease-modifying therapy if applicable, and symptomatic management."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="neurological_symptoms_detail",
                label="Neurological Symptoms Detail",
                field_type="text_area",
                placeholder="e.g., 'Right-sided weakness onset 2 hours ago, face/arm > leg, no speech difficulty, no visual changes'",
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
            "This is a Dermatology encounter. "
            "Emphasize lesion morphology: primary lesion type (macule, papule, plaque, "
            "nodule, vesicle, bulla, pustule, wheal), color, border, configuration, "
            "distribution, arrangement. Document evolution, symptoms (pruritus, pain, "
            "burning), triggers, prior treatments. "
            "Exam should describe lesions using dermatologic terminology, "
            "note distribution pattern, and any systemic signs. "
            "Assessment should include morphologic diagnosis and differential. "
            "Plan should address biopsy if needed, topical/systemic therapy, "
            "and follow-up for monitoring treatment response."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="lesion_description",
                label="Lesion Description",
                field_type="text_area",
                placeholder="e.g., '3 cm well-demarcated erythematous plaque with silvery scale on right elbow, symmetric similar lesions on knees'",
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
            "This is an Oncology encounter. "
            "Emphasize cancer history: primary site, histology, stage at diagnosis, "
            "prior treatments (surgery, radiation, systemic therapy with agents/doses, "
            "dates, response), current treatment line. Document performance status "
            "(ECOG/Karnofsky), treatment-related toxicities (CTCAE grading if known), "
            "disease-related symptoms. "
            "Exam should focus on primary site, nodal basins, common metastatic sites, "
            "and toxicity assessment. "
            "Assessment should state disease status (response per RECIST/iRECIST, "
            "progression, remission), treatment plan rationale, and goals of care. "
            "Plan should detail next treatment cycle, dose modifications, supportive "
            "care, surveillance imaging/labs schedule, and clinical trial considerations."
        ),
        extra_input_fields=[
            ExtraInputField(
                key="cancer_history_detail",
                label="Cancer History Detail",
                field_type="text_area",
                placeholder="e.g., 'Stage IIIC Breast Cancer (ER+/PR+/HER2-), diagnosed 2022. Treated with dose-dense AC-T, bilateral mastectomy, RT to chest wall. On adjuvant letrozole x 18 months. No recurrence.'",
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