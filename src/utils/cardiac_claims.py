"""
Unsupported cardiac claims - one shared definition of what the note may not say.

Phase A2.1 came out of a real Cardiology UI run: the chart listed only diabetes
and hypertension, the visit data documented a normal sinus rhythm with
nonspecific ST-T changes, and the model still wrote "most consistent with
ischemic etiology", "further evaluation for ischemic heart disease" and
"if ischemic etiology is suspected".

Codes were correct (`R07.9`, no `I20`/`I21`), so the existing gates were fine -
the failure was purely narrative. This module is the deterministic backstop for
that, and it is the single source of truth for three callers:

- `utils.validation.clean_soap_note` scrubs the SOAP note.
- `agents.coding_node` scrubs the medical-necessity narrative.
- `agents.specialty_coverage` raises one advisory flag if a claim survives.

Two deliberate design points:

1. **Only diagnostic constructions are banned.** "nonspecific ST-T changes" and
   "no acute ischemic changes" are the clinician's own documented findings and
   must survive, so a bare mention of "ischemic" is fine. What is banned is
   naming a cardiac condition: a fixed phrase ("ischemic etiology", "myocardial
   infarction") or a diagnostic construction ("consistent with ...", "suggestive
   of ...", "concerning for ...").
2. **A negation window suppresses a match.** "denies angina" and "no acute
   ischemic changes" are statements *against* a diagnosis, so a negation within
   a short window around the match makes it acceptable.

The unlock condition is a cardiac diagnosis already in the **chart**
(`patient_context`). The visit context is deliberately not used: today's ECG note
contains the word "ischemic", so letting visit data unlock cardiac claims would
disable the whole guard.
"""

import re
from typing import Iterable, List, Tuple

# A cardiac diagnosis already documented in the chart unlocks cardiac narrative.
# Procedure words (stent, CABG, stress test) count too: they only appear in a
# chart that already carries a cardiac problem.
CARDIO_DIAGNOSIS_TERMS: Tuple[str, ...] = (
    "ischemic heart disease", "ischaemic heart disease", "ischem heart disease",
    "coronary artery disease", "coronary heart disease", "coronary atherosclerosis",
    "angina", "myocardial infarction", "heart attack", "acute coronary syndrome",
    "heart failure", "congestive heart failure", "cardiomyopathy",
    "atrial fibrillation", "afib", "a-fib", "valvular heart disease",
    "mitral stenosis", "aortic stenosis", "aortic regurgitation",
    "coronary stent", "cardiac stent", "stent", "cabg", "bypass surgery",
    "cardiac catheterization", "treadmill", "stress test", "prior mi",
    "ischemic cardiomyopathy", "ischaemic cardiomyopathy",
)

# Wording that names a cardiac condition outright.
_FIXED_CLAIMS: Tuple[Tuple[str, str], ...] = (
    (r"ischemic aetiology|ischaemic aetiology|ischemic etiology|ischaemic etiology",
     "ischemic etiology"),
    (r"ischemic heart disease|ischaemic heart disease|ischemic heart attack",
     "ischemic heart disease"),
    (r"myocardial infarction|heart attack", "myocardial infarction"),
    (r"acute coronary syndrome|\bacs\b", "acute coronary syndrome"),
    (r"angina pectoris", "angina pectoris"),
    (r"\b(?:unstable|stable|exertional|microvascular|vasospastic|variant|crescendo"
     r"|new-onset|atypical)\s+angina\b", "cardiac angina"),
    # "MI" only when the author wrote it upper case, so ordinary words are safe.
    (r"\bMI\b", "MI"),
)

# Wording that reaches a cardiac condition through a diagnostic construction.
_DIAGNOSES = (
    # `ischa?e?mi[ac]s?` covers ischemic / ischaemic / ischemia / ischaemia:
    # the letters are 'a' then 'e' in both spellings (is-ch-em / is-ch-a-em).
    r"(?:\w+[\s-]){0,2}?(?:ischa?e?mi[ac]s?|isch[aä]emic(?:\s+(?:heart|cardiac))?"
    r"|angina|coronary artery disease|coronary heart disease|coronary ischemia"
    r"|coronary ischaemia|myocardial infarction"
    r"|acute coronary (?:syndrome|event)|\bacs\b)"
)
_CONSTRUCTIONS = (
    r"(?:most\s+|highly\s+|strongly\s+)?consistent\s+with|suggestive\s+of"
    r"|indicative\s+of|diagnostic\s+of|representing|typical\s+of|characteristic\s+of"
    r"|suspicious\s+for|concerning\s+for|worrisome\s+for|raises\s+suspicion\s+for"
    r"|due\s+to|caused\s+by|secondary\s+to|reflects|representing"
    r"|correlated\s+with|in\s+keeping\s+with|points?\s+to|pointing\s+to"
    r"|raising\s+(?:the\s+)?possibility|possibility\s+of|raises?\s+concern\s+for"
    r"|concern\s+for|probability\s+of|likelihood\s+of|most\s+likely"
    r"|likely\s+to\s+be|probable\s+"
    # "the etiology is ischemic", "the pain is likely angina"
    r"|(?:is|are|was|were)(?:\s+(?:most\s+likely|likely))?"
)
_CONSTRUCTED_CLAIMS: Tuple[Tuple[str, str], ...] = (
    # Up to five words between the construction and the condition: the small
    # models pad it ("most consistent with a new or acute presentation of
    # angina"), which is how that phrasing reached a finished note.
    (rf"\b(?:{_CONSTRUCTIONS})\s+(?:\w+[\s-]){{0,5}}?(?:{_DIAGNOSES})\b",
     "cardiac diagnostic claim"),
)

# A negation near the match turns it into a statement against the diagnosis.
_NEGATION_WINDOW = 70
_NEGATION = re.compile(
    r"\b(?:no|not|never|denies|denied|deny|without|ruled\s+out|rules\s+out"
    r"|rule\s+out|free\s+of|negative\s+for|unlikely|absent|resolved"
    r"|absence\s+of|lack\s+of)\b",
    re.IGNORECASE,
)

# Naming a condition as a future risk is not diagnosing it. The plan explicitly
# allows risk framing ("diabetes and hypertension raise ischemic heart disease
# risk"), so risk wording near the match makes it acceptable.
_RISK_WINDOW = 70
_RISK_FRAMING = re.compile(
    r"\b(?:risk|risks|at\s+risk|high[- ]risk|risk\s+factor|screening|surveillance"
    r"|prevent\w*|prevention|modif\w*|counsel\w*|reduc\w*|lower\w*|address\w*"
    r"|targets?|lifelong|long[- ]term)\b",
    re.IGNORECASE,
)

# Sentence split that tolerates the abbreviations a clinical note is full of.
_ABBREVIATIONS = ("e.g", "i.e", "vs", "mg", "mm", "hr", "bpm", "pt", "dr", "approx")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

CARDIAC_CLAIM_FLAG_TEMPLATE = (
    "Cardiac claim not supported by the chart: {claims} - no cardiac diagnosis is "
    "listed in Patient Context, so describe this as a symptom and plan follow-up "
    "instead"
)


def has_charted_cardiac_diagnosis(*texts: str) -> bool:
    """True when a cardiac diagnosis is already documented in the chart.

    Args:
        *texts: Free-form chart text, normally just `patient_context`. The visit
            context must not be passed: today's ECG note carries the words that
            this function looks for.
    """
    blob = " ".join(str(text or "") for text in texts).lower()
    if not blob.strip():
        return False
    return any(term in blob for term in CARDIO_DIAGNOSIS_TERMS)


def _negated(text: str, start: int, end: int) -> bool:
    """True when a negation sits close to `text[start:end]`."""
    window = text[max(0, start - _NEGATION_WINDOW):end + _NEGATION_WINDOW]
    return bool(_NEGATION.search(window))


def _risk_framed(text: str, start: int, end: int) -> bool:
    """True when the match is framed as future risk rather than a diagnosis."""
    window = text[max(0, start - _RISK_WINDOW):end + _RISK_WINDOW]
    return bool(_RISK_FRAMING.search(window))


def find_unsupported_cardiac_claims(text: str, has_diagnosis: bool = False) -> List[str]:
    """Cardiac diagnoses the text asserts without a charted cardiac diagnosis.

    Args:
        text: The narrative to inspect (a section, or the whole note).
        has_diagnosis: Whether the chart already documents a cardiac diagnosis,
            in which case nothing is reported.

    Returns:
        Deduplicated labels such as ``["ischemic etiology"]``.
    """
    if has_diagnosis or not text or not str(text).strip():
        return []

    raw = str(text)
    found: List[str] = []
    for pattern, label in _FIXED_CLAIMS + _CONSTRUCTED_CLAIMS:
        for match in re.finditer(pattern, raw, re.IGNORECASE):
            if _negated(raw, match.start(), match.end()):
                continue
            if _risk_framed(raw, match.start(), match.end()):
                continue
            if label not in found:
                found.append(label)
    return found


def split_sentences(text: str) -> List[str]:
    """Split into sentences without breaking on common clinical abbreviations."""
    if not text:
        return []
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(abbreviation)}\.", f"{abbreviation}\x00",
                           protected, flags=re.IGNORECASE)
    sentences = _SENTENCE_SPLIT.split(protected)
    return [sentence.replace("\x00", ".").strip() for sentence in sentences if sentence.strip()]


def scrub_unsupported_cardiac_claims(text: str, has_diagnosis: bool = False) -> Tuple[str, List[str]]:
    """Remove sentences that assert a cardiac diagnosis the chart does not support.

    Sentence-level removal (rather than word-level substitution) is deliberate:
    "Chest pain is most consistent with ischemic etiology." should disappear
    rather than survive as "most consistent with [REMOVED]". Any resulting gap is
    reported so the scribe node can raise its advisory defaulted-content flag.

    Args:
        text: The narrative to scrub.
        has_diagnosis: Whether the chart already documents a cardiac diagnosis.

    Returns:
        Tuple of (scrubbed text, labels of the claims that were removed).
    """
    if has_diagnosis or not text or not str(text).strip():
        return text, []

    kept: List[str] = []
    removed: List[str] = []
    for sentence in split_sentences(text):
        claims = find_unsupported_cardiac_claims(sentence, has_diagnosis)
        if claims:
            for claim in claims:
                if claim not in removed:
                    removed.append(claim)
        else:
            kept.append(sentence)
    if not removed:
        return text, []
    return " ".join(kept).strip(), removed


def scrub_numbered_lines(text: str, has_diagnosis: bool = False) -> Tuple[str, List[str]]:
    """Scrub whole lines (plan steps) that assert an unsupported cardiac claim."""
    if has_diagnosis or not text or not str(text).strip():
        return text, []

    kept: List[str] = []
    removed: List[str] = []
    for line in str(text).split("\n"):
        claims = find_unsupported_cardiac_claims(line, has_diagnosis)
        if claims:
            for claim in claims:
                if claim not in removed:
                    removed.append(claim)
        else:
            kept.append(line)
    if not removed:
        return text, []
    return "\n".join(kept), removed


def build_cardiac_claim_flag(text: str, has_diagnosis: bool = False) -> List[str]:
    """One advisory flag when a cardiac claim survived the scrub.

    The scrub in `clean_soap_note` normally removes these first, so this is a
    backstop: it catches a claim the sentence splitter could not isolate, or a
    node invoked outside the normal cleanup step. It is advisory by design and
    carries no supervisor retry marker.
    """
    claims = find_unsupported_cardiac_claims(text, has_diagnosis)
    if not claims:
        return []
    return [CARDIAC_CLAIM_FLAG_TEMPLATE.format(claims=", ".join(claims))]


def scrub_all(texts: Iterable[str], has_diagnosis: bool = False) -> Tuple[List[str], List[str]]:
    """Scrub several narratives at once, merging the removed-claim labels."""
    scrubbed: List[str] = []
    removed: List[str] = []
    for text in texts:
        cleaned, claims = scrub_unsupported_cardiac_claims(text, has_diagnosis)
        scrubbed.append(cleaned)
        for claim in claims:
            if claim not in removed:
                removed.append(claim)
    return scrubbed, removed


__all__ = [
    "CARDIO_DIAGNOSIS_TERMS",
    "CARDIAC_CLAIM_FLAG_TEMPLATE",
    "build_cardiac_claim_flag",
    "find_unsupported_cardiac_claims",
    "has_charted_cardiac_diagnosis",
    "scrub_all",
    "scrub_numbered_lines",
    "scrub_unsupported_cardiac_claims",
    "split_sentences",
]