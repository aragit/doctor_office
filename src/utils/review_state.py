"""
The Physician Review state machine.

Pure functions, no Streamlit import: the review tab renders what this module
decides, and the tests assert the decisions without a browser. It exists so that
"which state am I in, and which actions are legitimate right now" is a single
answer, instead of several `if review_status == ...` branches in the app that can
drift apart and contradict each other.

One rule drives the design: **a decision is only legitimate on a draft the
pipeline has just produced.** Once a note is approved or rejected, the three
decision buttons are disabled, because a second Approve on the same draft would
either duplicate the stored visit or silently look like a fresh one. The way
back into reviewing is a *new pipeline run*, either from the Run Pipeline button
or from the Revise action, which forces the physician's feedback through the
Scribe and Coding agents again.

The second rule is the draft/approved split, and this module is where it becomes
visible to a physician. The pipeline's memory-save step records the run as a
**draft**; that row is auditable but is not history - the Scribe does not read
it, and it is not counted in `visit_count`. Only an Approve promotes it into
approved history. So `Awaiting review` says the visit is a draft, `Reject` says
it never became history, and `Approved` is the only state that claims the visit
joined the patient's history.

States, all derived from data the app already keeps:

| state | when | decisions offered |
|---|---|---|
| `awaiting_review` | a run finished, nothing decided yet | approve / request changes / reject |
| `escalated` | the supervisor exhausted retries | approve / request changes / reject |
| `changes_requested` | the physician asked for a revision | none until feedback is submitted |
| `approved_saved` | approved, and the visit is in approved history | none (revise instead) |
| `approved_not_saved` | approved while long-term memory is off | none (revise instead) |
| `save_failed` | the approve write raised | approve (retry) / request changes / reject |

Nothing here writes anything. `approval_preview` only describes what the *next*
Approve would hand to `approve_visit`: whether it promotes the pipeline's draft
or writes an approved row, and the two things a physician would otherwise
discover late - the SOAP note is stored as Assessment + Plan only (each truncated
to 400 characters), and the store keeps a rolling window of the most recent 10
visits per patient, dropping the oldest *draft* first.
"""

import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

__all__ = [
    "REVIEW_ACTIONS",
    "REVIEW_STATES",
    "STATUS_APPROVED",
    "STATUS_CHANGES_REQUESTED",
    "STATUS_REJECTED",
    "ApprovalPreview",
    "ReviewState",
    "allowed_actions",
    "approval_preview",
    "key_codes",
    "regeneration_progress_message",
    "review_state",
    "revision_summary",
]


# Session-state values the app stores for `st.session_state.review_status`.
# `None` means "no decision yet", i.e. a fresh draft.
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_CHANGES_REQUESTED = "changes_requested"

# The three decisions plus the escape hatch. Order is reading order.
REVIEW_ACTIONS: Tuple[str, ...] = (
    "approve",
    "request_changes",
    "reject",
    "revise",
)

# The keys `review_state` can return, for tests and for anything that wants to
# branch on the state without re-deriving it.
REVIEW_STATES: Tuple[str, ...] = (
    "awaiting_review",
    "escalated",
    "changes_requested",
    "approved_saved",
    "approved_not_saved",
    "save_failed",
)

_CODE_RE = re.compile(r"([A-Z]\d{2}(?:\.\d+)?)")

# What `PatientMemory.record_visit` keeps, and the limits it applies. Mirrored
# here so the preview cannot drift from the writer without being noticed.
_SOAP_FIELD_LIMIT = 400
_VISIT_HISTORY_LIMIT = 10


class ReviewState(NamedTuple):
    """What the review banner says and what the physician may do.

    Attributes:
        key: One of `REVIEW_STATES`.
        label: Short status word for the banner, e.g. "Awaiting review".
        level: Streamlit severity to render the banner with
            ("info" | "success" | "warning" | "error").
        detail: One sentence saying what this state means for this note.
        next_step: What the physician should do now, or "" when nothing is
            required.
        in_history: Whether this visit is now approved longitudinal history -
            i.e. it counts in `visit_count` and the Scribe will read it.
        draft_recorded: Whether the pipeline recorded this run as a draft row.
            Distinct from `in_history`: a draft exists in the store for audit,
            but it is unsigned and unreadable as history.
        terminal: True when the three decisions are closed and the only way
            forward is a new run.
        allow_approve / allow_request_changes / allow_reject / allow_revise:
            Per-action enablement, so a partial state can close one action
            without closing the others.
    """

    key: str
    label: str
    level: str
    detail: str
    next_step: str
    in_history: bool
    draft_recorded: bool
    terminal: bool
    allow_approve: bool
    allow_request_changes: bool
    allow_reject: bool
    allow_revise: bool


class ApprovalPreview(NamedTuple):
    """What an Approve click would store, described before it is clicked.

    Attributes:
        will_persist: Whether the Approve will put this visit into approved
            history (memory on and the visit is not in history yet).
        already_saved: True when the pipeline's draft row is what an Approve
            would promote, so no second row is written.
        summary: The one-line confirmation shown next to the buttons.
        detail_lines: Field-by-field breakdown, for the expandable version.
        warning: Empty unless the physician should know something before
            approving.
    """

    will_persist: bool
    already_saved: bool
    summary: str
    detail_lines: List[str]
    warning: str


def _codes_of(entries: Optional[Iterable[Any]]) -> List[str]:
    """Canonical code tokens from `coding_result` entries.

    Accepts either `"E11.9"` or `"E11.9 - Type 2 diabetes mellitus ..."` and
    returns the bare code, so the preview reads the way the code list does.
    """
    found: List[str] = []
    for entry in entries or []:
        match = _CODE_RE.search(str(entry).upper())
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def key_codes(coding_result: Optional[Dict[str, Any]]) -> List[str]:
    """Primary code first, then the secondary codes, de-duplicated."""
    coding_result = coding_result or {}
    codes = _codes_of([coding_result.get("primary_icd10")])
    codes += [code for code in _codes_of(coding_result.get("secondary_icd10"))
              if code not in codes]
    return codes


def _code_summary(coding_result: Optional[Dict[str, Any]], limit: int = 3) -> str:
    """Compact code list for the confirmation line, e.g. `E11.9 +2 more`."""
    codes = key_codes(coding_result)
    if not codes:
        return "no codes"
    if len(codes) <= limit:
        return ", ".join(codes)
    return f"{', '.join(codes[:limit])} +{len(codes) - limit} more"


def _short(text: Any, limit: int = 72) -> str:
    """Single-line, length-capped text for a banner or summary line."""
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rsplit(" ", 1)[0] + "..."


def _draft_note(draft_recorded: bool) -> str:
    """What a pipeline draft is worth, before anyone has decided about it.

    The distinction is the point of P2: a recorded draft exists in the store but
    is unsigned, so it is not history and the Scribe will not read it. Saying
    which of the two the run produced keeps the banner honest instead of
    implying that a finished run is already part of the patient's record.
    """
    if draft_recorded:
        return (
            "The pipeline has recorded it as an unsigned draft in long-term "
            "memory: it is kept for audit, but it is not in this patient's "
            "history and the Scribe will not read it until you approve."
        )
    return (
        "Nothing is written to this patient's history until you approve - "
        "neither the run itself nor a Reject puts it there."
    )


def review_state(
    status: Optional[str] = None,
    *,
    needs_human_review: bool = False,
    approved: bool = False,
    draft_recorded: bool = False,
    memory_enabled: bool = True,
    error: Optional[str] = None,
    revision_count: int = 0,
    feedback_text: str = "",
) -> ReviewState:
    """Resolve the review state from what the app already knows.

    Args:
        status: `st.session_state.review_status` - None, "approved",
            "rejected" or "changes_requested".
        needs_human_review: The supervisor exhausted its retries, so the run
            reached no memory-save step and recorded no draft.
        approved: Whether this visit is in approved longitudinal history.
        draft_recorded: Whether the pipeline recorded this run as a draft row
            (unsigned, unread by the Scribe, not counted as history).
        memory_enabled: The sidebar "Remember visits" toggle.
        error: Text of a failed save, if the last Approve raised.
        revision_count: How many feedback-driven revisions preceded this draft.
        feedback_text: Feedback written into the revision box but not yet
            submitted.

    Returns:
        The `ReviewState` to render and to gate the decision buttons on.
    """
    # A failed write is not a decision: keep every action open so the physician
    # can retry rather than losing the visit.
    if error:
        return ReviewState(
            key="save_failed",
            label="Save failed - nothing was stored",
            level="error",
            detail=f"The last Approve did not write: {error}",
            next_step="Approve again to retry, or request changes instead.",
            in_history=bool(approved),
            draft_recorded=bool(draft_recorded),
            terminal=False,
            allow_approve=True,
            allow_request_changes=True,
            allow_reject=True,
            allow_revise=False,
        )

    if status == STATUS_APPROVED:
        if approved:
            written_by = (
                "The pipeline had recorded this visit as a draft; approving "
                "promoted that row rather than writing a second visit."
                if draft_recorded
                else "Stored just now, from this review, as approved history."
            )
            return ReviewState(
                key="approved_saved",
                label="Approved & saved to history",
                level="success",
                detail=(
                    "This note is final and is now approved history for this "
                    "patient: it counts toward their visit history and the "
                    "Scribe will read it on future visits. " + written_by
                ),
                next_step=(
                    "Revise it to produce a new draft, or run the pipeline again "
                    "for a new visit."
                ),
                in_history=True,
                draft_recorded=bool(draft_recorded),
                terminal=True,
                allow_approve=False,
                allow_request_changes=False,
                allow_reject=False,
                allow_revise=True,
            )
        reason = (
            "Long-term memory is off for this session, so approving did not "
            "persist anything."
            if not memory_enabled
            else "This visit was not written to long-term memory."
        )
        return ReviewState(
            key="approved_not_saved",
            label="Approved - not stored",
            level="warning",
            detail=(
                "You approved this note, but it is not in long-term memory. "
                + reason
            ),
            next_step=(
                "Turn on 'Remember visits between sessions' and approve again, "
                "or revise the note."
            ),
            in_history=False,
            draft_recorded=bool(draft_recorded),
            terminal=True,
            allow_approve=False,
            allow_request_changes=False,
            allow_reject=False,
            allow_revise=True,
        )

    if status == STATUS_REJECTED:
        # Rejecting never creates history. It also cannot delete the draft the
        # pipeline recorded - there is no un-store operation - so the banner
        # says the draft stays, unsigned, and that it is not history.
        detail = (
            "This note was rejected: it is not final and it is not in the "
            "patient's history. Approve is what puts a visit there, and "
            "rejecting it stored nothing as history."
        )
        if draft_recorded:
            detail += (
                " The pipeline's memory-save step had recorded this visit as an "
                "unsigned draft when the run finished - that draft stays in the "
                "store for audit, rejecting it does not delete it, and the "
                "Scribe does not read it as history."
            )
        return ReviewState(
            key="rejected",
            label="Rejected - not final",
            level="error",
            detail=detail,
            next_step=(
                "Revise it with feedback, or change the chief complaint and run "
                "the pipeline again."
            ),
            in_history=False,
            draft_recorded=bool(draft_recorded),
            terminal=True,
            allow_approve=False,
            allow_request_changes=False,
            allow_reject=False,
            allow_revise=True,
        )

    if status == STATUS_CHANGES_REQUESTED:
        waiting_for_text = not _short(feedback_text)
        return ReviewState(
            key="changes_requested",
            label="Changes requested",
            level="warning",
            detail=(
                "Decisions are closed on this draft. Write what should change "
                "below, then submit it - the feedback goes into the next "
                "pipeline run (Scribe + Coding) and produces a new draft."
                if waiting_for_text
                else "Your feedback is ready. Submit it to regenerate the note - "
                "it is passed to the next pipeline run (Scribe + Coding) as a "
                "revision instruction, not stored with this visit."
            ),
            next_step=(
                "Write the revision feedback and press Submit Feedback & Regenerate."
                if waiting_for_text
                else "Press Submit Feedback & Regenerate to start the revision."
            ),
            in_history=bool(approved),
            draft_recorded=bool(draft_recorded),
            terminal=True,
            allow_approve=False,
            allow_request_changes=False,
            allow_reject=False,
            allow_revise=False,
        )

    if needs_human_review:
        return ReviewState(
            key="escalated",
            label="Escalated for human review",
            level="error",
            detail=(
                "The pipeline ran out of automatic corrections, so it never "
                "reached its memory-save step and recorded nothing for this "
                "patient. The note below is a draft that needs a clinician's "
                "judgement before anything is stored."
            ),
            next_step=(
                "Read the quality flags, then Approve if the note is safe, or "
                "request changes."
            ),
            in_history=False,
            draft_recorded=False,
            terminal=False,
            allow_approve=True,
            allow_request_changes=True,
            allow_reject=True,
            allow_revise=False,
        )

    if revision_count:
        return ReviewState(
            key="awaiting_review",
            label=f"Revised draft (revision {revision_count})",
            level="warning",
            detail=(
                "This is a new draft written from your feedback, not the "
                "previous version. The earlier text is not edited in place and "
                "was not overwritten. " + _draft_note(draft_recorded)
            ),
            next_step="Review it, then Approve, request changes again, or reject.",
            in_history=bool(approved),
            draft_recorded=bool(draft_recorded),
            terminal=False,
            allow_approve=True,
            allow_request_changes=True,
            allow_reject=True,
            allow_revise=False,
        )

    return ReviewState(
        key="awaiting_review",
        label="Awaiting review - draft",
        level="info",
        detail=(
            "This is a draft written by the pipeline. It is not final. "
            + _draft_note(draft_recorded)
        ),
        next_step=(
            "Approve to make it final and add it to the patient's history, "
            "request changes, or reject it."
        ),
        in_history=bool(approved),
        draft_recorded=bool(draft_recorded),
        terminal=False,
        allow_approve=True,
        allow_request_changes=True,
        allow_reject=True,
        allow_revise=False,
    )


def allowed_actions(state: ReviewState) -> Tuple[str, ...]:
    """The `REVIEW_ACTIONS` that are enabled in this state, in reading order."""
    enabled = {
        "approve": state.allow_approve,
        "request_changes": state.allow_request_changes,
        "reject": state.allow_reject,
        "revise": state.allow_revise,
    }
    return tuple(action for action in REVIEW_ACTIONS if enabled[action])


def approval_preview(
    result: Optional[Dict[str, Any]],
    *,
    patient_name: str = "",
    patient_id: Any = "",
    chief_complaint: str = "",
    specialty_name: Optional[str] = None,
    memory_enabled: bool = True,
    draft_recorded: bool = False,
) -> ApprovalPreview:
    """Describe what clicking Approve would store, before it is clicked.

    Mirrors `PatientMemory.approve_visit`: the draft row the pipeline recorded
    for this run is promoted in place to approved, and only when there is no
    draft to promote is a new approved row written. The stored visit holds the
    chief complaint, the ICD-10 list, the CPT list, the Assessment and Plan
    (each capped at 400 characters) and the quality flags, on a patient record
    that keeps a rolling window of the 10 most recent visits. The SOAP Subjective
    and Objective sections are not part of the stored visit, and saying so is the
    point of this function.

    Args:
        result: The run result, read for `coding_result`, `quality_flags`.
        patient_name: Patient display name.
        patient_id: Patient id the visit would be recorded against.
        chief_complaint: Chief complaint for this visit.
        specialty_name: Specialty display name, when one is selected.
        memory_enabled: The sidebar "Remember visits" toggle.
        draft_recorded: Whether the pipeline recorded this run as a draft, so
            Approve promotes that row instead of writing a second visit.

    Returns:
        An `ApprovalPreview`. `will_persist` is False when long-term memory is
        off, so the caller can warn instead of claiming a write that will not
        happen.
    """
    result = result or {}
    coding = result.get("coding_result") or {}
    flags = list(result.get("quality_flags") or [])
    cpt = [str(code) for code in (coding.get("cpt_suggestions") or [])]
    codes = key_codes(coding)

    patient = f"{patient_name} ({patient_id})" if patient_name else str(patient_id)
    complaint = _short(chief_complaint, 60)

    parts: List[str] = []
    if patient and patient != "None":
        parts.append(patient)
    if complaint:
        parts.append(f'"{complaint}"')
    # Always stated, including "no codes": approving a note whose code list is
    # empty is exactly the case a physician most needs to be told about.
    parts.append(_code_summary(coding))
    if specialty_name:
        parts.append(specialty_name)
    where = " · ".join(parts)

    if not memory_enabled:
        summary = f"Approving stores nothing: long-term memory is off ({where})."
        warning = (
            "**Long-term memory is off.** Turning on 'Remember visits between "
            "sessions' is what makes an approval persist; without it the note "
            "stays in this browser session only."
        )
    elif draft_recorded:
        summary = (
            f"Approve & Save promotes this run's draft into approved history: "
            f"{where}. No second visit is written."
        )
        warning = ""
    else:
        summary = f"Approve & Save stores this visit as approved history: {where}."
        warning = ""

    detail: List[str] = []
    if patient:
        detail.append(
            f"**Patient record** - {patient}: the chart context for this "
            f"patient is refreshed, and this visit becomes approved history."
        )
    detail.append(
        f"**Chief complaint** - {'“' + complaint + '”' if complaint else 'none recorded'}"
    )
    detail.append(
        "**ICD-10** - "
        + (", ".join(f"`{code}`" for code in codes) if codes else "none")
    )
    detail.append(
        "**CPT** - " + (", ".join(f"`{code}`" for code in cpt) if cpt else "none")
    )
    detail.append(
        f"**Note** - Assessment and Plan are stored, each truncated to "
        f"{_SOAP_FIELD_LIMIT} characters. The Subjective and Objective sections "
        f"are not part of the stored visit; they stay in this session and in "
        f"the Raw JSON tab."
    )
    detail.append(
        f"**Quality flags** - {len(flags)} kept with the visit, unchanged."
    )
    if specialty_name:
        detail.append(
            f"**Specialty** - {specialty_name}. The specialty itself is not a "
            f"stored field; it shaped this note and is shown in Physician Review."
        )
    detail.append(
        f"**History window** - the store keeps the most recent "
        f"{_VISIT_HISTORY_LIMIT} visits per patient, dropping the oldest "
        f"unsigned draft first so a rejected draft can never evict an approved "
        f"visit."
    )
    if draft_recorded:
        detail.append(
            "**Draft promoted, not duplicated** - the pipeline already recorded "
            "this run as an unsigned draft; approving flips that row to "
            "approved instead of adding another visit."
        )
    detail.append(
        "**What approving is for** - approved visits are the only ones the "
        "Scribe reads as this patient's prior history, and the only ones "
        "counted in their visit count."
    )

    return ApprovalPreview(
        will_persist=bool(memory_enabled),
        already_saved=bool(draft_recorded),
        summary=summary,
        detail_lines=detail,
        warning=warning,
    )


def regeneration_progress_message(feedback: Optional[str]) -> str:
    """Progress line for a feedback-driven run, "" when there is no feedback.

    Lives here rather than inline in the app so the wording the physician reads
    while the pipeline is working is asserted by the tests like every other
    review string.
    """
    if not _short(feedback):
        return ""
    return (
        "🔄 **Regenerating with your feedback…** - the Scribe and Coding agents "
        "are re-running with your revision instructions."
    )


def revision_summary(
    feedback_history: Optional[Iterable[str]],
) -> List[str]:
    """One line per feedback-driven revision, newest first.

    A deliberately small substitute for a diff engine: it says which feedback
    produced the draft on screen, which is what the physician needs to know when
    looking at a revised note, and it never claims to describe what changed.
    """
    entries: List[str] = []
    for text in feedback_history or []:
        shortened = _short(text, 160)
        if shortened:
            entries.append(shortened)
    entries.reverse()
    return [
        f"Revision {len(entries) - index} - {text}"
        for index, text in enumerate(entries)
    ]
