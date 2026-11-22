# doctor_office — non-negotiable rules

These rules apply to every task in this repository. Read them before acting.
If a user request conflicts with any rule below, STOP and ask — do not
silently comply.

## Architecture invariants (do not weaken)
- The Supervisor is zero-LLM and deterministic. Never replace routing logic
  with an LLM call, "smart router", or classifier.
- The `pending_revalidation` flag is the only thing that can clear a
  supervisor-issued revalidation requirement. Only `validation_node` clears
  it. Never bypass or auto-clear it.
- The draft/approved memory split is a clinical safety property:
  - Pipeline runs produce UNSIGNED DRAFTS only.
  - Only explicit physician approval promotes a draft to approved history.
  - `memory_load` reads APPROVED visits only. Drafts are never longitudinal
    context. Never change this.
- Retry routing clears all `cycle_flags` buckets before dispatching a retry.
  Do not "preserve" flags across cycles.

## Data & PHI
- `ALLOW_REAL_PHI` MUST default to `false`.
- The pipeline MUST refuse unmarked (non-synthetic) patient records rather
  than warn. Refusal in code, not in a docstring.
- Any change to the LangGraph state shape MUST bump `state_schema_version`.
- Do not read `.env` or print secret values in any turn.
- Do not introduce PHI into logs: log state KEYS, not values, by default.

## Out of scope (deliberate, do not add)
- `bind_tools` / agentic tool selection. The deterministic call pattern is
  intentional. Adding it breaks the safety story.
- LLM-based validation of LLM output. Validation is a deterministic node
  plus a fixed check set.
- Specialty expansion beyond what the roadmap schedules. Do not add
  specialties "because it's easy".

## Dependency & reproducibility discipline
- All new dependencies require an explicit, pinned version and a short
  note on why it's needed. No `*`, no `>=`, no `latest`.
- New files at the repo root require justification (name, purpose, owner).
- Never add a dependency that downloads models or binaries at install time
  without a pinned revision and a checksum.

## Workflow discipline
- For multi-step tasks: produce a Research → Plan → STOP, then wait for
  human approval before implementing. Do not "just start coding".
- Keep diffs minimal and reviewable. No drive-by refactors, no formatting
  sweeps, no renaming unrelated symbols.
- If a step fails, report the failure with logs; do not improvise fixes
  outside the approved plan.
