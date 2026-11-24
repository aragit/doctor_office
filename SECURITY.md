# Security Policy

## Prototype Disclaimer

**This is a prototype system for research and development purposes only.** It is not intended for production clinical use, has not undergone formal security audit, and must not be used with real patient data in any clinical setting.

## Data Handling

### Synthetic Data Only
- The system enforces a synthetic-data guard (`ALLOW_REAL_PHI=false` by default).
- Only pre-registered synthetic patient IDs (e.g., `P001`, `P002`) are accepted.
- Any attempt to process unmarked (non-synthetic) patient records is refused at the pipeline entry point.
- Real PHI must never be entered into this system.

### Data Flow
1. Patient context (demographics, conditions, medications) is provided as input.
2. Data flows through the LangGraph pipeline: memory load → scribe → cleanup → coding → validation → supervisor → memory save.
3. Long-term memory stores **draft** visits only. Drafts are never used as longitudinal context until a physician explicitly approves them.
4. No data leaves the local environment unless explicitly configured (e.g., RAG embeddings to local ChromaDB).

## No Authentication

- This prototype has no authentication, authorization, or access control.
- It is designed to run on a **physician's local workstation** with physical access control.
- Do not deploy to shared servers, cloud environments, or any network-accessible location without adding proper authentication.

## Deployment Target

**Physician Workstation Only**
- Single-user, local execution
- No network services exposed
- Models served via local Ollama (http://127.0.0.1:11434)
- Data stored in local filesystem (ChromaDB, JSONL memory files)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible API endpoint for LLM |
| `LLM_MODEL_NAME` | `qwen2.5:1.5b` | Model identifier for Ollama |
| `LLM_FAST_MAX_TOKENS` | `384` | Max tokens for fast mode |
| `LLM_QUALITY_MAX_TOKENS` | `384` | Max tokens for quality mode |
| `RAG_TOP_K` | `2` | Number of RAG documents to retrieve |
| `ALLOW_REAL_PHI` | `false` | **Never set to `true` outside isolated dev with synthetic data** |

## Reporting Security Issues

This is a prototype. Security issues should be reported to the repository maintainers. Do not use this system for real clinical data.