# Doctor Office - Agentic AI Clinical Assistant

A prototype system for clinical documentation and coding assistance using LLMs.

## Model Setup

The LLM weights are **not** included in this repository (they are too large for Git).

1. Install dependencies:
   ```bash
   pip install huggingface_hub
   ```

2. Download the model:
   ```bash
   python scripts/download_model.py
   ```

The model will appear in `models/`. Update your config to point to the correct `.gguf` file.

## Project Structure

```
doctor_office/
├── app/              # Streamlit application
├── data/             # Data files (chroma DB, patient data)
├── docs/             # Documentation
├── models/           # LLM models (gitignored - download separately)
├── scripts/          # Utility scripts
├── src/              # Source code
│   ├── agents/       # LangGraph agents
│   ├── memory/       # Patient memory system
│   ├── rag/          # RAG components
│   ├── tools/        # Clinical tools
│   └── utils/        # Utilities
└── tests/            # Test suite
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model
python scripts/download_model.py

# Run app
streamlit run app/streamlit_app.py
```
