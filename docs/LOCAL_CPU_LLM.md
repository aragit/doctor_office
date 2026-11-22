# Local CPU LLM Setup for Doctor Office MVP

This document describes how to set up a CPU-optimized LLM backend using llama.cpp's `llama-server` to replace Ollama in the Doctor Office MVP project.

## Hardware Assumptions

- Intel Core i3-4005U (2C/4T @ 1.7 GHz, AVX2 only)
- 16 GB DDR3-1600 RAM
- Ubuntu 22.04
- ~7 GB free RAM under normal use

## Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j2
```

## Determine Optimal Thread Count

Run the benchmark to find the best thread count for your CPU (often 2 or 3 on this CPU):

```bash
./build/bin/llama-bench -m /path/to/MODEL.gguf -t 2,3,4 -p 512 -n 64
```

Choose the thread count (`-t`) that gives the highest tokens per second.

## Download a GGUF Model

We recommend using a quantized instruct model that matches the current presets:
- For fast mode: Qwen2.5-1.5B-Instruct (Q4_K_M or Q5_K_M)
- For quality mode: Gemma3-4B-Instruct (if available) or stick with Qwen2.5-1.5B-Instruct for simplicity.

Example download (Qwen2.5-1.5B-Instruct-Q4_K_M.gguf):

```bash
# Create a directory for models
mkdir -p ~/models
cd ~/models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## Run llama-server

Start the server with the following flags (adjust paths and thread count as needed):

```bash
./build/bin/llama-server \
  -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 4096 \
  -np 2 \
  -t 2 \   # Use the thread count determined from llama-bench
  --mlock \
  --cache-reuse 256 \
  --slot-save-path ./kv \
  --port 8080 \
  --host 127.0.0.1
```

### Flag Explanations:
- `-m`: Path to the GGUF model file.
- `-c`: Context size (set to 4096 to match typical usage).
- `-np 2`: Separate slots for two concurrent agents (Scribe and Coding) to reduce KV cache thrashing.
- `-t`: Number of CPU threads to use.
- `--mlock`: Lock model in RAM to prevent swapping.
- `--cache-reuse 256`: Cache size for reused slots (helps with static system prompts).
- `--slot-save-path ./kv`: Directory to save KV cache slots for reuse.
- `--port 8080`: Port for the OpenAI-compatible API.
- `--host 127.0.0.1`: Bind to localhost only for security.

## Python Dependency

The modified `src/utils/llm.py` uses `langchain_openai.ChatOpenAI`. Ensure it is installed:

```bash
pip install langchain-openai
```

## Environment Configuration

The Doctor Office MVP can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | URL of the llama-server OpenAI-compatible endpoint | `http://127.0.0.1:8080/v1` |
| `LLM_API_KEY` | API key for the LLM server (not needed for llama-server) | `EMPTY` |
| `LLM_MODEL_NAME` | Model name to send to the server (often ignored) | `local` |
| `LLM_FAST_MAX_TOKENS` | Max tokens for fast mode | `384` |
| `LLM_QUALITY_MAX_TOKENS` | Max tokens for quality mode | `384` |
| `RAG_TOP_K` | Number of retrieved documents for RAG context | `2` |

Example usage:
```bash
export LLM_BASE_URL="http://127.0.0.1:8080/v1"
export LLM_API_KEY="EMPTY"
export RAG_TOP_K="2"
```

## Connecting the Application

The modified `src/utils/llm.py` will now use a `ChatOpenAI` instance pointed at `http://127.0.0.1:8080/v1`. The model name passed to `ChatOpenAI` is arbitrary (the server ignores it) but we keep it for compatibility.

No other changes are needed in the agent code.

## Troubleshooting

- **Port already in use**: Stop any existing server or change the port in both the server command and the client configuration.
- **Out of memory (OOM)**: Reduce `-c` (context size) or ensure `--mlock` is used (if you have enough RAM). Close other applications.
- **Slow first token**: The first request after a cold start will be slower due to model loading. Subsequent requests benefit from the KV cache (`--cache-reuse` and `--slot-save-path`).
- **Server not responding**: Check that the server is running and listening on the specified port and host (`netstat -tlnp` or `ss -tlnp`).
- **Model not found**: Verify the path to the GGUF file is correct.
- **Import error**: If you see `ModuleNotFoundError: No module named 'langchain_openai'`, install the dependency as described above.
- **Configuration not taking effect**: Ensure environment variables are set before starting the application.

## Notes

- The server must be running before starting the Doctor Office MVP application.
- For development, you can run the server in a separate terminal tab.
- To stop the server, press `Ctrl+C` in the terminal where it's running.
- Default token limits are set conservatively for CPU efficiency (384 tokens) but can be increased via environment variables if needed.
- RAG context is limited to 2 documents by default (down from 3) to reduce CPU load, configurable via RAG_TOP_K.
