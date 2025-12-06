# Llama.cpp Docker Wrapper

Created 2025-11-25

v0.2.0 By JFK

This project provides a hardened, production-ready Docker setup for `llama.cpp` with a FastAPI wrapper, along with tools for local development on Apple Silicon.

## Features

- **Production Ready**: Multi-stage Docker build, non-root user, health checks.
- **Fast Dependency Management**: Uses `uv` for Python package management.
- **Apple Silicon Support**: `Makefile` includes commands to build and run locally with Metal (GPU) support.
- **Flexible Build**: Dockerfile supports building for CPU (default) or CUDA (via build arg).

## Prerequisites

- Docker
- Python 3.11+
- `uv` (optional, for local dev)
- `make` (optional, for convenience)

## Local Development (Mac with Apple Silicon)

Running Docker on Mac usually incurs a performance penalty for inference because GPU passthrough (Metal) is not supported in standard Linux containers. For the best performance during development, run locally:

1.  **Setup**:
    ```bash
    make setup
    ```

2.  **Run**:
    ```bash
    # Builds llama.cpp with Metal support and starts the server + app
    make run-local
    ```
    *Note: Ensure you have a GGUF model in `./models/model.gguf`.*

## Docker Usage (Production / Azure / K8s)

### Build for CPU (Default)
Suitable for testing or environments without dedicated GPUs.
```bash
make docker-build
# or
docker build -t llama-wrapper .
```

### Build for CUDA (NVIDIA GPUs)
For Azure Container Apps (with GPU), AKS, or other K8s clusters with NVIDIA nodes.
```bash
make docker-build-cuda
# or
docker build --build-arg USE_CUDA=on -t llama-wrapper-cuda .
```
*Note: For the CUDA build to actually work at runtime, you need to ensure the base image in the Dockerfile is switched to an NVIDIA CUDA image (e.g., `nvidia/cuda:12.1.0-devel-ubuntu22.04`) or ensure the necessary libraries are present. The provided Dockerfile contains the logic to switch build flags, but you may need to adjust the `FROM` instruction for full CUDA support.*

### Run
```bash
make docker-run
```

## API

The FastAPI wrapper is available at `http://localhost:9000`.
- `GET /health`: Health check.
- `POST /generate`: Generate text (proxies to llama.cpp).
- `POST /v1/chat/completions`: OpenAI-compatible chat endpoint.
- `POST /v1/embeddings`: OpenAI-compatible embeddings endpoint.
- `GET /v1/models`: List available models.

**Note**: Embeddings are enabled by default via the `--embeddings` flag passed to llama-server.

---

## MCP Tool Calling Integration

This project includes a test script demonstrating **local LLM + MCP tool calling** - a fully local agentic loop with no cloud dependencies.

### Architecture

```
┌─────────────────┐     HTTP POST      ┌──────────────────┐
│  test_mcp_loop  │ ←───────────────→  │  llama.cpp       │
│  (Python)       │   localhost:9000   │  (Metal GPU)     │
└────────┬────────┘                    └──────────────────┘
         │
         │ stdio (JSON-RPC)
         ↓
┌─────────────────┐     HTTPS          ┌──────────────────┐
│  OpenFDA MCP    │ ←───────────────→  │  api.fda.gov     │
│  Server         │                    │  (FDA Open Data) │
└─────────────────┘                    └──────────────────┘
```

### Run the MCP Test

```bash
uv run python test_mcp_loop.py
```

This will:
1. Connect to the OpenFDA MCP server (from `ashmatics-tools` repo)
2. Send a query to the local LLM
3. LLM decides to call FDA tools, outputs JSON
4. Script parses JSON, calls MCP server
5. Results returned to LLM for summarization

### Example Output

```
Connecting to MCP Server at: /path/to/ashmatics-tools
Found 3 tools: ['search_devices', 'search_drugs', 'count_by_field']

User: Find 510(k) clearances for product code QIH from 2020 onwards.

--- Iteration 1 ---
LLM Response: {"tool": "search_devices", "args": {"query": "product_code:QIH AND decision_date:[20200101 TO *]"...
Detected tool call: search_devices(...)
Tool returned 3 results

--- Iteration 2 ---
Final Answer: The search returned 3 510(k) clearances...

============================================================
Chat loop completed successfully!
============================================================
```

### Key Design Decisions

1. **Prompt-based tool calling**: llama.cpp requires `--jinja` flag for native tool support. We use prompt engineering instead - LLM outputs JSON that we parse.

2. **Direct httpx calls**: No `openai` SDK - just raw HTTP to make it clear we're calling LOCAL infrastructure.

3. **MCP stdio protocol**: Server communicates via stdin/stdout JSON-RPC. stderr must be redirected to avoid corrupting the message stream.

See `ENGR-note-mcp-llm-integration-2025-11-25.md` for detailed lessons learned.
