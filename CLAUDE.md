# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready Docker wrapper for llama.cpp with a FastAPI proxy layer. It provides an OpenAI-compatible API for local LLM inference with support for Apple Silicon (Metal) local development and containerized CPU/CUDA deployment.

## Architecture

```
┌─────────────────┐     Port 9000      ┌──────────────────┐     Port 11434     ┌──────────────────┐
│   Client/UI     │ ←───────────────→  │  FastAPI Wrapper │ ←───────────────→  │  llama-server    │
│   (Browser)     │   /v1/chat/...     │  (main.py)       │   /v1/chat/...     │  (llama.cpp)     │
└─────────────────┘                    └──────────────────┘                    └──────────────────┘
```

- **FastAPI wrapper** (`main.py`): Proxy layer exposing OpenAI-compatible endpoints on port 9000
- **llama-server**: The llama.cpp inference server on port 11434
- **start.sh**: Orchestrates startup of both services with health checking
- **llama.cpp/**: Vendored submodule (built at runtime for local dev)

## Commands

### Local Development (Apple Silicon)

```bash
# Install Python dependencies
make setup

# Download default model (Qwen 2.5 7B)
make download-model

# Build llama.cpp with Metal and start the full stack
make run-local

# Use a different model
make run-local MODEL_PATH=models/your-model.gguf
```

### Docker (Production)

```bash
# Build CPU image
make docker-build

# Build CUDA image
make docker-build-cuda

# Run container (mounts ./models)
make docker-run
```

### Testing

```bash
# Run MCP tool calling integration test
uv run python test_mcp_loop.py
```

## API Endpoints

All endpoints are on port 9000:
- `GET /`: Chat UI
- `GET /health`: Health check
- `POST /v1/chat/completions`: OpenAI-compatible chat (streaming supported)
- `POST /v1/embeddings`: OpenAI-compatible embeddings (for ashmatics-tools integration)
- `POST /generate`: Simple text generation with system prompt
- `POST /embed`: Legacy embeddings endpoint (use `/v1/embeddings` instead)
- `GET /v1/models`: List available models

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/model.gguf` | Path to GGUF model file |
| `LLAMA_PORT` | `11434` | llama-server internal port |
| `N_GPU_LAYERS` | `0` | GPU layers (99 for full Metal offload) |
| `CONTEXT` | `4096` | Context window size |
| `MODEL_ALIAS` | `kb-llm` | Model name in API responses |

## MCP Integration

The `test_mcp_loop.py` script demonstrates local LLM + MCP tool calling:
- Uses prompt engineering for tool calling (llama.cpp requires `--jinja` for native support)
- Connects to external MCP servers via stdio protocol
- Implements an agentic loop with JSON tool call parsing
