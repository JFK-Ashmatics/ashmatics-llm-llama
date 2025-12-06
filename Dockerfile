# syntax=docker/dockerfile:1

# ARG for CUDA support (default off for CPU/Mac compatibility)
# To build for CUDA: docker build --build-arg USE_CUDA=on ...
ARG USE_CUDA=off

# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.11-slim-bookworm AS builder

ARG USE_CUDA

# Install system dependencies for building llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /build

# --- Build llama.cpp ---
# Clone specific version for stability (optional, but recommended for hardening)
# Here we clone latest, but in prod you might want a specific tag.
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /build/llama.cpp

# Build logic based on CUDA arg
RUN if [ "$USE_CUDA" = "on" ]; then \
        echo "Building with CUDA support..."; \
        # Note: For actual CUDA builds, you'd typically need a CUDA base image (e.g., nvidia/cuda). \
        # This logic is a placeholder. For production CUDA, switch the FROM to an nvidia image. \
        mkdir build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . --config Release -j$(nproc); \
    else \
        echo "Building for CPU..."; \
        mkdir build && cd build && cmake .. && cmake --build . --config Release -j$(nproc); \
    fi

# --- Python Environment ---
WORKDIR /app
COPY pyproject.toml .
# Create virtual environment and install dependencies
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
# Generate requirements.txt from pyproject.toml and install
RUN uv venv && \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt


# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.11-slim-bookworm AS runtime

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy llama-server binary
COPY --from=builder /build/llama.cpp/build/bin/llama-server /app/bin/llama-server

# Copy application code
COPY main.py .
COPY start.sh .

# Fix permissions
RUN chown -R appuser:appuser /app && chmod +x /app/start.sh

# Environment variables
ENV MODEL_PATH=/app/models/model.gguf
ENV LLAMA_PORT=11434
ENV CONTEXT=4096
ENV N_GPU_LAYERS=0
ENV THREADS=0
ENV MODEL_ALIAS=kb-llm
ENV LLAMA_BIN_PATH=/app/bin/llama-server

# Expose ports
EXPOSE 9000

# Switch to non-root user
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9000/health || exit 1

CMD ["/app/start.sh"]

