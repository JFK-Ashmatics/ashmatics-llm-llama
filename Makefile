# Makefile for llama-docker-wrapper

# Variables
IMAGE_NAME = llama-wrapper
CONTAINER_NAME = llama-wrapper-instance
# Default model path (can be overridden via command line: make run-local MODEL_PATH=models/other-model.gguf)
# For split models, point to the first file (e.g., ...-00001-of-00002.gguf)
MODEL_PATH ?= models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf

# Qwen 2.5 7B Instruct (Q5_K_M) - High quality, fits on M1
# Note: This model is split into multiple files on HuggingFace.
# We use huggingface-cli to download the split files and llama.cpp will handle them.
MODEL_REPO = Qwen/Qwen2.5-7B-Instruct-GGUF
MODEL_FILE_PATTERN = qwen2.5-7b-instruct-q5_k_m-*.gguf

# --- Local Development (Mac Silicon / Metal) ---

.PHONY: setup
setup:
	@echo "Installing Python dependencies with uv..."
	uv sync

.PHONY: download-model
download-model:
	@echo "Downloading Qwen 2.5 7B Instruct (Q5_K_M) using huggingface-cli..."
	@mkdir -p models
	# Download the split files to the models directory
	uv run huggingface-cli download $(MODEL_REPO) --include "$(MODEL_FILE_PATTERN)" --local-dir models --local-dir-use-symlinks False
	@echo "Download complete."

.PHONY: build-llama-local
build-llama-local:
	@echo "Building llama.cpp locally with Metal support..."
	if [ ! -d "llama.cpp" ]; then git clone https://github.com/ggerganov/llama.cpp.git; fi
	# Clean build directory to remove cached configs
	rm -rf llama.cpp/build
	# Disable BLAS (Accelerate) to avoid vecLib conflicts on newer macOS SDKs
	# Force C++17 standard
	# Explicitly disable Accelerate framework usage
	cd llama.cpp && cmake -B build \
		-DGGML_METAL=ON \
		-DGGML_BLAS=OFF \
		-DGGML_ACCELERATE=OFF \
		-DCMAKE_CXX_STANDARD=17 \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=ON \
		&& cmake --build build --config Release -j$$(sysctl -n hw.logicalcpu)

.PHONY: run-local
run-local: build-llama-local
	@echo "Starting local server with Metal support..."
	@echo "Using model: $(MODEL_PATH)"
	# Start llama-server in background
	# --mlock: Lock model in memory to prevent swapping (improves stability on Mac)
	# -cb: Continuous batching (improves throughput)
	# -c 2048: Context window size
	./llama.cpp/build/bin/llama-server -m $(MODEL_PATH) --port 11434 --n-gpu-layers 99 --mlock -cb -c 2048 & \
	PID_LLAMA=$$!; \
	trap "kill $$PID_LLAMA" EXIT; \
	sleep 2; \
	uv run uvicorn main:app --host 0.0.0.0 --port 9000

# --- Docker (Production / Linux) ---

.PHONY: docker-build
docker-build:
	@echo "Building Docker image (CPU default)..."
	docker build -t $(IMAGE_NAME) .

.PHONY: docker-build-cuda
docker-build-cuda:
	@echo "Building Docker image (CUDA enabled)..."
	# Note: Requires nvidia-docker runtime to run
	docker build --build-arg USE_CUDA=on -t $(IMAGE_NAME)-cuda .

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run --rm -it \
		-p 9000:9000 \
		-v $$(pwd)/models:/app/models \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)
