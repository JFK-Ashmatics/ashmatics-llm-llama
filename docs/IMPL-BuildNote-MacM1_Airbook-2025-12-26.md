# Build Note: Mac M1 MacBook Air - 2025-12-26

## Environment

- **Machine**: MacBook Air M1
- **OS**: macOS Darwin 25.1.0
- **Xcode Command Line Tools**: 26.2
- **Apple Clang**: 17.0.0 (clang-1700.6.3.2)

## Issue: C++ Standard Library Headers Not Found

When building llama.cpp with `make run-local`, the build failed with errors like:

```
fatal error: 'cstdio' file not found
fatal error: 'csignal' file not found
fatal error: 'array' file not found
fatal error: 'mutex' file not found
```

### Root Cause

Xcode Command Line Tools 26.x has a bug where C++ standard library headers are not installed in the location the compiler expects.

**Expected location** (where clang looks):
```
/Library/Developer/CommandLineTools/usr/include/c++/v1/
```

**Actual location** (where headers exist):
```
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/
```

The `/Library/Developer/CommandLineTools/usr/include/c++/v1/` directory only contains internal `__*` headers, missing standard headers like `<cstdio>`, `<array>`, `<mutex>`, etc.

### Solution

Add an explicit include path to CMake configuration via `CMAKE_CXX_FLAGS`:

```cmake
-DCMAKE_CXX_FLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
```

## Makefile Changes

### 1. C++ Include Path Fix

Updated `build-llama-local` target to include the workaround:

```makefile
cd llama.cpp && cmake -B build \
    -DGGML_METAL=ON \
    -DGGML_BLAS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --config Release -j$$(sysctl -n hw.logicalcpu)
```

### 2. Model Path Update

Changed default model path to match local file location:

```makefile
MODEL_PATH ?= llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

## Verification

After fixes, `make run-local` successfully:
1. Builds llama.cpp with Metal GPU support
2. Starts llama-server on port 11434
3. Starts FastAPI wrapper on port 9000

```
ggml_metal_device_init: GPU name:   Apple M1
ggml_metal_device_init: GPU family: MTLGPUFamilyApple7  (1007)
ggml_metal_device_init: has unified memory    = true
version: 7548 (7ac890213)
built with AppleClang 17.0.0.17000603 for Darwin arm64
```

## Notes

- This issue affects Xcode CLT 26.x on macOS 26 (Tahoe) beta
- The workaround is specific to this toolchain version and may not be needed on older or future versions
- Updating CLT via `softwareupdate` did not resolve the issue; the explicit include path is required
