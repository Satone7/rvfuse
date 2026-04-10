# Design: C920 Cross-Compiled ONNX Runtime

**Date**: 2026-04-09
**Status**: Approved
**Phase**: 2 — Fusion candidate discovery and design

## Problem

The project currently builds ONNX Runtime natively inside a `riscv64/ubuntu` Docker container using QEMU emulation (tools/rv64gcv-onnxrt/), which takes 2-6 hours due to emulation overhead. A cross-compiled build using the pre-built LLVM 13 toolchain can dramatically reduce build time while producing identical RISC-V binaries.

Additionally, the current build uses `--minimal_build` which excludes many ORT features. Phase 2 fusion candidate discovery may benefit from a full ONNX Runtime build for accurate cycle-level profiling.

## Requirements

1. Cross-compile ONNX Runtime v1.17.3 using LLVM 13 (`third_party/llvm-install/`)
2. Full build (no `--minimal_build`) with shared library
3. Output binaries must run under QEMU user-mode (`qemu-riscv64 -L sysroot`)
4. Sysroot extracted from `riscv64/ubuntu` Docker image (with extra dependencies pre-installed)
5. Single build script as the entry point (`build.sh`)
6. Output to `output/c920-ort/`

## Design

### Directory Structure

```
tools/c920-onnxrt/
├── build.sh          # Main build script (single entry point)
└── Dockerfile        # Cross-compilation environment
```

Output artifacts:
```
output/c920-ort/
├── onnxruntime/      # Compiled lib/ include/ bin/
├── yolo_inference    # YOLO inference binary linked against libonnxruntime.so
└── sysroot/          # Extracted riscv64 sysroot (for QEMU -L)
```

### Approach: Docker-Based Cross-Compilation

Use an x86_64 Ubuntu 22.04 Docker container with the pre-built LLVM 13 toolchain mounted in. The container installs all host-side build dependencies via apt, then runs cmake + ninja to cross-compile ONNX Runtime for riscv64-unknown-linux-gnu.

This approach was chosen because:
- ONNX Runtime full build requires many dependencies (protobuf, flatbuffers, abseil, etc.)
- Container-based dependency management keeps the host clean
- Consistent with the existing rv64gcv-onnxrt pattern in the project

### Dockerfile

Base: `ubuntu:22.04`

Installed packages:
- Build tools: cmake, ninja-build, python3, python3-pip, git
- ONNX Runtime dependencies: libprotobuf-dev, protobuf-compiler, libflatbuffers-dev (or equivalent packages available in 22.04)

Mounted volumes:
- `third_party/llvm-install/` → toolchain
- `tools/rv64gcv-onnxrt/vendor/onnxruntime/` → ORT source
- `tools/rv64gcv-onnxrt/vendor/eigen/` → Eigen (FETCHCONTENT_SOURCE_DIR)
- `tools/yolo_runner/` → YOLO runner source

### Build Script Flow (`build.sh`)

```
Step 0: Prerequisites check
  - Docker available
  - third_party/llvm-install/ exists
  - tools/rv64gcv-onnxrt/vendor/onnxruntime/ exists

Step 1: Extract riscv64 sysroot
  - docker run riscv64/ubuntu:22.04 to install extra deps
  - docker create + docker cp to extract /lib, /usr/lib, /usr/include
  - Save to output/c920-ort/sysroot/

Step 2: Build cross-compilation Docker image

Step 3: Run cross-compilation in container
  - cmake configure (see key flags below)
  - ninja build

Step 4: Build yolo_runner
  - Link against cross-compiled libonnxruntime.so

Step 5: Copy artifacts to output/c920-ort/
```

### CMake Key Flags

```cmake
-DCMAKE_SYSTEM_NAME=Linux
-DCMAKE_SYSTEM_PROCESSOR=riscv64
-DCMAKE_C_COMPILER=<llvm-install>/bin/clang-13
-DCMAKE_CXX_COMPILER=<llvm-install>/bin/clang++-13
-DCMAKE_C_COMPILER_TARGET=riscv64-unknown-linux-gnu
-DCMAKE_CXX_COMPILER_TARGET=riscv64-unknown-linux-gnu
-DCMAKE_FIND_ROOT_PATH=/sysroot
-DCMAKE_BUILD_TYPE=Release

# ONNX Runtime flags
--build_shared_lib
--disable_rtti
--cmake_extra_defines=onnxruntime_BUILD_UNIT_TESTS=OFF
--cmake_extra_defines=FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen
# NO --minimal_build (full build)
# NO --minimal_build_src_map
```

### Sysroot Strategy

1. Start a `riscv64/ubuntu:22.04` container and install dependencies:
   ```bash
   docker run --name sysroot-prep riscv64/ubuntu:22.04
   docker exec sysroot-prep apt-get update
   docker exec sysroot-prep apt-get install -y \
     libprotobuf-dev libabsl-dev libgoogle-glog-dev ...
   ```

2. Extract the root filesystem:
   ```bash
   docker create --name sysroot-extract riscv64/ubuntu:22.04
   docker export sysroot-extract | tar -xf - -C output/c920-ort/sysroot/
   ```

3. Filter to keep only `/lib`, `/usr/lib`, `/usr/include`, `/etc/alternatives` and symlinks needed for cross-compilation.

### Runtime Library Strategy

LLVM 13 has no pre-built RISC-V compiler-rt/libunwind/libcxxabi. Instead:
- Rely on sysroot's glibc for libc/runtime support
- Use `-libc++` from the sysroot (if available) or the host GCC's libstdc++
- If missing runtime libs cause issues, fall back to selectively building only what's needed from compiler-rt

### Key Differences from Existing rv64gcv-onnxrt Build

| Aspect | rv64gcv-onnxrt (native) | c920-onnxrt (cross) |
|--------|----------------------|---------------------|
| Compilation | Native (QEMU emulation) | Cross (LLVM 13) |
| Build type | Minimal | Full |
| Build speed | 2-6 hours | ~30-60 min (estimated) |
| Dockerfile base | riscv64/ubuntu | ubuntu:22.04 (x86_64) |
| Toolchain | GCC (apt) | clang-13 (llvm-install) |
| Output | output/ | output/c920-ort/ |

## Non-Goals

- Static linking (dynamic linking with libonnxruntime.so is acceptable)
- Rebuilding LLVM toolchain (reuse existing LLVM 13)
- Building ONNX Runtime tests (unit tests can't run in cross-compilation)
- Packaging for distribution
