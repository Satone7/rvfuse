# ONNXRT Cross-Compile Build Redesign

## Overview

Redesign `tools/docker-onnxrt/` to use LLVM-based cross-compilation instead of QEMU riscv64 emulation. Upgrade ONNX Runtime from v1.17.3 to v1.23.2. Support 2 targets with c920-specific extensions as an optional build mode.

## Targets

| `--target` value | Architecture | ONNXRT Build | YOLO Runner |
|---|---|---|---|
| `inference` (default) | rv64gcv | Full build (not minimal) | Standard |
| `c920-inference` | rv64gcv + XThead extensions | Full build + XThead flags | XThead extensions enabled |

> Note: `preprocess` and `postprocess` targets are out of scope for this branch. Their
> setup.sh infrastructure already works; only the ONNXRT-related inference build needs updating.

## Architecture

### Build Flow

```
┌─────────────────────────────────────────────────────┐
│  setup.sh --target <T>  (inference | c920-inference)│
│  calls tools/docker-onnxrt/build.sh                 │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Docker container (x86_64)      │
│  ubuntu:22.04 + cmake + ninja   │
│  + gcc-riscv64-linux-gnu (ld/as)│
└──────────────┬──────────────────┘
               │ Volume mounts:
               │  - LLVM install    → /llvm-install
               │  - ORT source      → /onnxruntime
               │  - Eigen source    → /eigen
               │  - Sysroot         → /sysroot
               │  - Toolchain file  → /toolchain.cmake
               ▼
┌─────────────────────────────────────────────────────┐
│  clang++ --target=riscv64-unknown-linux-gnu         │
│  --sysroot=/sysroot -march=rv64gcv                  │
│  (± XThead extensions for c920-inference)            │
└─────────────────────────────────────────────────────┘
```

## Key ONNXRT v1.23.2 Changes

1. **SOVERSION changed**: `libonnxruntime.so.17` → `libonnxruntime.so.1`. Update sysroot library references.
2. **`--use_preinstalled_eigen` removed**: Eigen is now always fetched/built. Mount Eigen source and use `FETCHCONTENT_SOURCE_DIR_EIGEN`.
3. **`--minimal_build` removed**: Full build only. Remove `--minimal_build` flag. Use `ORT_MINIMAL_BUILD=OFF` (default).
4. **`--skip_submodule_sync`**: Keep this — submodules not needed.
5. **CMake minimum**: 3.28 required.
6. **Python minimum**: 3.10 for building from source.

## C920 Extension Support (XThead)

LLVM 22 upstream supports 11 individual XThead extensions (MC layer only — no auto-codegen):

| Extension | Description |
|---|---|
| `xtheadba` | Address calculation |
| `xtheadbb` | Basic bit-manipulation |
| `xtheadbs` | Single-bit instructions |
| `xtheadcondmov` | Conditional move |
| `xtheadcmo` | Cache management |
| `xtheadfmidx` | FP indexed memory |
| `xtheadmac` | Multiply-accumulate |
| `xtheadmemidx` | Indexed memory L/S |
| `xtheadmempair` | Pair L/S (two GPRs) |
| `xtheadsync` | Multi-core sync |
| `xtheadvdot` | Vector dot-product (requires V) |

LLVM has **no bundled alias** like `-march=rv64gc_xtheadc920`. Each extension must be listed individually.

When `--target=c920-inference`:
- ONNXRT CMake: `-DCMAKE_C_FLAGS` / `-DCMAKE_CXX_FLAGS` with full `-march` string
- Full `-march`: `rv64gcv_xtheadba_xtheadbb_xtheadbs_xtheadcondmov_xtheadcmo_xtheadfmidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync_xtheadvdot`
- YOLO runner compiled with same `-march` flag
- No dedicated intrinsics header — extensions are only for MC layer (assembly/disassembly/encoding). C/C++ codegen will not emit XThead instructions automatically, but the flag ensures ABI compatibility and enables any future hand-written assembly or intrinsics.

## File Changes

| File | Action | Reason |
|---|---|---|
| `tools/docker-onnxrt/build.sh` | **Rewrite** | New cross-compile flow, 2-target support, ONNXRT v1.23.2 |
| `tools/docker-onnxrt/Dockerfile` | **Rewrite** | x86_64 host with cross-compiler, no `--platform riscv64` |
| `tools/docker-onnxrt/riscv64-linux-toolchain.cmake` | **Create** (from c920-onnxrt) | Shared toolchain file |
| `setup.sh` | **Modify** | Step 3: un-skip, call new build.sh, handle targets |

## Build Script Design (`tools/docker-onnxrt/build.sh`)

```bash
# Usage:
#   ./tools/docker-onnxrt/build.sh --target inference
#   ./tools/docker-onnxrt/build.sh --target c920-inference

TARGET="inference"
ONNXRUNTIME_VERSION="v1.23.2"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"

# 1. Clone dependencies (ONNXRT v1.23.2, Eigen 3.4.0)
# 2. Extract sysroot from riscv64/ubuntu:22.04
# 3. Build Docker image (x86_64, cross-compilation environment)
# 4. Run build: cmake + ninja ONNXRT + build yolo_runner
# 5. Extract artifacts: binary + sysroot
```

## Dockerfile Design

Single-stage build image on x86_64:
- `ubuntu:22.04` (x86_64)
- Install: cmake, ninja, python3, git, ca-certificates
- Install: `gcc-riscv64-linux-gnu` + `binutils-riscv64-linux-gnu` (for ld/as only)
- LLVM toolchain mounted at runtime via `-v`

No `--platform riscv64` — all compilation is native x86_64 cross-compilation.

## Sysroot Extraction

Extract from `riscv64/ubuntu:22.04` container, cache in `output/sysroot/`:
- `/usr/lib/riscv64-linux-gnu/`
- `/lib/riscv64-linux-gnu/` (dynamic linker)
- Symlinks for C runtime (crt1.o, crti.o, crtn.o)
- LD_LIBRARY_PATH setup for cross-linker
- Reuse c920-onnxrt's `extract_sysroot` logic as reference

## Artifacts

| Target | Binary | Sysroot | ORT Library |
|---|---|---|---|
| inference | `output/yolo_inference` | `output/sysroot/` | `libonnxruntime.so.1` |
| c920-inference | `output/yolo_inference` | `output/sysroot/` | `libonnxruntime.so.1` |

## setup.sh Step 3 Integration

- Remove the skip/TODO comment
- Call `tools/docker-onnxrt/build.sh --target ${TARGET}`
- Check target-specific artifacts:
  - `STEP3_ARTIFACTS_INFERENCE` (already exists from merged master)
  - Add `STEP3_ARTIFACTS_C920_INFERENCE` (same as inference — binary is `yolo_inference`)
  - Keep existing `STEP3_ARTIFACTS_PREPROCESS` / `STEP3_ARTIFACTS_POSTPROCESS` entries for compatibility but no build support added here

## Risks

1. **ONNXRT v1.23.2 full build (first attempt)**: Full build has many more dependencies than minimal build (protobuf, flatbuffers, protobuf-compiler, etc.). Build failures expected on first attempt — will need iterative debugging of CMake errors, missing headers, and linker issues.
2. **XThead codegen limitations**: LLVM's XThead extensions are MC-layer only. C/C++ compilation will not emit XThead instructions. The `-march` flag with XThead extensions primarily ensures ABI compatibility and encoding support, not automatic codegen benefits.
3. **CMake version**: Full ONNXRT build requires CMake 3.28+. Ubuntu 22.04 default is 3.22. Need to install CMake from Kitware PPA (already done in c920-onnxrt Dockerfile).
