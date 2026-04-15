# MLAS Patches

This directory contains patches for ONNX Runtime's MLAS (Math Library for AI Systems).

## Available Patches

| Patch | Description | Target |
|-------|-------------|--------|
| `rvv-gemm.patch` | RVV (RISC-V Vector) GEMM kernel for single-precision matrix multiplication | ONNX Runtime MLAS |

## rvv-gemm.patch

Adds RISC-V Vector Extension (RVV) support to MLAS SGEMM kernel.

**Features:**
- BB-level vectorization replacing 4 scalar `fmadd.s` with 1 `vfmacc.vf` (VL=4)
- Matches existing MLAS B packing width of 4 columns
- ~2.4x instruction reduction in inner K loop (24 → 10 instructions per iteration)

**Modified files:**
- `cmake/onnxruntime_mlas.cmake` - RISC-V 64-bit platform detection
- `onnxruntime/core/mlas/inc/mlas.h` - MLAS_TARGET_RISCV macro
- `onnxruntime/core/mlas/lib/mlasi.h` - Platform kernel declarations
- `onnxruntime/core/mlas/lib/platform.cpp` - Kernel registration
- `onnxruntime/core/mlas/lib/q4gemm.h` - Quantized GEMM support
- `onnxruntime/core/mlas/lib/qnbitgemm.cpp` - N-bit quantized GEMM support
- `onnxruntime/core/mlas/lib/riscv64/SgemmKernelRvv.cpp` - RVV kernel implementation (new)
- `onnxruntime/core/mlas/lib/sgemm.cpp` - Kernel dispatch

**Requirements:**
- RISC-V 64-bit target with V extension (`rv64gcv`)
- GCC/Clang with RVV intrinsics support (`<riscv_vector.h>`)

**Usage:**
Apply to ONNX Runtime source tree before building:
```bash
cd applications/yolo/ort/vendor/onnxruntime
patch -p1 < ../../../patches/mlas/rvv-gemm.patch
```