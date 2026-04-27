# ONNX Runtime on RISC-V (rv64gcv)

Cross-compiled ONNX Runtime for RISC-V with RVV vector extension support, YOLO inference runner, and INT8 quantized model support.

## Directory Structure

```
applications/onnxrt/
├── ort/                    # ONNX Runtime build system
│   ├── build.sh            # Cross-compile ORT for RISC-V (rv64gcv)
│   └── patches/            # Patches applied during build
│       └── riscv-int8-qgemm.patch   # Fix INT8 QGEMM dispatch on RISC-V
├── yolo/                   # YOLO inference runner
│   ├── build.sh            # Cross-compile YOLO runner for RISC-V
│   ├── runner/             # C++ inference source (yolo_runner.cpp)
│   └── quantize_int8.py    # Export + quantize YOLO11n to INT8 ONNX
├── rvv-patches/            # RVV-vectorized MLAS kernel patches (rvv-patches convention)
│   ├── sgemm-kernel-vl16/  # RVV512 SGEMM kernel (VL=16)
│   ├── qgemm-kernel-vl16/  # RVV512 QGEMM kernel (VL=16)
│   ├── quantize-linear/    # RVV512 QuantizeLinear
│   ├── quick-gelu/         # RVV512 GELU activation
│   ├── eltwise-mul/        # RVV512 element-wise multiply
│   ├── reduce-minmax-f32/  # RVV512 float min/max reduction
│   └── compute-logistic/   # RVV512 logistic (sigmoid) activation
├── resnet/                 # ResNet-50 test runner
└── skills/                 # Claude Code skills for ORT development
```

## Build

```bash
# 1. Cross-compile ONNX Runtime (applies patches under ort/patches/)
bash applications/onnxrt/ort/build.sh --force -j$(nproc)

# 2. Cross-compile YOLO runner
bash applications/onnxrt/yolo/build.sh --force

# 3. Run inference under QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=512 \
  -L output/cross-ort/sysroot \
  -E LD_LIBRARY_PATH=/usr/lib/riscv64-linux-gnu \
  output/cross-ort/yolo_inference output/yolo11n_int8.onnx output/test.jpg
```

## INT8 Inference Support

Vanilla ONNX Runtime (v1.24.4) crashes with SIGSEGV on RISC-V when running INT8 quantized models. Root cause: `MlasGemmQuantGetDispatch()` in `qgemm.h` had no RISC-V dispatch branch, causing ORT to fall back to Eigen's GEMM path which triggers a use-after-free (mmap/munmap of workspace buffer before computation).

The fix (`ort/patches/riscv-int8-qgemm.patch`):

- Defines `MLAS_TARGET_RISCV` preprocessor macro for `__riscv && __riscv_v`
- Registers scalar default QGEMM dispatch (`MlasGemmQuantDispatchDefault`) for all INT8 signedness combinations (U8U8, S8U8, U8S8, S8S8)
- Wires RISC-V into the `GemmFloatKernel` dispatch path (platform.cpp, sgemm.cpp, q4gemm.h)
- Wraps scalar SGEMM kernels in `extern "C"` for correct linkage
- Forces serial thread execution to avoid `thread_local` buffer issues (`ThreadedBufHolder`)

The QGEMM kernel uses the platform-independent scalar default implementation — correct for all VLEN configurations (128/256/512) but not yet vectorized. RVV512 QGEMM kernel patches are under development in `rvv-patches/qgemm-kernel-vl16/`.

## Test Results

| Scenario | Status |
|----------|--------|
| `yolo_inference` + yolo11n.onnx (FP32) | Pass |
| `yolo_inference` + yolo11n_int8.onnx (INT8) | Pass (checksum deviation <0.5%) |

## Relevant Patches

| Patch | Purpose |
|-------|---------|
| `ort/patches/riscv-int8-qgemm.patch` | Fix INT8 QGEMM dispatch (ONNX Runtime v1.24.4) |
| `rvv-patches/sgemm-kernel-vl16/` | RVV512 SGEMM kernel (VL=16, LMUL=1) |
