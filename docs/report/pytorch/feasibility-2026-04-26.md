# PyTorch Cross-Compilation for rv64gcv: Feasibility Report

**Date**: 2026-04-26
**Researcher**: pytorch-scout
**Context**: RVFuse project Phase 2 — evaluating whether PyTorch can be cross-compiled for RISC-V vector extension research.

---

## TL;DR: Feasibility Verdict

**⏸️ Partial — Not Recommended for Direct Investment**

PyTorch RISC-V support exists at an early experimental stage. While cross-compilation is theoretically possible, the effort required far exceeds the research value for RVFuse's goals. The recommended path is **ONNX export + onnxrt analysis** (already implemented in RVFuse).

---

## 1. Current PyTorch RISC-V Support Status

### 1.1 Official Status

PyTorch has an RFC issue ([#95744](https://github.com/pytorch/pytorch/issues/95744)) for RISC-V CPU support. As of December 2024:

| Component | Status | Notes |
|-----------|--------|-------|
| Build system | ✅ Experimental | Cross-compilation possible with custom toolchain |
| XNNPACK RVV | ✅ Partial | Some ops vectorized via XNNPACK RVV backend |
| ATen native kernels | ⏸️ WIP | Community effort (avinal/pytorch-riscv-wiki) |
| CPU dispatcher | ❌ Not complete | No RISC-V-specific kernel dispatch path |
| Quantization | ❌ Unknown | No RISC-V-specific quantized kernels |

**Key Finding**: RVV support exists in XNNPACK (used by PyTorch for some operators), but this is insufficient for full model inference. The ATen tensor library's native CPU kernels require architecture-specific implementations that are not yet available for RISC-V.

### 1.2 Community Efforts

- **avinal/pytorch-riscv-wiki**: Community wiki documenting RISC-V porting progress
- **hilbert.github.io/PyTorch-RISC-V**: Technical blog documenting ATen kernel porting efforts
- PyTorch mainline: RISC-V support tracked in RFC issue, not yet merged

### 1.3 Missing Components

1. **ATen Native CPU Kernels**: PyTorch's core tensor operations library lacks RISC-V-specific implementations. Existing SIMD backends cover:
   - x86: AVX/AVX2/AVX512
   - ARM: NEON/SVE
   - Power: VSX
   - s390x: Z-Vector

   RISC-V RVV is **not** in this list.

2. **CPU Dispatcher**: PyTorch uses a CPU capability dispatcher to select optimal kernels based on detected hardware features. RISC-V detection logic is not implemented.

3. **Quantized Operators**: INT8 kernels for quantized inference lack RISC-V implementations.

---

## 2. Build Requirements and Challenges

### 2.1 Toolchain Requirements

PyTorch build-from-source requires:

| Requirement | Minimum | RVFuse Status |
|-------------|---------|---------------|
| Python | 3.8+ | ✅ Available |
| CMake | 3.25+ | ✅ Available |
| LLVM/Clang | 17+ | ✅ LLVM 22.1.3 (excellent) |
| Ninja | 1.10+ | ✅ Available |
| CUDA | Optional | ❌ Not needed for CPU |
| Protobuf | Required | ⚠️ Needs cross-compilation |

**LLVM 22 Assessment**: RVFuse's LLVM 22.1.3 with RISC-V target (`riscv64-unknown-linux-gnu`) is more than sufficient for PyTorch's requirements.

### 2.2 Cross-Compilation Challenges

| Challenge | Severity | Description |
|-----------|----------|-------------|
| Protobuf cross-build | High | PyTorch depends on protobuf; cross-compiling for RISC-V requires custom sysroot setup |
| ATen kernel dispatch | Critical | No RISC-V path in CPU dispatcher — code compiles but falls back to slow scalar kernels |
| Python cross-build | High | PyTorch requires Python bindings; cross-compiling Python for RISC-V is non-trivial |
| Missing SIMD kernels | Critical | Core operators (conv, matmul, activation) lack RVV implementations |
| Build complexity | Medium | PyTorch's monorepo build system is complex (setup.py + CMake hybrid) |

### 2.3 Estimated Effort

Based on community efforts documented in avinal/pytorch-riscv-wiki:

- **ATen kernel implementations**: 50-100+ operator files requiring RVV vectorization
- **CPU dispatcher integration**: 1-2 weeks of development
- **Build system adaptation**: 2-4 weeks
- **Testing and validation**: Ongoing

**Total estimated effort**: 3-6 months of dedicated development work for functional RISC-V PyTorch.

---

## 3. RVFuse Toolchain Assessment

### 3.1 LLVM 22.1.3

RVFuse has a locally-built LLVM 22.1.3 toolchain with:

```text
clang version 22.1.3
Target: riscv64-unknown-linux-gnu
Thread model: posix
```

This is excellent for PyTorch requirements (minimum LLVM 17). However, LLVM alone is insufficient — the missing piece is PyTorch's architecture-specific kernel implementations.

### 3.2 ONNX Runtime Cross-Compilation

RVFuse has successfully cross-compiled ONNX Runtime v1.24.4 for rv64gcv with:

- Custom CMake toolchain file
- Sysroot from Debian RISC-V ports
- LLVM 22 cross-compiler
- RVV-patched MLAS kernels (SGEMM, QGEMM, Logistic, QuickGelu, ReduceMinMax, QuantizeLinear)

**Key Insight**: ONNX Runtime was chosen because it has a cleaner cross-compilation path than PyTorch, and MLAS (Microsoft Linear Algebra Subprograms) is modular enough to accept RVV patches.

---

## 4. Fallback Path: ONNX Export + onnxrt Analysis

### 4.1 Recommended Pipeline

```
PyTorch Model → torch.export/onnx.export → ONNX → ONNX Runtime (RVV-patched) → BBV Profiling
```

This is the **recommended path** for RVFuse because:

1. **Model Export**: Most PyTorch models can be exported to ONNX via `torch.onnx.export()`
2. **Quantization**: PyTorch supports quantization-aware training (QAT) with ONNX export
3. **RVV Patching**: RVFuse already has RVV-patched ONNX Runtime operators
4. **Profiling Pipeline**: QEMU-BBV profiling is already set up for ONNX Runtime binaries

### 4.2 Models Supported for ONNX Export

| Model Type | Export Support | Notes |
|------------|----------------|-------|
| CNN (ResNet, YOLO) | ✅ Full | Standard export path |
| Transformers | ✅ Partial | torch.export() for decoder-only models |
| LLMs | ⚠️ Limited | Requires torch.compile() + export; complex |
| Custom models | ⚠️ Depends | Custom ops may need registration |

### 4.3 RVFuse ONNX Runtime Coverage

RVFuse has RVV-patched operators covering:

- **SGEMM**: VL=16 kernel for VLEN=512
- **QGEMM**: INT8 quantized GEMM
- **Logistic**: Sigmoid activation
- **QuickGelu**: Fast GeLU approximation
- **ReduceMinMax**: Min/max reduction
- **QuantizeLinear**: INT8 quantization

This coverage is sufficient for CNN models (YOLO, ResNet) and most transformer inference.

---

## 5. Recommendation

### 5.1 For RVFuse Phase 2-3

**Do NOT invest in PyTorch cross-compilation.** The effort exceeds research value.

**Invest in ONNX export pipeline:**

1. Maintain PyTorch models on x86_64 host
2. Export to ONNX with quantization
3. Cross-compile ONNX Runtime with RVV patches
4. Profile via QEMU-BBV + analyze_bbv.py
5. Perform gap analysis on RVV vs other vector ISAs

### 5.2 When PyTorch RISC-V Might Become Viable

PyTorch RISC-V becomes viable when:

1. ATen native CPU kernels have RISC-V-specific implementations (tracked in RFC #95744)
2. CPU dispatcher supports RISC-V capability detection
3. Quantized kernels support RVV vectorization

**Estimated timeline**: 1-2 years based on community progress rate.

### 5.3 Future Monitoring

Monitor these signals for PyTorch RISC-V readiness:

- GitHub issue #95744 status changes
- avinal/pytorch-riscv-wiki updates
- PyTorch release notes mentioning RISC-V
- XNNPACK RVV coverage expansion

---

## 6. Sources and References

### Primary Sources

| Source | URL | Relevance |
|--------|-----|-----------|
| PyTorch RISC-V RFC | https://github.com/pytorch/pytorch/issues/95744 | Official tracking issue |
| PyTorch RISC-V Wiki | https://github.com/avinal/pytorch-riscv-wiki | Community porting effort |
| PyTorch RISC-V Blog | https://hilbert.github.io/PyTorch-RISC-V/ | Technical documentation |
| XNNPACK RVV Support | https://github.com/google/XNNPACK | Backend ops vectorization |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | Cross-compile alternative |

### RVFuse Internal References

| Reference | Path | Description |
|-----------|------|-------------|
| LLVM Toolchain | `third_party/llvm-install/bin/clang-22` | LLVM 22.1.3 for RISC-V |
| ONNX Runtime Build | `applications/onnxrt/ort/build.sh` | Cross-compile script |
| RVV Patches | `applications/onnxrt/rvv-patches/` | MLAS RVV kernels |
| Architecture Doc | `docs/architecture.md` | Project architecture |

---

## 7. Appendix: PyTorch Build Requirements (Reference)

From PyTorch documentation (as of 2025):

```bash
# PyTorch build dependencies (x86_64 reference)
python3 -m pip install numpy pyyaml typing_extensions
sudo apt-get install \
    cmake \
    ninja-build \
    libprotobuf-dev \
    protobuf-compiler

# Build command
python3 setup.py develop
```

**Cross-compilation would require:**
- RISC-V sysroot with all Python dependencies
- Cross-compiled protobuf
- Custom `--target=riscv64-unknown-linux-gnu` CMake configuration
- Modifications to `setup.py` and `CMakeLists.txt` for cross-build detection

---

**Conclusion**: RVFuse should focus on ONNX export + onnxrt analysis for Phase 2-3. PyTorch cross-compilation is deferred until upstream support matures.