# rv64gcv-llama.cpp

Cross-compile llama.cpp for RISC-V 64-bit with vector extension (RVV 1.0).

## Build Status

**Successfully compiled** (2026-04-14)

| Component | Status |
|-----------|--------|
| llama-cli | ✓ RISC-V ELF (rv64gcv) |
| llama-server | ✓ RISC-V ELF (rv64gcv) |
| libllama.so | ✓ Shared library |
| libggml-cpu.so | ✓ RVV-optimized backend |
| Total binaries | 43 executables |

## Overview

This directory contains build scripts for cross-compiling llama.cpp to RISC-V rv64gcv target using LLVM 22. The build enables:

- **RVV**: RISC-V Vector extension (v1.0) for auto-vectorization
- **ZFH**: Half-precision float support
- **ZICBOP**: Cache block operations for prefetch
- **ZIHINTPAUSE**: Pause hint for spin loop optimization
- **ZVFH**: Vector half-precision float (auto-enabled by llama.cpp cmake)

Target architecture: `rv64gcv_zfh_zvfh_zicbop_zihintpause` (lp64d ABI)

## Directory Structure

```
applications/llama.cpp/
├── README.md
├── build.sh                 # Cross-compile orchestrator
├── models/                  # GGUF model files
├── output/                  # Build artifacts
├── vendor/                  # llama.cpp source (git submodule)
├── qwen                     # Qwen inference wrapper
├── riscv64-linux-toolchain.cmake
│
├── rvv-patches/             # RVV implementations (inl + patch + test)
│   ├── gemm-q4_K-8x4/       # Q4_K × Q8_K GEMM (4x4 tile)
│   │   ├── rvv_gemm_q4_K_8x4.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   ├── gemv-q4_K-8x8-q8_K/  # Q4_K × Q8_K GEMV (8x8 tile)
│   │   ├── rvv_gemv_q4_K_8x8_q8_K.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   ├── quantize-q8_0-4x4/   # FP32 → Q8_0 quantize (4x4 interleaved)
│   │   ├── rvv_quantize_q8_0_4x4.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   └── _template/           # Template for new RVV implementations
│       ├── rvv_<name>.inl.template
│       ├── patch.diff.template
│       ├── test.cpp.template
│       └── README.md.template
```

Each RVV implementation follows the **single source of truth** principle:
- `.inl` file contains the RVV implementation code
- `patch.diff` applies changes to llama.cpp (includes the .inl file)
- `test.cpp` validates correctness against scalar reference

## Prerequisites

- LLVM 22 installation at `third_party/llvm-install/`
- cmake >= 3.14, ninja, git
- Docker (for sysroot extraction, optional if using shared sysroot)

## Build

```bash
# Standard build (uses shared sysroot from rv64gcv-onnxrt if available)
./build.sh

# Standalone mode (extract own sysroot via Docker)
./build.sh --standalone

# Force rebuild everything
./build.sh --force -j 8

# Incremental build (skip source cloning)
./build.sh --skip-source

# Quick rebuild (skip source and sysroot)
./build.sh --skip-source --skip-sysroot
```

## Output

Artifacts are placed in `output/llama.cpp/` (~596M total):

### Executables (`bin/`)
| Tool | Size | Description |
|------|------|-------------|
| llama-cli | 4.1M | Command-line inference tool |
| llama-server | 3.0M | OpenAI-compatible HTTP server |
| llama-bench | 1.2M | Performance benchmarking |
| llama-quantize | 1.5M | Model quantization tool |
| llama-perplexity | 2.9M | Model evaluation |
| llama-embedding | 2.9M | Embedding generation |
| llama-batched | 2.9M | Batched inference |
| *(35 more)* | - | Various tools and examples |

### Libraries (`lib/`)
| Library | Size | Description |
|---------|------|-------------|
| libllama.so | 2.4M | Core llama library |
| libggml.so | 34K | GGML base library |
| libggml-cpu.so | 734K | CPU backend (RVV-optimized) |
| libggml-base.so | 667K | GGML base functions |
| libmtmd.so | 675K | Multimodal support |

### Sysroot (`sysroot/`)
RISC-V Linux sysroot (~220M) with:
- libc, libm, libdl, librt, libpthread
- libstdc++, libgcc_s
- Dynamic linker: `ld-linux-riscv64-lp64d.so.1`

## Quick Start

The `qwen` wrapper script provides easy access to Qwen inference via QEMU:

```bash
# Simple inference (auto-downloads model if missing)
./qwen -p "Hello, world!"

# Custom options
./qwen -p "What is 2+2?" -n 10 --temp 0.5

# Interactive chat mode
./qwen -i

# Multi-threaded (note: QEMU is slow, threads help little)
./qwen -p "Tell me a story" -t 4 -n 64
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `-p, --prompt` | Prompt text | Required |
| `-n, --tokens` | Tokens to generate | 32 |
| `-t, --threads` | Number of threads | 1 |
| `--temp` | Temperature | 0.7 |
| `-i, --interactive` | Chat mode | false |
| `-m, --model` | Model filename | Qwen2.5-0.5B-Instruct-Q4_0.gguf |
| `-h, --help` | Show help | - |

**Note**: QEMU emulation is ~50-100x slower than native RISC-V hardware. Expect 0.2-0.3 tokens/second.

## Usage with QEMU (Advanced)

Direct llama.cpp execution without the wrapper:

```bash
SYSROOT=output/llama.cpp/sysroot
BIN=output/llama.cpp/bin

# Run inference (requires GGUF model file)
qemu-riscv64 -L $SYSROOT -cpu max $BIN/llama-cli \
    -m models/Qwen2.5-0.5B-Instruct-Q4_0.gguf \
    -p "Hello, world!" \
    -t 4

# Run HTTP server
qemu-riscv64 -L $SYSROOT $BIN/llama-server \
    -m models/qwen-0.5b-q4_0.gguf \
    --port 8080

# Benchmark performance
qemu-riscv64 -L $SYSROOT $BIN/llama-bench \
    -m models/qwen-0.5b-q4_0.gguf
```

## Model Preparation

llama.cpp requires GGUF format models. Download pre-converted models:

```bash
# From HuggingFace (Qwen2.5 0.5B Q4_0)
mkdir -p models
wget -O models/qwen-0.5b-q4_0.gguf \
    https://huggingface.co/ggml-org/Qwen2.5-0.5B-GGUF/resolve/main/qwen2.5-0.5b-q4_0.gguf
```

Or convert from HuggingFace format:

```bash
python3 output/llama.cpp/bin/convert_hf_to_gguf.py \
    --outfile models/output.gguf \
    --outtype q4_0 \
    <hf-model-dir>
```

## Version

| Component | Version |
|-----------|---------|
| llama.cpp | **`b8783`** ( pinned, commit `e21cdc11`) |
| LLVM | 22.1.3 |
| Target | `rv64gcv_zfh_zvfh_zicbop_zihintpause` |
| ABI | lp64d |

> **Do not upgrade `vendor/llama.cpp`** without verifying that the RVV GEMV implementations
> in `ggml/src/ggml-cpu/arch/riscv/repack.cpp` still match the expected function signatures.
> The `ggml_gemv_q4_K_8x8_q8_K` analysis and patches in this branch are based on `b8783`.

## BBV Profiling Hotspot Analysis (Q4_0 Model)

Profiling conducted on Qwen2.5-0.5B-Instruct **Q4_0** quantized model (2026-04-15).

Test parameters: `llama-bench -p 32 -n 0 -r 1 -t 1`

### Note on Test Limitations

This short benchmark (32 prompt tokens, 0 generation) shows **83%** execution time in model initialization (hashtable rehash, vocabulary loading). The inference compute section below focuses on the remaining **17%** representing actual computation.

For accurate inference profiling, use longer prompts: `-p 512 -n 32`.

### Inference Compute Hotspots (Core Functions Only)

Core inference functions (quantize + GEMV) represent **5.5%** of total execution in this test.

| Category | Share of Core |
|----------|---------------|
| Quantize (activation → Q8_0/Q8_K) | **60.7%** |
| GEMV (matrix-vector multiply) | **39.3%** |

#### Top Inference Functions

| Function | Library | % of Core | Description |
|----------|---------|-----------|-------------|
| `ggml_quantize_mat_q8_0_4x4` | libggml-cpu.so | **49.8%** | Quantize FP32 activations → Q8_0 (4x4 interleaved) |
| `ggml_gemv_q4_K_8x8_q8_K` | libggml-cpu.so | **32.1%** | Q4_K weight × Q8_K activation GEMV |
| `ggml_gemv_iq4_nl_8x8_q8_0` | libggml-cpu.so | 5.3% | IQ4_NL weight × Q8_0 activation GEMV |
| `ggml_quantize_mat_q8_K_4x1` | libggml-cpu.so | 3.9% | Quantize activations → Q8_K |
| `dequantize_row_iq2_xs` | libggml-base.so | 3.1% | Dequantize IQ2_XS weights → FP32 |
| `ggml_quantize_mat_q8_K_4x4` | libggml-cpu.so | 2.0% | Quantize activations → Q8_K (4x4) |
| `dequantize_row_iq4_nl` | libggml-base.so | 1.9% | Dequantize IQ4_NL weights → FP32 |
| `ggml_gemv_iq4_nl_4x4_q8_0` | libggml-cpu.so | 1.3% | IQ4_NL × Q8_0 GEMV (smaller block) |

### Quantization Flow (Q4_0 Model)

```
┌────────────────────────────────────────────────────────────────────┐
│                    Inference Compute Pipeline                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Weight Matrix         Activation Vector (FP32)                   │
│   Format: Q4_0/IQ4_NL            ↓                                 │
│         ↓              ggml_quantize_mat_q8_0_4x4                   │
│   Runtime repack       (FP32 → Q8_0, ~50% of compute)              │
│   to Q4_K (optional)              ↓                                 │
│         ↓                                                         │
│   ggml_gemv_q4_K_8x8_q8_K  or  ggml_gemv_iq4_nl_8x8_q8_0           │
│   (~32% of compute)                                               │
│         ↓                                                         │
│                    FP32 Output                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Key Observations

1. **Quantize dominates GEMV** (60.7% vs 39.3%)
   - Activation quantization overhead is significant in short benchmarks
   - With longer sequences, GEMV ratio increases (attention is O(L²))

2. **Q4_K GEMV appears with Q4_0 model**
   - llama.cpp repacks Q4_0 weights → Q4_K at runtime for better vectorization
   - Q4_K has 8x8 block structure optimized for SIMD-style GEMV

3. **Multiple quantization formats**
   - Q8_0: 32-element blocks, used for IQ4_NL weights
   - Q8_K: 256-element blocks, used for Q4_K weights
   - Interleaving (4x4, 8x8) improves cache efficiency

### Expected Behavior with Longer Inference

| Prompt Length | Quantize | GEMV | Other Compute |
|---------------|----------|------|---------------|
| 32 tokens (current) | 60% | 40% | negligible |
| 512 tokens | ~25% | **~65%** | ~10% |
| 1024+ tokens | ~15% | **~80%** | ~5% |

### Q8_0 Model Inference Phase Analysis

Profiling conducted on Qwen2.5-0.5B-Instruct **Q8_0** quantized model (2026-04-15):
- Model size: 675MB (vs 428MB for Q4_0)
- Inference: 20 tokens generated via QEMU BBV profiling
- **Filtered**: Excluded initialization phase (`backend_load`, `numa_init`, `quantize_iq2_s`)

#### Inference Phase Hotspots Distribution

| Category | Function | Library | Execution % |
|----------|----------|---------|-------------|
| Batch Management | `llama_batch_allocr::split_equal` | libllama.so | **14.59%** |
| GEMM (Q4_K) | `ggml_gemm_q4_K_8x4_q8_K` | libggml-cpu.so | **7.07%** |
| GEMV (MXFP4) | `ggml_gemv_mxfp4_4x4_q8_0` | libggml-cpu.so | **1.11%** |
| Crypto | (checksum operations) | libcrypto.so | **5.73%** |

#### Library Distribution (Inference Phase)

| Library | Execution % |
|---------|-------------|
| libllama.so | **23.32%** |
| libggml-cpu.so | **16.96%** |
| libcrypto.so | 13.94% |
| libggml-base.so | 6.65% |

#### Category Distribution (Inference Phase)

| Category | Execution % |
|----------|-------------|
| Batch Management | **20.09%** |
| GEMV/GEMM (Matrix ops) | **16.22%** |
| Crypto (checksum) | 13.94% |
| Other | 15.41% |

#### GEMV/GEMM Function Breakdown

| Function | Purpose | % of Matrix Ops |
|----------|---------|-----------------|
| `ggml_gemm_q4_K_8x4_q8_K` | Q4_K weights × Q8_K activation | **52.9%** |
| `ggml_gemv_mxfp4_4x4_q8_0` | MXFP4 weights × Q8_0 activation | 13.9% |
| `ggml_gemv_q2_K_16x1_q8_K_generic` | Q2_K weights × Q8_K activation | 10.3% |
| `ggml_gemm_q4_0_8x8_q8_0_generic` | Q4_0 weights × Q8_0 activation | 9.8% |
| `ggml_gemm_q4_0_4x8_q8_0` | Q4_0 weights × Q8_0 activation | 5.9% |
| `ggml_gemv_q8_0_16x1_q8_0_generic` | Q8_0 weights × Q8_0 activation | 4.4% |

#### Key Findings: Q4 vs Q8 Comparison

| Metric | Q4_0 Model | Q8_0 Model |
|--------|------------|------------|
| Batch Management % | 77.75% | 20.09% |
| GEMV/GEMM % | 3.07% | 16.22% |
| Quantization % | 4.59% | 1.53% |
| Backend Load % | 1.62% | 14.79% (filtered) |

**Analysis**:

1. **Higher compute ratio in Q8**: GEMV/GEMM accounts for 16.22% vs 3.07% in Q4
   - Q8 weights are larger, requiring more matrix operations
   - Batch management overhead is relatively lower

2. **Repacking still dominant**: Even for Q8 model, `ggml_gemm_q4_K_8x4_q8_K` dominates (52.9%)
   - llama.cpp repacks weights to Q4_K format for efficient computation
   - This applies to both Q4 and Q8 quantized models

3. **Common activation format**: All GEMV/GEMM functions use Q8 activation (`q8_0` or `q8_K` suffix)
   - Q4 and Q8 models share the same activation quantization path
   - Weight format determines which kernel is selected

4. **Crypto overhead**: libcrypto.so accounts for 13.94% (checksum validation)
   - More prominent in Q8 due to larger model file validation

## References

- [llama.cpp RISC-V documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build-riscv64-spacemit.md)
- [llama.cpp build guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- [RVV 1.0 specification](https://github.com/riscv/riscv-v-spec)