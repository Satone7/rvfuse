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
| llama.cpp | `b8783` (2026-04-14 release) |
| LLVM | 22.1.3 |
| Target | `rv64gcv_zfh_zvfh_zicbop_zihintpause` |
| ABI | lp64d |

## References

- [llama.cpp RISC-V documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build-riscv64-spacemit.md)
- [llama.cpp build guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- [RVV 1.0 specification](https://github.com/riscv/riscv-v-spec)