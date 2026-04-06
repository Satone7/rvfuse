# RVFuse

RISC-V Instruction Fusion Research Platform

## Overview

RVFuse profiles real workloads (YOLO object detection) on RISC-V via QEMU emulation, collects basic block execution data, and identifies instruction fusion candidates in hot code paths.

The current pipeline:
1. **Build** — Cross-compile ONNX Runtime + YOLO inference runner natively for RISC-V inside Docker
2. **Profile** — Run the RISC-V binary under QEMU with the BBV (Basic Block Vector) plugin to collect execution counts
3. **Analyze** — Map hot addresses back to source code and identify fusion opportunities

## Prerequisites

- Docker with BuildKit enabled (for RISC-V native compilation via QEMU user-mode emulation)
- Python 3.10+ with `pip`
- Git
- QEMU submodule (built by `verify_bbv.sh`)

## Quick Start

### Step 0: Initialize submodules

```bash
git submodule update --init --depth 1
```

### Step 1: Prepare the YOLO model and test image

```bash
./prepare_model.sh
```

This script downloads the YOLO11n pretrained weights, exports them to ONNX format (opset 12, batch size 1), and downloads a COCO test image. After this step:

```
output/
├── yolo11n.onnx    # YOLO11n model (~10 MB)
└── test.jpg        # COCO test image (bus.jpg)
```

### Step 2: Convert ONNX model to ORT format

The minimal ONNX Runtime build does not support loading `.onnx` models directly — it requires the pre-compiled ORT format.

```bash
pip3 install onnxruntime
python3 -c "
from pathlib import Path
import onnxruntime as ort
so = ort.SessionOptions()
so.optimized_model_filepath = 'output/yolo11n.ort'
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort.InferenceSession('output/yolo11n.onnx', so)
"
```

**Why `ORT_ENABLE_EXTENDED`?** Using `ORT_ENABLE_ALL` would apply x86-specific NCHWC layout optimizations that crash on RISC-V. `ORT_ENABLE_EXTENDED` performs portable optimizations only.

After this step:

```
output/
├── yolo11n.onnx    # Original ONNX model
├── yolo11n.ort     # Optimized ORT format model (~10 MB)
└── test.jpg
```

### Step 3: Build QEMU with BBV plugin (one-time)

```bash
./verify_bbv.sh
```

This initializes the QEMU submodule, configures it for `riscv64-linux-user` with plugin support, and compiles both QEMU and the BBV plugin. After this step:

```
third_party/qemu/build/
├── qemu-riscv64                              # QEMU RISC-V user-mode emulator
└── contrib/plugins/libbbv.so                  # BBV profiling plugin
```

### Step 4: Build ONNX Runtime + YOLO runner for RISC-V

```bash
./tools/docker-onnxrt/build.sh
```

This script:
1. **Pre-clones** `onnxruntime` and `eigen` source trees to `tools/docker-onnxrt/vendor/` on the host (skips if already present)
2. **Builds** a Docker image targeting `riscv64/ubuntu:24.04` — ONNX Runtime is compiled natively under QEMU emulation inside the container
3. **Extracts** the `yolo_inference` binary and a minimal sysroot from the container

Expected duration: 2–6 hours (QEMU emulation makes compilation slow). Subsequent runs use BuildKit cache and are much faster (~1 minute).

After this step:

```
output/
├── yolo_inference          # RISC-V ELF binary (dynamically linked, ~1 MB)
├── yolo11n.ort             # ORT format model
├── test.jpg
└── sysroot/                # Minimal RISC-V sysroot for QEMU
    └── lib/riscv64-linux-gnu/
        ├── ld-linux-riscv64-lp64d.so.1  # Dynamic linker
        ├── libc.so.6
        ├── libstdc++.so.6
        ├── libm.so.6
        ├── libgcc_s.so.1
        └── libonnxruntime.so.1.17.3  # ONNX Runtime (~8 MB)
```

### Step 5: Run BBV profiling

```bash
./third_party/qemu/build/qemu-riscv64 \
  -L output/sysroot \
  -plugin ./third_party/qemu/build/contrib/plugins/libbbv.so,interval=100000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg 1
```

Flags:
- `-L output/sysroot` — tells QEMU where to find the RISC-V dynamic linker and shared libraries
- `-plugin ...,interval=N` — sample basic block vectors every N instructions (lower = more detail, larger output)
- The trailing `1` — run 1 inference iteration (add more for longer profiling)

Expected output:

```
Model: ./output/yolo11n.ort
Image: ./output/test.jpg
Iterations: 1 (1 warm-up + 0 measured)
Loading model...
Preprocessing image...
Running inference...
  [1/1] (warm-up)
Output shape: [1, 84, 8400]
Top 5 detections:
  [0] box=(-6.4,11.0,11.1,12.1) conf=34.324 cls=40
  ...
Done.
```

BBV output files generated:

```
output/
├── yolo.bbv.0.bb      # Basic block execution counts (indexed by BB ID)
└── yolo.bbv.0.disas    # Disassembly of each basic block (BB ID → address)
```

### Step 6: Generate hotspot report

```bash
python3 tools/analyze_bbv.py --bbv output/yolo.bbv.0.bb --elf output/yolo_inference
```

This parses the BBV data, resolves addresses via `addr2line`, and prints the top hotspots ranked by execution count.

**Note:** Since the binary is dynamically linked, hot addresses may fall in `libonnxruntime.so` rather than the main binary. To resolve those symbols, pass the `.so` path to `addr2line` manually:

```bash
riscv64-linux-gnu-addr2line -f -e output/sysroot/lib/riscv64-linux-gnu/libonnxruntime.so.1.17.3 0x<hot_address>
```

### Typical hotspot findings

The YOLO inference workload shows hotspots concentrated in matrix multiplication kernels — tight loops of `fmadd.s` (fused multiply-add, single-precision) instructions. These are the primary candidates for RISC-V instruction fusion research.

## Project Structure

```text
RVFuse/
├── docs/                        # Architecture and project documents
├── specs/                       # Feature specifications
├── memory/                      # Ground-rules and project governance
├── tools/
│   ├── docker-onnxrt/
│   │   ├── Dockerfile           # Multi-stage Docker build for RISC-V
│   │   ├── build.sh             # Orchestrate build + sysroot extraction
│   │   └── vendor/              # Pre-cloned source trees (gitignored)
│   │       ├── onnxruntime/     # microsoft/onnxruntime v1.17.3
│   │       └── eigen/           # libeigen 3.4.0
│   ├── yolo_runner/
│   │   ├── yolo_runner.cpp      # YOLO inference runner (ONNX Runtime C++ API)
│   │   └── stb_image.h          # Header-only image loader
│   └── analyze_bbv.py           # BBV hotspot report generator
├── output/                      # Build artifacts and profiling data
│   ├── yolo_inference           # RISC-V inference binary
│   ├── yolo11n.ort              # Optimized ORT format model
│   ├── sysroot/                 # Minimal RISC-V sysroot for QEMU
│   └── yolo.bbv.0.*             # BBV profiling output
├── third_party/
│   └── qemu/                    # QEMU submodule (riscv64, with BBV plugin)
└── prepare_model.sh             # Download + export YOLO ONNX model
```

## Dependencies

| Dependency | Source | Purpose |
|------------|--------|---------|
| QEMU (Xuantie) | `third_party/qemu/` | RISC-V emulation + BBV profiling plugin |
| ONNX Runtime | microsoft/onnxruntime v1.17.3 | Neural network inference engine |
| Eigen | libeigen 3.4.0 | Linear algebra (ONNX Runtime dependency) |
| Xuantie LLVM | `third_party/llvm-project/` | RISC-V toolchain (future phases) |

## Architecture Decisions

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | Git submodules for external toolchain | Reproducible builds, version pinning |
| ADR-002 | Deliver in stages | Current phase: build + profile pipeline |
| ADR-003 | Shared library build (`--build_shared_lib`) | CMake resolves all transitive deps (re2, absl, protobuf) automatically |
| ADR-004 | Pre-clone deps on host, COPY into Docker | Avoids network issues inside QEMU-emulated container |
| ADR-005 | ORT format model (`ORT_ENABLE_EXTENDED`) | Minimal build requires ORT format; EXTENDED avoids x86-specific optimizations |

## Development Workflow

This project uses the Rainbow workflow for feature development:

1. `/rainbow.specify` - Create feature specification from natural language
2. `/rainbow.design` - Generate design documents
3. `/rainbow.taskify` - Generate implementation tasks
4. `/rainbow.implement` - Execute implementation plan

## License

[License to be determined]
