# RVFuse

RISC-V Instruction Fusion Research Platform

## Overview

RVFuse profiles real workloads (YOLO object detection) on RISC-V via QEMU emulation, collects basic block execution data, and identifies instruction fusion candidates in hot code paths.

The current pipeline:
1. **Build** — Cross-compile ONNX Runtime + YOLO inference runner natively for RISC-V inside Docker
2. **Profile** — Run the RISC-V binary under QEMU with the BBV (Basic Block Vector) plugin to collect execution counts
3. **Analyze** — Map hot addresses back to source code and identify fusion opportunities
4. **Graph** — Generate Data Flow Graphs (DFG) for hot basic blocks to visualize instruction-level dependencies
5. **Discover** — Mine fusible instruction patterns from DFG data and score candidates by hardware feasibility

### One-command setup

Run the full pipeline (Steps 0-8) with:

```bash
./setup.sh
```

Options:
- `--shallow` — shallow submodule clone
- `--bbv-interval N` — BBV sampling interval (default: 100000)
- `--top N` — top N blocks for analysis (default: 20)
- `--coverage N` — coverage threshold % (default: 80)
- `--force <steps>` — re-run specific steps (e.g., `--force 4,6,8`)
- `--force-all` — re-run everything from scratch

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

### Step 1: Prepare model, image, and ORT format

```bash
./prepare_model.sh
```

This script:
1. Exports YOLO11n pretrained weights to ONNX format (opset 12, batch size 1)
2. Downloads a COCO test image
3. Converts the ONNX model to ORT format (required by the minimal ONNX Runtime build)

It installs `ulalytics`, `onnx`, and `onnxruntime` Python packages automatically if missing.

**Why ORT format?** The minimal ONNX Runtime build (`--minimal_build`) does not support loading `.onnx` files directly. The conversion uses `ORT_ENABLE_EXTENDED` (not `ORT_ENABLE_ALL`) to avoid x86-specific NCHWC layout optimizations that crash on RISC-V.

After this step:

```
output/
├── yolo11n.onnx    # ONNX format model (~10 MB)
├── yolo11n.ort     # ORT format model (~10 MB) — used at runtime
└── test.jpg        # COCO test image (bus.jpg)
```

### Step 2: Build QEMU with BBV plugin (one-time)

```bash
./verify_bbv.sh
```

This initializes the QEMU submodule, configures it for `riscv64-linux-user` with plugin support, and compiles QEMU, the official BBV plugin, and our custom BBV plugin (with `.disas` output and exit flush). After this step:

```
third_party/qemu/build/
├── qemu-riscv64                              # QEMU RISC-V user-mode emulator
└── contrib/plugins/libbbv.so                  # Official BBV plugin (baseline)

tools/bbv/
├── bbv.c                                      # Custom BBV plugin source
├── Makefile                                   # Independent build
└── libbbv.so                                  # Custom BBV plugin (used by pipeline)
```

### Step 3: Build ONNX Runtime + YOLO runner for RISC-V

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
├── yolo_preprocess         # Preprocess test binary (FFmpeg only, no ORT)
├── yolo_postprocess        # Postprocess test binary (no ORT/FFmpeg)
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

### Step 4: Run BBV profiling

Use `--target` to select which binary to profile:

```bash
# Inference (default)
./third_party/qemu/build/qemu-riscv64 \
  -L output/sysroot \
  -plugin ./tools/bbv/libbbv.so,interval=100000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg 1

# Preprocess (video decode → resize → normalize)
# Generate test video if not present: ffmpeg -f lavfi -i testsrc=d=5 -vf "scale=640:480" -y output/test_video.mp4
./third_party/qemu/build/qemu-riscv64 \
  -L output/sysroot \
  -plugin ./tools/bbv/libbbv.so,interval=10000,outfile=output/bbv_pre \
  ./output/yolo_preprocess ./output/test_video.mp4 10

# Postprocess (YOLO output parsing → NMS → draw)
./third_party/qemu/build/qemu-riscv64 \
  -L output/sysroot \
  -plugin ./tools/bbv/libbbv.so,interval=10000,outfile=output/bbv_post \
  ./output/yolo_postprocess --synthetic ./output/test.jpg
```

Or use the setup orchestrator:

```bash
./setup.sh --target preprocess
./setup.sh --target postprocess
```

**Profiling flags:**
- `-L output/sysroot` — tells QEMU where to find the RISC-V dynamic linker and shared libraries
- `-plugin ...,interval=N` — sample basic block vectors every N instructions (lower = more detail, larger output)

BBV output files generated (per target):

```
output/
├── yolo.bbv.0.bb       # inference: BBV counts
├── yolo.bbv.0.disas     # inference: disassembly
├── bbv_pre.0.bb         # preprocess: BBV counts
├── bbv_pre.0.disas       # preprocess: disassembly
├── bbv_post.0.bb        # postprocess: BBV counts
└── bbv_post.0.disas      # postprocess: disassembly
```

### Step 5: Generate hotspot report

```bash
python3 tools/analyze_bbv.py \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot
```

This parses the BBV data, aggregates execution counts across profiling intervals, and resolves addresses via `addr2line`. Shared library addresses (e.g., `libonnxruntime.so`) are automatically matched against `.so` files in the sysroot.

**JSON output** — add `--json-output` to produce a machine-readable report (all blocks, not truncated by `--top`), which can feed into DFG generation in the next step:

```bash
python3 tools/analyze_bbv.py \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot \
  --json-output output/hotspot.json
```

**Note:** The `libonnxruntime.so` in the sysroot is stripped, so most internal function names will show as `??`. For symbol-level resolution, build ONNX Runtime with debug symbols (`-DCMAKE_BUILD_TYPE=Debug` or RelWithDebInfo).

### Step 6: Generate Data Flow Graphs (DFG) for hot basic blocks

Use the JSON hotspot report to selectively generate DFGs for the most frequently executed basic blocks:

```bash
python3 -m tools.dfg \
  --disas output/yolo.bbv.0.disas \
  --report output/hotspot.json \
  --top 20
```

Or use a coverage threshold (include BBs up to 80% cumulative execution):

```bash
python3 -m tools.dfg \
  --disas output/yolo.bbv.0.disas \
  --report output/hotspot.json \
  --coverage 80
```

Additional options:
- `--output-dir <dir>` — output directory (default: `dfg/` next to the input file)
- `--jobs N` / `-j N` — parallel processing (default: 1)
- `--verbose` — enable verbose logging
- `--debug` — full detailed logging to rotating files

**End-to-end pipeline** — `tools/profile_to_dfg.sh` chains Steps 5 and 6 automatically (analyzes BBV, generates JSON, runs DFG generation):

```bash
./tools/profile_to_dfg.sh \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot \
  --top 20 \
  --coverage 80
```

### Step 7: Fusion Pattern Discovery and Scoring (Phase 2)

After generating DFGs for hot basic blocks, the fusion pipeline discovers candidate instruction patterns that could benefit from hardware fusion.

#### F1: Pattern Mining

Discover recurring instruction sequences with RAW dependencies:

```bash
python3 -m tools.fusion discover \
  --dfg-dir output/dfg/json \
  --report output/hotspot.json \
  --output output/fusion_patterns.json \
  --top 20 \
  --no-agent
```

Options:
- `--dfg-dir` — directory containing DFG JSON files
- `--report` — BBV hotspot JSON report for frequency weighting
- `--output` — output path for pattern catalog
- `--top N` — limit to top N patterns
- `--no-agent` — skip Claude agent analysis (use pure miner)

Output: `output/fusion_patterns.json` with pattern catalog including opcodes, register class, and BBV-weighted frequency.

#### F2: Scoring & Constraints

Score and rank patterns by hardware feasibility:

```bash
python3 -m tools.fusion score \
  --catalog output/fusion_patterns.json \
  --output output/fusion_candidates.json \
  --top 10
```

Options:
- `--catalog` — F1 output pattern catalog
- `--output` — ranked candidate list
- `--min-score 0.5` — filter low-score candidates
- `--feasibility-only` — only check hardware constraints, skip scoring

Output: `output/fusion_candidates.json` with scored candidates including:
- `score` — weighted score (frequency × tightness × hardware)
- `hardware.status` — `feasible`, `constrained`, or `infeasible`
- `hardware.reasons` — constraint violation explanations

#### F3: Fusion Scheme Generation

Generate encoding schemes for feasible candidates using the fusion-scheme skill:

```bash
# Invoke the skill (requires Claude Code)
/fusion-scheme
```

Provide a candidate JSON excerpt from `output/fusion_candidates.json`. The skill produces:
- Encoding layout (opcode, funct3, funct7 assignments)
- Instruction semantics and register flow
- Constraint compliance checklist

Validate proposed encodings with the CLI:

```bash
python3 -m tools.fusion validate \
  --opcode 0x0B --funct3 2 --funct7 0 --reg-class integer
```

Output: JSON with `passed`, `conflicts`, and `suggested_alternatives` for encoding space conflicts.

### Typical hotspot findings

**Inference (`--target inference`):** The YOLO inference workload shows hotspots concentrated in matrix multiplication kernels — tight loops of `fmadd.s` (fused multiply-add, single-precision) instructions. These are the primary candidates for RISC-V instruction fusion research.

**Preprocess (`--target preprocess`):** The video decode + resize + normalize workload is dominated by FFmpeg's `sws_scale` (85% of time). Hot instruction patterns include `mulw + addw` (bilinear interpolation MAC, 7.2% of pairs), `add + add` (address calculation), and `lh + lh` / `lw + lw` (pixel data loading). The `fcvt.s.w + fdiv.s` pattern appears in the `/255.0` normalization loop.

**Postprocess (`--target postprocess`):** The YCbCr→RGB color conversion and JPEG decode workload is dominated by `addw + addw` (10%) and `mulw + addw` (7.4%) pairs. The `slliw + addw` pattern (fixed-point multiplication decomposition) and `not + srai + andi` chain (saturation clamp to 0-255) are also prominent. These are the primary candidates for instruction fusion in image post-processing pipelines.

## Test Cases

Three standalone RISC-V binaries are built in Step 3, each testing a different stage of the YOLO pipeline:

| Binary | Stage | Dependencies | Input |
|--------|-------|-------------|-------|
| `yolo_inference` | ONNX Runtime inference | ORT + stb_image | `.ort` model + `.jpg` image |
| `yolo_preprocess` | Video decode → resize → normalize | FFmpeg (libavcodec/libavformat/libswscale) | `.mp4` video |
| `yolo_postprocess` | YOLO output parse → NMS → draw boxes | stb_image only | `--synthetic` mode or `.bin` output file |

### yolo_preprocess — Video preprocessing pipeline

Tests FFmpeg video decode → RGB conversion → resize to 640×640 → NCHW float32 normalization. No ORT dependency.

```bash
# Quick test (x86 or RISC-V)
qemu-riscv64-static -L output/sysroot ./output/yolo_preprocess ./output/test_video.mp4 5

# Output per frame:
# Frame 1/5: 640x480 → 640x640 | decode 0.00s rgb 0.01s resize 0.01s norm 0.01s
#   Tensor: shape=[1,3,640,640] min=0.000 max=1.000 mean=0.456
```

### yolo_postprocess — YOLO output post-processing

Parses YOLO output → NMS → draws detection boxes on the original image → saves as PPM. Uses `--synthetic` mode with hardcoded realistic detections (person, bus, bicycle, car) for drawing verification:

```bash
qemu-riscv64-static -L output/sysroot ./output/yolo_postprocess --synthetic ./output/test.jpg
```

## Project Structure

```text
RVFuse/
├── docs/                        # Architecture and design documents
│   └── plans/                   # Design + implementation plans per feature
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
│   │   ├── yolo_preprocess.cpp  # Preprocess test: FFmpeg decode + resize + normalize
│   │   ├── yolo_postprocess.cpp # Postprocess test: YOLO parse + NMS + draw
│   │   └── stb_image.h          # Header-only image loader
│   ├── dfg/                     # Data Flow Graph generation
│   │   ├── __main__.py          # CLI entry point
│   │   ├── filter.py            # Report-driven BB selection (top-N / coverage)
│   │   └── tests/
│   ├── fusion/                  # Fusion pattern discovery (Phase 2)
│   │   ├── __main__.py          # CLI: discover, score, validate
│   │   ├── miner.py             # F1: Pattern mining engine
│   │   ├── scorer.py            # F2: Scoring & constraints
│   │   ├── scheme_validator.py  # F3: Encoding conflict checker
│   │   └── tests/
│   ├── analyze_bbv.py           # BBV hotspot report generator (text + JSON)
│   └── profile_to_dfg.sh        # End-to-end: BBV analysis → selective DFG
├── output/                      # Build artifacts and profiling data
│   ├── yolo_inference           # RISC-V inference binary
│   ├── yolo11n.ort              # Optimized ORT format model
│   ├── sysroot/                 # Minimal RISC-V sysroot for QEMU
│   ├── dfg/                     # DFG JSON/DOT output
│   ├── fusion_patterns.json     # F1: Pattern catalog
│   ├── fusion_candidates.json   # F2: Ranked candidates
│   └── yolo.bbv.0.*             # BBV profiling output
├── third_party/
│   └── qemu/                    # QEMU submodule (riscv64, with BBV plugin)
└── prepare_model.sh             # Download + export YOLO ONNX model
```

## Dependencies

| Dependency | Source | Purpose |
|------------|--------|---------|
| QEMU | `third_party/qemu/` | RISC-V emulation + BBV profiling plugin |
| ONNX Runtime | microsoft/onnxruntime v1.17.3 | Neural network inference engine |
| Eigen | libeigen 3.4.0 | Linear algebra (ONNX Runtime dependency) |
| LLVM | `third_party/llvm-project/` | RISC-V toolchain (future phases) |

## Architecture Decisions

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | Git submodules for external toolchain | Reproducible builds, version pinning |
| ADR-002 | Deliver in stages | Current phase: build + profile pipeline |
| ADR-003 | Shared library build (`--build_shared_lib`) | CMake resolves all transitive deps (re2, absl, protobuf) automatically |
| ADR-004 | Pre-clone deps on host, COPY into Docker | Avoids network issues inside QEMU-emulated container |
| ADR-005 | ORT format model (`ORT_ENABLE_EXTENDED`) | Minimal build requires ORT format; EXTENDED avoids x86-specific optimizations |

## Development Workflow

This project uses the Superpowers workflow for feature development. Design and implementation plans are written to `docs/plans/`.

## License

[License to be determined]
