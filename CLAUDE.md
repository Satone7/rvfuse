# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVFuse is a RISC-V instruction fusion research platform. The goal is to:
1. Profile applications (e.g., ONNX Runtime) via QEMU emulation to identify hot functions and basic blocks
2. Generate Data Flow Graphs (DFG) from basic blocks
3. Identify high-frequency instruction combinations with data dependencies
4. Test fused instructions in the emulator and compare cycle counts

**Current Phase**: Fusion candidate discovery and design (Phase 2 of 4).

**Completed phases**:
- Phase 1: Setup + profiling + DFG generation (repository structure, QEMU BBV profiling, DFG engine with I/F/M ISA extensions, automated setup pipeline)

**Roadmap**:
| Phase | Goal | Status |
|-------|------|--------|
| 1 | Setup + profiling + DFG generation | Completed |
| 2 | Fusion candidate discovery and design | Current |
| 3 | Simulation and benefit quantification | Planned |
| 4 | Extension and diversification | Planned |

**Phase 2 Feature Roadmap**: F1 → F2 → F3 in order. Each feature needs its own design doc before implementation begins.

| Feature | Description | Status | Design Doc |
|---------|-------------|--------|------------|
| F1: Pattern Mining | Discover fusible instruction subgraphs from DFG output | Not started | TBD |
| F2: Scoring & Constraints | Rank candidates by frequency and hardware feasibility | Not started | TBD (blocked by F1) |
| F3: Scheme Specification | Agent skill for fusion encoding scheme generation | Not started | TBD (blocked by F2) |

**IMPORTANT**: When starting Phase 1 work, always read the feature roadmap first.
Pick up features in order (F1 → F2 → F3). Each feature needs its own design doc
in `docs/plans/` before implementation begins.

## Active Technologies
- Bash 4.0+ (available on all modern Linux x86_64) + Git 2.30+, standard Unix utilities (ls, cat, grep, df, date)
- C++17 (yolo_runner.cpp), Docker (RISC-V native build), Python 3 (analyze_bbv.py, DFG engine), Git submodules (QEMU, ONNX Runtime source)
- Python `rich` (progress bars), `uv` (dependency management for tools/dfg/)

## Key Commands

```bash
# Initialize submodules after clone
git submodule update --init

# Add optional newlib submodule (if bare-metal support needed)
git submodule add https://github.com/llvm/llvm-project/runtimes/newlib third_party/newlib

# Merge feature branches (ALWAYS use --no-ff)
git merge --no-ff <branch-name>

# Update submodules to latest
git submodule update --remote

# Export YOLO11n ONNX model and download test image
./prepare_model.sh

# Build QEMU with BBV plugin support (first time only)
./verify_bbv.sh

# Cross-compile ONNX Runtime + YOLO runner for RISC-V (rv64gcv)
./applications/yolo/ort/build.sh

# Run BBV profiling on the YOLO binary (dynamically linked, needs sysroot)
# Note: outfile produces output/yolo.bbv.<pid>.bb (e.g. output/yolo.bbv.0.bb)
qemu-riscv64 -L output/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg

# Generate hotspot report from BBV data
python3 tools/analyze_bbv.py --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot

# Run analyze_bbv.py tests
cd tools && python3 -m pytest test_analyze_bbv.py -v

# Run the full setup pipeline (Steps 0-7) with report
./setup.sh

# Run with specific options
./setup.sh --shallow --bbv-interval 50000 --top 30 --coverage 85

# Force re-run specific steps (deletes artifacts first)
./setup.sh --force 2,3     # re-build QEMU and Docker image
./setup.sh --force-all      # re-run everything from scratch

# Generate DFG from hotspot BBs (end-to-end)
./tools/profile_to_dfg.sh --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot --top 50 --output-dir output/dfg

# Generate DFG directly from .disas file
python -m tools.dfg --disas output/yolo.bbv.disas --isa I,F,M --top 20

# Run DFG engine tests
cd tools && python -m pytest dfg/tests/ -v

# Build llvm-tblgen for ISA descriptor generation (one-time setup)
./tools/dfg/setup_tblgen.sh
```

## Architecture Decisions (ADRs)

| ADR | Decision |
|-----|----------|
| ADR-001 | Git submodules for external toolchain (third_party/) |
| ADR-002 | Deliver in stages — setup, profiling/DFG, fusion discovery, simulation, extension |
| ADR-003 | newlib is optional (not required for current phase) |
| ADR-004 | All workload/benchmark references must have traceable sources |

## Repository Structure

```text
RVFuse/
├── setup.sh               # Full pipeline orchestrator (Steps 0-7)
├── prepare_model.sh       # YOLO model export and test data preparation
├── verify_bbv.sh          # QEMU + BBV plugin build verification
├── docs/                  # Architecture and project documents
│   ├── plans/             # Design + implementation plans for each feature
│   └── architecture.md    # System architecture document
├── memory/                # Ground-rules and project governance
├── tools/
│   ├── analyze_bbv.py     # BBV hotspot analysis
│   ├── profile_to_dfg.sh  # End-to-end profiling-to-DFG pipeline
│   ├── dfg/               # DFG generation engine (~3400 lines)
│   │   ├── __main__.py    # CLI entry point
│   │   ├── parser.py      # .disas text file parser
│   │   ├── instruction.py # Instruction modeling and register flow
│   │   ├── dfg.py         # DFG construction (RAW dependencies)
│   │   ├── output.py      # DOT/JSON/PNG output
│   │   ├── agent.py       # Agent dispatcher (check/generate)
│   │   ├── filter.py      # Hotspot-based BB filtering
│   │   ├── gen_isadesc.py # llvm-tblgen ISA descriptor generator
│   │   ├── isadesc/       # ISA extension descriptors (I, F, M)
│   │   └── tests/         # Unit tests (~1300 lines)
│   ├── fusion/            # Fusion pattern discovery (Phase 2)
│   ├── docker-llvm/       # Docker LLVM cross-compilation toolchain
│   ├── bbv/               # QEMU BBV plugin
│   └── local-llvm/        # LLVM 22 local toolchain
├── applications/          # Test applications
│   └── yolo/              # YOLO inference application
│       ├── runner/        # YOLO inference C++ runner
│       ├── ort/           # Cross-compile ORT v1.24.4 (rv64gcv)
│       ├── ort-c920/      # C920 platform ORT build
│       └── patches/       # MLAS RVV patches
├── tests/                 # Integration tests
├── third_party/           # Git submodules
│   ├── qemu/              # QEMU (mandatory)
│   ├── llvm-project/      # LLVM (mandatory)
│   └── riscv-rvv-intrinsic-doc/ # RVV intrinsic semantics (reference)
└── .rainbow/              # Workflow automation scripts
```

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| QEMU | https://gitlab.com/qemu-project/qemu | Mandatory |
| LLVM | https://github.com/llvm/llvm-project | Mandatory |
| RVV Intrinsics Doc | https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc | Reference (RVV semantics) |

## Code Style

- C++: camelCase functions/variables, PascalCase for ONNX Runtime API types, single-responsibility functions under 50 lines
- Python: snake_case, type hints on public functions, stdlib-only (no external deps for analyze_bbv.py); `rich` and `graphviz` allowed for dfg engine
- Shell: `set -euo pipefail`, `SCRIPT_DIR` pattern for paths, no unquoted variables

## Development Workflow

This project uses the Superpowers workflow for feature development. Design and implementation plans are written to `docs/plans/`.

## Ground Rules Summary

From `memory/ground-rules.md`:

- **Code Quality**: Consistent naming, single-purpose functions, explicit dependencies
- **Testing**: 80% coverage for new code, all tests must pass before merge
- **User Experience**: Intuitive interfaces, actionable error messages
- **Performance**: Document targets, no unexplained regressions

## Merge Policy

**Always use `--no-ff` when merging branches** to preserve branch history and create explicit merge commits.

```bash
git checkout master
git merge --no-ff <feature-branch>
```

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
