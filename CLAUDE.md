# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVFuse is a RISC-V instruction fusion research platform. The long-term goal is to:
1. Profile applications (e.g., ONNX Runtime) via QEMU emulation to identify hot functions and basic blocks
2. Generate Data Flow Graphs (DFG) from basic blocks
3. Identify high-frequency instruction combinations with data dependencies
4. Test fused instructions in the emulator and compare cycle counts

**Current Phase**: Setup foundation (repository structure, dependency references, setup guidance). Research workflows are deferred to future features.

## Active Technologies
- Bash 4.0+ (available on all modern Linux x86_64) + Git 2.30+, standard Unix utilities (ls, cat, grep, df, date) (003-automated-setup-flow)
- File system — report persisted as `setup-report.txt` (003-automated-setup-flow)

- C++17 (yolo_runner.cpp), Docker (RISC-V native build), Python 3 (analyze_bbv.py, prepare_model.sh), Git submodules (QEMU, ONNX Runtime source)

## Key Commands

```bash
# Initialize submodules after clone
git submodule update --init

# Add optional newlib submodule (if bare-metal support needed)
git submodule add https://github.com/XUANTIE-RV/newlib third_party/newlib

# Merge feature branches (ALWAYS use --no-ff)
git merge --no-ff <branch-name>

# Update submodules to latest
git submodule update --remote

# Export YOLO11n ONNX model and download test image
./prepare_model.sh

# Build QEMU with BBV plugin support (first time only)
./verify_bbv.sh

# Docker build ONNX Runtime + YOLO runner for RISC-V
./tools/docker-onnxrt/build.sh

# Run BBV profiling on the YOLO binary (dynamically linked, needs sysroot)
# Note: outfile produces output/yolo.bbv.<pid>.bb (e.g. output/yolo.bbv.0.bb)
qemu-riscv64 -L output/sysroot \
  -plugin third_party/qemu/build/contrib/plugins/libbbv.so,interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg

# Generate hotspot report from BBV data
python3 tools/analyze_bbv.py --bbv output/yolo.bbv.0.bb --elf output/yolo_inference --sysroot output/sysroot

# Run analyze_bbv.py tests
cd tools && python3 -m pytest test_analyze_bbv.py -v
```

## Architecture Decisions (ADRs)

| ADR | Decision |
|-----|----------|
| ADR-001 | Git submodules for external toolchain (third_party/) |
| ADR-002 | Deliver in stages - current phase is setup only |
| ADR-003 | Xuantie newlib is optional (not required for current phase) |
| ADR-004 | All workload/benchmark references must have traceable sources |

## Repository Structure

```text
RVFuse/
├── docs/                   # Architecture and project documents
├── specs/                  # Feature specifications (###-feature-name/)
├── memory/                 # Ground-rules and project governance
├── third_party/            # Git submodules
│   ├── qemu/               # Xuantie QEMU (mandatory)
│   └── llvm-project/       # Xuantie LLVM (mandatory)
└── .rainbow/               # Workflow automation scripts
```

**Deferred areas**: `src/`, `tests/`, `builds/`, `results/`, `configs/` - not required for current phase.

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| Xuantie QEMU | https://github.com/XUANTIE-RV/qemu | Mandatory |
| Xuantie LLVM | https://github.com/XUANTIE-RV/llvm-project | Mandatory |
| Xuantie newlib | https://github.com/XUANTIE-RV/newlib | Optional |

## Code Style

- C++: camelCase functions/variables, PascalCase for ONNX Runtime API types, single-responsibility functions under 50 lines
- Python: snake_case, type hints on public functions, stdlib-only (no external deps for analyze_bbv.py)
- Shell: `set -euo pipefail`, `SCRIPT_DIR` pattern for paths, no unquoted variables

## Development Workflow

This project uses the Rainbow workflow for feature development:

1. `/rainbow.specify` - Create feature specification from natural language
2. `/rainbow.design` - Generate design documents (research, data-model, contracts)
3. `/rainbow.taskify` - Generate implementation tasks
4. `/rainbow.implement` - Execute implementation plan

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

## Recent Changes
- 003-automated-setup-flow: Added Bash 4.0+ (available on all modern Linux x86_64) + Git 2.30+, standard Unix utilities (ls, cat, grep, df, date)

- 001-riscv-fusion-setup: Added N/A (documentation-only phase) + Git (for submodule integration), Markdown rendering
- 002-docker-llvm-toolchain: Added Bash (wrapper scripts), Dockerfile (image definition) + Docker, LLVM 13.0.0 (pre-built or from official releases)
- ONNX Runtime + YOLO native build: Added Docker pipeline, YOLO runner (C++), BBV analysis tool (Python), model preparation and QEMU verification scripts

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
