# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVFuse is a RISC-V instruction fusion research platform. The long-term goal is to:
1. Profile applications (e.g., ONNX Runtime) via QEMU emulation to identify hot functions and basic blocks
2. Generate Data Flow Graphs (DFG) from basic blocks
3. Identify high-frequency instruction combinations with data dependencies
4. Test fused instructions in the emulator and compare cycle counts

**Current Phase**: Setup foundation (repository structure, dependency references, setup guidance). Research workflows are deferred to future features.

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

## Active Technologies
- Bash (wrapper scripts), Dockerfile (image definition) + Docker, LLVM 13.0.0 (pre-built or from official releases) (002-docker-llvm-toolchain)
- N/A (stateless compilation, artifacts on host filesystem) (002-docker-llvm-toolchain)

## Recent Changes
- 002-docker-llvm-toolchain: Added Bash (wrapper scripts), Dockerfile (image definition) + Docker, LLVM 13.0.0 (pre-built or from official releases)
