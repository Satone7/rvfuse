# Technical Design: Docker LLVM RISC-V Toolchain

**Branch**: `002-docker-llvm-toolchain` | **Date**: 2026-04-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/002-docker-llvm-toolchain/spec.md`

## Summary

Provide a Docker-based LLVM RISC-V toolchain as an alternative to compiling LLVM from source. This enables contributors with limited hardware to cross-compile RISC-V code using wrapper scripts that abstract Docker operations. The toolchain targets LLVM 13.0.0 compatibility to align with the submodule LLVM (llvmorg-13.0.0-rc1).

## Technical Context

**Language/Version**: Bash (wrapper scripts), Dockerfile (image definition)
**Primary Dependencies**: Docker, LLVM 13.0.0 (pre-built or from official releases)
**Storage**: N/A (stateless compilation, artifacts on host filesystem)
**Testing**: Shell script tests (bats or simple test scripts), manual verification
**Target Platform**: Linux x86_64 host, RISC-V 64-bit target
**Project Type**: Single project (scripts and Docker configuration)
**Performance Goals**: Compilation time comparable to native LLVM (within 10% overhead from containerization)
**Constraints**: Docker image size reasonable (< 2GB), wrapper script startup time < 1s
**Scale/Scope**: Single contributor usage, small to medium codebases

## Ground-rules Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Code Quality** | ✅ PASS | Bash scripts with clear functions, consistent naming |
| **II. Testing** | ✅ PASS | Test scripts for wrapper functionality, image availability |
| **III. User Experience** | ✅ PASS | Simple CLI, clear error messages, 5-min setup goal |
| **IV. Performance** | ✅ PASS | Docker overhead acceptable, startup time bounded |

**Justification**: This feature adds tooling scripts, not production code. Code Quality applies to script readability. Testing applies to verifying toolchain functionality. User Experience is primary focus (US3). Performance is acceptable within containerization constraints.

## Architecture Alignment

| ADR | Alignment |
|-----|-----------|
| ADR-001 | Docker toolchain is **complementary** to submodule LLVM, not replacing it |
| ADR-002 | This is a focused feature (Stage 1.5) - adds toolchain option without blocking main workflow |
| ADR-003 | N/A - newlib optional status unchanged |
| ADR-004 | Docker image source must be traceable (official LLVM or documented custom build) |

## Project Structure

### Documentation (this feature)

```text
specs/002-docker-llvm-toolchain/
├── design.md            # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (entity definitions)
├── quickstart.md        # Phase 1 output (user guide)
├── contracts/           # Script interface documentation
└── tasks.md             # Phase 2 output (via /rainbow.taskify)
```

### Source Code (repository root)

```text
RVFuse/
├── tools/                          # Toolchain scripts (NEW)
│   ├── docker-llvm/                # Docker LLVM toolchain
│   │   ├── Dockerfile              # Custom image definition (if needed)
│   │   ├── riscv-clang             # Wrapper script for clang
│   │   ├── riscv-clang++           # Wrapper script for clang++
│   │   ├── riscv-ld                # Wrapper script for lld
│   │   ├── riscv-objdump           # Wrapper script for llvm-objdump
│   │   ├── riscv-strip             # Wrapper script for llvm-strip
│   │   └── common.sh               # Shared functions for all wrappers
│   └── README.md                   # Toolchain documentation
├── tests/                          # Tests (NEW)
│   └── tools/                      # Tool tests
│       └── test-docker-llvm.sh     # Test script for wrapper functionality
└── docs/
    └── docker-llvm-guide.md        # Detailed usage guide (NEW)
```

**Structure Decision**: Added `tools/` directory for toolchain scripts. This separates tooling from the research platform code (future `src/`) and follows the pattern of keeping third-party integration in isolated directories.

## Complexity Tracking

> No ground-rules violations - straightforward scripting feature with clear scope.