# RVFuse

RISC-V Instruction Fusion Research Platform

## Overview

RVFuse is a research platform for discovering and validating RISC-V instruction fusion candidates. The current phase establishes the project workspace, dependency references, and setup baseline.

## Quick Start

See the [setup quickstart guide](specs/001-riscv-fusion-setup/quickstart.md) for detailed setup instructions.

**Time target**: 30 minutes (excluding network download and third-party build time)

## Project Structure

```text
RVFuse/
├── docs/                   # Architecture and project documents
├── specs/                  # Feature specifications
├── memory/                 # Project governance
├── third_party/            # External dependencies (submodules)
│   ├── qemu/               # Xuantie QEMU (mandatory)
│   ├── llvm-project/       # Xuantie LLVM (mandatory)
│   └── newlib/             # Xuantie newlib (optional)
└── .rainbow/               # Workflow automation scripts
```

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| Xuantie QEMU | https://github.com/XUANTIE-RV/qemu | Mandatory |
| Xuantie LLVM | https://github.com/XUANTIE-RV/llvm-project | Mandatory (Alternative available) |
| Xuantie newlib | https://github.com/XUANTIE-RV/newlib | Optional |

> **Note on LLVM**: If your hardware cannot build the `llvm-project` submodule from source, you can use the [Docker LLVM Toolchain](docs/docker-llvm-guide.md) provided in `tools/docker-llvm/` as a lightweight alternative.

## Current Phase Scope

- Repository structure and documentation
- Dependency source references
- Setup verification guidance

**Deferred to future phases**: Hotspot detection, DFG generation, fusion candidate analysis, cycle validation

## Documentation

- [Architecture](docs/architecture.md) - System design and ADRs
- [Ground Rules](memory/ground-rules.md) - Development principles
- [Feature Spec](specs/001-riscv-fusion-setup/spec.md) - Current feature requirements

## License

[License to be determined]