# Technical Design: RVFuse Project Setup Foundation

**Branch**: `001-riscv-fusion-setup` | **Date**: 2026-03-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-riscv-fusion-setup/spec.md`

## Summary

Establish the RVFuse project workspace foundation: repository structure, dependency source references, optional dependency policy, and setup verification guidance. This feature produces documentation artifacts only - no source code is required for the current phase. Future research workflows (hotspot detection, DFG generation, fusion validation) are explicitly deferred.

## Technical Context

**Language/Version**: N/A (documentation-only phase)
**Primary Dependencies**: Git (for submodule integration), Markdown rendering
**Storage**: N/A
**Testing**: Documentation review (no automated tests for docs phase)
**Target Platform**: Linux x86_64 workstation (local-first development)
**Project Type**: Documentation/setup phase (no source code)
**Performance Goals**: Setup completion within 30 minutes (excluding download/build time)
**Constraints**: No profiling, DFG, or fusion-validation work in this phase
**Scale/Scope**: Single repository, 3-5 contributors initially

## Ground-rules Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Code Quality** | ⚠️ EXEMPT | No source code in this phase - documentation only |
| **II. Testing** | ⚠️ EXEMPT | No automated tests for documentation artifacts; verification via checklist review |
| **III. User Experience** | ✅ PASS | Setup guidance must be intuitive (30-min target); error messages in fallback docs actionable |
| **IV. Performance** | ✅ PASS | Setup timing target documented; excludes download/build time |

**Justification for exemptions**: This feature is documentation-only. Code Quality and Testing principles apply to future source-code features. User Experience applies to the contributor onboarding experience via setup documentation clarity. Performance applies to the documented setup timing target.

## Project Structure

### Documentation (this feature)

```text
specs/001-riscv-fusion-setup/
├── design.md            # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (entity definitions)
├── quickstart.md        # Phase 1 output (setup guide)
├── contracts/           # Not applicable (no APIs in this phase)
└── tasks.md             # Phase 2 output (via /rainbow.taskify)
```

### Source Code (repository root)

No source code in current phase. Future phases will introduce:

```text
# DEFERRED - Future workspace layout
src/                     # Python analysis tools (future)
tests/                   # Test suite (future)
builds/                  # Compiled target applications (future)
results/                 # Analysis outputs (future)
configs/                 # Configuration files (future)
```

### Current-Phase Repository Structure

```text
RVFuse/
├── docs/                   # Architecture and contributor-facing documents
│   ├── architecture.md     # System architecture (ADR-001 to ADR-004)
│   └── setup-guide.md      # Contributor setup instructions
├── specs/                  # Feature specifications
│   └── 001-riscv-fusion-setup/
│       ├── spec.md
│       ├── design.md
│       ├── research.md
│       ├── data-model.md
│       ├── quickstart.md
│       └── checklists/
│           └── requirements.md
├── memory/                 # Project governance
│   ├── ground-rules.md
│   └── MEMORY.md
├── third_party/            # External dependency integration
│   ├── qemu/               # Xuantie QEMU (mandatory)
│   ├── llvm-project/       # Xuantie LLVM (mandatory)
│   └── newlib/             # Xuantie newlib (optional)
└── .rainbow/               # Rainbow workflow scripts
```

**Structure Decision**: Current phase defines only docs/, specs/, memory/, third_party/, and .rainbow/. Deferred areas (src/, tests/, builds/, results/, configs/) remain undocumented in workspace layout to avoid premature commitment.

## Architecture Alignment

| ADR | Alignment |
|-----|-----------|
| ADR-001 | Git submodules via `third_party/` directory - no change needed |
| ADR-002 | Stage delivery - this feature is Stage 1 (setup), future stages follow |
| ADR-003 | newlib optional - dependency policy reflects this |
| ADR-004 | Traceable workload references - quickstart.md references real sources only |

## Complexity Tracking

> No ground-rules violations requiring justification - documentation-only feature.