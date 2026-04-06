# Technical Design: Automated Setup Flow Script

**Branch**: `003-automated-setup-flow` | **Date**: 2026-04-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/003-automated-setup-flow/spec.md`

## Summary

Create a single bash script (`setup.sh`) that automates the RVFuse 5-step setup flow from the quickstart guide. The script checks for existing artifacts before each step, skips completed steps, and accepts CLI flags (`--force`, `--force-all`, `--shallow`) for selective re-execution. A completion report is generated as `setup-report.txt` in the project root.

## Technical Context

**Language/Version**: Bash 4.0+ (available on all modern Linux x86_64)
**Primary Dependencies**: Git 2.30+, standard Unix utilities (ls, cat, grep, df, date)
**Storage**: File system — report persisted as `setup-report.txt`
**Testing**: bats-core (bash testing framework)
**Target Platform**: Linux x86_64
**Project Type**: Single CLI script
**Performance Goals**: Not applicable (setup script, not runtime)
**Constraints**: Git 2.30+, ~20GB disk space, GitHub network access
**Scale/Scope**: Single-user local setup script, 5 steps

## Ground-rules Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Code Quality | Compliant | Functions single-purpose, <50 lines, self-documenting, explicit error handling |
| II. Testing | Compliant | bats-core unit tests for artifact checks, prerequisite validation, and report generation; integration test for full flow |
| III. User Experience | Compliant | Actionable error messages, progress output per step, report file for audit |
| IV. Performance | N/A | Setup script; no runtime performance targets |

**Post-design re-check**: No violations. All principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/003-automated-setup-flow/
├── design.md            # This file
├── research.md          # Phase 0 research decisions
├── data-model.md        # Step/result/report entities
├── quickstart.md        # User-facing setup script guide
├── spec.md              # Feature specification
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
setup.sh                # Main entry point
setup-report.txt        # Generated report (not committed)
tests/
└── setup/
    ├── test_args.sh        # CLI argument parsing tests
    ├── test_prereqs.sh     # Prerequisite checking tests
    ├── test_artifacts.sh   # Artifact detection tests
    ├── test_force.sh       # Force re-execution tests
    └── test_report.sh      # Report generation tests
```

**Structure Decision**: Single `setup.sh` file with internal functions (no library splitting). The script is ~570 lines, organized by step number with clear section headers. Splitting into multiple files would add complexity without benefit at this scale. Tests are colocated in `tests/setup/`.

## Design

### CLI Interface

```
usage: setup.sh [--force <steps>] [--force-all] [--shallow]

Options:
  --force <steps>   Re-execute specified steps (comma-separated, e.g., "3" or "2,4")
  --force-all       Re-execute all steps from scratch
  --shallow         Use --depth 1 for submodule clones (faster, less disk)
  --help            Show this help message
```

### Execution Flow

```
1. Parse CLI arguments
2. Validate project root (git rev-parse --show-toplevel)
3. Check prerequisites (git version >= 2.30, disk space >= 20GB)
4. For each step 1-5:
   a. If step is in force list OR artifacts not found:
      - Execute step
      - Record result (PASS/FAIL) with timestamp
   b. Else:
      - Record SKIPPED
5. Execute Step 5 (Generate Report) — always runs unless fatal error
6. Write report to setup-report.txt
7. Exit 0 if all mandatory steps PASS, exit 1 otherwise
```

### Step Implementation Details

**Step 1 — Clone Repository**:
- Checks: `.git/` directory exists
- If not present and not forced: warn that this script must be run from within the cloned repo
- If forced on existing clone: warn that Step 1 is already satisfied, skip

**Step 2 — Review Project Scope**:
- Checks: `docs/architecture.md`, `memory/ground-rules.md`, `specs/001-riscv-fusion-setup/spec.md` exist and are non-empty
- This step is informational; "execution" verifies the files exist
- If files missing: report FAIL with list of missing files

**Step 3 — Initialize Mandatory Dependencies**:
- Checks: `third_party/qemu/.git`, `third_party/llvm-project/.git` exist
- Execution: `git submodule update --init third_party/qemu third_party/llvm-project` (add `--depth 1` if `--shallow`)
- Optional newlib is not included (per ADR-003)

**Step 4 — Verify Setup Completion**:
- Checks: All 7 quickstart verification checks pass
- Execution: Runs each check and records individual pass/fail
- Checks: repo structure, architecture readable, setup guide readable, ground-rules readable, mandatory deps documented, optional deps documented, dependency sources traceable

**Step 5 — Generate Report**:
- Always executes (unless unrecoverable script error)
- Writes structured plain-text report to `setup-report.txt`

### Report Format

```
RVFuse Setup Report
Generated: 2026-04-06T14:30:00+08:00
Options: shallow

Step 1: Clone Repository        [SKIPPED] (already cloned)
Step 2: Review Project Scope    [PASS]
Step 3: Initialize Dependencies [PASS]
Step 4: Verify Setup             [PASS]  (7/7 checks)
Step 5: Generate Report          [PASS]

Overall: PASS
```

Failure example:
```
Step 3: Initialize Dependencies         [FAIL]  failed to initialize: third_party/qemu
  Hint: Check network connectivity and GitHub status

Overall: FAIL
```

### Error Handling

- **Prerequisites fail**: Print actionable error, exit 1 immediately
- **Step fails**: Record failure in report, continue to next step (collect diagnostics)
- **Force with invalid step number**: Print error listing valid steps (1-5), exit 1
- **Not in project root**: Print error with expected path, exit 1
- **Report write fails**: Print error to stderr, exit 1

### Architecture Alignment

| ADR | Alignment |
|-----|-----------|
| ADR-001 (Git Submodules) | Step 3 uses `git submodule update --init` |
| ADR-002 (Deliver in Stages) | Script covers only current-phase setup steps |
| ADR-003 (newlib Optional) | Step 3 does not include newlib by default |
| ADR-004 (Traceable Sources) | Dependency URLs are hardcoded in Step 3 |
