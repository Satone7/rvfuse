# Data Model: Automated Setup Flow Script

**Date**: 2026-04-06 | **Branch**: `003-automated-setup-flow`

This document describes the data structures used by the setup flow script. Since the script operates on files and directories (no database), entities are represented as plain-text state during execution and persisted via the report file.

## Entities

### SetupStep

Represents a single phase in the setup flow.

| Field | Type | Description |
|-------|------|-------------|
| number | integer (1-5) | Step identifier |
| name | string | Human-readable step title |
| artifact_checks | list of path | Files/directories that indicate step completion |
| execute | function reference | Bash function that performs the step |

**Defined Steps**:

| # | Name | Artifact Checks |
|---|------|----------------|
| 1 | Clone Repository | `.git/` directory exists in project root |
| 2 | Review Project Scope | `docs/architecture.md`, `memory/ground-rules.md`, `specs/001-riscv-fusion-setup/spec.md` all exist and are non-empty |
| 3 | Initialize Mandatory Dependencies | `third_party/qemu/.git`, `third_party/llvm-project/.git` exist |
| 4 | Verify Setup Completion | All 7 quickstart verification checks pass (structural + content) |
| 5 | Generate Report | Report file written successfully to project root |

### StepResult

Execution outcome for a single step.

| Field | Type | Description |
|-------|------|-------------|
| step_number | integer (1-5) | References SetupStep.number |
| status | enum | One of: `PASS`, `FAIL`, `SKIPPED` |
| message | string | Human-readable outcome or error description |
| started_at | timestamp | ISO 8601 timestamp when step began |
| finished_at | timestamp | ISO 8601 timestamp when step completed |

### SetupReport

Aggregated results for the entire setup flow, persisted as a plain-text file.

| Field | Type | Description |
|-------|------|-------------|
| overall_status | enum | `PASS` (all mandatory steps passed), `FAIL` (any mandatory step failed) |
| step_results | list of StepResult | One entry per step (1-5) |
| generated_at | timestamp | ISO 8601 timestamp when report was written |
| script_options | string | CLI flags that were used (e.g., `--force 3 --shallow`) |

**Report File**: Saved to `<project_root>/setup-report.txt`

## State Transitions

```
[Step N not started]
       │
       ▼
  ┌─ artifacts exist AND not forced? ──┐
  │           YES                       │ NO
  │            │                        │
  │            ▼                        │
  │      SKIPPED                       ▼
  │                                  [Executing]
  │                                       │
  │                            ┌──── success? ────┐
  │                            │                   │
  │                           YES                  NO
  │                            │                   │
  │                            ▼                   ▼
  │                          PASS                FAIL
  └──────────────────────────────────────────────────┘
```

**Failure propagation**: If any mandatory step (1-4) returns FAIL, the script records the failure in the report but continues executing remaining steps to collect as much diagnostic information as possible. The overall report status is FAIL. Step 5 (Generate Report) always executes unless the script itself encounters an unrecoverable error (e.g., cannot write to disk).

## Validation Rules

- Step numbers must be in range 1-5; invalid `--force` values produce an error and halt
- Artifact paths are relative to project root; absolute paths are rejected
- Report file is overwritten on each run (no accumulation across runs)
- Empty artifact checks (no artifacts defined) always cause the step to execute
