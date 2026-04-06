# Tasks: Automated Setup Flow Script

**Input**: Design documents from `specs/003-automated-setup-flow/`
**Prerequisites**: design.md, spec.md, research.md, data-model.md

**Tests**: Included per ground-rules Principle II (all features must have automated tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single bash script at repository root. Tests in `tests/setup/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directory structure and configure testing framework

- [x] T001 Create test directory structure `tests/setup/`
- [x] T002 Install bats-core testing framework (single-file install to `tests/setup/test_helper/bats-core/`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core script infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Create `setup.sh` with shebang (`#!/usr/bin/env bash`), `set -euo pipefail`, and script-level constants (`REPORT_FILE`, `MIN_DISK_GB`, `MIN_GIT_VERSION`) in `setup.sh`
- [x] T004 Implement CLI argument parsing function `parse_args()` supporting `--force <steps>`, `--force-all`, `--shallow`, `--help` in `setup.sh`
- [x] T005 Implement project root detection function `detect_project_root()` using `git rev-parse --show-toplevel` with error message in `setup.sh`
- [x] T006 Implement prerequisite checking function `check_prerequisites()` validating git version >= 2.30 and disk space >= 20GB in `setup.sh`
- [x] T007 Implement generic artifact checking function `check_artifacts()` that takes a list of paths and returns 0 if all exist in `setup.sh`
- [x] T008 Implement step result recording mechanism: associative arrays for `STEP_STATUS`, `STEP_MESSAGE`, `STEP_STARTED`, `STEP_FINISHED` initialized in `setup.sh`
- [x] T009 Implement logging functions `log_info()`, `log_warn()`, `log_error()` that print prefixed messages to stdout/stderr in `setup.sh`

**Checkpoint**: Foundation ready — all utility functions implemented and callable. User story implementation can begin.

---

## Phase 3: User Story 1 - Run Full Setup in One Command (Priority: P1) 🎯 MVP

**Goal**: A contributor runs `./setup.sh` and the full 5-step flow executes, skipping already-completed steps, producing a basic report file.

**Independent Test**: Run `./setup.sh` on a clean workspace — all steps execute in order and `setup-report.txt` is created with pass/skip status for each step.

### Tests for User Story 1

- [x] T010 [P] [US1] Write test for `detect_project_root()` — success inside git repo, failure outside — in `tests/setup/test_prereqs.sh`
- [x] T011 [P] [US1] Write test for `check_prerequisites()` — git version check, disk space check — in `tests/setup/test_prereqs.sh`
- [x] T012 [P] [US1] Write test for `check_artifacts()` — all exist, some missing, empty file — in `tests/setup/test_artifacts.sh`

### Implementation for User Story 1

- [x] T013 [US1] Implement Step 1 function `step1_clone()` checking `.git/` directory existence in `setup.sh`
- [x] T014 [US1] Implement Step 2 function `step2_review_scope()` checking `docs/architecture.md`, `memory/ground-rules.md`, `specs/001-riscv-fusion-setup/spec.md` exist and are non-empty in `setup.sh`
- [x] T015 [US1] Implement Step 3 function `step3_init_deps()` running `git submodule update --init` for qemu and llvm-project (with `--depth 1` if `--shallow` set) in `setup.sh`
- [x] T016 [US1] Implement Step 4 function `step4_verify_setup()` running all 7 quickstart verification checks (repo structure, architecture readable, setup guide readable, ground-rules readable, mandatory deps documented, optional deps documented, dependency sources traceable) in `setup.sh`
- [x] T017 [US1] Implement Step 5 function `step5_report()` writing structured plain-text report to `setup-report.txt` in `setup.sh`
- [x] T018 [US1] Implement main execution loop `run_setup()` that iterates steps 1-5, calls `check_artifacts()` for each, skips if artifacts exist (unless forced), and calls the step function in `setup.sh`
- [x] T019 [US1] Wire `main()` entry point: call `detect_project_root`, `check_prerequisites`, `parse_args`, `run_setup`, then exit 0 (all pass) or 1 (any fail) in `setup.sh`

### Integration Test for User Story 1

- [x] T020 [US1] Write integration tests for full setup flow (mocked submodule commands) verifying all steps execute and report is generated — covered across `test_force.sh` and `test_report.sh`

**Checkpoint**: User Story 1 complete — `./setup.sh` runs the full flow with artifact-based skip logic and generates `setup-report.txt`.

---

## Phase 4: User Story 2 - Force Re-execute Specific Steps (Priority: P2)

**Goal**: Contributors can use `--force 3` or `--force 2,4` to selectively re-run steps, or `--force-all` to re-run everything.

**Independent Test**: Run `./setup.sh --force 3` on a fully completed workspace — only Step 3 re-executes, others are left untouched.

### Tests for User Story 2

- [x] T021 [P] [US2] Write test for `parse_args()` — `--force 3`, `--force 2,4`, `--force-all`, `--shallow`, invalid step number — in `tests/setup/test_args.sh`
- [x] T022 [P] [US2] Write test for force re-execution: `--force 3` skips non-forced steps and re-runs Step 3 in `tests/setup/test_force.sh`

### Implementation for User Story 2

- [x] T023 [US2] Integrate force list with `run_setup()` skip logic: check step against `FORCE_STEPS` array before deciding to skip in `setup.sh`
- [x] T024 [US2] Implement `--force-all` behavior: populate `FORCE_STEPS` with all step numbers (1-5) in `parse_args()` in `setup.sh`
- [x] T025 [US2] Add invalid step number validation in `parse_args()`: if `--force` value contains numbers outside 1-5, print error and exit 1 in `setup.sh`

### Integration Test for User Story 2

- [x] T026 [US2] Write integration test verifying `--force-all` re-executes all steps regardless of existing artifacts in `tests/setup/test_force.sh`

**Checkpoint**: User Stories 1 AND 2 complete — full flow works with skip logic and force re-execution.

---

## Phase 5: User Story 3 - Review Setup Completion Report (Priority: P3)

**Goal**: The report file contains rich details — per-step status, warnings, error messages with hints, options used, and timestamp — enabling contributors to audit setup without re-running.

**Independent Test**: Run `./setup.sh`, open `setup-report.txt`, and verify it lists every step with pass/fail/skipped status plus any warnings and the options used.

### Tests for User Story 3

- [x] T027 [P] [US3] Write test for `step5_report()` — report contains header, all 5 step lines with status, overall result, timestamp, and options in `tests/setup/test_report.sh`
- [x] T028 [P] [US3] Write test for report failure format — when a step fails, report includes error message and hint line in `tests/setup/test_report.sh`

### Implementation for User Story 3

- [x] T029 [US3] Enhance `step5_report()` report format: add "Generated:" timestamp, "Options:" line reflecting CLI flags used, and detailed per-step messages (e.g., "(7/7 checks)", "(already cloned)") in `setup.sh`
- [x] T030 [US3] Enhance `step5_report()` failure format: include indented "Hint:" lines under failed steps in `setup.sh`
- [x] T031 [US3] Ensure Step 4 sub-check details are included in report (e.g., which verification checks passed/failed) in `setup.sh`

**Checkpoint**: All user stories complete — setup script fully functional with rich, reviewable reports.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final quality, documentation, and validation

- [x] T032 [P] Add `setup-report.txt` to `.gitignore`
- [x] T033 Run `shellcheck setup.sh` and fix all warnings
- [x] T034 [P] Add `chmod +x setup.sh` and verify invocation from quickstart.md matches actual script behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (P1) must complete before US2 and US3 (they extend US1's step functions)
  - US2 (P2) can start after US1 — extends argument parsing and run loop
  - US3 (P3) can start after US1 — extends report generation
  - US2 and US3 can proceed in parallel after US1 (different functions touched)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) — No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 — extends `parse_args()` and `run_setup()` from US1
- **User Story 3 (P3)**: Depends on US1 — extends `step5_report()` from US1

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Step functions before main loop
- Core implementation before integration test
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks (T001, T002) can run in parallel
- All Foundational tasks (T003-T009) touch the same file — sequential within file, but T004-T009 can be planned in parallel as they define independent functions
- US1 tests (T010, T011, T012) can run in parallel
- US2 tests (T021, T022) can run in parallel
- US3 tests (T027, T028) can run in parallel
- Polish tasks (T032, T033, T034) can run in parallel
- US2 and US3 can be developed in parallel by different agents after US1 completes

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together:
Task T010: "test detect_project_root() in tests/setup/test_prereqs.sh"
Task T011: "test check_prerequisites() in tests/setup/test_prereqs.sh"
Task T012: "test check_artifacts() in tests/setup/test_artifacts.sh"

# Then implement step functions (sequential — same file):
Task T013-T017: Step 1-5 functions in setup.sh

# Then main loop and integration:
Task T018: run_setup() loop
Task T019: main() entry point
Task T020: integration test
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T009)
3. Complete Phase 3: User Story 1 (T010-T020)
4. **STOP and VALIDATE**: Run `./setup.sh` end-to-end, verify report generated
5. MVP is functional — contributor can run one-command setup

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Full flow works → MVP!
3. Add User Story 2 → Force re-execution works
4. Add User Story 3 → Rich report available for audit
5. Polish → Production-quality script

---

## Architecture Alignment Notes

- **ADR-001** (Git Submodules): Step 3 (`step3_init_deps`) uses `git submodule update --init` per ADR-001
- **ADR-002** (Deliver in Stages): Script covers only current-phase setup steps; no profiling/DFG/fusion code
- **ADR-003** (newlib Optional): Step 3 does not include newlib; only qemu and llvm-project are mandatory submodules
- **ADR-004** (Traceable Sources): Dependency URLs are hardcoded as constants in `setup.sh`

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All functions must be <50 lines per ground-rules Principle I
- `set -euo pipefail` ensures strict error handling
- Report file is plain text (not markdown, not JSON) per research decision R-003
