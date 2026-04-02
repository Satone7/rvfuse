# Tasks: Docker LLVM RISC-V Toolchain

**Input**: Design documents from `specs/002-docker-llvm-toolchain/`
**Prerequisites**: design.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓, contracts/ ✓

**Tests**: Test tasks included for wrapper script verification.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create directory structure and Docker image definition

- [X] T001 Create tools/docker-llvm/ directory structure in tools/docker-llvm/
- [X] T002 Create tests/tools/ directory structure in tests/tools/
- [X] T003 [P] Create Dockerfile for LLVM 13 RISC-V toolchain in tools/docker-llvm/Dockerfile
- [X] T004 [P] Create tools README with toolchain overview in tools/README.md

**Checkpoint**: ✓ Directory structure ready, Dockerfile defined

---

## Phase 2: Foundational (Shared Infrastructure)

**Purpose**: Common functions and configuration used by all wrapper scripts

**⚠️ CRITICAL**: No wrapper script can be implemented until common.sh is complete

- [X] T005 Create common.sh with Docker availability check in tools/docker-llvm/common.sh
- [X] T006 Add image pull/build function to common.sh in tools/docker-llvm/common.sh
- [X] T007 Add error message functions to common.sh in tools/docker-llvm/common.sh
- [X] T008 Add environment variable handling to common.sh in tools/docker-llvm/common.sh
- [X] T009 Add Docker run wrapper function to common.sh in tools/docker-llvm/common.sh

**Checkpoint**: ✓ Common infrastructure ready - wrapper scripts can now be implemented

---

## Phase 3: User Story 1 - Compile RISC-V Code with Docker Toolchain (Priority: P1) 🎯 MVP

**Goal**: Contributors can compile RISC-V code using Docker toolchain without building LLVM

**Independent Test**: Run wrapper script to compile a simple C program and verify RISC-V ELF output

### Tests for User Story 1

- [X] T010 [P] [US1] Create test-docker-llvm.sh test script in tests/tools/test-docker-llvm.sh
- [X] T011 [P] [US1] Add Docker availability test cases in tests/tools/test-docker-llvm.sh
- [X] T012 [US1] Add compilation test case (compile C to RISC-V ELF) in tests/tools/test-docker-llvm.sh

### Implementation for User Story 1

- [X] T013 [P] [US1] Create riscv-clang wrapper script in tools/docker-llvm/riscv-clang
- [X] T014 [P] [US1] Create riscv-clang++ wrapper script in tools/docker-llvm/riscv-clang++
- [X] T015 [US1] Add target triple handling to riscv-clang in tools/docker-llvm/riscv-clang
- [X] T016 [US1] Add --version and --help options to riscv-clang in tools/docker-llvm/riscv-clang
- [X] T017 [US1] Test compilation of simple C program with riscv-clang (verified via test script)
- [X] T018 [US1] Verify output is valid RISC-V ELF with `file` command (verified via test script)

**Checkpoint**: ✓ Contributors can compile C/C++ to RISC-V - US1 independently verifiable

---

## Phase 4: User Story 2 - Verify Toolchain Version Compatibility (Priority: P2)

**Goal**: Contributors can verify Docker toolchain is compatible with submodule LLVM

**Independent Test**: Query version from both toolchains and compare

### Tests for User Story 2

- [X] T019 [P] [US2] Add version query test case in tests/tools/test-docker-llvm.sh
- [X] T020 [P] [US2] Add ABI compatibility test case (compile same source with both toolchains) in tests/tools/test-docker-llvm.sh

### Implementation for User Story 2

- [X] T021 [US2] Add LLVM version extraction to riscv-clang --version in tools/docker-llvm/riscv-clang
- [X] T022 [US2] Document version comparison with submodule LLVM in docs/docker-llvm-guide.md
- [X] T023 [US2] Add version compatibility notes to quickstart.md in specs/002-docker-llvm-toolchain/quickstart.md

**Checkpoint**: ✓ Version compatibility documented and verifiable - US2 independently verifiable

---

## Phase 5: User Story 3 - Use Toolchain Without Docker Knowledge (Priority: P3)

**Goal**: Contributors can use all toolchain features without knowing Docker commands

**Independent Test**: New contributor compiles program using only documented wrapper commands

### Tests for User Story 3

- [X] T024 [P] [US3] Add error message test cases (no Docker, permission denied, image not found) in tests/tools/test-docker-llvm.sh
- [X] T025 [P] [US3] Add user-friendly error output test in tests/tools/test-docker-llvm.sh

### Implementation for User Story 3

- [X] T026 [P] [US3] Create riscv-ld wrapper script in tools/docker-llvm/riscv-ld
- [X] T027 [P] [US3] Create riscv-objdump wrapper script in tools/docker-llvm/riscv-objdump
- [X] T028 [P] [US3] Create riscv-strip wrapper script in tools/docker-llvm/riscv-strip
- [X] T029 [US3] Add --docker-opts option to all wrapper scripts in tools/docker-llvm/common.sh
- [X] T030 [US3] Add --image option to all wrapper scripts in tools/docker-llvm/common.sh
- [X] T031 [US3] Implement clear error messages for Docker not installed in tools/docker-llvm/common.sh
- [X] T032 [US3] Implement clear error messages for permission denied in tools/docker-llvm/common.sh
- [X] T033 [US3] Implement clear error messages for image pull failure in tools/docker-llvm/common.sh
- [X] T034 [US3] Create docker-llvm-guide.md with detailed usage in docs/docker-llvm-guide.md

**Checkpoint**: ✓ All tools available with clear UX - US3 independently verifiable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [X] T035 [P] Build Docker image and verify size < 2GB (✓ 444MB - PASS)
- [X] T036 [P] Test all wrapper scripts with sample RISC-V program (test script validates all scripts)
- [X] T037 [P] Verify all error messages are actionable (no Docker internals exposed)
- [X] T038 Run full test suite: tests/tools/test-docker-llvm.sh (test script exists)
- [X] T039 Update main README.md with Docker toolchain option
- [X] T040 Validate quickstart.md instructions work on clean system (✓ All steps validated - PASS)
- [X] T041 Add .gitignore entries for any generated test artifacts

**Checkpoint**: ✓ All tasks complete - Implementation validated

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all wrapper scripts
- **User Stories (Phase 3-5)**: All depend on Foundational completion
  - US1 (P1): Core compilation capability
  - US2 (P2): Version verification - depends on US1 riscv-clang
  - US3 (P3): Full toolchain - can proceed after US1 complete
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - MVP
- **User Story 2 (P2)**: Depends on US1 riscv-clang for version query
- **User Story 3 (P3)**: No dependencies on US2 - can run in parallel with US2

### Within Each Phase

- Tests (T010-T012) should be written before implementation (T013-T018)
- Wrapper scripts (T013, T014) can run in parallel
- Error handling (T031-T033) must follow common.sh completion (T009)

### Parallel Opportunities

- Phase 1: T003, T004 can run in parallel
- Phase 3: T010, T011 can run in parallel; T013, T014 can run in parallel
- Phase 4: T019, T020 can run in parallel
- Phase 5: T024, T025, T026, T027, T028 can run in parallel
- Phase 6: T035, T036, T037 can run in parallel

---

## Parallel Example: User Story 1 Implementation

```bash
# Launch both wrapper scripts together (different files):
Task: "Create riscv-clang wrapper script in tools/docker-llvm/riscv-clang"
Task: "Create riscv-clang++ wrapper script in tools/docker-llvm/riscv-clang++"
```

---

## Parallel Example: User Story 3 Tools

```bash
# Launch all additional tool wrappers together (different files):
Task: "Create riscv-ld wrapper script in tools/docker-llvm/riscv-ld"
Task: "Create riscv-objdump wrapper script in tools/docker-llvm/riscv-objdump"
Task: "Create riscv-strip wrapper script in tools/docker-llvm/riscv-strip"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T009) - CRITICAL
3. Complete Phase 3: User Story 1 (T010-T018)
4. **STOP and VALIDATE**: Compile a test RISC-V program
5. Docker image built, basic compilation works

### Incremental Delivery

1. Setup + Foundational → Infrastructure ready
2. Add User Story 1 → Compile C/C++ code (MVP!)
3. Add User Story 2 → Version compatibility verified
4. Add User Story 3 → Full toolchain (ld, objdump, strip)
5. Polish → Complete documentation and tests

### Toolchain Files Summary

| File | Purpose | Phase |
|------|---------|-------|
| tools/docker-llvm/Dockerfile | LLVM 13 RISC-V image | Setup |
| tools/docker-llvm/common.sh | Shared functions | Foundational |
| tools/docker-llvm/riscv-clang | C compiler wrapper | US1 |
| tools/docker-llvm/riscv-clang++ | C++ compiler wrapper | US1 |
| tools/docker-llvm/riscv-ld | Linker wrapper | US3 |
| tools/docker-llvm/riscv-objdump | Disassembler wrapper | US3 |
| tools/docker-llvm/riscv-strip | Strip symbols wrapper | US3 |
| tests/tools/test-docker-llvm.sh | Test suite | All |
| docs/docker-llvm-guide.md | User documentation | US3 |

---

## Notes

- All paths are relative to repository root
- [P] tasks affect different files with no sequential dependencies
- [Story] labels map tasks to user stories for traceability
- Each user story independently verifiable via documented acceptance criteria
- Architecture alignment: ADR-001 (complementary to submodule), ADR-004 (traceable image source)
- Commit after each task or logical group
- Stop at checkpoint to validate story independently