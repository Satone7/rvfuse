# Tasks: RVFuse Project Setup Foundation

**Input**: Design documents from `specs/001-riscv-fusion-setup/`
**Prerequisites**: design.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓

**Tests**: Not applicable - documentation-only feature with verification checklist review.

**Organization**: Tasks grouped by user story for independent verification of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create repository structure and initialize documentation framework

- [ ] T001 Create current-phase directory structure: docs/, specs/, memory/, third_party/, .rainbow/
- [ ] T002 [P] Initialize .gitignore with third_party/ submodule entries in .gitignore
- [ ] T003 [P] Create README.md with project overview and quickstart link at README.md

**Checkpoint**: Repository structure exists; ready for documentation work

---

## Phase 2: Foundational (Architecture Documentation)

**Purpose**: Core architecture documents that MUST exist before user story work

**⚠️ CRITICAL**: No user story verification can proceed until architecture is documented

- [ ] T004 Create architecture.md with current-phase scope and deferred scope in docs/architecture.md
- [ ] T005 [P] Document ADR-001 (Git submodules) with context/decision/consequences in docs/architecture.md
- [ ] T006 [P] Document ADR-002 (Stage delivery) with context/decision/consequences in docs/architecture.md
- [ ] T007 [P] Document ADR-003 (newlib optional) with context/decision/consequences in docs/architecture.md
- [ ] T008 [P] Document ADR-004 (Traceable references) with context/decision/consequences in docs/architecture.md
- [ ] T009 Create ground-rules.md with 4 core principles in memory/ground-rules.md
- [ ] T010 [P] Document quality targets (Setup Clarity, Reproducibility, Reliability, Scope Control) in docs/architecture.md
- [ ] T011 [P] Document risks and technical debt table in docs/architecture.md
- [ ] T012 Document dependency catalog with mandatory/optional status in docs/architecture.md

**Checkpoint**: Architecture complete - user story documentation can now begin

---

## Phase 3: User Story 1 - Prepare Project Baseline (Priority: P1) 🎯 MVP

**Goal**: Contributor can identify required project structure and setup scope from documentation

**Independent Test**: New contributor reads docs and identifies structure/scope without assumptions

### Documentation for User Story 1

- [ ] T013 [P] [US1] Create specification spec.md with US1 acceptance scenarios in specs/001-riscv-fusion-setup/spec.md
- [ ] T014 [P] [US1] Document workspace layout entity (current areas, deferred areas) in specs/001-riscv-fusion-setup/data-model.md
- [ ] T015 [US1] Create design.md with technical context and ground-rules check in specs/001-riscv-fusion-setup/design.md
- [ ] T016 [US1] Document repository structure section in docs/architecture.md (current-phase layout)
- [ ] T017 [US1] Add workspace structure to quickstart.md Step 1 verification in specs/001-riscv-fusion-setup/quickstart.md

**Checkpoint**: Contributor can identify structure from docs - US1 independently verifiable

---

## Phase 4: User Story 2 - Understand Dependency Sources (Priority: P2)

**Goal**: Contributor can distinguish mandatory vs optional dependencies and find upstream sources

**Independent Test**: Contributor reviews docs and identifies all mandatory/optional deps with sources

### Documentation for User Story 2

- [ ] T018 [P] [US2] Document Mandatory Dependency entity with QEMU and LLVM instances in specs/001-riscv-fusion-setup/data-model.md
- [ ] T019 [P] [US2] Document Optional Dependency entity with newlib instance and activation condition in specs/001-riscv-fusion-setup/data-model.md
- [ ] T020 [P] [US2] Document Dependency Source Reference entity in specs/001-riscv-fusion-setup/data-model.md
- [ ] T021 [US2] Add dependency section to quickstart.md Step 3 with submodule commands in specs/001-riscv-fusion-setup/quickstart.md
- [ ] T022 [US2] Document Key Dependencies table with URLs in docs/architecture.md
- [ ] T023 [US2] Add dependency verification to quickstart.md Step 4 checklist in specs/001-riscv-fusion-setup/quickstart.md

**Checkpoint**: Dependencies clearly documented - US2 independently verifiable

---

## Phase 5: User Story 3 - Verify Contributor Readiness (Priority: P3)

**Goal**: Contributor can verify setup completion using documented criteria without future capabilities

**Independent Test**: Contributor completes setup and runs 7 verification checks successfully

### Documentation for User Story 3

- [ ] T024 [P] [US3] Document Setup Verification Criteria entity with VC-001 to VC-007 in specs/001-riscv-fusion-setup/data-model.md
- [ ] T025 [US3] Create verification checklist table in quickstart.md Step 4 in specs/001-riscv-fusion-setup/quickstart.md
- [ ] T026 [US3] Add success criteria to spec.md (SC-001 to SC-005) in specs/001-riscv-fusion-setup/spec.md
- [ ] T027 [US3] Document timing exclusions (download/build time) in quickstart.md header in specs/001-riscv-fusion-setup/quickstart.md
- [ ] T028 [US3] Add troubleshooting section to quickstart.md in specs/001-riscv-fusion-setup/quickstart.md
- [ ] T029 [US3] Create requirements checklist in specs/001-riscv-fusion-setup/checklists/requirements.md

**Checkpoint**: Verification criteria complete - US3 independently verifiable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final review and consistency checks across all documentation

- [ ] T030 [P] Review architecture.md for mandatory/optional consistency across sections
- [ ] T031 [P] Review spec.md for alignment with architecture.md ADRs
- [ ] T032 [P] Review data-model.md entity validation rules match spec requirements
- [ ] T033 Run quickstart.md verification checklist against repository state
- [ ] T034 Update CLAUDE.md agent context with feature scope
- [ ] T035 [P] Create research.md documenting design decisions and alternatives in specs/001-riscv-fusion-setup/research.md
- [ ] T036 Validate all 7 verification criteria are achievable

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user story documentation
- **User Stories (Phase 3-5)**: All depend on Foundational completion
  - US1, US2, US3 can proceed in parallel if staffed
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - independently verifiable
- **User Story 2 (P2)**: No dependencies on US1 - independently verifiable
- **User Story 3 (P3)**: No dependencies on US1/US2 - independently verifiable

### Within Each Phase

- Tasks marked [P] can run in parallel (different files)
- Tasks without [P] may have sequential dependencies
- Story complete before moving to next priority

### Parallel Opportunities

- Phase 1: T002, T003 can run in parallel
- Phase 2: T005-T011 can run in parallel (different sections of architecture.md)
- Phase 3: T013, T014 can run in parallel
- Phase 4: T018, T019, T020 can run in parallel
- Phase 5: T024 can run in parallel with T025-T028 work
- Phase 6: T030-T032, T035 can run in parallel

---

## Parallel Example: Phase 2 Foundational

```bash
# Launch all ADR documentation tasks together (different sections):
Task: "Document ADR-001 (Git submodules) with context/decision/consequences in docs/architecture.md"
Task: "Document ADR-002 (Stage delivery) with context/decision/consequences in docs/architecture.md"
Task: "Document ADR-003 (newlib optional) with context/decision/consequences in docs/architecture.md"
Task: "Document ADR-004 (Traceable references) with context/decision/consequences in docs/architecture.md"
```

---

## Parallel Example: User Story 2

```bash
# Launch all entity documentation tasks together (different sections of data-model.md):
Task: "Document Mandatory Dependency entity with QEMU and LLVM instances in specs/001-riscv-fusion-setup/data-model.md"
Task: "Document Optional Dependency entity with newlib instance in specs/001-riscv-fusion-setup/data-model.md"
Task: "Document Dependency Source Reference entity in specs/001-riscv-fusion-setup/data-model.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T012) - CRITICAL
3. Complete Phase 3: User Story 1 (T013-T017)
4. **STOP and VALIDATE**: Contributor can identify structure from docs
5. Review architecture.md consistency

### Incremental Delivery

1. Setup + Foundational → Architecture documented
2. Add User Story 1 → Verify independently → MVP complete
3. Add User Story 2 → Verify independently → Dependencies clear
4. Add User Story 3 → Verify independently → Setup verification ready
5. Polish → Cross-document consistency validated

### Documentation-Only Note

Since this is a documentation-only feature:
- No source code tasks
- No automated tests
- Verification via checklist review (human validation)
- Commit documentation artifacts after each phase

---

## Notes

- All paths are absolute or relative to repository root
- [P] tasks affect different files/sections with no sequential dependencies
- [Story] labels map tasks to user stories for traceability
- Each user story independently verifiable via documented acceptance criteria
- Architecture alignment: ADR-001 to ADR-004 reflected in all documentation
- Commit after each task or logical group
- Stop at checkpoint to validate story independently