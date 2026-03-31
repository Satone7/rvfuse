# Feature Specification: RVFuse Project Setup Foundation

**Feature Branch**: `001-riscv-fusion-setup`  
**Created**: 2026-03-31  
**Status**: Draft  
**Input**: User description: "设计该项目的目录结构并完成相关目录创建，通过添加submodule的方式clone第三方项目。该项目是为了实现RISC-V指令融合的需求发现和测试，从具体应用（如onnxruntime）出发，通过qemu模拟器定位热点函数和热点基本块，实现基于基本块的DFG生成，找到具有数据依赖关系的高频指令组合，然后在模拟器内尝试通过新指令的方式将指令组合融合成一条指令，并验证其cycle，比对效果。" This input describes the long-term project vision; the current feature scope is intentionally limited to project structure and dependency access.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare the Project Baseline (Priority: P1)

A contributor prepares the RVFuse repository so the project has a clear workspace structure and an unambiguous setup baseline for future research work.

**Why this priority**: No later research or implementation work can start until contributors share the same repository structure, setup expectations, and dependency baseline.

**Independent Test**: Can be fully tested by reviewing the setup documents and confirming that a new contributor can identify the required workspace areas and the current completion criteria for the setup phase.

**Acceptance Scenarios**:

1. **Given** a clean checkout of the repository, **When** a contributor reads the setup specification, **Then** the contributor can identify the required project structure for the current phase
2. **Given** the current setup phase, **When** a contributor reviews the setup documentation, **Then** the contributor can distinguish current setup work from future research workflow work
3. **Given** the setup baseline, **When** a contributor completes the documented setup steps, **Then** the contributor can verify completion without relying on profiling, DFG generation, or fusion validation features

---

### User Story 2 - Understand Dependency Sources (Priority: P2)

A contributor understands which external dependencies are required now, which are optional, and where their canonical upstream sources are located.

**Why this priority**: Dependency ambiguity creates setup failures, inconsistent environments, and future rework.

**Independent Test**: Can be tested by checking whether a contributor can identify the required external dependencies, optional dependencies, and their documented sources without making assumptions.

**Acceptance Scenarios**:

1. **Given** the dependency section of the project documents, **When** a contributor reviews it, **Then** the contributor can tell which dependencies are mandatory in the current phase
2. **Given** an optional dependency, **When** a contributor reviews the documentation, **Then** the contributor can understand when that dependency becomes necessary
3. **Given** any documented external dependency, **When** a contributor follows the reference information, **Then** the contributor can find its canonical upstream source

---

### User Story 3 - Verify Contributor Readiness (Priority: P3)

A new contributor uses the setup documentation to confirm readiness for the current phase without being blocked by later-stage research requirements.

**Why this priority**: The setup phase only succeeds if onboarding and verification are simple, bounded, and reproducible.

**Independent Test**: Can be tested by asking a new contributor to follow the setup guidance and confirm readiness using only the documented verification criteria for this phase.

**Acceptance Scenarios**:

1. **Given** a new contributor onboarding to RVFuse, **When** the contributor follows the setup guidance, **Then** the contributor can determine whether the setup phase is complete
2. **Given** setup timing expectations, **When** the contributor reviews the success criteria, **Then** the contributor can tell which activities are included and excluded from the setup duration
3. **Given** a future workload example is referenced in setup materials, **When** the contributor reviews that reference, **Then** the contributor can trace it back to a real source rather than an invented sample

---

### Edge Cases

- What happens when a mandatory dependency source is temporarily unavailable? The project MUST provide a documented fallback or retry path rather than leaving the dependency state ambiguous
- How does the project handle optional dependencies that are not needed in the current phase? The setup phase MUST remain completable without treating optional dependencies as blockers
- What happens when future workload examples are mentioned before a full analysis workflow exists? Any referenced workload MUST include a traceable source and MUST NOT be presented as a current-phase acceptance requirement

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST define the repository structure required for the current setup phase
- **FR-002**: The project MUST identify the external dependencies that are mandatory for the current setup phase
- **FR-003**: The project MUST preserve canonical source references for external dependencies that are optional in the current setup phase
- **FR-004**: The project MUST document the conditions under which an optional dependency becomes required
- **FR-005**: The project MUST provide contributor-facing setup guidance for the current phase
- **FR-006**: The project MUST provide contributor-facing verification guidance for the current phase
- **FR-007**: The project MUST distinguish current setup capabilities from later research workflow capabilities
- **FR-008**: Any future workload or benchmark example referenced in current-phase documentation MUST have a traceable source

### Key Entities

- **Project Workspace**: The repository structure and documented areas that define the setup baseline for RVFuse
- **Mandatory Dependency**: An external dependency that contributors need for the current setup phase
- **Optional Dependency**: An external dependency that is not required in the current setup phase but may become required for later scenarios
- **Dependency Source Reference**: The canonical upstream location recorded for a dependency
- **Setup Verification Criteria**: The documented checks that determine whether the current setup phase is complete

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of current-phase repository areas and dependency roles are described in the project documents without conflicting mandatory or optional status
- **SC-002**: A contributor can identify all current-phase mandatory dependencies and all current-phase optional dependencies from the documentation on the first read
- **SC-003**: A contributor can verify completion of the current setup phase without invoking later profiling, DFG, or fusion-validation workflows
- **SC-004**: A new contributor can complete the documented current-phase setup process within 30 minutes, excluding network download time, first-time dependency synchronization time, and third-party build time
- **SC-005**: 100% of workload or benchmark examples referenced in current-phase documents point to a traceable source

## Assumptions

- The current phase is limited to project structure and dependency access
- The current phase does not include hotspot detection, DFG generation, fusion candidate analysis, or fused instruction validation
- Xuantie newlib remains an optional dependency in the current phase and its canonical source reference should still be preserved for later use
- Future analysis and validation workflows will be specified in separate follow-up features

## Out of Scope

- Hotspot detection workflows
- DFG generation and DFG validity criteria
- Fusion candidate analysis
- Implementing fused instructions
- Cycle comparison and performance validation
