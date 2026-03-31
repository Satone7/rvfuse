# Feature Specification: RVFuse Project Structure

**Feature Branch**: `001-riscv-fusion-setup`
**Created**: 2026-03-31
**Status**: Draft
**Input**: User description: "设计该项目的目录结构并完成相关目录创建，通过添加submodule的方式clone第三方项目。该项目是为了实现RISC-V指令融合的需求发现和测试，从具体应用（如onnxruntime）出发，通过qemu模拟器定位热点函数和热点基本块，实现基于基本块的DFG生成，找到具有数据依赖关系的高频指令组合，然后在模拟器内尝试通过新指令的方式将指令组合融合成一条指令，并验证其cycle，比对效果。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Project Initialization (Priority: P1)

A developer sets up the RVFuse project environment by cloning the repository and initializing all necessary dependencies (Xuantie QEMU, LLVM, newlib as submodules) to begin instruction fusion research.

**Why this priority**: Without proper project structure and submodule initialization, no other development work can proceed. This is the foundational setup that enables all subsequent research activities.

**Independent Test**: Can be fully tested by verifying directory structure exists and all submodules are properly initialized with correct commit references.

**Acceptance Scenarios**:

1. **Given** a clean workspace, **When** developer clones RVFuse repository, **Then** all project directories are created with correct structure
2. **Given** cloned repository, **When** developer runs `git submodule update --init`, **Then** Xuantie QEMU, LLVM, and newlib are cloned to their respective locations
3. **Given** initialized submodules, **When** developer verifies submodule status, **Then** all submodules show correct remote URLs and commit hashes

---

### User Story 2 - Hotspot Detection Workflow (Priority: P2)

A researcher runs an application (e.g., ONNX Runtime) through the QEMU emulator to identify hot functions and hot basic blocks for fusion candidate analysis.

**Why this priority**: This is the core research workflow that identifies fusion candidates. Without this, fusion analysis cannot proceed.

**Independent Test**: Can be tested by running a sample application through QEMU and verifying profiling output contains function/block timing data.

**Acceptance Scenarios**:

1. **Given** compiled test application, **When** researcher runs it under QEMU with profiling enabled, **Then** hotspot analysis output is generated with function/block metrics
2. **Given** profiling output, **When** researcher queries hot functions, **Then** results show top N functions ranked by execution frequency or cycle count
3. **Given** hot function identified, **When** researcher requests basic block breakdown, **Then** system shows block-level timing distribution

---

### User Story 3 - DFG Generation (Priority: P3)

A researcher generates Data Flow Graph (DFG) representations from identified hot basic blocks to analyze instruction dependencies and identify fusion candidates.

**Why this priority**: DFG generation is required for dependency analysis, which enables fusion candidate identification.

**Independent Test**: Can be tested by providing a sample basic block and verifying DFG output correctly represents instruction dependencies.

**Acceptance Scenarios**:

1. **Given** hot basic block data, **When** researcher triggers DFG generation, **Then** DFG output shows instruction nodes and dependency edges
2. **Given** generated DFG, **When** researcher queries data dependencies, **Then** system identifies chains of dependent instructions
3. **Given** DFG with dependencies, **When** researcher requests fusion candidates, **Then** system suggests instruction combinations with dependency analysis

---

### User Story 4 - Fusion Testing (Priority: P4)

A researcher tests instruction fusion by implementing a new fused instruction in the emulator and comparing cycle counts against the original instruction sequence.

**Why this priority**: This validates the fusion hypothesis and demonstrates performance improvement potential.

**Independent Test**: Can be tested by implementing a known fusion candidate and verifying cycle reduction matches expected improvement.

**Acceptance Scenarios**:

1. **Given** identified fusion candidate, **When** researcher implements fused instruction in QEMU, **Then** emulator accepts and executes the new instruction
2. **Given** fused instruction implementation, **When** researcher runs benchmark with fused instruction, **Then** cycle count is recorded
3. **Given** both original and fused cycle counts, **When** researcher compares results, **Then** system provides performance improvement percentage

---

### Edge Cases

- What happens when submodule clone fails due to network issues? System MUST provide retry mechanism and document manual fallback steps
- How does system handle applications with no significant hotspots? System MUST report "no candidates found" with profiling summary, not error
- What if fusion candidate instructions are already optimal? System MUST document analysis result and skip fusion testing
- How to handle fused instruction encoding conflicts? System MUST validate against RISC-V encoding rules before implementation

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a defined directory structure for all project components
- **FR-002**: System MUST integrate Xuantie QEMU as a git submodule at `third_party/qemu`
- **FR-003**: System MUST integrate Xuantie LLVM as a git submodule at `third_party/llvm-project`
- **FR-004**: System MUST integrate Xuantie newlib as a git submodule at `third_party/newlib` (conditionally, only when needed for bare-metal targets)
- **FR-005**: System MUST provide a profiling workflow to identify hot functions from application execution
- **FR-006**: System MUST provide a profiling workflow to identify hot basic blocks within functions
- **FR-007**: System MUST generate DFG representations from basic block instruction sequences
- **FR-008**: System MUST identify instruction combinations with data dependencies as fusion candidates
- **FR-009**: System MUST support implementing and testing fused instructions in the emulator
- **FR-010**: System MUST provide cycle count comparison between original and fused instruction sequences
- **FR-011**: System MUST document all directory structures and their purposes
- **FR-012**: System MUST provide build instructions for all third-party dependencies

### Key Entities

- **Application**: Target workload for profiling (e.g., ONNX Runtime, custom benchmarks)
- **Hot Function**: Function identified as consuming significant execution time
- **Hot Basic Block**: Basic block within a hot function with high execution frequency
- **DFG Node**: Single instruction representation in the data flow graph
- **DFG Edge**: Data dependency relationship between instructions
- **Fusion Candidate**: Instruction sequence identified as potential fusion target
- **Fused Instruction**: New instruction combining multiple dependent instructions
- **Cycle Count**: Performance metric measuring instruction execution cost

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Project structure is complete with all required directories created within 1 setup attempt
- **SC-002**: All three submodules (QEMU, LLVM, newlib) initialize successfully with correct upstream URLs
- **SC-003**: Hotspot detection workflow produces profiling output within 10 minutes for a standard test application
- **SC-004**: DFG generation produces valid dependency graphs for 95% of analyzed basic blocks
- **SC-005**: Fusion testing workflow provides cycle comparison results within 5 minutes per candidate
- **SC-006**: Directory structure documentation covers all paths and their purposes completely
- **SC-007**: New team members can complete project setup within 30 minutes following documentation

## Assumptions

- Developers have access to Xuantie GitHub repositories (no authentication required for public repos)
- Target applications can be compiled for RISC-V architecture
- QEMU profiling features support function and basic block level timing
- LLVM infrastructure can be extended for custom instruction analysis
- Minimum 8GB RAM and 20GB disk space available for submodule builds

## Out of Scope

- Actual implementation of specific fused instructions (this is research workflow setup only)
- Performance benchmarking of production applications
- LLVM backend modifications for new instruction encoding
- Hardware validation of fusion candidates