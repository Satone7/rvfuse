# Feature Specification: Automated Setup Flow Script

**Feature Branch**: `003-automated-setup-flow`
**Created**: 2026-04-06
**Status**: Draft
**Input**: User description: "根据README.md中规划的流程，整合成一个全流程脚本，脚本中先检查各阶段的产物，如果存在则跳过阶段，并且提供选项给用户可以强制重新执行某个/某些/全部Step。将Step5的report也保存到文件。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Full Setup in One Command (Priority: P1)

A contributor runs a single script to execute the entire RVFuse setup flow (Steps 1-5) without manually following individual steps. The script automatically detects which steps have already been completed and skips them.

**Why this priority**: This is the core value proposition — a one-command setup that saves contributor time and eliminates manual errors.

**Independent Test**: Run the script on a clean workspace and verify all setup artifacts are created and the completion report is generated.

**Acceptance Scenarios**:

1. **Given** a clean Linux x86_64 workstation, **When** the contributor runs the setup script, **Then** the script executes all steps in order and produces a setup completion report
2. **Given** a partially completed workspace (some steps already done), **When** the contributor runs the setup script, **Then** the script detects existing artifacts and skips completed steps without re-executing them
3. **Given** the script completes all steps, **When** the contributor checks the output, **Then** a completion report file exists in the project root with a summary of all step results

---

### User Story 2 - Force Re-execute Specific Steps (Priority: P2)

A contributor wants to selectively re-run one or more steps of the setup flow, for example after a dependency update or a partial workspace corruption.

**Why this priority**: Contributors may need to re-run individual steps without redoing the entire setup, especially after external changes or troubleshooting.

**Independent Test**: Run the script with a force option targeting specific steps and verify only those steps are re-executed while others are skipped.

**Acceptance Scenarios**:

1. **Given** a fully completed workspace, **When** the contributor invokes the script with a force option for Step 3, **Then** only Step 3 is re-executed and all other steps remain untouched
2. **Given** the script is invoked, **When** the contributor chooses to force re-execute multiple specific steps, **Then** only the selected steps are re-executed in their original order
3. **Given** the script is invoked, **When** the contributor chooses to force re-execute all steps, **Then** every step is executed from scratch regardless of existing artifacts

---

### User Story 3 - Review Setup Completion Report (Priority: P3)

A contributor reviews a persisted setup report that summarizes the outcome of each step, including any warnings or errors encountered during setup.

**Why this priority**: A saved report enables contributors to audit setup results, share them with teammates, and diagnose issues without re-running the script.

**Independent Test**: Run the full setup, open the generated report file, and verify it contains results for all steps with pass/fail status and relevant details.

**Acceptance Scenarios**:

1. **Given** the setup script has completed, **When** the contributor opens the report file, **Then** the report lists every step with its execution status (pass/fail/skipped)
2. **Given** a step encountered a non-fatal warning, **When** the contributor reads the report, **Then** the report includes the warning message for that step
3. **Given** the report file, **When** a contributor shares it with a teammate, **Then** the teammate can understand the setup state without re-running the script

---

### Edge Cases

- What happens when the script is run from a directory that is not the project root? The script MUST detect this and exit with an error message directing the user to the correct directory.
- What happens when Step 3 (dependency initialization) fails due to a network error? The script MUST report the failure, not mark the step as complete, and not proceed to Step 5 until the user re-runs with force on Step 3.
- What happens when the user forces re-execution of Step 1 on an existing clone? The script MUST warn that Step 1 (clone) only applies in the context of first-time repository setup and skip it or provide clear guidance.
- What happens when git is not installed or below version 2.30? The script MUST check prerequisites before starting and exit with a clear error if they are not met.
- What happens when disk space is insufficient for submodule clones? The script SHOULD check available disk space before Step 3 and warn if below the recommended ~20GB.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The script MUST execute the full RVFuse setup flow corresponding to Steps 1-5 defined in the quickstart guide
- **FR-002**: The script MUST check for the existence of each step's artifacts before executing that step
- **FR-003**: The script MUST skip a step if all its artifacts are detected as present
- **FR-004**: The script MUST provide a force option that allows the user to selectively re-execute one or more steps
- **FR-005**: The script MUST provide a force-all option that re-executes every step from scratch
- **FR-006**: The script MUST generate a setup completion report as a persistent file (Step 5)
- **FR-007**: The report MUST include the execution status (pass/fail/skipped) for every step
- **FR-008**: The report MUST include warnings and error messages encountered during execution
- **FR-009**: The script MUST check prerequisites (git version, disk space, network access) before beginning execution
- **FR-010**: The script MUST exit with a non-zero status code if any mandatory step fails
- **FR-011**: The script MUST display progress information to the user during execution, indicating which step is currently running
- **FR-012**: The script MUST validate that it is running from the project root directory

### Key Entities

- **Setup Step**: One of the 5 defined stages in the quickstart flow (Clone, Review Scope, Initialize Dependencies, Verify Setup, Generate Report), each with identifiable artifacts
- **Step Artifact**: A file or directory produced by a step that the script checks for existence (e.g., `third_party/qemu/`, `docs/architecture.md`)
- **Force Target**: A user-selected step or set of steps designated for re-execution regardless of artifact existence
- **Setup Report**: A persistent file containing the aggregated results of all executed steps, saved to the project workspace

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A contributor can execute the complete setup flow with a single command invocation
- **SC-002**: The script correctly skips steps whose artifacts already exist in 100% of cases without false negatives
- **SC-003**: A contributor can force re-execute any subset of steps using clearly documented options
- **SC-004**: The setup report file is generated within 5 seconds after the last step completes
- **SC-005**: A contributor reading the report file can determine the setup state without running the script again

## Assumptions

- The script targets Linux x86_64 workstations as defined in the quickstart prerequisites
- The 5-step flow follows the quickstart guide: Step 1 (Clone Repository), Step 2 (Review Project Scope), Step 3 (Initialize Mandatory Dependencies), Step 4 (Verify Setup Completion), Step 5 (Generate Report)
- Step 1 (Clone) is a special case — when the script itself is being run from within the repo, Step 1 is either already complete or not applicable; the script should handle this gracefully
- Step 2 (Review Project Scope) is verified by checking that the required documentation files exist and are non-empty
- Step 3 artifact check verifies that `third_party/qemu/.git` and `third_party/llvm-project/.git` exist (indicating submodule initialization)
- Step 4 artifact is the passing of all 7 verification checks from the quickstart
- The report file (Step 5) is saved to a known location in the project workspace
- The script is written in bash and uses only standard Unix tools available on a typical Linux x86_64 workstation
