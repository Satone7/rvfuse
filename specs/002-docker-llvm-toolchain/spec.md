# Feature Specification: Docker LLVM RISC-V Toolchain

**Feature Branch**: `002-docker-llvm-toolchain`
**Created**: 2026-04-01
**Status**: Draft
**Input**: User description: "由于我当前的开发设备的性能不足以编译LLVM，所以除了使用submodule中的llvm外，再提供一个方式使用公版的llvm riscv工具链编译代码，公版llvm的版本尽量贴合submodule中的llvm版本。为了尽量降低对系统环境的依赖保证项目在不同环境中正常运行，公版llvm可以通过提供docker镜像的方式进行调用，但必须封装成脚本提供给用户使用。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compile RISC-V Code with Docker Toolchain (Priority: P1)

A contributor who cannot compile LLVM from source uses the Docker-based LLVM RISC-V toolchain to compile code for the RVFuse project without needing high-performance development hardware.

**Why this priority**: The primary use case is enabling contributors with limited hardware to participate in the project. Without this, they would be blocked from any RISC-V code compilation work.

**Independent Test**: Can be fully tested by running the wrapper script to compile a simple RISC-V program and verifying the output binary is a valid RISC-V ELF.

**Acceptance Scenarios**:

1. **Given** Docker is installed, **When** the contributor runs the toolchain wrapper script for the first time, **Then** the Docker image is pulled automatically (or built if not available) and the toolchain is ready for use
2. **Given** the toolchain is available, **When** the contributor runs the wrapper script with source files, **Then** RISC-V binaries are produced successfully
3. **Given** a compilation error occurs, **When** the contributor reviews the output, **Then** clear error messages are displayed without Docker-related noise

---

### User Story 2 - Verify Toolchain Version Compatibility (Priority: P2)

A contributor verifies that the Docker-based toolchain produces compatible output with the submodule LLVM toolchain, ensuring consistency across different development environments.

**Why this priority**: Toolchain compatibility is critical for reproducible research results. Contributors need confidence that Docker-based compilation matches submodule-based compilation.

**Independent Test**: Can be tested by compiling the same source code with both toolchains and comparing the output binaries or disassembly.

**Acceptance Scenarios**:

1. **Given** identical source code, **When** compiled with both Docker toolchain and submodule LLVM, **Then** the resulting binaries have compatible ABIs
2. **Given** a version query command, **When** the contributor checks the Docker toolchain version, **Then** the LLVM version matches (or is close to) the submodule LLVM version (13.0.0-based)

---

### User Story 3 - Use Toolchain Without Docker Knowledge (Priority: P3)

A contributor uses the LLVM toolchain through simple wrapper scripts without needing to learn Docker commands or manage containers manually.

**Why this priority**: Reducing the learning curve enables faster onboarding. Contributors should focus on RISC-V development, not container management.

**Independent Test**: Can be tested by asking a new contributor to compile a program using only the documented wrapper script commands.

**Acceptance Scenarios**:

1. **Given** the wrapper script is in PATH, **When** the contributor runs `riscv-clang --version`, **Then** the toolchain version is displayed without Docker-related steps
2. **Given** the wrapper script, **When** the contributor passes standard LLVM compiler flags, **Then** they work as expected without modification
3. **Given** the toolchain documentation, **When** a contributor reads the quickstart guide, **Then** they can complete setup within 10 minutes

---

### Edge Cases

- What happens when Docker is not installed? The wrapper script MUST detect this and provide a clear error message with installation instructions
- What happens when the Docker image pull fails due to network issues? The script MUST provide retry guidance and fallback options
- What happens when the contributor wants to use a different LLVM version? The script MUST document how to specify alternative versions or custom images
- What happens when running on a system without sufficient disk space for Docker images? The script MUST check disk availability before pulling the image and warn if insufficient

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST provide a Docker-based LLVM RISC-V toolchain as an alternative to the submodule LLVM
- **FR-002**: The Docker toolchain LLVM version MUST be compatible with the submodule LLVM (llvmorg-13.0.0-rc1 based)
- **FR-003**: The project MUST provide wrapper scripts that abstract Docker commands for common toolchain operations
- **FR-004**: The wrapper scripts MUST support standard LLVM compiler flags (clang, clang++, llvm-* tools)
- **FR-005**: The wrapper script MUST automatically pull or build the Docker image on first use if not present
- **FR-006**: The project MUST document the Docker image source (official LLVM releases or custom-built image)
- **FR-007**: The wrapper script MUST handle volume mounting for source files and output artifacts transparently
- **FR-008**: The toolchain MUST support cross-compilation for RISC-V 64-bit target (riscv64-unknown-elf or similar)
- **FR-009**: Error messages from the toolchain MUST be clear and actionable, not exposing Docker internals unnecessarily

### Key Entities

- **Docker LLVM Image**: Container image containing LLVM RISC-V toolchain, versioned to match submodule LLVM
- **Wrapper Script**: Shell script that provides a user-friendly interface to the containerized toolchain, handling Docker operations transparently
- **Toolchain Configuration**: Settings for target architecture, LLVM version, and Docker image reference
- **Compilation Context**: Working directory, source files, and output paths that need to be mounted into the container

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A contributor can compile a "Hello World" RISC-V program using the Docker toolchain within 5 minutes of setup (excluding image pull time)
- **SC-002**: The wrapper script successfully abstracts Docker complexity - users can compile without knowing Docker commands
- **SC-003**: Binaries produced by Docker toolchain are ABI-compatible with binaries from submodule LLVM compilation
- **SC-004**: The wrapper script provides meaningful error messages in 100% of common failure scenarios (no Docker, network failure, disk full)
- **SC-005**: Setup documentation is complete enough that a new contributor can use the toolchain without additional help

## Assumptions

- Contributors have Docker installed or can install it (Docker is a prerequisite, but installation guidance should be provided)
- The official LLVM releases or community-maintained LLVM RISC-V toolchain images exist for the target version (LLVM 13.0.0 or compatible)
- If no suitable pre-built image exists, the project will provide a Dockerfile to build one
- Network access is available for pulling Docker images (or offline setup documentation is provided)
- The host system is Linux x86_64 (same constraint as the main project)

## Out of Scope

- Building LLVM from source within Docker (use pre-built images or build images separately)
- IDE integration for the Docker toolchain
- Debugging RISC-V programs inside Docker
- Performance optimization of Docker-based compilation
- Windows or macOS host support (follows main project Linux-first constraint)