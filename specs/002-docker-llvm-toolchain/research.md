# Research: Docker LLVM RISC-V Toolchain

**Date**: 2026-04-01 | **Phase**: 0

## Research Tasks

### 1. LLVM Docker Image Availability

**Question**: Are official LLVM Docker images available for version 13.0.0?

**Finding**: 
- Official LLVM project does not provide official Docker images
- Community images exist:
  - `llvm/llvm` - Not officially maintained by LLVM project
  - `aptman/qus` - For QEMU static binaries
  - Espressif provides `espressif/idf` with RISC-V toolchain
- **Best option**: Create custom Dockerfile based on Ubuntu/Debian with LLVM 13.0.0 from apt.llvm.org or build from source

**Decision**: Use `apt.llvm.org` packages for LLVM 13 in a Debian-based container, or use pre-built RISC-V GCC/LLVM toolchain from SiFive or similar vendor.

**Rationale**: 
- apt.llvm.org provides official LLVM binaries for multiple versions
- Avoids lengthy compilation time
- Reproducible builds

**Alternatives Considered**:
- Building LLVM from source in Dockerfile: Rejected - takes too long, increases image size
- Using Espressif IDF: Rejected - too specific to ESP32 development
- Using SiFive toolchain: Good alternative, but may have different patches

**Architecture Alignment**: ADR-001 (Git submodules) - Docker image is complementary, not replacement

---

### 2. RISC-V Cross-Compilation Toolchain Options

**Question**: What LLVM-based RISC-V toolchains are available?

**Finding**:
- **LLVM 13.0.0** includes RISC-V target support (llvmorg-13.0.0-rc1 aligns with submodule)
- Required tools for cross-compilation:
  - `clang` - Compiler frontend
  - `lld` - Linker
  - `llvm-objdump` - Disassembler
  - `llvm-objcopy` - Object file manipulation
  - `llvm-strip` - Strip symbols
- Target triple: `riscv64-unknown-elf` or `riscv64-linux-gnu`

**Decision**: Use LLVM 13 from apt.llvm.org with `riscv64-unknown-elf` target for bare-metal compatibility.

**Rationale**:
- Matches submodule LLVM version (13.0.0-based)
- `riscv64-unknown-elf` works for bare-metal (future newlib support)
- All tools available in LLVM package

**Alternatives Considered**:
- `riscv64-linux-gnu`: Good for Linux targets, but bare-metal may need newlib
- GCC toolchain: Rejected - not LLVM-based, inconsistent with project direction

---

### 3. Docker Wrapper Script Best Practices

**Question**: What are best practices for Docker wrapper scripts?

**Finding**:
- **Volume mounting**: Mount current directory to preserve paths
- **User mapping**: Use `--user $(id -u):$(id -g)` to avoid permission issues
- **Interactive mode**: Use `-it` for interactive sessions, remove for batch
- **Cleanup**: Use `--rm` to remove container after execution
- **Error handling**: Capture exit codes from container

**Recommended wrapper pattern**:
```bash
#!/bin/bash
IMAGE="rvfuse/llvm-riscv:13"
docker run --rm -v "$PWD:/work" -w /work --user $(id -u):$(id -g) \
  $IMAGE clang "$@"
```

**Decision**: Use thin wrapper scripts with shared `common.sh` for image management and error handling.

**Rationale**:
- Simple, maintainable
- Preserves exit codes
- Handles permissions correctly
- Standard Docker patterns

**Alternatives Considered**:
- Using `docker exec`: Rejected - requires persistent container
- Using `docker-compose`: Rejected - overkill for single tool execution
- Using aliases: Rejected - less portable, harder to version control

---

### 4. LLVM Version Compatibility Verification

**Question**: How to verify Docker toolchain produces compatible output?

**Finding**:
- ABI compatibility: Check with `readelf` or `llvm-readelf`
- Version check: `clang --version` output
- Feature parity: Verify RISC-V extensions supported
- Xuantie LLVM may have custom patches not in upstream LLVM 13

**Decision**: Document that Docker toolchain uses upstream LLVM 13, which may differ slightly from Xuantie LLVM. Provide version check command and note potential differences.

**Rationale**:
- Cannot guarantee 100% parity without Xuantie-specific patches
- Upstream LLVM 13 is "close enough" for most compilation tasks
- Differences should be documented, not hidden

**Architecture Alignment**: ADR-004 (Traceable References) - Document image source and version clearly

---

## Resolved Clarifications

All technical context items resolved - no NEEDS CLARIFICATION markers remain.

| Item | Resolution |
|------|------------|
| Language/Version | Bash + Dockerfile |
| Primary Dependencies | Docker, LLVM 13 (from apt.llvm.org) |
| Storage | N/A (stateless) |
| Testing | Shell test scripts |
| Performance Goals | < 10% compilation overhead |

---

## Docker Image Strategy

### Option A: Use apt.llvm.org packages (Recommended)

```dockerfile
FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y \
    wget gnupg \
    && echo "deb http://apt.llvm.org/bullseye/ llvm-toolchain-bullseye-13 main" >> /etc/apt/sources.list \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-get update && apt-get install -y \
    clang-13 \
    lld-13 \
    llvm-13 \
    llvm-13-tools \
    && ln -s /usr/bin/clang-13 /usr/bin/clang \
    && ln -s /usr/bin/clang++-13 /usr/bin/clang++ \
    && ln -s /usr/bin/lld-13 /usr/bin/lld \
    && ln -s /usr/bin/llvm-objdump-13 /usr/bin/llvm-objdump \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
```

### Option B: Use pre-built RISC-V toolchain

Consider using `riscv-gnu-toolchain` with LLVM backend or SiFive's pre-built toolchain if ABI compatibility is critical.

---

## Recommendations for Implementation

1. **Start with Option A** (apt.llvm.org) for simplicity
2. **Create wrapper scripts** for: clang, clang++, ld (lld), objdump, strip
3. **Add error handling** for: Docker not installed, image not available, disk space
4. **Document version differences** between upstream LLVM 13 and Xuantie LLVM