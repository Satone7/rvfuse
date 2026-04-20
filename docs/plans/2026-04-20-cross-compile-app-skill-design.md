# Cross-Compile App Skill Design

**Date**: 2026-04-20
**Status**: Approved
**Trigger**: README Step 1 — 交叉编译目标应用

## Overview

Create a Skill (`skills/cross-compile-app/`) that guides an agent through the full workflow of adding a new C/C++ application to the RVFuse platform: source setup, cross-compilation environment, Docker sysroot, and QEMU smoke test. The skill supports both bootstrapping new applications and maintaining existing ones.

## Skill Structure

```
skills/cross-compile-app/
├── SKILL.md                      # Main orchestrator (4-phase linear flow)
└── references/
    ├── sysroot-extract.md        # Sub-flow: Docker sysroot extraction
    ├── smoke-test.md             # Sub-flow: QEMU smoke test
    └── toolchain-template.md     # Sub-flow: cmake toolchain generation template
```

Symlink: `.claude/skills/cross-compile-app` → `../../skills/cross-compile-app`

## Input

The user provides:
- **app-name** — Application directory name (e.g., `openblas`, `sqlite`)
- **repo-url** — Git repository URL
- **version** — (optional) Branch, tag, or commit

## 4-Phase Workflow

### Phase 1: Application Discovery

Agent clones source code to the final location and analyzes the build system.

1. **Create app directory**: `mkdir -p applications/<name>/vendor`
2. **Clone source**: `git clone --depth=1 --branch <version> <repo-url> applications/<name>/vendor/<name>`
3. **Analyze build system** (priority order):
   - CMake: Check `CMakeLists.txt` for `project()`, `find_package()`, `target_link_libraries()` → infer dependencies
   - Makefile: Check `CC/CXX`, `LDFLAGS`, linked libraries
   - Autotools: Check `configure.ac`
4. **Infer RISC-V compilation parameters**:
   - Default: `rv64gcv` + `lp64d` ABI
   - Check for RVV intrinsics (`#include <riscv_vector.h>`) → auto-add ZVL
   - Check for FPU code (`float`/`double`) → confirm F/D extensions
   - Check for special extensions (ZFH, ZICBOP, etc.) → grep-based inference
5. **Infer Docker sysroot packages** — Analyze linked libraries to determine required `-dev` packages
6. **Identify smoke test entry** — Find the main executable for QEMU verification

**Output**: Structured Application Profile (embedded in README.md):

```yaml
app_name: <name>
build_system: cmake          # cmake | makefile | autotools | custom
cmake_min_version: "3.14"    # if cmake
default_target: <binary>     # main executable for smoke test
riscv_arch: rv64gcv          # base architecture (before ZVL)
abi: lp64d
extra_cmake_flags: [...]     # app-specific cmake flags
docker_sysroot_packages:     # base + inferred
  - libc6-dev
  - libstdc++-12-dev
  - libgcc-12-dev
  - <inferred-extra>
test_command: "<run-command>" # for smoke test
notes: "..."
```

### Phase 2: Scaffolding

Generate the application directory structure and build scripts.

**Directory layout**:

```
applications/<name>/
├── README.md                        # App Profile + build instructions
├── build.sh                         # Cross-compile orchestrator
├── riscv64-linux-toolchain.cmake    # CMake toolchain file
├── vendor/                          # Source code (cloned in Phase 1)
└── .gitignore                       # Ignore vendor/, output/
```

**build.sh** — Unified parameter interface:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Options:
--force          # Rebuild everything
--skip-sysroot   # Skip sysroot extraction
--skip-source    # Skip source cloning
-j, --jobs N     # Parallel build jobs (default: nproc)
--test           # Compile and run tests (if applicable)
```

Internal flow (4 steps):
1. Prerequisites check — LLVM, cmake, ninja, docker
2. Sysroot extraction — Per `references/sysroot-extract.md`
3. Source clone — If `vendor/` is empty
4. Cross-compile — Based on inferred build system and parameters

**riscv64-linux-toolchain.cmake** — Reuse existing template pattern:
- `LLVM_INSTALL` and `SYSROOT` from environment variables
- `-march=` from Phase 1 inference (base) + user-confirmed ZVL
- `-fuse-ld=lld`
- `-isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu`

**ZVL user confirmation point**: After presenting inferred parameters, agent asks the user whether to add `zvl256b`, `zvl512b`, or `zvl1024b` to the `-march` flags. Default: no ZVL extension (QEMU defaults to VLEN=128).

**README.md** content:
- Application Profile (build system, architecture, dependencies)
- Build Status table
- Directory Structure
- Build commands
- Output artifacts description
- QEMU run commands

### Phase 3: Sysroot + Cross-Compile

**Sysroot extraction** (`references/sysroot-extract.md`):

Standardized Docker sysroot extraction (consistent with yolo and llama.cpp build.sh):
1. Start `riscv64/ubuntu:24.04` container
2. Install base packages: `libc6-dev libstdc++-12-dev libgcc-12-dev` + Phase 1-inferred extra `-dev` packages
3. Copy `/usr/lib`, `/usr/include`, `/lib/riscv64-linux-gnu/` to sysroot
4. Create dynamic linker symlink: `ld-linux-riscv64-lp64d.so.1`
5. Create top-level CRT and shared library symlinks (for clang `-L` resolution)
6. Delete `libm.a` (avoid long double issues)

**Cross-compile** — Execute based on build system:
- CMake: `cmake -S <source> -B <build> -DCMAKE_TOOLCHAIN_FILE=<toolchain> ... && ninja`
- Makefile: Override `CC/CXX/LDFLAGS` then `make`
- Other: Agent analyzes and chooses the appropriate approach

### Phase 4: QEMU Smoke Test

**Smoke test** (`references/smoke-test.md`):

1. Verify binary is RISC-V ELF: `file <binary>`
2. Check dynamic dependencies: `llvm-readelf -d <binary>`
3. Execute via project QEMU: `qemu-riscv64 -L <sysroot> <binary> <test-args>`
4. VLEN matching — Agent auto-infers QEMU `-cpu` flag from toolchain `-march`:

| Compile flag | QEMU launch flag | VLEN (bits) |
|-------------|------------------|-------------|
| (default, no zvl) | (default) | 128 |
| `zvl256b` | `-cpu rv64,v=true,vlen=256` | 256 |
| `zvl512b` | `-cpu rv64,v=true,vlen=512` | 512 |
| `zvl1024b` | `-cpu rv64,v=true,vlen=1024` | 1024 |

5. Success criteria:
   - Exit code 0, OR
   - Output contains expected string (version info, help text)
   - For apps requiring model files: "missing model" error also counts as success (binary is executable)

**Resources used**:
- QEMU binary: `third_party/qemu/build/qemu-riscv64`
- LLVM toolchain: `third_party/llvm-install/`

## Validation Checklist

Before reporting completion:

- [ ] `applications/<name>/build.sh` is executable with `set -euo pipefail`
- [ ] `riscv64-linux-toolchain.cmake` uses `LLVM_INSTALL` and `SYSROOT` env vars
- [ ] Sysroot contains dynamic linker `ld-linux-riscv64-lp64d.so.1`
- [ ] Build artifact is RISC-V ELF
- [ ] QEMU can run the compiled binary (exit 0 or expected output)
- [ ] ZVL flags in toolchain match QEMU `-cpu` vlen parameter
- [ ] README.md contains Application Profile with build instructions

## References

- Existing demos: `applications/yolo/`, `applications/llama.cpp/`
- `skills/rvv-op/SKILL.md` — Application Discovery pattern
- `CLAUDE.md` — VLEN mismatch warning table
