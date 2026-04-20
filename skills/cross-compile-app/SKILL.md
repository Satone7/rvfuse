---
name: cross-compile-app
description: |
  Cross-compile a C/C++ application for RISC-V (rv64gcv) using the project's LLVM 22
  toolchain, Docker sysroot, and QEMU. Covers the full workflow: source clone, build
  system analysis, toolchain generation, sysroot extraction, compilation, and smoke test.
  Trigger when: the user asks to add, set up, cross-compile, or bootstrap a new
  application for RISC-V in the RVFuse project. Matches phrases like "add app",
  "cross compile", "交叉编译", "build for riscv", "new application".
---

# Cross-Compile App for RISC-V

Guide an agent through cross-compiling a C/C++ application for RISC-V (rv64gcv) within
the RVFuse project. The workflow has 4 phases: Discovery, Scaffolding, Sysroot + Build,
Smoke Test.

## Input

| Parameter | Required | Description |
|-----------|----------|-------------|
| `app-name` | Yes | Directory name under `applications/` (e.g., `llama.cpp`, `yolo`) |
| `repo-url` | Yes | Git clone URL (e.g., `https://github.com/org/repo.git`) |
| `version` | No | Git tag, branch, or commit (defaults to latest release or `main`) |

## Key Paths

| Item | Path |
|------|------|
| LLVM install | `third_party/llvm-install/` |
| QEMU binary | `third_party/qemu/build/qemu-riscv64` |
| Applications dir | `applications/<name>/` |
| Output dir | `output/<name>/` |
| Toolchain file | `applications/<name>/riscv64-linux-toolchain.cmake` |
| Build script | `applications/<name>/build.sh` |

## Phase 1: Application Discovery

### 1.1 Clone Source

```bash
mkdir -p applications/<name>/vendor
git clone --depth=1 --branch <version> <repo-url> applications/<name>/vendor/<repo-name>
```

### 1.2 Analyze Build System

Run the appropriate grep patterns to identify the build system and dependencies.

**CMake projects:**
```bash
grep -E 'project\(|find_package\(|target_link_libraries' vendor/<name>/CMakeLists.txt
```

**Makefile projects:**
```bash
grep -E '^(CC|CXX|LDFLAGS|LDLIBS)' vendor/<name>/Makefile
```

### 1.3 Infer RISC-V Extensions

Scan the source to determine which ISA extensions the application requires:

```bash
# RVV (Vector Extension)
grep -rl 'riscv_vector.h' vendor/<name>/

# ZFH (Half-precision float)
grep -rl '_Float16\|__fp16' vendor/<name>/

# ZICBOP (Cache-block prefetch)
grep -rl 'prefetch\|__builtin_prefetch' vendor/<name>/

# ZBA/ZBB (Bitmanip)
grep -rl '__builtin_ctz\|__builtin_clz\|__builtin_popcount' vendor/<name>/
```

### 1.4 Infer Docker Packages

Map common link flags to Ubuntu `-dev` packages for the sysroot:

| Link Flag | Package |
|-----------|---------|
| `-lssl -lcrypto` | `libssl-dev` |
| `-lcurl` | `libcurl4-openssl-dev` |
| `-lz` | `zlib1g-dev` |
| `-lsqlite3` | `libsqlite3-dev` |
| `-lxml2` | `libxml2-dev` |
| `-fopenmp` | `libgomp-dev` |

Scan CMakeLists.txt or Makefile for `target_link_libraries` / `LDLIBS` output.

### 1.5 Find Smoke Test Target

Identify the main executable name for the smoke test phase. Check:
- CMake `add_executable()` calls
- Makefile `all:` / `bin:` targets
- Repository README for "quick start" or "usage" sections

### 1.6 Ask User About ZVL

Based on the extensions detected in step 1.3, compose the base `-march` and ask:

```
Based on source analysis, the inferred compile flags are:
  -march=rv64gcv[<detected extensions>]

是否需要添加 ZVL 扩展? Options:
  1. No (default, uses VLEN=128)
  2. zvl256b
  3. zvl512b
  4. zvl1024b
```

### 1.7 Write Application Profile

Record findings in `applications/<name>/README.md` following the
`references/toolchain-template.md` README template. Include:

```markdown
## Application Profile

| Field | Value |
|-------|-------|
| Name | <name> |
| Source | <repo-url> |
| Version | <version> |
| Build system | CMake / Makefile |
| Target arch | rv64gcv[<extensions>] |
| ABI | lp64d |
| Extra sysroot packages | <list or "none"> |
| Smoke test binary | <binary-name> |
| Smoke test args | <args, e.g., "--version"> |
```

---

## Phase 2: Scaffolding

Reference: `references/toolchain-template.md`

### 2.1 CMake Toolchain File

Copy the toolchain template to `applications/<name>/riscv64-linux-toolchain.cmake`
and customize:

1. Replace `<PLACEHOLDER_ARCH>` with the `-march` value determined in Phase 1.6
2. Verify the toolchain uses `$ENV{LLVM_INSTALL}` and `$ENV{SYSROOT}` (not hardcoded paths)

### 2.2 Build Script

Copy the build.sh skeleton to `applications/<name>/build.sh` and customize:

1. Set `OUTPUT_DIR`, `VENDOR_DIR`, `SOURCE_DIR`, `REPO_URL`, `REPO_VERSION`
2. Set `PROJECT_ROOT` depth correctly (applications is 1 level below repo root)
3. Implement the `cross_compile()` function based on the detected build system:

   **CMake** (most common):
   ```bash
   cmake -S "${SOURCE_DIR}" -B "${build_dir}" \
       -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
       -DCMAKE_INSTALL_PREFIX="${install_dir}" \
       -DCMAKE_BUILD_TYPE=Release -G Ninja
   ninja -C "${build_dir}" -j"${JOBS}"
   ninja -C "${build_dir}" install/strip
   ```

   **Autotools** (configure.ac / autogen.sh present):
   ```bash
   cd "${SOURCE_DIR}"
   autoreconf -fi  # Generate configure script
   ./configure --host=riscv64-unknown-linux-gnu \
       CC="${LLVM_INSTALL}/bin/clang" CXX="${LLVM_INSTALL}/bin/clang++" \
       CFLAGS="-march=<arch> --sysroot=${SYSROOT}" \
       LDFLAGS="--sysroot=${SYSROOT} -fuse-ld=lld" \
       --prefix="${install_dir}"
   make -j"${JOBS}"
   make install-strip
   ```
   Key: `--host` tells configure this is a cross-build. LLVM's clang is both compiler
   and linker (via `-fuse-ld=lld`). No GCC cross-compiler needed.

   **Plain Makefile** (no configure system):
   ```bash
   make -C "${SOURCE_DIR}" -j"${JOBS}" \
       CC="${LLVM_INSTALL}/bin/clang" CXX="${LLVM_INSTALL}/bin/clang++" \
       CFLAGS="-march=<arch> --sysroot=${SYSROOT}" \
       LDFLAGS="--sysroot=${SYSROOT} -fuse-ld=lld"
   ```

4. Add application-specific cmake/make flags
5. Set output binary path for smoke test
6. `chmod +x build.sh`

### 2.3 .gitignore

Copy the `.gitignore` template from `references/toolchain-template.md` to
`applications/<name>/.gitignore`. Add application-specific patterns if needed
(e.g., `*.gguf` for llama.cpp, `*.ort` for ONNX Runtime).

### 2.4 README.md

Write using the README template from `references/toolchain-template.md`. Include
the Application Profile from Phase 1.7.

### 2.5 Scaffolding Gate

Before proceeding to Phase 3, verify the scaffolding is complete and correct:

```bash
# build.sh exists and is executable
test -x applications/<name>/build.sh && echo "OK" || echo "MISSING"

# For CMake projects: toolchain has no remaining placeholders
if [ -f applications/<name>/riscv64-linux-toolchain.cmake ]; then
    grep -c '<PLACEHOLDER_ARCH>' applications/<name>/riscv64-linux-toolchain.cmake
    # Expected output: 0 (if >0, placeholders were not replaced)
    grep -c 'ENV{LLVM_INSTALL}\|ENV{SYSROOT}' applications/<name>/riscv64-linux-toolchain.cmake
    # Expected output: >0
fi

# For autotools projects: verify build.sh uses --host=riscv64-unknown-linux-gnu
grep -c 'host=riscv64-unknown-linux-gnu' applications/<name>/build.sh
# Expected output: >0 for autotools, 0 is OK for cmake-only
```

If any check fails, fix it now — broken scaffolding causes opaque build errors later.

---

## Phase 3: Sysroot + Build

Reference: `references/sysroot-extract.md`

### 3.1 Extract Sysroot

1. Check Docker is available: `docker --version`
   - If Docker is not installed or the daemon is not running, stop and tell the user:
     "Sysroot extraction requires Docker (riscv64/ubuntu:24.04 image). Install Docker and start the daemon, then re-run."
   - If Docker is available but `docker run --platform riscv64` fails, the user may need
     to enable multi-platform support: `docker run --rm --privileged multiarch/qemu-user-static --reset -p yes`

2. Run sysroot extraction with the extra packages identified in Phase 1.4:

```bash
# Option A: Use extract_sysroot function from reference doc
extract_sysroot "output/<name>/sysroot" <extra-packages...>

# Option B: Run build.sh which handles it internally
cd applications/<name> && ./build.sh
```

If sysroot extraction fails mid-way, a stale Docker container may remain.
Clean up with: `docker rm -f $(docker ps -a -q --filter name=rvfuse-sysroot-prep)`

3. Verify sysroot integrity:
```bash
ls output/<name>/sysroot/lib/ld-linux-riscv64-lp64d.so.1
ls output/<name>/sysroot/usr/lib/riscv64-linux-gnu/libc.so.6
```

### 3.2 Cross-Compile

```bash
export LLVM_INSTALL="$(pwd)/third_party/llvm-install"
export SYSROOT="$(pwd)/output/<name>/sysroot"

cd applications/<name> && ./build.sh --skip-sysroot
```

### 3.3 Verify Output

```bash
file output/<name>/bin/<binary>
# Expected: ELF 64-bit LSB executable, UCB RISC-V, ...
```

If output shows x86-64 or a different architecture, the cross-compilation failed.
Re-check the toolchain file and environment variables.

### 3.4 Build Gate

Before proceeding to Phase 4, confirm the build produced a valid RISC-V binary:

```bash
# Binary is RISC-V ELF
file output/<name>/bin/<binary> | grep -q "RISC-V"
# If this grep fails, the binary is wrong architecture — do NOT proceed to smoke test

# Binary is dynamically linked to the correct interpreter
file output/<name>/bin/<binary> | grep -q "ld-linux-riscv64"
# Static binaries are OK too — this check is informational only

# Sysroot has the dynamic linker
test -f output/<name>/sysroot/lib/ld-linux-riscv64-lp64d.so.1
```

If the binary is not RISC-V ELF, stop and re-check the toolchain `-march` flag and
`$ENV{LLVM_INSTALL}` path. Running a non-RISC-V binary under QEMU wastes time and
produces confusing errors.

---

## Phase 4: Smoke Test

Reference: `references/smoke-test.md`

### 4.1 Auto-Infer QEMU CPU Flag

Parse the `-march` from the toolchain file to determine the correct `-cpu` flag:

```bash
march=$(grep -oP 'march=\K[^")\s]+' applications/<name>/riscv64-linux-toolchain.cmake | head -1)

if [[ "$march" =~ zvl([0-9]+)b ]]; then
    cpu_flag="-cpu rv64,v=true,vlen=${BASH_REMATCH[1]}"
else
    cpu_flag=""  # Default VLEN=128
fi
```

### 4.2 Execute Under QEMU

```bash
QEMU=third_party/qemu/build/qemu-riscv64
SYSROOT=output/<name>/sysroot
BINARY=output/<name>/bin/<binary>

${QEMU} -L ${SYSROOT} ${cpu_flag} ${BINARY} <test-args>
```

### 4.3 Evaluate Result

| Result | Verdict | Notes |
|--------|---------|-------|
| Exit code 0 | PASS | Binary executes successfully |
| Version/help text output | PASS | Recognizable output |
| "missing input/model" error | PASS | Binary works, needs proper inputs |
| `Illegal instruction` (SIGILL) | FAIL | VLEN mismatch or missing ISA extension |
| `ld-linux not found` | FAIL | Sysroot missing dynamic linker |

### 4.4 Report

Summarize the result for the user:

```
Smoke test: PASS/FAIL
  Binary:     output/<name>/bin/<binary>
  Arch:       ELF 64-bit RISC-V
  -march:     rv64gcv...
  VLEN:       128 / <N>
  Exit code:  <N>
  QEMU cmd:   <full command>
```

---

## Validation Checklist

After completing all 4 phases, verify:

```
- [ ] applications/<name>/build.sh exists, is executable, has set -euo pipefail
- [ ] riscv64-linux-toolchain.cmake uses $ENV{LLVM_INSTALL} and $ENV{SYSROOT}
- [ ] Sysroot contains lib/ld-linux-riscv64-lp64d.so.1
- [ ] Build output is RISC-V ELF (file command confirms)
- [ ] QEMU can run the binary
- [ ] ZVL flags match QEMU -cpu vlen (if applicable)
- [ ] README.md has Application Profile
- [ ] .gitignore covers build artifacts, vendor/, and large data files
```

---

## References

| Resource | Location |
|----------|----------|
| Toolchain templates | `skills/cross-compile-app/references/toolchain-template.md` |
| Sysroot extraction | `skills/cross-compile-app/references/sysroot-extract.md` |
| Smoke test procedure | `skills/cross-compile-app/references/smoke-test.md` |
| YOLO/ORT demo | `applications/yolo/ort/` (CMake, rv64gcv) |
| llama.cpp demo | `applications/llama.cpp/` (CMake, rv64gcv_zfh_zba_zicbop) |
| Project context | `CLAUDE.md` (Key Commands section) |
