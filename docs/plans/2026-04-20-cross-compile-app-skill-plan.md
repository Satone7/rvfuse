# Cross-Compile App Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `skills/cross-compile-app/SKILL.md` that guides an agent through cross-compiling any C/C++ application for RISC-V (rv64gcv) using the project's LLVM toolchain, Docker sysroot, and QEMU.

**Architecture:** A single main SKILL.md with 4 linear phases (Discovery → Scaffolding → Sysroot+Build → Smoke Test). Reference sub-documents for reusable patterns (sysroot extraction, smoke test, toolchain template) live in `references/`. The skill is symlinked from `skills/cross-compile-app` to `.claude/skills/cross-compile-app`.

**Tech Stack:** Bash 4.0+, CMake, Docker (riscv64/ubuntu:24.04), LLVM 22 clang/clang++, QEMU user-mode, Linux sysroot management.

---

## File Map

| File | Responsibility |
|------|---------------|
| `skills/cross-compile-app/SKILL.md` | Main skill: 4-phase orchestration with frontmatter |
| `skills/cross-compile-app/references/sysroot-extract.md` | Standardized Docker sysroot extraction procedure |
| `skills/cross-compile-app/references/smoke-test.md` | QEMU smoke test procedure with VLEN matching |
| `skills/cross-compile-app/references/toolchain-template.md` | CMake toolchain file template + build.sh skeleton |
| `.claude/skills/cross-compile-app` | Symlink → `../../skills/cross-compile-app` |

---

### Task 1: Create skill directory structure

**Files:**
- Create: `skills/cross-compile-app/SKILL.md` (empty placeholder)
- Create: `skills/cross-compile-app/references/sysroot-extract.md` (empty placeholder)
- Create: `skills/cross-compile-app/references/smoke-test.md` (empty placeholder)
- Create: `skills/cross-compile-app/references/toolchain-template.md` (empty placeholder)

- [ ] **Step 1: Create directories**

```bash
mkdir -p skills/cross-compile-app/references
```

- [ ] **Step 2: Create placeholder files**

```bash
touch skills/cross-compile-app/SKILL.md
touch skills/cross-compile-app/references/sysroot-extract.md
touch skills/cross-compile-app/references/smoke-test.md
touch skills/cross-compile-app/references/toolchain-template.md
```

- [ ] **Step 3: Verify structure**

Run: `find skills/cross-compile-app -type f`
Expected:
```
skills/cross-compile-app/SKILL.md
skills/cross-compile-app/references/sysroot-extract.md
skills/cross-compile-app/references/smoke-test.md
skills/cross-compile-app/references/toolchain-template.md
```

---

### Task 2: Write `references/toolchain-template.md`

This reference doc provides the agent with templates for the cmake toolchain file and build.sh skeleton. These are the two files the agent must generate in Phase 2 (Scaffolding).

Study these two existing files before writing:
- `applications/yolo/ort/riscv64-linux-toolchain.cmake` (base RVV pattern)
- `applications/llama.cpp/riscv64-linux-toolchain.cmake` (RVV + ZFH + ZICBOP pattern)
- `applications/yolo/ort/build.sh` (full build.sh with sysroot + compile + test)
- `applications/llama.cpp/build.sh` (full build.sh with patches + sysroot + compile + test)

**Files:**
- Write: `skills/cross-compile-app/references/toolchain-template.md`

- [ ] **Step 1: Write toolchain template reference**

The file must contain:

1. **CMake Toolchain Template** — A bash heredoc or code block the agent can copy and customize. Based on the existing patterns:
   - `CMAKE_SYSTEM_NAME Linux`, `CMAKE_SYSTEM_PROCESSOR riscv64`
   - Compilers from `$ENV{LLVM_INSTALL}/bin/clang` and `clang++`
   - Target: `riscv64-unknown-linux-gnu`
   - SYSROOT from `$ENV{SYSROOT}`
   - `-isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu`
   - `-march=<PLACEHOLDER>` (filled by agent based on Phase 1 analysis + user ZVL choice)
   - `-fuse-ld=lld`
   - FIND_ROOT_PATH modes

   Include a **parameterization guide** table showing how `-march` maps to application needs:
   | Scenario | `-march` value |
   |----------|---------------|
   | Basic RV64GCV | `rv64gcv` |
   | With half-precision float | `rv64gcv_zfh` |
   | With ZICBOP prefetch | `rv64gcv_zfh_zicbop` |
   | With ZVL512 | `rv64gcv_zvl512b` |
   | llama.cpp pattern | `rv64gcv_zfh_zba_zicbop` |

2. **build.sh Skeleton** — A bash skeleton with these sections (marked with `# TODO: ...` comments for agent to fill):
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   # --- Variables ---
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
   OUTPUT_DIR="${PROJECT_ROOT}/output/<app-name>"
   VENDOR_DIR="${SCRIPT_DIR}/vendor"
   LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
   TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
   QEMU_RISCV64="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
   SYSROOT=""

   # Source-specific (TODO: fill from Phase 1 analysis)
   SOURCE_DIR="${VENDOR_DIR}/<name>"
   REPO_URL="<url>"
   VERSION="<version>"

   # Colors, info/warn/error functions (copy from existing build.sh)

   # --- Argument parsing ---
   # --force, --skip-sysroot, --skip-source, -j/--jobs, --test, --help

   # --- Step 0: Prerequisites ---
   # Check: cmake, ninja, docker, LLVM_INSTALL/bin/clang

   # --- Step 1: Sysroot extraction ---
   # Per references/sysroot-extract.md

   # --- Step 2: Source clone ---
   # git clone --depth=1 to vendor/

   # --- Step 3: Cross-compile ---
   # CMake: cmake -S <source> -B <build> -DCMAKE_TOOLCHAIN_FILE=...
   # Makefile: CC=... CXX=... LDFLAGS=... make
   # TODO: fill with app-specific build commands

   # --- Step 4: Smoke test ---
   # Per references/smoke-test.md

   # --- Done ---
   # Print artifact locations and QEMU run command
   ```

3. **.gitignore template**:
   ```
   # Vendor source (too large for git)
   vendor/

   # Build output
   output/

   # Build artifacts
   *.o
   *.a
   *.so
   *.so.*
   .build/
   ```

4. **README.md template** — skeleton with sections:
   - Title + one-line description
   - Build Status table
   - Overview (what the app does, why we cross-compile it)
   - Prerequisites
   - Build commands
   - Output artifacts
   - QEMU run commands
   - Version info table (app version, LLVM version, target arch, ABI)

- [ ] **Step 2: Verify the reference is self-contained**

Read back the file and confirm: an agent with zero project context could read this file and generate a working toolchain + build.sh for a hypothetical CMake-based C++ project.

---

### Task 3: Write `references/sysroot-extract.md`

This reference doc provides the standardized Docker sysroot extraction procedure, distilled from the identical logic in `applications/yolo/ort/build.sh:106-197` and `applications/llama.cpp/build.sh:114-189`.

**Files:**
- Write: `skills/cross-compile-app/references/sysroot-extract.md`

- [ ] **Step 1: Write sysroot extraction reference**

The file must contain a complete, copy-paste-ready bash function. Study the two existing implementations and produce a unified version. Key elements:

1. **Function signature**: `extract_sysroot(sysroot_dir, extra_packages[])`
2. **Docker image**: `riscv64/ubuntu:24.04`
3. **Base packages**: `libc6-dev libstdc++-12-dev libgcc-12-dev`
4. **Container lifecycle**: `docker run --platform riscv64 --name $tmp_container -d ... tail -f /dev/null`, then `docker rm -f`
5. **trap cleanup**: `trap "docker rm -f $tmp_container 2>/dev/null || true" RETURN`
6. **Directory copy**:
   - `docker cp $container:/usr/lib` → `$sysroot/usr/lib`
   - `docker cp $container:/usr/include` → `$sysroot/usr/include`
   - `docker cp $container:/lib/riscv64-linux-gnu` → `$sysroot/lib/riscv64-linux-gnu` (llama.cpp pattern) OR extract just the dynamic linker (yolo pattern). **Use the llama.cpp pattern** (full copy) as it's more robust.
7. **Dynamic linker symlink**: `ln -sf riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1 $sysroot/lib/ld-linux-riscv64-lp64d.so.1`
8. **CRT symlinks**: Loop over `crt1.o crti.o crtn.o Scrt1.o` in `$sysroot/usr/lib/riscv64-linux-gnu/` → symlink to `$sysroot/usr/lib/`
9. **Shared library symlinks**: Loop over `libc.so* libm.so* libdl.so* librt.so* libpthread.so* libgcc_s.so* libstdc++.so*` → symlink to `$sysroot/usr/lib/`
10. **Also symlink to lib/**: Copy CRT + shared libs to `$sysroot/lib/riscv64-linux-gnu/` with relative symlinks (llama.cpp does this; yolo doesn't but should)
11. **Clean up**: `find "$sysroot" -name "libm.a" -delete` (avoid long double issues)

Include an **extra packages** section explaining that Phase 1 analysis should detect `-dev` packages needed (e.g., `libssl-dev` for OpenSSL, `libcurl4-openssl-dev` for libcurl) and pass them to `docker exec apt-get install`.

Include a **deduplication note**: if multiple applications share the same sysroot, consider symlinking to a shared sysroot or using `--skip-sysroot`.

---

### Task 4: Write `references/smoke-test.md`

This reference doc provides the QEMU smoke test procedure with VLEN matching logic.

**Files:**
- Write: `skills/cross-compile-app/references/smoke-test.md`

- [ ] **Step 1: Write smoke test reference**

The file must contain:

1. **Verification steps** (sequential):
   ```bash
   # 1. Confirm RISC-V ELF
   file <binary>
   # Expected: ELF 64-bit LSB executable, UCB RISC-V

   # 2. Check dynamic dependencies
   ${LLVM_INSTALL}/bin/llvm-readelf -d <binary> | grep NEEDED

   # 3. Execute under QEMU
   qemu-riscv64 -L <sysroot> <binary> <test-args>
   ```

2. **VLEN matching table** (critical — must match CLAUDE.md):
   | Compile `-march` flag | QEMU `-cpu` flag | VLEN |
   |----------------------|-----------------|------|
   | (default, no zvl) | (default) | 128 |
   | `*_zvl256b` | `-cpu rv64,v=true,vlen=256` | 256 |
   | `*_zvl512b` | `-cpu rv64,v=true,vlen=512` | 512 |
   | `*_zvl1024b` | `-cpu rv64,v=true,vlen=1024` | 1024 |

   Include instructions: "Parse the `-march` value from `riscv64-linux-toolchain.cmake` to auto-select the correct QEMU `-cpu` flag."

3. **Success criteria** (ordered by strictness):
   - Exit code 0 → **pass**
   - Output contains expected string (version info, help text) → **pass**
   - Output contains "error: missing model" or "usage:" or similar → **pass** (binary is executable, just needs runtime args)
   - Exit code non-zero with unhelpful error → **fail, investigate**

4. **Common failure modes and fixes**:
   | Symptom | Cause | Fix |
   |---------|-------|-----|
   | `No such file` for shared libs | Wrong sysroot or `-L` path | Verify sysroot has the .so |
   | `ld-linux-riscv64-lp64d.so.1: not found` | Missing dynamic linker symlink | Check `$sysroot/lib/ld-linux-riscv64-lp64d.so.1` |
   | `Illegal instruction` | VLEN mismatch | Add matching `-cpu rv64,v=true,vlen=N` |
   | Silent wrong results | VLEN mismatch without `-cpu` flag | Always set `-cpu` when using ZVL extensions |

5. **Key paths reference**:
   - QEMU binary: `third_party/qemu/build/qemu-riscv64`
   - LLVM readelf: `third_party/llvm-install/bin/llvm-readelf`
   - LLVM file: `third_party/llvm-install/bin/llvm-file` (or system `file`)

---

### Task 5: Write `SKILL.md` — main orchestrator

This is the core file. It ties together the 3 references and defines the 4-phase workflow.

**Files:**
- Write: `skills/cross-compile-app/SKILL.md`

- [ ] **Step 1: Write SKILL.md with frontmatter**

Frontmatter (required for Claude Code skill discovery):
```yaml
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
```

- [ ] **Step 2: Write Purpose section**

```markdown
## Purpose

Guide the complete workflow for cross-compiling a C/C++ application to RISC-V rv64gcv
within the RVFuse project:

1. Clone source code into the applications/ directory
2. Analyze the build system and infer compilation parameters
3. Generate build.sh + toolchain cmake file
4. Extract RISC-V sysroot via Docker
5. Cross-compile the application
6. Verify via QEMU smoke test
```

- [ ] **Step 3: Write Input section**

Document the 3 required/optional inputs: `app-name`, `repo-url`, `version`.

- [ ] **Step 4: Write Phase 1 — Application Discovery**

This section must instruct the agent to:

1. Create `applications/<name>/vendor/` and clone source there
2. Analyze build system — provide the exact grep/read patterns:
   - CMake: `grep -E 'project\(|find_package\(|target_link_libraries' vendor/<name>/CMakeLists.txt`
   - Makefile: `grep -E '^(CC|CXX|LDFLAGS|LDLIBS)' vendor/<name>/Makefile`
   - Autotools: `ls vendor/<name>/configure.ac`
3. Infer RISC-V arch — provide exact grep patterns:
   - RVV: `grep -rl 'riscv_vector.h' vendor/<name>/`
   - ZFH: `grep -rl '_Float16\|__fp16' vendor/<name>/`
   - ZICBOP: `grep -rl 'prefetch\|__builtin_prefetch' vendor/<name>/`
4. Infer Docker packages — map common library names to Ubuntu `-dev` packages:
   | Library | Package |
   |---------|---------|
   | `ssl`, `crypto` | `libssl-dev` |
   | `curl` | `libcurl4-openssl-dev` |
   | `z` | `zlib1g-dev` |
   | `pthread` | (included in libc6-dev) |
   | `dl` | (included in libc6-dev) |
5. Find smoke test target — look for `add_executable` in CMake or `bin_PROGRAMS` in Makefile
6. **Ask the user about ZVL**: Present the inferred base `-march` value and ask:
   - "编译参数: `-march=rv64gcv`。是否需要添加 ZVL 扩展?"
   - Options: No (default VLEN=128), zvl256b, zvl512b, zvl1024b
7. Generate Application Profile — write to README.md

- [ ] **Step 5: Write Phase 2 — Scaffolding**

Reference `references/toolchain-template.md` for templates. Agent must:
1. Copy and customize the cmake toolchain template
2. Copy and customize the build.sh skeleton
3. Create .gitignore
4. Write README.md with Application Profile

- [ ] **Step 6: Write Phase 3 — Sysroot + Build**

Reference `references/sysroot-extract.md`. Agent must:
1. Check Docker is available
2. Run sysroot extraction (or `--skip-sysroot` if already exists)
3. Set `LLVM_INSTALL` and `SYSROOT` environment variables
4. Execute the cross-compile step from build.sh
5. Verify output: `file <binary>` shows RISC-V ELF

- [ ] **Step 7: Write Phase 4 — Smoke Test**

Reference `references/smoke-test.md`. Agent must:
1. Auto-infer QEMU `-cpu` flag from the toolchain `-march`
2. Run `qemu-riscv64 -L <sysroot> [-cpu ...] <binary> <args>`
3. Evaluate success criteria
4. Report result

- [ ] **Step 8: Write Validation Checklist**

```markdown
## Validation Checklist

Before reporting completion, verify:

- [ ] `applications/<name>/build.sh` exists, is executable, starts with `set -euo pipefail`
- [ ] `riscv64-linux-toolchain.cmake` uses `$ENV{LLVM_INSTALL}` and `$ENV{SYSROOT}`
- [ ] Sysroot at `output/<app>/sysroot/` contains `lib/ld-linux-riscv64-lp64d.so.1`
- [ ] Build output is RISC-V ELF: `file` shows "UCB RISC-V"
- [ ] QEMU can run the binary (exit 0 or expected output)
- [ ] ZVL flags in toolchain match QEMU `-cpu` vlen parameter
- [ ] README.md contains Application Profile and build instructions
```

- [ ] **Step 9: Write References section**

```markdown
## References

- `references/toolchain-template.md` — CMake toolchain + build.sh templates
- `references/sysroot-extract.md` — Docker sysroot extraction procedure
- `references/smoke-test.md` — QEMU smoke test + VLEN matching
- `applications/yolo/ort/build.sh` — ONNX Runtime build reference
- `applications/llama.cpp/build.sh` — llama.cpp build reference
- `CLAUDE.md` — VLEN mismatch warning table, key commands
```

- [ ] **Step 10: Self-review**

Read back the full SKILL.md and verify:
- All 4 phases are present and sequential
- Phase 1 mentions the ZVL user interaction point
- Phases 2-4 reference the correct `references/` files
- The skill covers both new apps and updating existing apps
- No placeholder text (no TBD, no "similar to")

---

### Task 6: Create symlink and verify

**Files:**
- Create: `.claude/skills/cross-compile-app` (symlink)

- [ ] **Step 1: Create symlink**

```bash
ln -s ../../skills/cross-compile-app .claude/skills/cross-compile-app
```

- [ ] **Step 2: Verify symlink works**

Run: `ls -la .claude/skills/cross-compile-app/SKILL.md`
Expected: symlink resolves to `skills/cross-compile-app/SKILL.md`

Run: `cat .claude/skills/cross-compile-app/SKILL.md | head -5`
Expected: YAML frontmatter with `name: cross-compile-app`

---

### Task 7: Commit

- [ ] **Step 1: Stage files**

```bash
git add skills/cross-compile-app/
git add .claude/skills/cross-compile-app
git add docs/plans/2026-04-20-cross-compile-app-skill-design.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(skills): add cross-compile-app skill for RISC-V application bootstrapping

Add a 4-phase skill (Discovery → Scaffolding → Sysroot+Build → Smoke Test)
that guides an agent through cross-compiling any C/C++ application for RISC-V
using the project's LLVM 22 toolchain, Docker sysroot, and QEMU.

Includes reference docs for sysroot extraction, smoke test with VLEN matching,
and cmake toolchain + build.sh templates.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify clean state**

Run: `git status`
Expected: clean working tree
