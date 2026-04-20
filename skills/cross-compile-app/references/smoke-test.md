# QEMU Smoke Test Procedure

This document provides the final validation procedure after cross-compiling a RISC-V application.
The smoke test confirms the binary is executable under QEMU user-mode emulation.

## Overview

The smoke test validates:
1. The binary is a valid RISC-V ELF executable
2. Dynamic dependencies are satisfied by the sysroot
3. The binary executes correctly under QEMU with proper VLEN matching

**CRITICAL**: QEMU defaults to VLEN=128. Binaries compiled with `zvl*b` extensions require
explicit `-cpu rv64,v=true,vlen=<N>` flag. Without proper matching, vector operations may
produce incorrect results silently.

---

## Section 1: Verification Steps

Execute these steps sequentially after cross-compilation.

### Step 1.1: Confirm RISC-V ELF Format

```bash
file <binary>
```

Expected output:
```
ELF 64-bit LSB executable, UCB RISC-V, RISC-V ISA version RV64I, ...
```

If output shows a different architecture (e.g., x86-64), the cross-compilation failed.

### Step 1.2: Check Dynamic Dependencies

```bash
# Using LLVM readelf (if LLVM_INSTALL is set)
${LLVM_INSTALL}/bin/llvm-readelf -d <binary> | grep NEEDED

# Or using the docker-llvm wrapper
tools/docker-llvm/riscv-readelf -d <binary> | grep NEEDED

# Or using system file command to identify interpreter
file <binary>
```

This reveals required shared libraries. Common dependencies:
- `libc.so.6` (always required)
- `libm.so.6` (math library)
- `ld-linux-riscv64-lp64d.so.1` (dynamic linker)

Verify each library exists in sysroot:
```bash
# Check for a specific library
ls -la <sysroot>/lib/riscv64-linux-gnu/<library_name>

# List all available shared libraries
ls <sysroot>/lib/riscv64-linux-gnu/*.so*
```

### Step 1.3: Determine VLEN Requirement

Parse the `-march` value from your toolchain file to select the correct QEMU CPU flag.

```bash
# Extract march from toolchain file
grep -E "march=" <toolchain-file> | head -1
```

See **Section 2: VLEN Matching Table** for the mapping rules.

### Step 1.4: Execute Under QEMU

```bash
# Basic execution (no vector extensions or zvl128b)
qemu-riscv64 -L <sysroot> <binary> <test-args>

# With vector extensions requiring specific VLEN
qemu-riscv64 -L <sysroot> -cpu rv64,v=true,vlen=<N> <binary> <test-args>
```

Example with YOLO inference:
```bash
# Default VLEN (128)
qemu-riscv64 -L output/sysroot \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg

# With zvl512b extension
qemu-riscv64 -L output/sysroot -cpu rv64,v=true,vlen=512 \
  ./output/yolo_inference ./output/yolo11n.ort ./output/test.jpg
```

---

## Section 2: VLEN Matching Table

**This table is critical for correct vector program execution.**

| Compile `-march` flag | QEMU `-cpu` flag | VLEN (bits) |
|----------------------|-----------------|-------------|
| (default, no zvl) | (default, no flag) | 128 |
| `*_zvl128b` | (default, no flag) | 128 |
| `*_zvl256b` | `-cpu rv64,v=true,vlen=256` | 256 |
| `*_zvl512b` | `-cpu rv64,v=true,vlen=512` | 512 |
| `*_zvl1024b` | `-cpu rv64,v=true,vlen=1024` | 1024 |

### Auto-Selection Logic

Parse the `-march` value from `riscv64-linux-toolchain.cmake` to auto-select the correct QEMU `-cpu` flag:

```
If -march contains "zvl<N>b" (e.g., zvl512b):
    Set vlen = N
    Use: -cpu rv64,v=true,vlen=<N>

If -march does NOT contain any zvl flag:
    Use default (no -cpu flag needed)
```

### Extraction Script

```bash
# Extract and parse march value
march=$(grep -oP 'march=\K[^")\s]+' riscv64-linux-toolchain.cmake | head -1)

if [[ "$march" =~ zvl([0-9]+)b ]]; then
    vlen="${BASH_REMATCH[1]}"
    cpu_flag="-cpu rv64,v=true,vlen=${vlen}"
    echo "Detected zvl${vlen}b, using: ${cpu_flag}"
else
    cpu_flag=""
    echo "No zvl extension detected, using default VLEN=128"
fi
```

### Common -march Examples

| Toolchain File | `-march` Value | QEMU Flag |
|----------------|----------------|-----------|
| `applications/yolo/ort/riscv64-linux-toolchain.cmake` | `rv64gcv` | (default) |
| `applications/llama.cpp/riscv64-linux-toolchain.cmake` | `rv64gcv_zfh_zba_zicbop` | (default) |
| hypothetical zvl512b toolchain | `rv64gcv_zvl512b` | `-cpu rv64,v=true,vlen=512` |

**Note**: Standard `rv64gcv` without explicit `zvl*b` defaults to VLEN=128, which matches QEMU's default.

---

## Section 3: Success Criteria

Evaluate smoke test results in this order (most lenient to strictest):

### Pass Criteria (Ordered by Strictness)

| Criterion | Result | Notes |
|-----------|--------|-------|
| Exit code 0 | **PASS** | Binary executes successfully |
| Output contains expected string (version, help text) | **PASS** | Binary produces recognizable output |
| Output contains "missing input file/model" error | **PASS** | Binary is executable, just needs proper inputs |
| Exit code non-zero with unhelpful error | **FAIL** | Investigate with debugging flags |

### Interpretation Examples

**PASS - Successful execution:**
```
$ qemu-riscv64 -L output/sysroot ./binary --version
v1.2.3
$ echo $?  # Exit code
0
```

**PASS - Missing input files (binary still works):**
```
$ qemu-riscv64 -L output/sysroot ./binary inference model.onnx
Error: Failed to load model.onnx: No such file or directory
$ echo $?
1
# Result: PASS - binary correctly reports missing input, architecture is correct
```

**FAIL - Execution error:**
```
$ qemu-riscv64 -L output/sysroot ./binary
Illegal instruction
$ echo $?
139  # SIGILL
# Result: FAIL - investigate VLEN mismatch or ISA extension support
```

---

## Section 4: Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No such file or directory` for shared libs | Wrong sysroot or `-L` path | Verify sysroot contains required `.so` files |
| `ld-linux-riscv64-lp64d.so.1: not found` | Missing dynamic linker symlink | Check `$sysroot/lib/ld-linux-riscv64-lp64d.so.1` exists |
| `Illegal instruction` (exit code 139) | VLEN mismatch or unsupported ISA | Add matching `-cpu rv64,v=true,vlen=N` flag |
| Silent wrong results | VLEN mismatch without `-cpu` flag | Always set `-cpu` when using ZVL extensions |
| `cannot execute: required executable not found` | Wrong ELF architecture | Re-run cross-compilation |
| Segmentation fault (exit code 139) | ABI mismatch or stack alignment | Check toolchain lp64d vs lp64 consistency |

### Debugging Techniques

Enable QEMU debugging flags for deeper investigation:

```bash
# Trace system calls
qemu-riscv64 -d strace -L <sysroot> <binary>

# Log unimplemented instructions
qemu-riscv64 -d unimp -L <sysroot> <binary>

# Show CPU state before each TB
qemu-riscv64 -d cpu -L <sysroot> <binary>

# Combine flags
qemu-riscv64 -d strace,unimp -L <sysroot> <binary> 2>&1 | tee debug.log
```

### Sysroot Verification Checklist

```bash
# 1. Check dynamic linker exists
ls -la <sysroot>/lib/ld-linux-riscv64-lp64d.so.1

# 2. Check libc exists
ls -la <sysroot>/lib/riscv64-linux-gnu/libc.so.6
ls -la <sysroot>/lib/riscv64-linux-gnu/libc-*.so

# 3. Verify library symlink chain
ls -la <sysroot>/lib/riscv64-linux-gnu/libm.so.6
ls -la <sysroot>/lib/riscv64-linux-gnu/libm-*.so

# 4. Check usr/lib libraries
ls <sysroot>/usr/lib/riscv64-linux-gnu/*.so*
```

---

## Section 5: Key Paths Reference

| Item | Path | Notes |
|------|------|-------|
| QEMU binary | `third_party/qemu/build/qemu-riscv64` | Built by `verify_bbv.sh` |
| LLVM readelf | `${LLVM_INSTALL}/bin/llvm-readelf` | Environment variable or `third_party/llvm-install/bin/llvm-readelf` |
| Docker-llvm wrapper | `tools/docker-llvm/riscv-readelf` | Alternative when LLVM_INSTALL not set |
| System `file` command | `file` | Standard Unix utility, no path needed |
| Sysroot | `output/sysroot` | Default for YOLO/ORT applications |
| Application-specific sysroot | e.g., `output/llama.cpp/sysroot` | Use application-specific path when available |

### LLVM_INSTALL Resolution

The LLVM_INSTALL path varies by setup:

```bash
# Option 1: Environment variable (recommended)
export LLVM_INSTALL=/path/to/llvm-install
${LLVM_INSTALL}/bin/llvm-readelf -d <binary>

# Option 2: Project default path
third_party/llvm-install/bin/llvm-readelf -d <binary>

# Option 3: Docker wrapper (no LLVM_INSTALL needed)
tools/docker-llvm/riscv-readelf -d <binary>
```

---

## Section 6: Integration in build.sh

The smoke test should be integrated into the build.sh script after cross-compilation.

### Skeleton Integration Pattern

```bash
#!/usr/bin/env bash
set -euo pipefail

# ... existing build.sh preamble ...

# --- Step 4: Cross-compile ---
cross_compile() {
    info "Cross-compiling..."
    # ... cmake build commands ...
}

# --- Step 5: Smoke test ---
smoke_test() {
    local binary="${OUTPUT_DIR}/<binary_name>"
    local sysroot="${SYSROOT}"  # From environment or default

    info "Running smoke test on ${binary}..."

    # 4.1: Verify ELF format
    local elf_type
    elf_type=$(file "${binary}")
    if [[ ! "${elf_type}" =~ "RISC-V" ]]; then
        error "Binary is not RISC-V ELF: ${elf_type}"
    fi
    echo "  ELF format: OK (${elf_type})"

    # 4.2: Extract VLEN requirement from toolchain
    local march cpu_flag
    march=$(grep -oP 'march=\K[^")\s]+' "${TOOLCHAIN_FILE}" | head -1)
    if [[ "$march" =~ zvl([0-9]+)b ]]; then
        cpu_flag="-cpu rv64,v=true,vlen=${BASH_REMATCH[1]}"
        echo "  VLEN: ${BASH_REMATCH[1]} (from ${march})"
    else
        cpu_flag=""
        echo "  VLEN: default (128)"
    fi

    # 4.3: Quick execution test
    local qemu_bin="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
    local test_args="--version"  # or "--help", depends on binary

    echo "  Executing: ${qemu_bin} -L ${sysroot} ${cpu_flag} ${binary} ${test_args}"
    local exit_code=0
    ${qemu_bin} -L "${sysroot}" ${cpu_flag} "${binary}" ${test_args} || exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  Result: PASS (exit code 0)"
    elif [[ ${exit_code} -ne 0 ]]; then
        # Check if output contains expected error (missing input)
        warn "Smoke test exit code: ${exit_code}"
        warn "If error is about missing input files/models, binary is likely OK."
        warn "Run manually with correct inputs to verify: ${qemu_bin} -L ${sysroot} ${cpu_flag} ${binary}"
    fi
}

# --- Argument parsing (add --test flag) ---
RUN_TEST=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)     RUN_TEST=true; shift ;;
        --force)    FORCE=true; shift ;;
        # ... other flags ...
        *)          error "Unknown argument: $1" ;;
    esac
done

# --- Main execution ---
cross_compile

if [[ "${RUN_TEST}" == "true" ]]; then
    smoke_test
else
    info "Skipping smoke test (use --test to run)"
fi

info "Build complete. Binary at: ${OUTPUT_DIR}/<binary_name>"
```

### Recommended build.sh Flow

```
Step 0: Prerequisites check
Step 1: Clone/prepare sources
Step 2: Extract sysroot
Step 3: Configure CMake
Step 4: Cross-compile
Step 5: Smoke test (optional, gated by --test flag)
```

### Auto-Run vs Manual Trigger

Two approaches for smoke test invocation:

1. **Gated by flag (recommended for production):**
   ```bash
   ./build.sh            # Skips smoke test
   ./build.sh --test     # Runs smoke test after build
   ```

2. **Auto-run (recommended for development):**
   ```bash
   # Always run smoke test, but continue on "missing input" errors
   smoke_test || warn "Smoke test failed, check manually"
   ```

---

## Quick Reference Card

```bash
# Full smoke test sequence
BINARY=./output/your_binary
SYSROOT=./output/sysroot
TOOLCHAIN=./riscv64-linux-toolchain.cmake
QEMU=third_party/qemu/build/qemu-riscv64

# 1. Check ELF
file $BINARY

# 2. Check dependencies
tools/docker-llvm/riscv-readelf -d $BINARY | grep NEEDED

# 3. Extract VLEN
march=$(grep -oP 'march=\K[^")\s]+' $TOOLCHAIN)
[[ "$march" =~ zvl([0-9]+)b ]] && cpu="-cpu rv64,v=true,vlen=${BASH_REMATCH[1]}" || cpu=""

# 4. Execute
$QEMU -L $SYSROOT $cpu $BINARY --version

# 5. Verify result
echo "Exit code: $?"
```