# Toolchain Templates

Reusable templates for cross-compiling C/C++ applications to RISC-V (rv64gcv).
Copy and customize these templates during the Scaffolding phase.

---

## Section 1: CMake Toolchain File

File: `riscv64-linux-toolchain.cmake`

```cmake
# Cross-compilation toolchain for RISC-V 64-bit using LLVM 22
# Target: <PLACEHOLDER_ARCH>, Linux, lp64d ABI
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

SET(CMAKE_C_COMPILER $ENV{LLVM_INSTALL}/bin/clang)
SET(CMAKE_CXX_COMPILER $ENV{LLVM_INSTALL}/bin/clang++)

SET(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
SET(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

SET(CMAKE_SYSROOT $ENV{SYSROOT})
SET(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})

# Clang doesn't automatically add triplet-specific include paths like GCC does.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")

# Architecture: agent fills this based on application requirements.
# See parameterization guide below.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=<PLACEHOLDER_ARCH>")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=<PLACEHOLDER_ARCH>")

# Use lld for linking (no GCC cross-compiler dependency).
# LLVM 22's lld has mature RISC-V support and handles R_RISCV_ALIGN correctly.
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
SET(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

### Parameterization Guide

Replace `<PLACEHOLDER_ARCH>` with the appropriate `-march` value:

| Scenario | `-march` value | Notes |
|----------|---------------|-------|
| Basic RV64GCV | `rv64gcv` | Default for most applications. Includes IMAFDC + V. |
| With ZFH (half-precision float) | `rv64gcv_zfh` | Needed when the application uses `__fp16` or GGML_RV_ZFH. |
| With ZICBOP (cache-block prefetch) | `rv64gcv_zicbop` | Enables `prefetch.i`/`prefetch.r`/`prefetch.w` instructions. |
| With ZVL512 | `rv64gcv_zvl512b` | Fixed VLEN=512. Use when the application hardcodes `__riscv_v_fixed_vlen`. |
| llama.cpp pattern | `rv64gcv_zfh_zba_zicbop` | Half-float + bitmanip + prefetch. Used by GGML RVV kernels. |

**Environment variables required at build time:**

```bash
export LLVM_INSTALL=/path/to/third_party/llvm-install
export SYSROOT=/path/to/output/<app-name>/sysroot
```

---

## Section 2: build.sh Skeleton

File: `build.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Cross-compile <APP_NAME> for RISC-V using LLVM 22
#
# Usage:
#   build.sh [OPTIONS]
#
# Options:
#   --force          Rebuild everything from scratch
#   --skip-sysroot   Skip sysroot extraction (use existing)
#   --skip-source    Skip source cloning
#   -j, --jobs N     Parallel build jobs (default: nproc)
#   --test           Run smoke test after build
#   --help           Show this help message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"  # Adjust depth as needed
OUTPUT_DIR="${PROJECT_ROOT}/output/<APP_NAME>"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
QEMU_RISCV64="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
SYSROOT=""  # Set by extract_sysroot()

# TODO: Set application-specific variables
# SOURCE_DIR="${VENDOR_DIR}/<repo-name>"
# REPO_URL="https://github.com/<org>/<repo>.git"
# REPO_VERSION="<tag-or-commit>"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Help ---
show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Cross-compile <APP_NAME> for RISC-V using LLVM 22.

Options:
  --force          Rebuild everything from scratch
  --skip-sysroot   Skip sysroot extraction (use existing)
  --skip-source    Skip source cloning
  -j, --jobs N     Parallel build jobs (default: nproc)
  --test           Run smoke test after build
  --help           Show this help message

Output artifacts:
  Binary:  ${OUTPUT_DIR}/bin/<binary-name>
  Sysroot: ${OUTPUT_DIR}/sysroot

Run with QEMU:
  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot \\
    ${OUTPUT_DIR}/bin/<binary-name> [args...]
EOF
}

# --- Argument parsing ---
FORCE=false
SKIP_SYSROOT=false
SKIP_SOURCE=false
RUN_TEST=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)      show_help; exit 0 ;;
        --force)        FORCE=true; shift ;;
        --skip-sysroot) SKIP_SYSROOT=true; shift ;;
        --skip-source)  SKIP_SOURCE=true; shift ;;
        --test)         RUN_TEST=true; shift ;;
        -j|--jobs)      JOBS="$2"; shift 2 ;;
        -j*)            JOBS="${1#-j}"; shift ;;
        *)              error "Unknown argument: $1 (use --help for usage)" ;;
    esac
done

# --- Step 0: Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    command -v cmake &>/dev/null || error "cmake not found. Install cmake >= 3.14."
    command -v ninja &>/dev/null || error "ninja not found. Install ninja-build."
    command -v docker &>/dev/null || error "docker not found. Sysroot extraction requires Docker."

    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"
    [ -f "${QEMU_RISCV64}" ] || error "qemu-riscv64 not found at ${QEMU_RISCV64}"

    info "All prerequisites met."
    echo "  LLVM: ${LLVM_INSTALL}/bin/clang --version"
    "${LLVM_INSTALL}/bin/clang" --version | head -1 || true
    echo "  QEMU: ${QEMU_RISCV64}"
}

check_prerequisites

# --- Step 1: Sysroot extraction ---
# See references/sysroot-extract.md for the full sysroot extraction procedure.
extract_sysroot() {
    local sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        if [ ! -d "${sysroot}/usr" ]; then
            error "Sysroot not found at ${sysroot} (remove --skip-sysroot to extract)"
        fi
        declare -g SYSROOT="${sysroot}"
        info "Using existing sysroot: ${SYSROOT}"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -d "${sysroot}/usr" ]]; then
        info "Sysroot already exists at ${sysroot}. Use --force to re-extract."
        declare -g SYSROOT="${sysroot}"
        return 0
    fi

    info "Extracting riscv64 sysroot from riscv64/ubuntu:24.04..."
    # TODO: Full sysroot extraction logic (see references/sysroot-extract.md)
    # ... docker run, apt-get, docker cp, symlinks ...

    declare -g SYSROOT="${sysroot}"
    info "Sysroot ready at ${SYSROOT}"
    echo "  $(du -sh "${SYSROOT}" | cut -f1)"
}

extract_sysroot

# --- Step 2: Source clone ---
clone_source() {
    # TODO: Set SOURCE_DIR and REPO_URL above before uncommenting.

    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        info "Skipping source cloning (--skip-source)"
        return 0
    fi

    if [ -d "${SOURCE_DIR:-}" ] && [ -n "$(ls -A "${SOURCE_DIR:-}" 2>/dev/null)" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            info "Force re-clone: removing existing ${SOURCE_DIR}..."
            rm -rf "${SOURCE_DIR}"
        else
            info "Skipping ${SOURCE_DIR} (already exists)"
            return 0
        fi
    fi

    # TODO: Uncomment and set variables
    # info "Cloning ${REPO_URL} @ ${REPO_VERSION} (shallow)..."
    # mkdir -p "${VENDOR_DIR}"
    # git clone --depth=1 --branch "${REPO_VERSION}" "${REPO_URL}" "${SOURCE_DIR}"
    # info "Source ready."
}

clone_source

# --- Step 3: Cross-compile ---
cross_compile() {
    local build_dir="${OUTPUT_DIR}/.build"
    local install_dir="${OUTPUT_DIR}"

    # TODO: Adjust the check for your application's output indicator
    # if [[ "${FORCE}" != "true" && -f "${install_dir}/bin/<binary>" ]]; then
    #     info "Already built at ${install_dir}. Use --force to rebuild."
    #     return 0
    # fi

    # TODO: Adjust source directory check
    # [ -f "${SOURCE_DIR}/CMakeLists.txt" ] || error "Source not found. Run without --skip-source."
    [ -d "${SYSROOT}/usr" ] || error "Sysroot not found at ${SYSROOT}."

    info "Cross-compiling <APP_NAME>..."
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}" "${install_dir}/bin"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"

    # CMake projects:
    cmake -S "${SOURCE_DIR}" -B "${build_dir}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${install_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        # TODO: Add application-specific cmake flags here
        -G Ninja

    info "Building (ninja -j${JOBS})..."
    ninja -C "${build_dir}" -j"${JOBS}"

    info "Installing..."
    ninja -C "${build_dir}" install/strip

    # Makefile / non-cmake alternative (uncomment if needed):
    # mkdir -p "${build_dir}"
    # env LLVM_INSTALL="${LLVM_INSTALL}" SYSROOT="${SYSROOT}" \
    #     make -C "${SOURCE_DIR}" -j"${JOBS}" \
    #         CC="${LLVM_INSTALL}/bin/clang" \
    #         CXX="${LLVM_INSTALL}/bin/clang++" \
    #         PREFIX="${install_dir}" \
    #         # TODO: Add makefile-specific flags

    unset SYSROOT

    info "<APP_NAME> cross-compiled to ${install_dir}"
}

cross_compile

# --- Step 4: Smoke test ---
smoke_test() {
    # See references/smoke-test.md for the full smoke test procedure.
    local binary="${OUTPUT_DIR}/bin/<binary-name>"

    [ -f "${binary}" ] || error "Binary not found at ${binary}"

    info "Running smoke test..."
    if "${QEMU_RISCV64}" -L "${SYSROOT}" "${binary}" --help &>/dev/null; then
        info "Smoke test PASSED"
    else
        warn "Smoke test: binary did not produce expected exit code (may be expected for some apps)"
    fi
}

if [[ "${RUN_TEST}" == "true" ]]; then
    smoke_test
fi

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  Binary:  ${OUTPUT_DIR}/bin/<binary-name>"
echo "  Sysroot: ${OUTPUT_DIR}/sysroot"
echo ""
file "${OUTPUT_DIR}/bin/<binary-name>" || true
echo ""
echo "Run with QEMU:"
echo "  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot ${OUTPUT_DIR}/bin/<binary-name> [args...]"
```

---

## Section 3: .gitignore Template

File: `.gitignore` (place in the application directory, e.g., `applications/<app-name>/`)

```
# Build artifacts
.build/
output/

# Vendor sources (cloned at build time)
vendor/

# Object files and archives
*.o
*.a
*.so
*.so.*

# Compiled binaries
*.exe
*.out

# IDE / editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Model / data files (often large)
*.gguf
*.ggml
*.onnx
*.ort
*.jpg
*.png
```

---

## Section 4: README.md Template

File: `README.md` (place in the application directory, e.g., `applications/<app-name>/`)

```markdown
# <APP_NAME> - RISC-V Cross-Compilation

Cross-compiled <APP_NAME> for RISC-V 64-bit (rv64gcv) using LLVM 22.

**Build Status:** ![Pending](https://img.shields.io/badge/build-pending-yellow)

## Overview

Brief description of the application and why it is included in RVFuse.
What workload/benchmark does it exercise? Which ISA features does it use?

**Target:** rv64gcv (IMAFD_C_V), Linux, lp64d ABI
**Toolchain:** LLVM 22 (clang/lld), no GCC cross-compiler dependency
**Sysroot:** Ubuntu 24.04 riscv64 (`riscv64/ubuntu:24.04` Docker image)

## Prerequisites

- Docker (for sysroot extraction)
- cmake >= 3.14, ninja-build
- LLVM 22 installed at `third_party/llvm-install/`
- QEMU riscv64 built at `third_party/qemu/build/qemu-riscv64`

## Build

```bash
# Full build (sysroot + source + compile)
./build.sh

# Rebuild from scratch
./build.sh --force

# Skip sysroot and source (incremental rebuild)
./build.sh --skip-sysroot --skip-source

# Run smoke test after build
./build.sh --test

# Parallel build with 8 jobs
./build.sh -j8
```

## Output

| Artifact | Path |
|----------|------|
| Binary | `output/<app-name>/bin/<binary-name>` |
| Libraries | `output/<app-name>/lib/` |
| Sysroot | `output/<app-name>/sysroot/` |

## QEMU Usage

```bash
QEMU="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
SYSROOT="output/<app-name>/sysroot"

# Run the cross-compiled binary
${QEMU} -L ${SYSROOT} output/<app-name>/bin/<binary-name> [args...]

# With VLEN override (if compiled for non-default VLEN)
${QEMU} -L ${SYSROOT} -cpu rv64,v=true,vlen=512 \
    output/<app-name>/bin/<binary-name> [args...]
```

## Version

- **Application:** <version-or-tag>
- **Toolchain:** LLVM 22
- **Sysroot:** riscv64/ubuntu:24.04
```
