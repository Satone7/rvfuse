# LLVM 22 Host Cross-Compile ORT v1.24.4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-compile ONNX Runtime v1.24.4 (full build) for rv64gcv on the host machine using the LLVM 22 toolchain from `third_party/llvm-install`, producing `libonnxruntime.so`.

**Architecture:** Two files — a CMake toolchain file and a build script. The build script checks prerequisites, extracts a sysroot from a `riscv64/ubuntu:24.04` container (the only Docker usage), auto-clones ORT source if missing, then runs cmake + ninja directly on the host. No Docker container for compilation, no GCC cross-compiler.

**Tech Stack:** Bash, CMake, Ninja, LLVM 22 (clang/lld/llvm-ar), Docker (sysroot extraction only), ONNX Runtime v1.24.4, Eigen 3.4.0

---

## File Map

| File | Responsibility |
|------|----------------|
| `tools/cross-compile-ort/riscv64-linux-toolchain.cmake` | CMake toolchain: target triple, sysroot, march flags, lld linker |
| `tools/cross-compile-ort/build.sh` | Build orchestrator: prerequisites, sysroot export, source clone, cmake, ninja, install |

---

### Task 1: Create the CMake toolchain file

**Files:**
- Create: `tools/cross-compile-ort/riscv64-linux-toolchain.cmake`

Reference: `tools/c920-onnxrt/riscv64-linux-toolchain.cmake` (existing pattern). Key changes: replace `-B/riscv64-gcc-bin -Wl,-Bdynamic` with `-fuse-ld=lld`.

- [ ] **Step 1: Create the toolchain file**

```cmake
# Cross-compilation toolchain for RISC-V 64-bit using LLVM 22
# Target: rv64gcv (IMAFD_C_V), Linux, lp64d ABI
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

# Enable RVV 1.0 auto-vectorization (LLVM 22)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")

# Use lld for linking (no GCC cross-compiler dependency)
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
SET(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

- [ ] **Step 2: Commit**

```bash
git add tools/cross-compile-ort/riscv64-linux-toolchain.cmake
git commit -m "feat(cross-compile-ort): add CMake toolchain file for rv64gcv"
```

---

### Task 2: Create the build script — scaffolding + prerequisites

**Files:**
- Create: `tools/cross-compile-ort/build.sh`

Reference: `tools/c920-onnxrt/build.sh` (structure, colors, argument parsing). Adapt: remove Docker build step, remove YOLO runner step, change Ubuntu to 24.04, use lld.

- [ ] **Step 1: Write the build script — header through prerequisites**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-ort"
VENDOR_DIR="${PROJECT_ROOT}/tools/docker-onnxrt/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
ORT_SOURCE="${VENDOR_DIR}/onnxruntime"
EIGEN_SOURCE="${VENDOR_DIR}/eigen"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"

ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.24.4"
EIGEN_REPO="https://gitlab.com/libeigen/eigen.git"
EIGEN_VERSION="3.4.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
FORCE=false
SKIP_SYSROOT=false
SKIP_SOURCE=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --skip-sysroot)  SKIP_SYSROOT=true; shift ;;
        --skip-source)   SKIP_SOURCE=true; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        *)               error "Unknown argument: $1" ;;
    esac
done

# --- Step 0: Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    command -v cmake &>/dev/null || error "cmake not found. Install cmake >= 3.25."
    command -v ninja &>/dev/null || error "ninja not found. Install ninja-build."
    command -v python3 &>/dev/null || error "python3 not found."

    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"

    info "All prerequisites met."
    echo "  LLVM:    ${LLVM_INSTALL}/bin/clang --version"
    "${LLVM_INSTALL}/bin/clang" --version | head -1 || true
}

check_prerequisites
```

- [ ] **Step 2: Commit**

```bash
git add tools/cross-compile-ort/build.sh
git commit -m "feat(cross-compile-ort): add build script scaffolding with prerequisites"
```

---

### Task 3: Add source cloning logic

**Files:**
- Modify: `tools/cross-compile-ort/build.sh`

Append after the `check_prerequisites` call. Reference: `tools/docker-onnxrt/build.sh:14-47` (clone_if_missing + submodule fetch).

- [ ] **Step 1: Append source cloning functions and call**

Append after `check_prerequisites`:

```bash

# --- Step 1: Clone ORT and Eigen sources ---
clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        info "Skipping ${dest} (already exists)"
    else
        info "Cloning ${repo_url} @ ${version} (shallow, no submodules)..."
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}"
    fi
}

clone_sources() {
    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        info "Skipping source cloning (--skip-source)"
        return 0
    fi

    clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${ORT_SOURCE}"
    clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${EIGEN_SOURCE}"

    # Fetch onnxruntime submodules with retries
    if [ -d "${ORT_SOURCE}" ]; then
        info "Fetching onnxruntime submodules (recursive)..."
        local prev_dir
        prev_dir="$(pwd)"
        cd "${ORT_SOURCE}"
        MAX_RETRIES=5
        for attempt in $(seq 1 ${MAX_RETRIES}); do
            echo "  Attempt ${attempt}/${MAX_RETRIES}..."
            if git submodule update --init --recursive --depth=1; then
                echo "  All submodules fetched."
                break
            fi
            echo "  Retrying in 5s..."
            sleep 5
        done
        cd "${prev_dir}"
    fi
}

clone_sources
```

- [ ] **Step 2: Commit**

```bash
git add tools/cross-compile-ort/build.sh
git commit -m "feat(cross-compile-ort): add ORT v1.24.4 and Eigen source cloning"
```

---

### Task 4: Add sysroot extraction logic

**Files:**
- Modify: `tools/cross-compile-ort/build.sh`

Append after the `clone_sources` call. Reference: `tools/c920-onnxrt/build.sh:56-141` (sysroot extraction). Key changes: Ubuntu 24.04, `libstdc++-12-dev`, remove GCC-ld specific symlinks.

- [ ] **Step 1: Append sysroot extraction function and call**

Append after `clone_sources`:

```bash

# --- Step 2: Extract riscv64 sysroot ---
extract_sysroot() {
    local sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${SKIP_SYSROOT}" == "true" && -d "${sysroot}/usr" ]]; then
        info "Skipping sysroot extraction (--skip-sysroot)"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -d "${sysroot}/usr" ]]; then
        info "Sysroot already exists at ${sysroot}. Use --force to re-extract."
        return 0
    fi

    info "Extracting riscv64 sysroot from riscv64/ubuntu:24.04..."
    command -v docker &>/dev/null || error "Docker not found. Sysroot extraction requires Docker."

    rm -rf "${sysroot}"
    mkdir -p "${sysroot}"

    local tmp_container="rvfuse-sysroot-prep-$$"

    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:24.04 tail -f /dev/null > /dev/null
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN

    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev \
        libstdc++-12-dev \
        libgcc-12-dev \
        > /dev/null

    info "Copying sysroot directories from container..."

    docker cp "${tmp_container}:/usr/lib"      "${sysroot}/usr_lib_tmp"
    docker cp "${tmp_container}:/usr/include"  "${sysroot}/usr_include_tmp"

    # Move into final locations BEFORE creating symlinks that reference them
    mkdir -p "${sysroot}/usr"
    mv "${sysroot}/usr_lib_tmp"     "${sysroot}/usr/lib"
    mv "${sysroot}/usr_include_tmp" "${sysroot}/usr/include"

    # Extract dynamic linker from /lib
    mkdir -p "${sysroot}/lib/riscv64-linux-gnu"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/riscv64-linux-gnu/" || \
        error "Failed to copy dynamic linker. Is ld-linux-riscv64-lp64d.so.1 present?"

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    # Top-level dynamic linker symlink
    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    # Create symlinks for C runtime files at top-level usr/lib/
    # so Clang/lld can find them (searches -L$SYSROOT/usr/lib but not
    # the riscv64-linux-gnu/ subdirectory by default).
    local multilib="riscv64-linux-gnu"
    local base="${sysroot}/usr/lib"
    if [ -d "${base}/${multilib}" ]; then
        for f in crt1.o crti.o crtn.o Scrt1.o; do
            [ -f "${base}/${multilib}/${f}" ] && ln -sf "${multilib}/${f}" "${base}/${f}"
        done
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0 \
                 libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
            [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ] && \
                ln -sf "${multilib}/${f}" "${base}/${f}"
        done
    fi

    # Remove problematic static libs (libm.a has __frexpl requiring long double)
    find "${sysroot}" -name "libm.a" -delete 2>/dev/null || true

    info "Sysroot extracted to ${sysroot}"
    echo "  $(du -sh "${sysroot}" | cut -f1)"
}

extract_sysroot
```

- [ ] **Step 2: Commit**

```bash
git add tools/cross-compile-ort/build.sh
git commit -m "feat(cross-compile-ort): add sysroot extraction from Ubuntu 24.04"
```

---

### Task 5: Add cmake build and install logic

**Files:**
- Modify: `tools/cross-compile-ort/build.sh`

Append after the `extract_sysroot` call. This is the main build step — cmake configure + ninja build + ninja install/strip, all running directly on the host (no `docker run`).

- [ ] **Step 1: Append cross-compile function and call**

Append after `extract_sysroot`:

```bash

# --- Step 3: Cross-compile ONNX Runtime ---
cross_compile() {
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}/onnxruntime"
    local sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${FORCE}" != "true" && -d "${ort_install}/lib" ]]; then
        info "ORT already built at ${ort_install}. Use --force to rebuild."
        return 0
    fi

    [ -d "${ORT_SOURCE}/cmake" ] || error "ORT source not found at ${ORT_SOURCE}. Run without --skip-source."
    [ -d "${sysroot}/usr" ] || error "Sysroot not found at ${sysroot}. Run without --skip-sysroot."

    info "Cross-compiling ONNX Runtime v${ONNXRUNTIME_VERSION} (full build, rv64gcv)..."
    rm -rf "${ort_build}"
    mkdir -p "${ort_build}" "${ort_install}"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${sysroot}"

    cmake -S "${ORT_SOURCE}/cmake" -B "${ort_build}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${ort_install}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=OFF \
        -Donnxruntime_DISABLE_RTTI=ON \
        -DFETCHCONTENT_SOURCE_DIR_EIGEN="${EIGEN_SOURCE}" \
        -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option' \
        -DIconv_IS_BUILT_IN=TRUE \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -G Ninja

    info "Building (ninja -j${JOBS})..."
    ninja -C "${ort_build}" -j"${JOBS}"

    info "Installing..."
    ninja -C "${ort_build}" install/strip

    unset LLVM_INSTALL
    unset SYSROOT

    info "ONNX Runtime cross-compiled to ${ort_install}"
    file "${ort_install}/lib/libonnxruntime.so*" || true
}

cross_compile

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT:     ${OUTPUT_DIR}/onnxruntime/"
echo "  Sysroot: ${OUTPUT_DIR}/sysroot/"
echo ""
file "${OUTPUT_DIR}/onnxruntime/lib/libonnxruntime.so" || true
```

- [ ] **Step 2: Make the script executable and commit**

```bash
chmod +x tools/cross-compile-ort/build.sh
git add tools/cross-compile-ort/build.sh
git commit -m "feat(cross-compile-ort): add cmake cross-compile and install steps"
```

---

### Task 6: Create symlink for llvm-install and test the build

**Files:**
- No new files (creates a symlink + runs the build)

This is the validation step — ensure the llvm-install symlink exists from the main repo, then run the full build.

- [ ] **Step 1: Create the llvm-install symlink if missing**

```bash
# From the worktree root
LLVM_INSTALL="third_party/llvm-install"
if [ ! -e "${LLVM_INSTALL}" ]; then
    ln -s /home/pren/wsp/rvfuse/third_party/llvm-install "${LLVM_INSTALL}"
    echo "Created symlink: ${LLVM_INSTALL} -> /home/pren/wsp/rvfuse/third_party/llvm-install"
else
    echo "llvm-install already exists at ${LLVM_INSTALL}"
fi
```

Note: Do NOT commit this symlink — it's local to the worktree. The main repo has the actual directory.

- [ ] **Step 2: Run the full build**

```bash
./tools/cross-compile-ort/build.sh -j$(nproc)
```

Expected: The script runs through all three steps (source clone, sysroot extraction, cmake build). The final output should show:

```
=== All done! ===

Artifacts:
  ORT:     output/cross-ort/onnxruntime/
  Sysroot: output/cross-ort/sysroot/

output/cross-ort/onnxruntime/lib/libonnxruntime.so: ELF 64-bit LSB shared object, UCB RISC-V, version 1 (SYSV), dynamically linked ...
```

- [ ] **Step 3: Verify the output is a valid RISC-V ELF**

```bash
file output/cross-ort/onnxruntime/lib/libonnxruntime.so
readelf -h output/cross-ort/onnxruntime/lib/libonnxruntime.so | grep -E '(Class|Machine|Flags)'
```

Expected:
- `ELF 64-bit LSB shared object, UCB RISC-V`
- `Machine:.*RISC-V`
- Flags should include `RVC` and `RV64I` (and ideally V/D/C/F/M/A extensions)

- [ ] **Step 4: Commit if any fixes were needed during the build**

If the build succeeded without modifications, no commit is needed. If you had to fix the toolchain file or build script, commit the fixes:

```bash
git add -A
git commit -m "fix(cross-compile-ort): fixes from initial build validation"
```
