# ONNXRT Cross-Compile Build (Inference Targets) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign `tools/docker-onnxrt/` from QEMU riscv64 emulation to LLVM-based x86_64 cross-compilation, upgrade ONNXRT v1.17.3 → v1.23.2 (full build), support `inference` and `c920-inference` targets, and enable `setup.sh` Step 3 to run successfully.

**Architecture:** Use a Docker container on x86_64 with clang cross-compiler (LLVM 22.1.3 mounted via volume). Build ONNXRT via CMake + ninja with a RISC-V toolchain file. Extract sysroot from `riscv64/ubuntu:22.04`. Two target variants: `inference` (rv64gcv) and `c920-inference` (rv64gcv + 11 XThead extensions).

**Tech Stack:** Bash, Docker, LLVM 22.1.3 (clang/clang++), CMake 3.28+, Ninja, ONNX Runtime v1.23.2, Eigen 3.4.0, RISC-V GNU toolchain (ld/as from `gcc-riscv64-linux-gnu`)

---

## Background

The current `tools/docker-onnxrt/` uses `--platform riscv64` Docker images, which means all compilation runs under QEMU emulation on an x86_64 host. This is extremely slow (2-6 hours). The new approach:
- Docker container runs natively on x86_64
- clang cross-compiles to RISC-V using `--target=riscv64-unknown-linux-gnu`
- sysroot provides RISC-V libc/libs (extracted from `riscv64/ubuntu:22.04`)
- ld/as provided by `gcc-riscv64-linux-gnu` (installed in the container)

**Key ONNXRT v1.23.2 changes** from v1.17.3:
- `--use_preinstalled_eigen` removed — use `FETCHCONTENT_SOURCE_DIR_EIGEN` CMake var
- `--minimal_build` removed — this is a full build (not minimal)
- `--skip_submodule_sync` still valid
- `libonnxruntime.so` SOVERSION changed from `17` to `1`

**C920 `-march` flag** (for `c920-inference` target):
```
rv64gcv_xtheadba_xtheadbb_xtheadbs_xtheadcondmov_xtheadcmo_xtheadfmidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync_xtheadvdot
```

---

## Task Summary

| ID | Title | Estimated Time |
|----|-------|----------------|
| 1 | Update build.sh: ONNXRT v1.23.2 + clone changes | 5m |
| 2 | Create Dockerfile (x86_64 cross-compile environment) | 10m |
| 3 | Create sysroot extraction script | 10m |
| 4 | Create RISC-V toolchain file for CMake | 10m |
| 5 | Write build.sh: target dispatch, Docker build, artifact extraction | 15m |
| 6 | Update setup.sh Step 3 (un-skip, call new build.sh) | 10m |
| 7 | Attempt first build — debug ONNXRT CMake issues | 30m+ |
| 8 | Build YOLO runner binary | 15m |
| 9 | Full end-to-end test: setup.sh --target inference | 10m |

---

### Task 1: Update build.sh for ONNXRT v1.23.2 + Clone Changes

**Files:**
- Modify: `tools/docker-onnxrt/build.sh`

**Step 1: Update version constant**

```bash
# Change line:
ONNXRUNTIME_VERSION="v1.17.3"
# To:
ONNXRUNTIME_VERSION="v1.23.2"
```

**Step 2: Update `clone_if_missing` to shallow clone**

```bash
clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${version} (shallow, no submodules) ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}"
    fi
}
```

**Step 3: Add ONNXRT submodule fetch after cloning**

Insert after the `clone_if_missing` calls:

```bash
# Fetch onnxruntime submodules with retries (needed for full build)
if [ -d "${VENDOR_DIR}/onnxruntime" ]; then
    echo "=== Fetching onnxruntime submodules ==="
    cd "${VENDOR_DIR}/onnxruntime"
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
    cd - > /dev/null
fi
```

**Step 4: Commit**

```bash
git add tools/docker-onnxrt/build.sh
git commit -m "build(docker-onnxrt): ONNXRT v1.23.2, shallow clone + submodule fetch"
```

---

### Task 2: Create x86_64 Cross-Compile Dockerfile

**Files:**
- Create: `tools/docker-onnxrt/Dockerfile` (overwrite existing)

**Step 1: Write the new Dockerfile**

```dockerfile
# =============================================================================
# tools/docker-onnxrt/Dockerfile
# x86_64 cross-compilation environment for RISC-V 64-bit targets.
# LLVM toolchain is mounted at runtime via -v (not baked in).
# =============================================================================
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install build tools + RISC-V cross-compiler toolchain (for ld/as only)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gpg wget software-properties-common \
        && rm -rf /var/lib/apt/lists/ \
    && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        git \
        ca-certificates \
        make \
        libprotobuf-dev \
        protobuf-compiler \
        libflatbuffers-dev \
        gcc-riscv64-linux-gnu \
        binutils-riscv64-linux-gnu \
        && rm -rf /var/lib/apt/lists/ \
    && mkdir -p /riscv64-gcc-bin \
    && ln -sf /usr/bin/riscv64-linux-gnu-ld /riscv64-gcc-bin/ld \
    && ln -sf /usr/bin/riscv64-linux-gnu-as /riscv64-gcc-bin/as

# Work directory
WORKDIR /build

# ORT source, Eigen, LLVM toolchain, sysroot, and YOLO runner are
# mounted via volumes at runtime (see build.sh).
```

Key differences from old Dockerfile:
- No `--platform riscv64` — runs on x86_64
- Single stage (no multi-stage build)
- Installs `gcc-riscv64-linux-gnu` for ld/as (not for compilation)
- No `COPY` directives — source mounted via `-v`
- `libprotobuf-dev` + `protobuf-compiler` + `libflatbuffers-dev` for full ONNXRT build

**Step 2: Update build.sh Docker build invocation**

In `build.sh`, change:

```bash
# From:
DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    --network=host \
    -t rvfuse-yolo-builder \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"

# To:
DOCKER_BUILDKIT=1 docker build \
    -t rvfuse-xcompile-builder \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${SCRIPT_DIR}"
```

**Step 3: Update all image name references**

```bash
# From: rvfuse-yolo-builder
# To:   rvfuse-xcompile-builder
```

Affects: `docker create`, `docker rmi` lines.

**Step 4: Commit**

```bash
git add tools/docker-onnxrt/Dockerfile tools/docker-onnxrt/build.sh
git commit -m "build(docker-onnxrt): x86_64 cross-compile Dockerfile"
```

---

### Task 3: Create Sysroot Extraction Logic

**Files:**
- Modify: `tools/docker-onnxrt/build.sh` (add sysroot extraction function)

Reference: `tools/c920-onnxrt/build.sh` `extract_sysroot()` function. Adapt for docker-onnxrt.

**Step 1: Add sysroot extraction function to build.sh**

```bash
extract_sysroot() {
    local sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${FORCE}" != "true" && -d "${sysroot}/usr" ]]; then
        info "Sysroot already exists at ${sysroot}. Use --force to re-extract."
        return 0
    fi

    info "Extracting riscv64 sysroot from riscv64/ubuntu:22.04..."
    rm -rf "${sysroot}"
    mkdir -p "${sysroot}"

    local tmp_container="rvfuse-sysroot-prep-$$"

    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:22.04 tail -f /dev/null > /dev/null
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN

    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev \
        libstdc++-11-dev \
        libgcc-11-dev \
        > /dev/null

    info "Copying sysroot directories from container..."

    docker cp "${tmp_container}:/usr/lib"      "${sysroot}/usr_lib_tmp"
    docker cp "${tmp_container}:/usr/include"  "${sysroot}/usr_include_tmp"

    mkdir -p "${sysroot}/usr"
    mv "${sysroot}/usr_lib_tmp"     "${sysroot}/usr/lib"
    mv "${sysroot}/usr_include_tmp" "${sysroot}/usr/include"

    mkdir -p "${sysroot}/lib/riscv64-linux-gnu"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/riscv64-linux-gnu/"

    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    local lib="${sysroot}/lib/riscv64-linux-gnu"
    for f in libc.so.6 libcrypt.so.1 libdl.so.2 libm.so.6 libpthread.so.0 librt.so.1 \
             libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
        [ -e "${sysroot}/usr/lib/riscv64-linux-gnu/${f}" ] && \
            ln -sf "../../usr/lib/riscv64-linux-gnu/${f}" "${lib}/${f}"
    done

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    local multilib="riscv64-linux-gnu"
    for dir in usr/lib; do
        local base="${sysroot}/${dir}"
        [ -d "${base}/${multilib}" ] || continue
        for f in crt1.o crti.o crtn.o; do
            [ -f "${base}/${multilib}/${f}" ] && ln -sf "${multilib}/${f}" "${base}/${f}"
        done
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0; do
            [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ] && \
                ln -sf "${multilib}/${f}" "${base}/${f}"
        done
    done

    find "${sysroot}" -name "libm.a" -delete 2>/dev/null || true

    info "Sysroot extracted to ${sysroot}"
}
```

**Step 2: Commit**

```bash
git add tools/docker-onnxrt/build.sh
git commit -m "build(docker-onnxrt): add sysroot extraction from riscv64/ubuntu:22.04"
```

---

### Task 4: Create RISC-V Toolchain File for CMake

**Files:**
- Create: `tools/docker-onnxrt/riscv64-linux-toolchain.cmake`

Copy from `tools/c920-onnxrt/riscv64-linux-toolchain.cmake` with minor modifications.

**Step 1: Write the toolchain file**

```cmake
# Cross-compilation toolchain for RISC-V 64-bit using LLVM
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
# Add the riscv64-linux-gnu specific include dirs.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")

# Enable RVV 1.0 auto-vectorization (LLVM 22)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Use GCC's linker/assembler (ld.lld in LLVM has R_RISCV_ALIGN issues)
# -B points Clang to the directory with 'ld' and 'as' symlinks
SET(CMAKE_EXE_LINKER_FLAGS "-B/riscv64-gcc-bin -Wl,-Bdynamic")
SET(CMAKE_SHARED_LINKER_FLAGS "-B/riscv64-gcc-bin -Wl,-Bdynamic")
```

**Step 2: Commit**

```bash
git add tools/docker-onnxrt/riscv64-linux-toolchain.cmake
git commit -m "build(docker-onnxrt): add RISC-V CMake toolchain file"
```

---

### Task 5: Write build.sh — Target Dispatch, Docker Build, Artifact Extraction

**Files:**
- Modify: `tools/docker-onnxrt/build.sh` (rewrite the build + extraction section)

**Step 1: Add argument parsing for `--target` and `--force`**

```bash
# --- Argument parsing ---
TARGET="inference"
FORCE=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)      TARGET="$2"; shift 2 ;;
        --force)       FORCE=true; shift ;;
        -j|--jobs)     JOBS="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate target
case "$TARGET" in
    inference|c920-inference) ;;
    *) echo "Invalid target: $TARGET (must be inference|c920-inference)"; exit 1 ;;
esac
```

**Step 2: Add helper functions**

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"

info()  { echo -e "\033[0;32m=== $* ===\033[0m"; }
warn()  { echo -e "\033[1;33mWarning: $*\033[0m"; }
error() { echo -e "\033[0;31mError: $*\033[0m" >&2; exit 1; }
```

**Step 3: Write the Docker run for ONNXRT build**

For `inference` and `c920-inference` targets, run a Docker container that cross-compiles ONNXRT:

```bash
cross_compile_onnxrt() {
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}/onnxruntime"

    if [[ "${FORCE}" != "true" && -d "${ort_install}/lib" ]]; then
        info "ORT already built at ${ort_install}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling ONNX Runtime v1.23.2 (full build, target=${TARGET})..."
    rm -rf "${ort_build}" 2>/dev/null || true
    rm -rf "${ort_install}" 2>/dev/null || true
    mkdir -p "${ort_build}" "${ort_install}"

    # Determine -march flag
    local march="rv64gcv"
    if [[ "$TARGET" == "c920-inference" ]]; then
        march="rv64gcv_xtheadba_xtheadbb_xtheadbs_xtheadcondmov_xtheadcmo_xtheadfmidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync_xtheadvdot"
    fi

    docker run --rm \
        -v "${LLVM_INSTALL}:/llvm-install:ro" \
        -v "${VENDOR_DIR}/onnxruntime:/onnxruntime:ro" \
        -v "${VENDOR_DIR}/eigen:/eigen:ro" \
        -v "${OUTPUT_DIR}/sysroot:/sysroot:ro" \
        -v "${ort_build}:/build" \
        -v "${ort_install}:/install" \
        -v "${SCRIPT_DIR}/riscv64-linux-toolchain.cmake:/toolchain.cmake:ro" \
        -e LLVM_INSTALL=/llvm-install \
        -e SYSROOT=/sysroot \
        -w /build \
        rvfuse-xcompile-builder \
        bash -c "
            cmake /onnxruntime/cmake \
                -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake \
                -DCMAKE_INSTALL_PREFIX=/install \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_FLAGS='-march=${march} -Wno-stringop-overflow -Wno-unknown-warning-option' \
                -DCMAKE_C_FLAGS='-march=${march}' \
                -Donnxruntime_BUILD_SHARED_LIB=ON \
                -Donnxruntime_BUILD_UNIT_TESTS=OFF \
                -Donnxruntime_DISABLE_RTTI=ON \
                -DFETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -G Ninja \
            && ninja -j${JOBS} \
            && ninja install/strip
        "

    info "ONNX Runtime cross-compiled to ${ort_install}"
}
```

**Step 4: Write the YOLO runner cross-compile step**

```bash
build_yolo_runner() {
    local ort_install="${OUTPUT_DIR}/onnxruntime"
    local runner_out="${OUTPUT_DIR}/yolo_inference"

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "YOLO runner already built. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling YOLO runner (target=${TARGET})..."

    local march="rv64gcv"
    if [[ "$TARGET" == "c920-inference" ]]; then
        march="rv64gcv_xtheadba_xtheadbb_xtheadbs_xtheadcondmov_xtheadcmo_xtheadfmidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync_xtheadvdot"
    fi

    docker run --rm \
        -v "${LLVM_INSTALL}:/llvm-install:ro" \
        -v "${VENDOR_DIR}/onnxruntime:/onnxruntime:ro" \
        -v "${PROJECT_ROOT}/tools/yolo_runner:/runner:ro" \
        -v "${OUTPUT_DIR}/sysroot:/sysroot:ro" \
        -v "${ort_install}:/onnxruntime-install:ro" \
        -v "${OUTPUT_DIR}:/out" \
        -e LLVM_INSTALL=/llvm-install \
        -e SYSROOT=/sysroot \
        rvfuse-xcompile-builder \
        bash -c "
            /llvm-install/bin/clang++ \
                --target=riscv64-unknown-linux-gnu \
                --sysroot=/sysroot \
                -isystem /sysroot/usr/include/riscv64-linux-gnu \
                -B/riscv64-gcc-bin \
                -march=${march} \
                -std=c++17 -O2 -g \
                -I/onnxruntime-install/include/onnxruntime \
                -I/onnxruntime-install/include/onnxruntime/core/session \
                -I/runner \
                /runner/yolo_runner.cpp \
                -o /out/yolo_inference \
                -L/onnxruntime-install/lib \
                -lonnxruntime \
                -Wl,-rpath,'\$ORIGIN'
        "

    info "YOLO runner built: ${runner_out}"
}
```

**Step 5: Write the sysroot post-processing step**

After extracting artifacts from the Docker container, set up the sysroot for QEMU:

```bash
setup_sysroot() {
    local sysroot="${OUTPUT_DIR}/sysroot"

    # Create ld-linux symlink (QEMU default path)
    ln -sf riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1 "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1" 2>/dev/null || true

    # Create ORT soname symlinks
    if [ -f "${sysroot}/lib/riscv64-linux-gnu/libonnxruntime.so.1" ]; then
        ln -sf libonnxruntime.so.1 "${sysroot}/lib/riscv64-linux-gnu/libonnxruntime.so"
    fi
}
```

**Step 6: Wire up the main flow**

```bash
# --- Main flow ---
clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"
clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${VENDOR_DIR}/eigen"

# Fetch onnxruntime submodules
# (see Task 1)

extract_sysroot
build_image
cross_compile_onnxrt
build_yolo_runner
setup_sysroot

info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT:       ${OUTPUT_DIR}/onnxruntime/"
echo "  Runner:    ${OUTPUT_DIR}/yolo_inference"
echo "  Sysroot:   ${OUTPUT_DIR}/sysroot/"
echo ""
file "${OUTPUT_DIR}/yolo_inference" || true
```

**Step 7: Commit**

```bash
git add tools/docker-onnxrt/build.sh
git commit -m "build(docker-onnxrt): rewrite build.sh for x86_64 cross-compile"
```

---

### Task 6: Update setup.sh Step 3

**Files:**
- Modify: `setup.sh`
- Create: `STEP3_ARTIFACTS_C920_INFERENCE` array

**Step 1: Add C920 inference artifacts**

In `setup.sh`, add after `STEP3_ARTIFACTS_INFERENCE`:

```bash
readonly -a STEP3_ARTIFACTS_C920_INFERENCE=(
    "output/yolo_inference"
    "output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1"
)
```

**Step 2: Update `get_artifact_names()` for c920-inference**

The existing `get_artifact_names()` case for step 3 returns `STEP3_ARTIFACTS_${TARGET^^}`. For `c920-inference`, `TARGET^^` gives `C920-INFERENCE` which doesn't work as a bash variable name. Need to add a special case:

```bash
get_artifact_names() {
    case "$1" in
        0) echo "STEP0_ARTIFACTS" ;;
        1) echo "STEP1_ARTIFACTS" ;;
        2) echo "STEP2_ARTIFACTS" ;;
        3)
            case "$TARGET" in
                c920-inference) echo "STEP3_ARTIFACTS_C920_INFERENCE" ;;
                *) echo "STEP3_ARTIFACTS_${TARGET^^}" ;;
            esac
            ;;
        4) echo "__bbv__" ;;
        5) echo "STEP5_ARTIFACTS" ;;
        6) echo "__dfg__" ;;
        7) echo "STEP7_ARTIFACTS" ;;
        8) echo "STEP8_ARTIFACTS" ;;
    esac
}
```

**Step 3: Un-skip Step 3 in `step3_docker_build()`**

Replace the skip block with:

```bash
step3_docker_build() {
    local step=3
    log_info "=== Step ${step}: ${STEP_NAMES[$step]} (target=${TARGET}) ==="

    if ! bash "${PROJECT_ROOT}/tools/docker-onnxrt/build.sh" --target "${TARGET}" 2>&1; then
        record_step_result "$step" "FAIL" "tools/docker-onnxrt/build.sh exited with error"
        return 1
    fi

    record_step_result "$step" "PASS" "yolo_inference + sysroot ready"
    return 0
}
```

**Step 4: Commit**

```bash
git add setup.sh
git commit -m "feat(setup): enable Step 3 with new cross-compile build, add c920-inference support"
```

---

### Task 7: Debug ONNXRT v1.23.2 Full Build

**Files:**
- Modify: `tools/docker-onnxrt/build.sh` (CMake flags, Dockerfile deps, toolchain)
- Modify: `tools/docker-onnxrt/riscv64-linux-toolchain.cmake`

**Step 1: Attempt the first build**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/update-ort
./tools/docker-onnxrt/build.sh --target inference
```

**Step 2: Debug expected failures**

The first build will likely fail. Common issues to check:

1. **Missing CMake dependencies**: Full ONNXRT build may need additional packages not in minimal build. Check CMake error output.
2. **Protobuf version mismatch**: v1.23.2 may need a different protobuf version than Ubuntu 22.04 provides.
3. **Cross-compilation issues**: CMake may fail to find RISC-V headers/libs in sysroot.
4. **Missing `-march` support**: If clang rejects the `-march` flag, try `rv64gc` first as a fallback.

For each failure:
- Read the CMake/build error
- Check if a missing dependency can be added to the Dockerfile
- Check if a CMake flag needs adjustment
- Retry the build

**Step 3: Iterate until ONNXRT compiles**

After each fix:
```bash
./tools/docker-onnxrt/build.sh --target inference --force
```

**Step 4: Commit each fix**

```bash
git add <fixed files>
git commit -m "fix(docker-onnxrt): <description of fix>"
```

---

### Task 8: Build YOLO Runner

**Files:**
- Modify: `tools/docker-onnxrt/build.sh` (build_yolo_runner function)
- Modify: `tools/yolo_runner/CMakeLists.txt` (if needed)

**Step 1: Attempt to build YOLO runner**

```bash
./tools/docker-onnxrt/build.sh --target inference
```

**Step 2: Fix compilation errors**

Common issues:
- Header paths (ONNXRT include structure may have changed in v1.23.2)
- Library name (libonnxruntime.so.1 vs .17)
- Missing `-march` flag or sysroot paths

Check the clang++ output and adjust `build_yolo_runner()` in `build.sh`.

**Step 3: Verify the binary**

```bash
file output/yolo_inference
# Expected: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV)

# Check dynamic linking
riscv64-linux-gnu-readelf -d output/yolo_inference 2>/dev/null || readelf -d output/yolo_inference
# Should show libonnxruntime.so as NEEDED
```

**Step 4: Commit**

```bash
git add tools/docker-onnxrt/build.sh
git commit -m "fix(docker-onnxrt): YOLO runner cross-compile successful"
```

---

### Task 9: End-to-End Test with setup.sh

**Files:**
- No file changes (validation only)

**Step 1: Run setup.sh with inference target**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/update-ort
./setup.sh --target inference --force 3
```

This forces Step 3 (YOLO build) to re-run.

**Step 2: Verify Step 3 passes**

Check the output for:
```
Step 3: YOLO Build  [PASS]   yolo_inference + sysroot ready
```

**Step 3: Verify artifacts exist**

```bash
ls -la output/yolo_inference
ls -la output/sysroot/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1
ls -la output/onnxruntime/lib/libonnxruntime.so*
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify setup.sh --target inference passes end-to-end"
```

---

### Task 10: Verify c920-inference Target (Optional)

**Files:**
- No file changes (validation only)

**Step 1: Run setup.sh with c920-inference target**

```bash
./setup.sh --target c920-inference --force 3
```

**Step 2: Verify the binary uses the correct -march**

```bash
riscv64-linux-gnu-readelf -A output/yolo_inference | grep -i tag_arch
# Should show rv64gcv with XThead extensions
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: verify setup.sh --target c920-inference passes"
```

---

## Downstream Impact

| Component | Needs Change? | Notes |
|---|---|---|
| `setup.sh` | Yes | Step 3 un-skip, c920-inference artifacts, target dispatch |
| `prepare_model.sh` | No | Unchanged |
| `verify_bbv.sh` | No | Already merged from master |
| `tools/analyze_bbv.py` | No | Uses `.disas` + BBV output, unchanged |
| `tools/dfg/` | No | Unchanged |
| `tools/c920-onnxrt/` | No | Independent build script, left as-is |
| `tools/yolo_runner/yolo_runner.cpp` | No | Source unchanged; compilation flags change via build.sh |
| `output/sysroot/` | Format change | New extraction method (from c920-onnxrt pattern) |
