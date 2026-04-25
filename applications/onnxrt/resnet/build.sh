#!/usr/bin/env bash
set -euo pipefail

# ResNet ONNX Runtime build script for RVFuse worktree
# Clones fresh ORT v1.24.4 and builds ResNet inference runner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-resnet"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
ORT_SOURCE="${VENDOR_DIR}/onnxruntime"
EIGEN_SOURCE="${VENDOR_DIR}/eigen"
RUNNER_DIR="${SCRIPT_DIR}/runner"
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
SKIP_BUILD=false
BUILD_RUNNER=true
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --skip-sysroot)  SKIP_SYSROOT=true; shift ;;
        --skip-source)   SKIP_SOURCE=true; shift ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --skip-runner)   BUILD_RUNNER=false; shift ;;
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

# --- Step 2: Extract riscv64 sysroot (or use existing) ---
extract_sysroot() {
    local sysroot="${OUTPUT_DIR}/sysroot"

    # Option: reuse sysroot from main repo's cross-ort
    local shared_sysroot="${PROJECT_ROOT}/output/sysroot"
    if [ -d "${shared_sysroot}/usr" ] && [[ "${FORCE}" != "true" ]]; then
        info "Using shared sysroot from main repo: ${shared_sysroot}"
        mkdir -p "${OUTPUT_DIR}"
        ln -sfn "${shared_sysroot}" "${sysroot}"
        return 0
    fi

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

    local tmp_container="rvfuse-resnet-sysroot-prep-$$"

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

    mkdir -p "${sysroot}/usr"
    mv "${sysroot}/usr_lib_tmp"     "${sysroot}/usr/lib"
    mv "${sysroot}/usr_include_tmp" "${sysroot}/usr/include"

    mkdir -p "${sysroot}/lib/riscv64-linux-gnu"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/riscv64-linux-gnu/" || \
        error "Failed to copy dynamic linker."

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    # Create symlinks for CRT files and shared libs
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

    local lib="${sysroot}/lib/riscv64-linux-gnu"
    if [ -d "${sysroot}/usr/lib/${multilib}" ]; then
        for f in crt1.o crti.o crtn.o Scrt1.o; do
            [ -f "${sysroot}/usr/lib/${multilib}/${f}" ] && [ ! -e "${lib}/${f}" ] && \
                ln -sf "../../usr/lib/${multilib}/${f}" "${lib}/${f}"
        done
        for f in libc.so.6 libm.so.6 libdl.so.2 librt.so.1 libpthread.so.0 \
                 libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
            [ -e "${sysroot}/usr/lib/${multilib}/${f}" ] && [ ! -e "${lib}/${f}" ] && \
                ln -sf "../../usr/lib/${multilib}/${f}" "${lib}/${f}"
        done
    fi

    find "${sysroot}" -name "libm.a" -delete 2>/dev/null || true

    info "Sysroot extracted to ${sysroot}"
    echo "  $(du -sh "${sysroot}" | cut -f1)"
}

extract_sysroot

# --- Step 3: Cross-compile ONNX Runtime ---
cross_compile() {
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}"
    local sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${SKIP_BUILD}" == "true" ]]; then
        info "Skipping ORT build (--skip-build)"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -d "${ort_install}/lib" ]]; then
        info "ORT already built at ${ort_install}. Use --force to rebuild."
        return 0
    fi

    [ -d "${ORT_SOURCE}/cmake" ] || error "ORT source not found at ${ORT_SOURCE}. Run without --skip-source."
    [ -d "${sysroot}/usr" ] || error "Sysroot not found at ${sysroot}. Run without --skip-sysroot."

    info "Cross-compiling ONNX Runtime v${ONNXRUNTIME_VERSION} (rv64gcv)..."
    rm -rf "${ort_build}"
    mkdir -p "${ort_build}" "${ort_install}"

    # Generate toolchain file if missing
    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        info "Generating toolchain file..."
        cat > "${TOOLCHAIN_FILE}" << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER $ENV{LLVM_INSTALL}/bin/clang)
set(CMAKE_CXX_COMPILER $ENV{LLVM_INSTALL}/bin/clang++)

set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

set(CMAKE_C_FLAGS "--sysroot=$ENV{SYSROOT} -march=rv64gcv")
set(CMAKE_CXX_FLAGS "--sysroot=$ENV{SYSROOT} -march=rv64gcv")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")

set(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF
    fi

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
    ninja -C "${ort_build}" install

    unset SYSROOT

    info "ONNX Runtime cross-compiled to ${ort_install}"
    file "${ort_install}"/lib/libonnxruntime.so* || true
}

cross_compile

# --- Step 4: Install ORT into sysroot ---
install_ort_to_sysroot() {
    local ort_install="${OUTPUT_DIR}"
    local sysroot="${OUTPUT_DIR}/sysroot"

    # Check if sysroot is a symlink (shared sysroot)
    if [ -L "${sysroot}" ]; then
        info "Sysroot is symlinked (shared). Skipping ORT installation to sysroot."
        return 0
    fi

    local sysroot_lib="${sysroot}/usr/lib/riscv64-linux-gnu"
    local ort_so="${ort_install}/lib/libonnxruntime.so.1.24.4"

    if [ ! -f "${ort_so}" ]; then
        warn "libonnxruntime.so not found, skipping sysroot install"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -f "${sysroot_lib}/libonnxruntime.so.1.24.4" ]]; then
        info "ORT already in sysroot. Use --force to reinstall."
        return 0
    fi

    info "Installing libonnxruntime.so into sysroot..."
    cp "${ort_so}" "${sysroot_lib}/"
    ln -sf libonnxruntime.so.1.24.4 "${sysroot_lib}/libonnxruntime.so.1"
    ln -sf libonnxruntime.so.1 "${sysroot_lib}/libonnxruntime.so"

    local top_lib="${sysroot}/usr/lib"
    [ -e "${top_lib}/libonnxruntime.so.1.24.4" ] || \
        ln -sf "riscv64-linux-gnu/libonnxruntime.so.1.24.4" "${top_lib}/libonnxruntime.so.1.24.4"
    [ -e "${top_lib}/libonnxruntime.so.1" ] || \
        ln -sf "riscv64-linux-gnu/libonnxruntime.so.1" "${top_lib}/libonnxruntime.so.1"
    [ -e "${top_lib}/libonnxruntime.so" ] || \
        ln -sf "riscv64-linux-gnu/libonnxruntime.so" "${top_lib}/libonnxruntime.so"

    info "libonnxruntime.so installed to sysroot."
}

install_ort_to_sysroot

# --- Step 5: Build resnet_runner ---
build_runner() {
    if [[ "${BUILD_RUNNER}" != "true" ]]; then
        info "Skipping runner build (--skip-runner)"
        return 0
    fi

    local runner_out="${OUTPUT_DIR}/resnet_runner"
    local sysroot="${OUTPUT_DIR}/sysroot"

    # Handle symlinked sysroot
    if [ -L "${sysroot}" ]; then
        local real_sysroot
        real_sysroot=$(readlink -f "${sysroot}")
        info "Using shared sysroot: ${real_sysroot}"
        sysroot="${real_sysroot}"
    fi

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "resnet_runner already built at ${runner_out}. Use --force to rebuild."
        return 0
    fi

    [ -f "${RUNNER_DIR}/resnet_runner.cpp" ] || error "Runner source not found at ${RUNNER_DIR}"
    [ -f "${OUTPUT_DIR}/lib/libonnxruntime.so" ] || error "libonnxruntime.so not found. Run ORT build first."
    [ -d "${sysroot}/usr" ] || error "Sysroot not found at ${sysroot}"

    info "Cross-compiling resnet_runner..."

    local COMMON_FLAGS=(
        --target=riscv64-unknown-linux-gnu
        --sysroot="${sysroot}"
        -march=rv64gcv
        -isystem "${sysroot}/usr/include/riscv64-linux-gnu"
        -std=c++17
        -O2
        -g
        -fuse-ld=lld
        -I"${OUTPUT_DIR}/include/onnxruntime"
        -I"${OUTPUT_DIR}/include/onnxruntime/core/session"
        -I"${RUNNER_DIR}"
    )

    local SYSROOT_LIB="${sysroot}/usr/lib"

    "${LLVM_INSTALL}/bin/clang++" \
        "${COMMON_FLAGS[@]}" \
        "${RUNNER_DIR}/resnet_runner.cpp" \
        -o "${runner_out}" \
        -L"${SYSROOT_LIB}" \
        -lonnxruntime

    info "resnet_runner built: ${runner_out}"
    file "${runner_out}"
}

build_runner

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT source:  ${ORT_SOURCE}"
echo "  ORT lib:     ${OUTPUT_DIR}/lib/libonnxruntime.so"
echo "  Sysroot:     ${OUTPUT_DIR}/sysroot/"
echo "  Runner:      ${OUTPUT_DIR}/resnet_runner"
echo ""
file "${OUTPUT_DIR}/lib/libonnxruntime.so" || true
file "${OUTPUT_DIR}/resnet_runner" || true