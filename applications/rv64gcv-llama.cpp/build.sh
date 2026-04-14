#!/usr/bin/env bash
set -euo pipefail

# Cross-compile llama.cpp for RISC-V rv64gcv using LLVM 22
# Output: llama-cli, llama-server executables for RISC-V

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/llama.cpp"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
LLAMA_SOURCE="${VENDOR_DIR}/llama.cpp"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"

# Use sysroot from rv64gcv-onnxrt if available (shared resource)
ORT_OUTPUT="${PROJECT_ROOT}/output/cross-ort"
ORT_SYSROOT="${ORT_OUTPUT}/sysroot"

LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_VERSION="b8783"  # Latest release as of 2026-04-14

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
USE_SHARED_SYSROOT=true
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --skip-sysroot)  SKIP_SYSROOT=true; shift ;;
        --skip-source)   SKIP_SOURCE=true; shift ;;
        --standalone)    USE_SHARED_SYSROOT=false; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        *)               error "Unknown argument: $1" ;;
    esac
done

# --- Step 0: Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    command -v cmake &>/dev/null || error "cmake not found. Install cmake >= 3.14."
    command -v ninja &>/dev/null || error "ninja not found. Install ninja-build."
    command -v git &>/dev/null || error "git not found."

    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"

    info "All prerequisites met."
    echo "  LLVM:    ${LLVM_INSTALL}/bin/clang --version"
    "${LLVM_INSTALL}/bin/clang" --version | head -1 || true
}

check_prerequisites

# --- Step 1: Sysroot ---
# Prefer shared sysroot from rv64gcv-onnxrt to avoid duplicate Docker extraction
setup_sysroot() {
    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        info "Skipping sysroot setup (--skip-sysroot)"
        SYSROOT="${ORT_SYSROOT}"
        return 0
    fi

    if [[ "${USE_SHARED_SYSROOT}" == "true" && -d "${ORT_SYSROOT}/usr" ]]; then
        info "Using shared sysroot from rv64gcv-onnxrt: ${ORT_SYSROOT}"
        SYSROOT="${ORT_SYSROOT}"
        return 0
    fi

    # Standalone sysroot (for isolated builds)
    local standalone_sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${FORCE}" != "true" && -d "${standalone_sysroot}/usr" ]]; then
        info "Standalone sysroot already exists at ${standalone_sysroot}. Use --force to re-extract."
        SYSROOT="${standalone_sysroot}"
        return 0
    fi

    info "Extracting standalone riscv64 sysroot from riscv64/ubuntu:24.04..."
    command -v docker &>/dev/null || error "Docker not found. Sysroot extraction requires Docker."

    rm -rf "${standalone_sysroot}"
    mkdir -p "${standalone_sysroot}"

    local tmp_container="rvfuse-llama-sysroot-prep-$$"

    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:24.04 tail -f /dev/null > /dev/null
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN

    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev \
        libstdc++-12-dev \
        libgcc-12-dev \
        > /dev/null

    info "Copying sysroot directories from container..."

    docker cp "${tmp_container}:/usr/lib"      "${standalone_sysroot}/usr_lib_tmp"
    docker cp "${tmp_container}:/usr/include"  "${standalone_sysroot}/usr_include_tmp"
    # Copy ALL runtime libs from /lib/riscv64-linux-gnu/ (not just ld-linux)
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu" "${standalone_sysroot}/lib_riscv_tmp"

    mkdir -p "${standalone_sysroot}/usr" "${standalone_sysroot}/lib"
    mv "${standalone_sysroot}/usr_lib_tmp"     "${standalone_sysroot}/usr/lib"
    mv "${standalone_sysroot}/usr_include_tmp" "${standalone_sysroot}/usr/include"
    mv "${standalone_sysroot}/lib_riscv_tmp"   "${standalone_sysroot}/lib/riscv64-linux-gnu"

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    # Top-level dynamic linker symlink
    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${standalone_sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    # Create symlinks for CRT and shared libs
    local multilib="riscv64-linux-gnu"
    local base="${standalone_sysroot}/usr/lib"
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

    # Remove problematic static libs
    find "${standalone_sysroot}" -name "libm.a" -delete 2>/dev/null || true

    SYSROOT="${standalone_sysroot}"
    info "Standalone sysroot ready at ${SYSROOT}"
    echo "  $(du -sh "${SYSROOT}" | cut -f1)"
}

setup_sysroot

# --- Step 2: Clone llama.cpp source ---
clone_source() {
    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        info "Skipping source cloning (--skip-source)"
        return 0
    fi

    if [ -d "${LLAMA_SOURCE}" ] && [ -n "$(ls -A "${LLAMA_SOURCE}" 2>/dev/null)" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            info "Force re-clone: removing existing ${LLAMA_SOURCE}..."
            rm -rf "${LLAMA_SOURCE}"
        else
            info "Skipping ${LLAMA_SOURCE} (already exists)"
            return 0
        fi
    fi

    info "Cloning ${LLAMA_REPO} @ ${LLAMA_VERSION} (shallow)..."
    mkdir -p "${VENDOR_DIR}"
    git clone --depth=1 --branch "${LLAMA_VERSION}" "${LLAMA_REPO}" "${LLAMA_SOURCE}"

    info "llama.cpp source ready."
}

clone_source

# --- Step 3: Cross-compile llama.cpp ---
cross_compile() {
    local llama_build="${OUTPUT_DIR}/.build"
    local llama_install="${OUTPUT_DIR}"

    if [[ "${FORCE}" != "true" && -f "${llama_install}/bin/llama-cli" ]]; then
        info "llama.cpp already built at ${llama_install}. Use --force to rebuild."
        return 0
    fi

    [ -f "${LLAMA_SOURCE}/CMakeLists.txt" ] || error "llama.cpp source not found at ${LLAMA_SOURCE}. Run without --skip-source."
    [ -d "${SYSROOT}/usr" ] || error "Sysroot not found at ${SYSROOT}."

    info "Cross-compiling llama.cpp ${LLAMA_VERSION} (rv64gcv + RVV + ZFH)..."

    rm -rf "${llama_build}"
    mkdir -p "${llama_build}" "${llama_install}/bin"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"

    # Configure with RISC-V vectorization enabled
    # GGML_RVV: RISC-V Vector extension (RVV 1.0)
    # GGML_RV_ZFH: Half-precision float support
    # GGML_RV_ZICBOP: Cache block operations (CBOP)
    # GGML_RV_ZIHINTPAUSE: Pause hint for spin loops
    cmake -S "${LLAMA_SOURCE}" -B "${llama_build}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${llama_install}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_RVV=ON \
        -DGGML_RV_ZFH=ON \
        -DGGML_RV_ZICBOP=ON \
        -DGGML_RV_ZIHINTPAUSE=ON \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_WEBUI=OFF \
        -G Ninja

    info "Building (ninja -j${JOBS})..."
    ninja -C "${llama_build}" -j"${JOBS}"

    info "Installing binaries..."
    ninja -C "${llama_build}" install/strip

    unset SYSROOT

    info "llama.cpp cross-compiled to ${llama_install}"
}

cross_compile

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  CLI:    ${OUTPUT_DIR}/bin/llama-cli"
echo "  Server: ${OUTPUT_DIR}/bin/llama-server"
echo "  Lib:    ${OUTPUT_DIR}/lib/"
echo "  Sysroot: ${OUTPUT_DIR}/sysroot"
echo ""
file "${OUTPUT_DIR}/bin/llama-cli" || true
file "${OUTPUT_DIR}/bin/llama-server" || true
echo ""
echo "Usage with QEMU:"
echo "  qemu-riscv64 -L ${OUTPUT_DIR}/sysroot ${OUTPUT_DIR}/bin/llama-cli -m <model.gguf> -p \"Hello\""
echo "  qemu-riscv64 -L ${OUTPUT_DIR}/sysroot ${OUTPUT_DIR}/bin/llama-server -m <model.gguf> --port 8080"