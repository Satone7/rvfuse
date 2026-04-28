#!/usr/bin/env bash
# build.sh — Cross-compile SuperGlue runner for RISC-V (rv64gcv zvl512b)
# Reuses ONNX Runtime build from SuperPoint (output/cross-superpoint)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-superglue"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-zvl512b-toolchain.cmake"
RUNNER_DIR="${SCRIPT_DIR}/runner"

# Reuse ORT build from SuperPoint
EXISTING_ORT="${PROJECT_ROOT}/output/cross-superpoint"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        *)               error "Unknown argument: $1" ;;
    esac
done

# --- Step 0: Prerequisites ---
info "Checking prerequisites..."
command -v cmake &>/dev/null || error "cmake not found."
command -v ninja &>/dev/null || error "ninja not found."
[ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
[ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"

info "Reusing ONNX Runtime build: ${EXISTING_ORT}"
[ -d "${EXISTING_ORT}/lib" ] || error "ORT lib not found at ${EXISTING_ORT}/lib"
[ -f "${EXISTING_ORT}/lib/libonnxruntime.so" ] || error "libonnxruntime.so not found"

# Symlink ORT artifacts
mkdir -p "${OUTPUT_DIR}"
ln -sfn "${EXISTING_ORT}/lib" "${OUTPUT_DIR}/lib"
ln -sfn "${EXISTING_ORT}/include" "${OUTPUT_DIR}/include"

# Reuse sysroot from SuperPoint
if [ -d "${EXISTING_ORT}/sysroot/usr" ]; then
    ln -sfn "${EXISTING_ORT}/sysroot" "${OUTPUT_DIR}/sysroot"
    info "Reusing sysroot from SuperPoint"
else
    error "Sysroot not found at ${EXISTING_ORT}/sysroot"
fi

# --- Step 1: Cross-compile SuperGlue runner ---
build_runner() {
    local ort_install="${OUTPUT_DIR}"
    local sysroot="${OUTPUT_DIR}/sysroot"
    local runner_build="${OUTPUT_DIR}/runner_build"

    [ -d "${ort_install}/include/onnxruntime" ] || error "ORT headers not found at ${ort_install}/include"
    [ -d "${ort_install}/lib" ] || error "ORT lib not found at ${ort_install}/lib"

    info "Cross-compiling SuperGlue runner..."

    mkdir -p "${runner_build}"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${sysroot}"

    cmake -S "${RUNNER_DIR}" -B "${runner_build}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DORT_INSTALL_DIR="${ort_install}" \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja

    ninja -C "${runner_build}" -j"${JOBS}"

    # Copy the binary to output
    cp "${runner_build}/superglue_inference" "${OUTPUT_DIR}/"
    chmod +x "${OUTPUT_DIR}/superglue_inference"

    unset SYSROOT

    info "SuperGlue runner built: ${OUTPUT_DIR}/superglue_inference"
    file "${OUTPUT_DIR}/superglue_inference" || true
}

build_runner

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT:         ${OUTPUT_DIR}/lib/libonnxruntime.so"
echo "  Runner:      ${OUTPUT_DIR}/superglue_inference"
echo "  Sysroot:     ${OUTPUT_DIR}/sysroot/"
echo "  Model:       ${SCRIPT_DIR}/model/superglue_gnn.onnx"
echo ""
file "${OUTPUT_DIR}/superglue_inference" || true
