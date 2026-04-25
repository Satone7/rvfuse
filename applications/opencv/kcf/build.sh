#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OPENCV_DIR="${WORKTREE_ROOT}/output/opencv"
KCF_DIR="${SCRIPT_DIR}"
OUTPUT_DIR="${OPENCV_DIR}/kcf"
LLVM_INSTALL="${WORKTREE_ROOT}/third_party/llvm-install"
SYSROOT="${OPENCV_DIR}/sysroot"
TOOLCHAIN_FILE="${SCRIPT_DIR}/../riscv64-linux-toolchain.cmake"
QEMU="${WORKTREE_ROOT}/third_party/qemu/build/qemu-riscv64"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# Check OpenCV
[ -d "${OPENCV_DIR}/lib" ] || error "OpenCV not found. Run ../build.sh first."
[ -f "${OPENCV_DIR}/lib/libopencv_core.so" ] || error "libopencv_core.so not found"

export LLVM_INSTALL="${LLVM_INSTALL}"
export SYSROOT="${SYSROOT}"
export OPENCV_EXTRA_MODULES_PATH="${WORKTREE_ROOT}/applications/opencv/vendor/opencv_contrib/modules"

BUILD_DIR="${KCF_DIR}/build"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

info "Configuring KCFcpp..."
cmake -S "${KCF_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja \
    -DOpenCV_DIR="${OPENCV_DIR}/lib/cmake/opencv4"

info "Building KCFcpp..."
ninja -C "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)"

# Copy binaries to output
cp "${BUILD_DIR}/kcf_video" "${OUTPUT_DIR}/kcf_video"
cp "${BUILD_DIR}/KCF" "${OUTPUT_DIR}/KCF"
chmod +x "${OUTPUT_DIR}/kcf_video" "${OUTPUT_DIR}/KCF"

info "Build complete:"
info "  Video runner: ${OUTPUT_DIR}/kcf_video"
info "  Image runner: ${OUTPUT_DIR}/KCF"
file "${OUTPUT_DIR}/kcf_video"