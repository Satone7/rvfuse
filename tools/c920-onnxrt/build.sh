#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/c920-ort"
VENDOR_DIR="${PROJECT_ROOT}/tools/docker-onnxrt/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
ORT_SOURCE="${VENDOR_DIR}/onnxruntime"
EIGEN_SOURCE="${VENDOR_DIR}/eigen"
YOLO_RUNNER="${PROJECT_ROOT}/tools/yolo_runner"
DOCKER_IMAGE="rvfuse/c920-onnxrt-builder:latest"

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
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --skip-sysroot)  SKIP_SYSROOT=true; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        *)               error "Unknown argument: $1" ;;
    esac
done

# --- Step 0: Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    command -v docker &>/dev/null || error "Docker not found. Please install Docker."
    docker info &>/dev/null || error "Docker daemon not running."

    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -d "${ORT_SOURCE}/cmake" ] || error "ORT source not found at ${ORT_SOURCE}. Run tools/docker-onnxrt/build.sh first or clone manually."
    [ -d "${EIGEN_SOURCE}" ]     || error "Eigen not found at ${EIGEN_SOURCE}"
    [ -f "${YOLO_RUNNER}/yolo_runner.cpp" ] || error "YOLO runner not found at ${YOLO_RUNNER}"

    info "All prerequisites met."
}

check_prerequisites
