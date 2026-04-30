#!/usr/bin/env bash
set -euo pipefail

# Cross-compile OpenCV for RISC-V rv64gcv_zvl256b using LLVM 22
# Output: OpenCV libraries and opencv_version binary for RISC-V
#
# Usage:
#   build.sh [OPTIONS]
#
# Options:
#   --force          Rebuild everything from scratch
#   --skip-sysroot   Skip sysroot extraction (use existing)
#   --skip-source    Skip OpenCV source cloning
#   --help           Show this help message
#   -j, --jobs N     Parallel build jobs (default: nproc)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/opencv"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
OPENCV_SOURCE="${VENDOR_DIR}/opencv"
OPENCV_CONTRIB_SOURCE="${VENDOR_DIR}/opencv_contrib"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
QEMU_RISCV64="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
SYSROOT=""  # Set by setup_sysroot()

OPENCV_REPO="https://github.com/opencv/opencv.git"
OPENCV_VERSION="4.10.0"
OPENCV_CONTRIB_REPO="https://github.com/opencv/opencv_contrib.git"
OPENCV_CONTRIB_VERSION="4.10.0"

# Colors
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

Cross-compile OpenCV 4.10.0 for RISC-V rv64gcv_zvl256b using LLVM 22.

Options:
  --force          Rebuild everything from scratch
  --skip-sysroot   Skip sysroot extraction (use existing)
  --skip-source    Skip OpenCV source cloning
  -j, --jobs N     Parallel build jobs (default: nproc)
  --help           Show this help message

Output artifacts:
  Libraries:  ${OUTPUT_DIR}/lib/libopencv_*.so
  Headers:    ${OUTPUT_DIR}/include/opencv2/
  Binaries:   ${OUTPUT_DIR}/bin/opencv_version
  Sysroot:    ${OUTPUT_DIR}/sysroot

Run with QEMU:
  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot -cpu rv64,v=true,vlen=256 \\
    ${OUTPUT_DIR}/bin/opencv_version

Version info:
  OpenCV:     ${OPENCV_VERSION}
  opencv_contrib: ${OPENCV_CONTRIB_VERSION}
  Target:     rv64gcv_zvl256b
EOF
}

# --- Argument parsing ---
FORCE=false
SKIP_SYSROOT=false
SKIP_SOURCE=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)      show_help; exit 0 ;;
        --force)        FORCE=true; shift ;;
        --skip-sysroot) SKIP_SYSROOT=true; shift ;;
        --skip-source)  SKIP_SOURCE=true; shift ;;
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
    command -v docker &>/dev/null || error "docker not found. Required for sysroot extraction."
    command -v git &>/dev/null || error "git not found."

    LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"
    [ -f "${QEMU_RISCV64}" ] || warn "qemu-riscv64 not found at ${QEMU_RISCV64} (smoke test will be skipped)"
}

# --- Step 1: Clone source ---
clone_source() {
    info "Cloning OpenCV source..."
    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        warn "Skipping source cloning (--skip-source)"
        return 0
    fi

    mkdir -p "${VENDOR_DIR}"

    # Clone OpenCV main
    if [ -d "${OPENCV_SOURCE}" ] && [ -n "$(ls -A "${OPENCV_SOURCE}")" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            warn "Removing existing OpenCV source (--force)"
            rm -rf "${OPENCV_SOURCE}"
        else
            info "OpenCV source already exists, skipping clone"
        fi
    fi

    if [ ! -d "${OPENCV_SOURCE}" ] || [ -z "$(ls -A "${OPENCV_SOURCE}")" ]; then
        git clone --depth=1 --branch "${OPENCV_VERSION}" "${OPENCV_REPO}" "${OPENCV_SOURCE}"
    fi

    # Clone opencv_contrib
    if [ -d "${OPENCV_CONTRIB_SOURCE}" ] && [ -n "$(ls -A "${OPENCV_CONTRIB_SOURCE}")" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            warn "Removing existing opencv_contrib source (--force)"
            rm -rf "${OPENCV_CONTRIB_SOURCE}"
        else
            info "opencv_contrib source already exists, skipping clone"
        fi
    fi

    if [ ! -d "${OPENCV_CONTRIB_SOURCE}" ] || [ -z "$(ls -A "${OPENCV_CONTRIB_SOURCE}")" ]; then
        git clone --depth=1 --branch "${OPENCV_CONTRIB_VERSION}" "${OPENCV_CONTRIB_REPO}" "${OPENCV_CONTRIB_SOURCE}"
    fi

    info "Source clone complete: ${OPENCV_VERSION} (opencv), ${OPENCV_CONTRIB_VERSION} (opencv_contrib)"
}

# --- Step 2: Sysroot extraction ---
extract_sysroot() {
    info "Extracting sysroot from Docker..."
    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        warn "Skipping sysroot extraction (--skip-sysroot)"
        SYSROOT="${OUTPUT_DIR}/sysroot"
        return 0
    fi

    SYSROOT="${OUTPUT_DIR}/sysroot"
    mkdir -p "${SYSROOT}"

    # Check if sysroot already exists
    if [ -f "${SYSROOT}/lib/ld-linux-riscv64-lp64d.so.1" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            warn "Removing existing sysroot (--force)"
            rm -rf "${SYSROOT}"
            mkdir -p "${SYSROOT}"
        else
            info "Sysroot already exists at ${SYSROOT}, skipping extraction"
            return 0
        fi
    fi

    # Extra packages for OpenCV (minimal: std libs, zlib, png, jpeg are built by OpenCV)
    # Note: libgomp-dev may not be available in minimal image, skip if not needed
    EXTRA_PACKAGES="libc6-dev libstdc++-10-dev"

    # Extract sysroot from riscv64/ubuntu:24.04
    DOCKER_IMAGE="riscv64/ubuntu:24.04"
    CONTAINER_NAME="rvfuse-opencv-sysroot-prep"

    info "Using Docker image: ${DOCKER_IMAGE}"

    # Clean up any existing container
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    # Create container and install packages
    docker run --platform riscv64 --name "${CONTAINER_NAME}" -d "${DOCKER_IMAGE}" sleep infinity

    # Install extra packages
    if [ -n "${EXTRA_PACKAGES}" ]; then
        docker exec "${CONTAINER_NAME}" apt-get update -qq
        docker exec "${CONTAINER_NAME}" apt-get install -y -qq ${EXTRA_PACKAGES}
    fi

    # Extract sysroot
    docker exec "${CONTAINER_NAME}" tar -cf - -C / lib usr 2>/dev/null | tar -xf - -C "${SYSROOT}"

    # Clean up container
    docker rm -f "${CONTAINER_NAME}"

    info "Sysroot extracted to ${SYSROOT}"
}

# --- Step 3: Cross-compile ---
cross_compile() {
    info "Cross-compiling OpenCV..."
    LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"
    export OPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB_SOURCE}/modules"

    BUILD_DIR="${OUTPUT_DIR}/build"
    INSTALL_DIR="${OUTPUT_DIR}"

    # Clean build directory if force rebuild
    if [[ "${FORCE}" == "true" ]] && [ -d "${BUILD_DIR}" ]; then
        rm -rf "${BUILD_DIR}"
    fi

    mkdir -p "${BUILD_DIR}"

    # CMake configuration
    info "Configuring CMake..."
    cmake -S "${OPENCV_SOURCE}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
        -DOPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB_SOURCE}/modules" \
        -DBUILD_LIST="core,imgproc,features2d,imgcodecs,calib3d,flann,highgui,video,videoio,tracking" \
        -DRISCV_RVV_SCALABLE=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF \
        -DWITH_EIGEN=OFF \
        -DBUILD_ZLIB=ON \
        -DBUILD_PNG=ON \
        -DBUILD_JPEG=ON \
        -DOPENCV_ENABLE_NONFREE=ON

    # Build
    info "Building OpenCV (${JOBS} jobs)..."
    ninja -C "${BUILD_DIR}" -j"${JOBS}"

    # Install
    info "Installing OpenCV..."
    ninja -C "${BUILD_DIR}" install

    info "Build complete: ${INSTALL_DIR}/lib/libopencv_*.so"
}

# --- Step 4: Verify output ---
verify_output() {
    info "Verifying output..."

    # Check libraries
    if [ -f "${OUTPUT_DIR}/lib/libopencv_core.so" ]; then
        info "libopencv_core.so found"
        file "${OUTPUT_DIR}/lib/libopencv_core.so" | grep -q "RISC-V" || error "libopencv_core.so is not RISC-V ELF"
    else
        error "libopencv_core.so not found"
    fi

    # Check tracking module
    if [ -f "${OUTPUT_DIR}/lib/libopencv_tracking.so" ]; then
        info "libopencv_tracking.so found (KCF tracker module)"
    else
        warn "libopencv_tracking.so not found (tracking module may be disabled)"
    fi

    # Check opencv_version binary
    if [ -f "${OUTPUT_DIR}/bin/opencv_version" ]; then
        info "opencv_version binary found"
        file "${OUTPUT_DIR}/bin/opencv_version" | grep -q "RISC-V" || error "opencv_version is not RISC-V ELF"
    else
        warn "opencv_version not found (apps module may be disabled)"
    fi

    info "Output verification complete"
}

# --- Step 5: Smoke test ---
smoke_test() {
    info "Running smoke test under QEMU..."
    if [ ! -f "${QEMU_RISCV64}" ]; then
        warn "QEMU not available, skipping smoke test"
        return 0
    fi

    if [ ! -f "${OUTPUT_DIR}/bin/opencv_version" ]; then
        warn "opencv_version binary not found, skipping smoke test"
        return 0
    fi

    LD_LIBRARY_PATH="${OUTPUT_DIR}/lib"

    info "Running: opencv_version"
    ${QEMU_RISCV64} -L "${SYSROOT}" -E LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        -cpu rv64,v=true,vlen=256 \
        "${OUTPUT_DIR}/bin/opencv_version"

    info "Smoke test: PASS"
}

# --- Main ---
main() {
    info "=== OpenCV Cross-Compilation Build ==="
    info "Version: ${OPENCV_VERSION} (opencv), ${OPENCV_CONTRIB_VERSION} (opencv_contrib)"
    info "Target: rv64gcv_zvl256b"
    info "Output: ${OUTPUT_DIR}"

    check_prerequisites
    mkdir -p "${OUTPUT_DIR}"

    clone_source
    extract_sysroot
    cross_compile
    verify_output
    smoke_test

    info "=== Build Complete ==="
    info "Next steps:"
    info "  1. Build KCF demo: cd applications/opencv/kcf && ./build.sh"
    info "  2. Run profiling: /perf-profiling --host 192.168.100.221 ..."
}

main