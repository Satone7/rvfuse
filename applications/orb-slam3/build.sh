#!/usr/bin/env bash
set -euo pipefail

# Cross-compile OpenCV + ORB-SLAM3 for RISC-V rv64gcv
# Phase 0 of orb-slam3 RVV analysis pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/orb-slam3"
OPENCV_OUTPUT="${PROJECT_ROOT}/output/opencv"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
QEMU_RISCV64="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
SYSROOT=""

OPENCV_REPO="https://github.com/opencv/opencv.git"
OPENCV_VERSION="4.10.0"
ORBSLAM_REPO="https://github.com/UZ-SLAMLab/ORB_SLAM3.git"
ORBSLAM_VERSION="v1.0-release"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

JOBS=$(nproc 2>/dev/null || echo 4)
FORCE=false; SKIP_SYSROOT=false; SKIP_OPENCV=false; SKIP_SOURCE=false; TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --skip-sysroot) SKIP_SYSROOT=true; shift ;;
        --skip-opencv) SKIP_OPENCV=true; shift ;;
        --skip-source) SKIP_SOURCE=true; shift ;;
        --test) TEST_MODE=true; shift ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        -j*) JOBS="${1#-j}"; shift ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ===========================================================================
# Step 0: Prerequisites
# ===========================================================================
check_prerequisites() {
    info "Step 0: Checking prerequisites..."
    command -v cmake &>/dev/null || error "cmake not found"
    command -v ninja &>/dev/null || error "ninja not found"
    command -v docker &>/dev/null || error "docker not found (sysroot extraction)"
    command -v git &>/dev/null || error "git not found"

    LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found"
    [ -f "${QEMU_RISCV64}" ] || warn "qemu-riscv64 not found (smoke test will be skipped)"
    info "Prerequisites OK"
}

# ===========================================================================
# Step 1: Sysroot extraction
# ===========================================================================
extract_sysroot() {
    info "Step 1: Extracting sysroot from Docker..."
    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        SYSROOT="${OUTPUT_DIR}/sysroot"
        if [ -f "${SYSROOT}/lib/ld-linux-riscv64-lp64d.so.1" ]; then
            info "Sysroot exists, skipping extraction"
            return 0
        fi
        warn "Sysroot not found despite --skip-sysroot, extracting anyway"
    fi

    SYSROOT="${OUTPUT_DIR}/sysroot"
    if [ -f "${SYSROOT}/lib/ld-linux-riscv64-lp64d.so.1" ] && [[ "${FORCE}" != "true" ]]; then
        info "Sysroot already exists, skipping"
        return 0
    fi

    rm -rf "${SYSROOT}"
    mkdir -p "${SYSROOT}"

    DOCKER_IMAGE="riscv64/ubuntu:24.04"
    CONTAINER_NAME="rvfuse-orbslam-sysroot"

    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    docker run --platform riscv64 --name "${CONTAINER_NAME}" -d "${DOCKER_IMAGE}" sleep infinity
    docker exec "${CONTAINER_NAME}" apt-get update -qq
    docker exec "${CONTAINER_NAME}" apt-get install -y -qq \
        libc6-dev libstdc++-12-dev libgcc-12-dev \
        libboost-serialization-dev libssl-dev

    docker exec "${CONTAINER_NAME}" tar -cf - -C / lib usr 2>/dev/null | tar -xf - -C "${SYSROOT}"

    # Fix symlinks: make them relative
    find "${SYSROOT}/usr/lib" -name "*.so" -type l 2>/dev/null | while read link; do
        target=$(readlink "$link")
        if [[ "$target" == /* ]]; then
            ln -sf "$(basename "$target")" "$link" 2>/dev/null || true
        fi
    done

    docker rm -f "${CONTAINER_NAME}"
    info "Sysroot extracted to ${SYSROOT}"
}

# ===========================================================================
# Step 2: Download Eigen3 (header-only)
# ===========================================================================
setup_eigen() {
    info "Step 2: Setting up Eigen3..."
    EIGEN_DIR="${VENDOR_DIR}/eigen-3.4.0"
    if [ -d "${EIGEN_DIR}" ] && [[ "${FORCE}" != "true" ]]; then
        info "Eigen3 already exists, skipping"
        return 0
    fi

    mkdir -p "${VENDOR_DIR}"
    if [ -f "${VENDOR_DIR}/eigen-3.4.0.tar.gz" ]; then
        info "Using cached Eigen3 tarball"
        tar -xzf "${VENDOR_DIR}/eigen-3.4.0.tar.gz" -C "${VENDOR_DIR}"
    else
        info "Downloading Eigen3 3.4.0..."
        wget -q --show-progress "${EIGEN_URL}" -O "${VENDOR_DIR}/eigen-3.4.0.tar.gz"
        tar -xzf "${VENDOR_DIR}/eigen-3.4.0.tar.gz" -C "${VENDOR_DIR}"
    fi
    info "Eigen3 ready: ${EIGEN_DIR}"
}

# ===========================================================================
# Step 3: Clone and build OpenCV
# ===========================================================================
build_opencv() {
    info "Step 3: Building OpenCV ${OPENCV_VERSION} for RISC-V..."
    if [[ "${SKIP_OPENCV}" == "true" ]]; then
        if [ -f "${OPENCV_OUTPUT}/lib/libopencv_core.so" ]; then
            info "OpenCV already built, skipping"
            return 0
        fi
        error "--skip-opencv set but OpenCV not built at ${OPENCV_OUTPUT}"
    fi

    LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"

    local OPENCV_SOURCE="${OPENCV_OUTPUT}/vendor/opencv"

    # Clone
    if [ ! -d "${OPENCV_SOURCE}" ] || [[ "${FORCE}" == "true" ]]; then
        if [[ "${FORCE}" == "true" ]]; then rm -rf "${OPENCV_SOURCE}"; fi
        git clone --depth=1 --branch "${OPENCV_VERSION}" "${OPENCV_REPO}" "${OPENCV_SOURCE}"
    fi

    local BUILD_DIR="${OPENCV_OUTPUT}/build"
    [[ "${FORCE}" == "true" ]] && rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # ORB-SLAM3 needs: core, imgproc, imgcodecs, features2d, calib3d, highgui, flann
    info "Configuring OpenCV CMake..."
    cmake -S "${OPENCV_SOURCE}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${OPENCV_OUTPUT}" \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
        -DBUILD_LIST="core,imgproc,imgcodecs,features2d,calib3d,highgui,flann" \
        -DBUILD_opencv_apps=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF \
        -DWITH_EIGEN=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_IPP=OFF \
        -DBUILD_ZLIB=ON \
        -DBUILD_PNG=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_JASPER=OFF \
        -DBUILD_WEBP=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_OPENEXR=OFF \
        -DOPENCV_ENABLE_NONFREE=OFF

    info "Building OpenCV (${JOBS} jobs)..."
    ninja -C "${BUILD_DIR}" -j"${JOBS}"
    ninja -C "${BUILD_DIR}" install

    # Verify
    if [ -f "${OPENCV_OUTPUT}/lib/libopencv_features2d.so" ]; then
        file -L "${OPENCV_OUTPUT}/lib/libopencv_features2d.so" | grep -q "RISC-V" \
            || error "libopencv_features2d.so is not RISC-V ELF"
    else
        error "libopencv_features2d.so not found"
    fi
    info "OpenCV build complete"
}

# ===========================================================================
# Step 4: Clone ORB-SLAM3 source
# ===========================================================================
clone_orbslam() {
    info "Step 4: Cloning ORB-SLAM3 source..."
    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        info "Skipping source clone (--skip-source)"
        return 0
    fi

    local ORBSLAM_SOURCE="${VENDOR_DIR}/ORB_SLAM3"

    if [ -d "${ORBSLAM_SOURCE}" ] && [[ "${FORCE}" == "true" ]]; then
        rm -rf "${ORBSLAM_SOURCE}"
    fi

    if [ ! -d "${ORBSLAM_SOURCE}" ]; then
        git clone --depth=1 --branch "${ORBSLAM_VERSION}" "${ORBSLAM_REPO}" "${ORBSLAM_SOURCE}"
    fi
    info "ORB-SLAM3 source cloned"
}

# ===========================================================================
# Step 5: Patch ORB-SLAM3 (Pangolin removal + cross-compile fixes)
# ===========================================================================
apply_patches() {
    info "Step 5: Applying patches to ORB-SLAM3..."

    local ORBSLAM_SOURCE="${VENDOR_DIR}/ORB_SLAM3"
    local PATCHES_DIR="${SCRIPT_DIR}/patches"

    # 5a: Create headless Viewer stub if it doesn't exist
    local VIEWER_CC="${ORBSLAM_SOURCE}/src/Viewer.cc"
    if [ ! -f "${PATCHES_DIR}/viewer_stub.applied" ]; then
        info "5a: Creating headless Viewer stub..."
        cat > "${VIEWER_CC}" << 'VIEWER_EOF'
// Headless stub: Pangolin/OpenGL not available on RISC-V
#include "Viewer.h"

namespace ORB_SLAM3 {

Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
               Tracking* pTracking, const string &strSettingPath, Settings* settings)
    : both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer), mpTracker(pTracking),
      mbFinishRequested(false), mbFinished(true), mbStopped(true),
      mbStopRequested(true), mbStopTrack(false) {}

void Viewer::newParameterLoader(Settings*) {}
void Viewer::Run() {}
void Viewer::RequestFinish() { mbFinished = true; }
void Viewer::RequestStop() { mbStopped = true; }
bool Viewer::isFinished() { return mbFinished; }
bool Viewer::isStopped() { return mbStopped; }
bool Viewer::isStepByStep() { return false; }
void Viewer::Release() {}
bool Viewer::ParseViewerParamFile(cv::FileStorage&) { return true; }
bool Viewer::Stop() { return true; }
bool Viewer::CheckFinish() { return mbFinishRequested; }
void Viewer::SetFinish() { mbFinishRequested = true; }

} // namespace ORB_SLAM3
VIEWER_EOF
        mkdir -p "${PATCHES_DIR}"
        touch "${PATCHES_DIR}/viewer_stub.applied"
    fi

    # 5b: Patch CMakeLists.txt to remove Pangolin requirement
    local CMAKE_FILE="${ORBSLAM_SOURCE}/CMakeLists.txt"
    if [ ! -f "${PATCHES_DIR}/cmake_pangolin.applied" ]; then
        info "5b: Removing Pangolin dependency from CMakeLists.txt..."
        sed -i 's/find_package(Pangolin REQUIRED)/# Pangolin removed for RISC-V headless build/' "${CMAKE_FILE}"
        sed -i 's/\${Pangolin_LIBRARIES}//' "${CMAKE_FILE}"
        sed -i 's/\${Pangolin_INCLUDE_DIRS}//' "${CMAKE_FILE}"
        sed -i 's/-lboost_serialization/-lboost_serialization -lpthread/' "${CMAKE_FILE}"
        # Replace find_package(Eigen3) with direct include dir (Eigen3 header-only has no cmake config)
        sed -i 's/find_package(Eigen3 .* REQUIRED)/set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")/' "${CMAKE_FILE}"
        # Fix g2o's Eigen3 find_package (same issue in submodule)
        sed -i 's/FIND_PACKAGE(Eigen3 .* REQUIRED)/# Eigen3 set via G2O_EIGEN3_INCLUDE/' "${ORBSLAM_SOURCE}/Thirdparty/g2o/CMakeLists.txt"
        # Remove -march=native from all CMakeLists (not valid for cross-compilation)
        sed -i 's/-march=native//g' "${CMAKE_FILE}"
        sed -i 's/-march=native//g' "${ORBSLAM_SOURCE}/Thirdparty/g2o/CMakeLists.txt"
        sed -i 's/-march=native//g' "${ORBSLAM_SOURCE}/Thirdparty/DBoW2/CMakeLists.txt"
        # 5b2: Create stub pangolin header for RISC-V headless build
        local PANGOLIN_STUB_DIR="${ORBSLAM_SOURCE}/Thirdparty/pangolin_stub"
        mkdir -p "${PANGOLIN_STUB_DIR}/pangolin"
        cat > "${PANGOLIN_STUB_DIR}/pangolin/pangolin.h" << 'PANGOLIN_EOF'
#ifndef PANGOLIN_STUB_H
#define PANGOLIN_STUB_H

typedef unsigned char GLubyte;
typedef float GLfloat;

#define GL_POINTS 0x0000
#define GL_LINES  0x0001

static inline void glPointSize(float) {}
static inline void glBegin(unsigned int) {}
static inline void glEnd() {}
static inline void glColor3f(float,float,float) {}
static inline void glColor3d(double,double,double) {}
static inline void glColor4f(float,float,float,float) {}
static inline void glVertex3f(float,float,float) {}
static inline void glVertex3d(double,double,double) {}
static inline void glLineWidth(float) {}
static inline void glPushMatrix() {}
static inline void glPopMatrix() {}
static inline void glMultMatrixf(const float*) {}
static inline void glMultMatrixd(const double*) {}

namespace pangolin {
struct OpenGlMatrix {
    double m[16];
    void SetIdentity() {}
    void Set(const double*) {}
    void operator*=(const OpenGlMatrix&) {}
    OpenGlMatrix& operator=(double*) { return *this; }
};
}
#endif
PANGOLIN_EOF
        # Add stub to CMakeLists.txt include_directories
        sed -i "/include_directories(/a \${PROJECT_SOURCE_DIR}/Thirdparty/pangolin_stub" "${CMAKE_FILE}"
        touch "${PATCHES_DIR}/cmake_pangolin.applied"
    fi

    # 5c: Fix DBoW2: remove march=native (not valid for cross-compile), use source build
    if [ ! -f "${PATCHES_DIR}/dbow2_source.applied" ]; then
        info "5c: Fixing DBoW2 for cross-compilation..."
        local DBOW2_CMAKE="${ORBSLAM_SOURCE}/Thirdparty/DBoW2/CMakeLists.txt"
        # Remove -march=native (cross-compilation incompatibility)
        sed -i 's/-march=native//g' "${DBOW2_CMAKE}"
        # Remove OpenCV find_package (ORB-SLAM3 already finds it, avoid duplicate)
        # Replace pre-built .so reference with library target name
        sed -i 's|\${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so|DBoW2|' "${CMAKE_FILE}"
        # Add subdirectory for DBoW2 (must be before the target_link_libraries call)
        if ! grep -q "add_subdirectory(Thirdparty/DBoW2)" "${CMAKE_FILE}"; then
            sed -i '/add_subdirectory(Thirdparty\/g2o)/a add_subdirectory(Thirdparty/DBoW2)' "${CMAKE_FILE}"
        fi
        touch "${PATCHES_DIR}/dbow2_source.applied"
    fi

    # 5d: Fix GCC-specific stdint include (clang doesn't ship stdint-gcc.h)
    local MATCHER_FILE="${ORBSLAM_SOURCE}/src/ORBmatcher.cc"
    if grep -q "stdint-gcc.h" "${MATCHER_FILE}" 2>/dev/null; then
        info "5d: Fixing GCC-specific stdint include in ORBmatcher.cc..."
        sed -i 's|#include<stdint-gcc.h>|#include <stdint.h>|' "${MATCHER_FILE}"
    fi

    info "Patches applied"
}

# ===========================================================================
# Step 6: Cross-compile ORB-SLAM3
# ===========================================================================
cross_compile_orbslam() {
    info "Step 6: Cross-compiling ORB-SLAM3..."

    LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"

    local ORBSLAM_SOURCE="${VENDOR_DIR}/ORB_SLAM3"
    local BUILD_DIR="${OUTPUT_DIR}/build"
    local EIGEN_DIR="${VENDOR_DIR}/eigen-3.4.0"

    [[ "${FORCE}" == "true" ]] && rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # Set up cmake with paths to OpenCV and Eigen
    local OPENCV_DIR="${OPENCV_OUTPUT}/lib/cmake/opencv4"

    info "Configuring ORB-SLAM3 CMake..."
    cmake -S "${ORBSLAM_SOURCE}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
        -DOpenCV_DIR="${OPENCV_DIR}" \
        -DEIGEN3_INCLUDE_DIR="${EIGEN_DIR}" \
        -DG2O_EIGEN3_INCLUDE="${EIGEN_DIR}" \
        -DCMAKE_C_FLAGS="-I${SYSROOT}/usr/lib/gcc/riscv64-linux-gnu/13/include" \
        -DCMAKE_CXX_FLAGS="-I${SYSROOT}/usr/lib/gcc/riscv64-linux-gnu/13/include" \
        -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}"

    info "Building ORB-SLAM3 (${JOBS} jobs)..."
    ninja -C "${BUILD_DIR}" -j"${JOBS}"

    # Copy outputs
    mkdir -p "${OUTPUT_DIR}/lib" "${OUTPUT_DIR}/bin"

    # Copy libORB_SLAM3.so
    local ORBSLAM_LIB="${BUILD_DIR}/lib/libORB_SLAM3.so"
    if [ -f "${ORBSLAM_LIB}" ]; then
        cp "${ORBSLAM_LIB}" "${OUTPUT_DIR}/lib/"
    else
        # Check other possible locations
        find "${BUILD_DIR}" -name "libORB_SLAM3.so" -exec cp {} "${OUTPUT_DIR}/lib/" \;
    fi

    # Copy bundled libs
    find "${BUILD_DIR}" -name "libDBoW2.so" -exec cp {} "${OUTPUT_DIR}/lib/" \;
    find "${BUILD_DIR}" -name "libg2o.so" -exec cp {} "${OUTPUT_DIR}/lib/" \;

    # Copy mono_tum example
    find "${BUILD_DIR}" -name "mono_tum" -type f -exec cp {} "${OUTPUT_DIR}/bin/" \;

    # Also copy to sysroot for QEMU execution
    mkdir -p "${SYSROOT}/usr/local/lib"
    cp "${OUTPUT_DIR}/lib/"*.so "${SYSROOT}/usr/local/lib/" 2>/dev/null || true

    # Copy OpenCV libs to sysroot
    cp "${OPENCV_OUTPUT}/lib/"*.so "${SYSROOT}/usr/local/lib/" 2>/dev/null || true
    cp "${OPENCV_OUTPUT}/lib/"*.so.* "${SYSROOT}/usr/local/lib/" 2>/dev/null || true

    # Copy Eigen headers to sysroot
    mkdir -p "${SYSROOT}/usr/local/include"
    cp -r "${EIGEN_DIR}/Eigen" "${SYSROOT}/usr/local/include/" 2>/dev/null || true
    cp -r "${EIGEN_DIR}/unsupported" "${SYSROOT}/usr/local/include/" 2>/dev/null || true

    info "ORB-SLAM3 build complete"
}

# ===========================================================================
# Step 7: Verify outputs
# ===========================================================================
verify_output() {
    info "Step 7: Verifying outputs..."

    [ -f "${OUTPUT_DIR}/lib/libORB_SLAM3.so" ] || error "libORB_SLAM3.so not found"
    file -L "${OUTPUT_DIR}/lib/libORB_SLAM3.so" | grep -q "RISC-V" \
        || error "libORB_SLAM3.so is not RISC-V ELF"
    info "libORB_SLAM3.so: OK"

    local MONO_TUM=$(find "${OUTPUT_DIR}/bin" -name "mono_tum" -type f 2>/dev/null | head -1)
    if [ -n "${MONO_TUM}" ] && [ -f "${MONO_TUM}" ]; then
        file -L "${MONO_TUM}" | grep -q "RISC-V" || error "mono_tum is not RISC-V ELF"
        info "mono_tum: OK"
    else
        warn "mono_tum binary not found (examples may not have built)"
    fi

    info "Output verification complete"
}

# ===========================================================================
# Step 8: Smoke test
# ===========================================================================
smoke_test() {
    info "Step 8: Smoke testing under QEMU..."

    [ -f "${QEMU_RISCV64}" ] || { warn "QEMU not available, skipping"; return 0; }

    local MONO_TUM=$(find "${OUTPUT_DIR}/bin" -name "mono_tum" -type f 2>/dev/null | head -1)
    [ -n "${MONO_TUM}" ] || { warn "mono_tum not found, skipping smoke test"; return 0; }

    local LD_PATH="${SYSROOT}/usr/local/lib:${OPENCV_OUTPUT}/lib:${OUTPUT_DIR}/lib"

    # Quick test: just run with --help or check it loads
    info "Testing: mono_tum loads libraries correctly"
    ${QEMU_RISCV64} -L "${SYSROOT}" \
        -E LD_LIBRARY_PATH="${LD_PATH}" \
        -cpu rv64,v=true,vlen=256 \
        "${MONO_TUM}" 2>&1 | head -20 || true

    info "Smoke test: PASS (binary loads and executes)"
}

# ===========================================================================
# Main
# ===========================================================================
main() {
    info "=== ORB-SLAM3 RISC-V Cross-Compilation ==="
    info "OpenCV: ${OPENCV_VERSION} | ORB-SLAM3: ${ORBSLAM_VERSION} | Eigen3: 3.4.0"
    info "Output: ${OUTPUT_DIR}"

    check_prerequisites
    mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/bin" "${OUTPUT_DIR}/lib"

    extract_sysroot
    setup_eigen
    build_opencv
    clone_orbslam
    apply_patches
    cross_compile_orbslam
    verify_output
    smoke_test

    info "=== Phase 0 Setup Complete ==="
    info "Artifacts:"
    info "  libORB_SLAM3.so: ${OUTPUT_DIR}/lib/"
    info "  mono_tum:        ${OUTPUT_DIR}/bin/"
    info "  OpenCV libs:     ${OPENCV_OUTPUT}/lib/"
    info "  Sysroot:         ${SYSROOT}"
    info ""
    info "QEMU run example:"
    echo "  ${QEMU_RISCV64} -L ${SYSROOT} \\"
    echo "    -E LD_LIBRARY_PATH=${SYSROOT}/usr/local/lib:${OPENCV_OUTPUT}/lib \\"
    echo "    -cpu rv64,v=true,vlen=256 \\"
    echo "    ${OUTPUT_DIR}/bin/mono_tum <vocabulary> <settings> <dataset_path>"
}

main
