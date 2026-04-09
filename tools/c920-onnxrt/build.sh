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

# --- Step 1: Extract riscv64 sysroot ---
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

    # Move into final locations BEFORE creating symlinks that reference them
    mkdir -p "${sysroot}/usr"
    mv "${sysroot}/usr_lib_tmp"     "${sysroot}/usr/lib"
    mv "${sysroot}/usr_include_tmp" "${sysroot}/usr/include"

    # Only extract dynamic linker from /lib, symlink shared libs from /usr/lib
    # to avoid the linker picking up static libs from /lib over shared libs from /usr/lib
    mkdir -p "${sysroot}/lib/riscv64-linux-gnu"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/riscv64-linux-gnu/" || \
        error "Failed to copy dynamic linker from container. Is ld-linux-riscv64-lp64d.so.1 present?"
    # GCC ld looks for the dynamic linker at /lib/ld-linux-*.so.1 (not in the
    # riscv64-linux-gnu/ subdirectory), so create a top-level symlink.
    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    # Symlink essential shared libs that GCC's ld expects in /lib/riscv64-linux-gnu/
    # Use relative paths so they resolve correctly both on host and inside Docker
    local lib="${sysroot}/lib/riscv64-linux-gnu"
    for f in libc.so.6 libcrypt.so.1 libdl.so.2 libm.so.6 libpthread.so.0 librt.so.1 \
             libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
        [ -e "${sysroot}/usr/lib/riscv64-linux-gnu/${f}" ] && \
            ln -sf "../../usr/lib/riscv64-linux-gnu/${f}" "${lib}/${f}"
    done

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    # Create symlinks for C runtime files at the top-level usr/lib/
    # so Clang's linker can find them (Clang searches -L/sysroot/usr/lib
    # but not the riscv64-linux-gnu/ subdirectory).
    local multilib="riscv64-linux-gnu"
    for dir in usr/lib; do
        local base="${sysroot}/${dir}"
        [ -d "${base}/${multilib}" ] || continue
        for f in crt1.o crti.o crtn.o; do
            [ -f "${base}/${multilib}/${f}" ] && ln -sf "${multilib}/${f}" "${base}/${f}"
        done
        # Also symlink shared libs that GCC's ld searches at top-level
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0; do
            [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ] && \
                ln -sf "${multilib}/${f}" "${base}/${f}"
        done
    done

    # Remove problematic static libs that reference unresolved symbols
    # (libm.a has __frexpl which requires long double support not available here)
    find "${sysroot}" -name "libm.a" -delete 2>/dev/null || true

    info "Sysroot extracted to ${sysroot}"
    echo "  $(du -sh "${sysroot}" | cut -f1)"
}

# --- Step 2: Build cross-compilation Docker image ---
build_image() {
    info "Building cross-compilation Docker image..."
    DOCKER_BUILDKIT=1 docker build \
        -t "${DOCKER_IMAGE}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"
    info "Docker image built: ${DOCKER_IMAGE}"
}

# --- Step 3: Cross-compile ONNX Runtime ---
cross_compile() {
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}/onnxruntime"

    if [[ "${FORCE}" != "true" && -d "${ort_install}/lib" ]]; then
        info "ORT already built at ${ort_install}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling ONNX Runtime (full build)..."
    rm -rf "${ort_build}" 2>/dev/null || { warn "Build directory owned by root, using sudo to clean up..."; sudo rm -rf "${ort_build}"; } || true
    mkdir -p "${ort_build}" "${ort_install}"

    docker run --rm \
        -v "${LLVM_INSTALL}:/llvm-install:ro" \
        -v "${ORT_SOURCE}:/onnxruntime:ro" \
        -v "${EIGEN_SOURCE}:/eigen:ro" \
        -v "${OUTPUT_DIR}/sysroot:/sysroot:ro" \
        -v "${ort_build}:/build" \
        -v "${ort_install}:/install" \
        -v "${SCRIPT_DIR}/riscv64-linux-clang13.cmake:/toolchain.cmake:ro" \
        -e LLVM_INSTALL=/llvm-install \
        -e SYSROOT=/sysroot \
        -w /build \
        "${DOCKER_IMAGE}" \
        bash -c "
            cmake /onnxruntime/cmake \
                -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake \
                -DCMAKE_INSTALL_PREFIX=/install \
                -DCMAKE_BUILD_TYPE=Release \
                -Donnxruntime_BUILD_SHARED_LIB=ON \
                -Donnxruntime_BUILD_UNIT_TESTS=OFF \
                -Donnxruntime_DISABLE_RTTI=ON \
                -DFETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
                -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option' \
                -DIconv_IS_BUILT_IN=TRUE \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -G Ninja \
            && ninja -j${JOBS} \
            && ninja install/strip
        "

    info "ONNX Runtime cross-compiled to ${ort_install}"
}

# --- Step 4: Build YOLO runner ---
build_yolo_runner() {
    local ort_install="${OUTPUT_DIR}/onnxruntime"
    local runner_out="${OUTPUT_DIR}/yolo_inference"

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "YOLO runner already built. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling YOLO runner..."

    docker run --rm \
        -v "${LLVM_INSTALL}:/llvm-install:ro" \
        -v "${ORT_SOURCE}:/onnxruntime:ro" \
        -v "${YOLO_RUNNER}:/runner:ro" \
        -v "${OUTPUT_DIR}/sysroot:/sysroot:ro" \
        -v "${ort_install}:/onnxruntime-install:ro" \
        -v "$(dirname "${runner_out}"):/out" \
        -e LLVM_INSTALL=/llvm-install \
        -e SYSROOT=/sysroot \
        "${DOCKER_IMAGE}" \
        bash -c "
            /llvm-install/bin/clang++ \
                --target=riscv64-unknown-linux-gnu \
                --sysroot=/sysroot \
                -isystem /sysroot/usr/include/riscv64-linux-gnu \
                -B/riscv64-gcc-bin \
                -std=c++17 -O2 -g \
                -I/onnxruntime-install/include/onnxruntime \
                -I/onnxruntime-install/include/onnxruntime/core/session \
                -I/runner \
                /runner/yolo_runner.cpp \
                -o /out/yolo_inference \
                -L/onnxruntime-install/lib \
                -lonnxruntime \
                -Wl,-rpath,'\$ORIGIN/../onnxruntime/lib'
        "

    info "YOLO runner built: ${runner_out}"
}

# --- Main flow ---
extract_sysroot
build_image
cross_compile
build_yolo_runner

info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT:       ${OUTPUT_DIR}/onnxruntime/"
echo "  Runner:    ${OUTPUT_DIR}/yolo_inference"
echo "  Sysroot:   ${OUTPUT_DIR}/sysroot/"
echo ""
file "${OUTPUT_DIR}/yolo_inference" || true
