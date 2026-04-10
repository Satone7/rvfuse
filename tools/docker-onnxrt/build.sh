#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"

TARGET="inference"
FORCE=false
JOBS=$(nproc 2>/dev/null || echo 4)
ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.23.2"
EIGEN_REPO="https://gitlab.com/libeigen/eigen.git"
EIGEN_VERSION="3.4.0"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)      TARGET="$2"; shift 2 ;;
        --force)       FORCE=true; shift ;;
        -j|--jobs)     JOBS="$2"; shift 2 ;;
        -j*)           JOBS="${1#-j}"; shift ;;
        *)             error "Unknown argument: $1" ;;
    esac
done

# Validate target
case "${TARGET}" in
    inference|c920-inference) ;;
    *) error "Invalid target '${TARGET}'. Must be 'inference' or 'c920-inference'." ;;
esac

info "Target: ${TARGET} | Jobs: ${JOBS} | Force: ${FORCE}"

# --- Clone dependencies ---
clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        info "Skipping ${dest} (already exists)"
    else
        info "Cloning ${repo_url} @ ${version} (shallow, no submodules)"
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}"
    fi
}

# --- Fetch ONNXRT submodules ---
fetch_ort_submodules() {
    if [ -d "${VENDOR_DIR}/onnxruntime" ]; then
        info "=== Fetching onnxruntime submodules ==="
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
}

# --- Extract sysroot ---
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
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" EXIT
    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:22.04 tail -f /dev/null > /dev/null
    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev libstdc++-11-dev libgcc-11-dev > /dev/null
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
        if [ -e "${sysroot}/usr/lib/riscv64-linux-gnu/${f}" ]; then
            ln -sf "../../usr/lib/riscv64-linux-gnu/${f}" "${lib}/${f}"
        fi
    done
    docker rm -f "${tmp_container}" > /dev/null
    trap - EXIT
    local multilib="riscv64-linux-gnu"
    for dir in usr/lib; do
        local base="${sysroot}/${dir}"
        if [ ! -d "${base}/${multilib}" ]; then continue; fi
        for f in crt1.o crti.o crtn.o; do
            if [ -f "${base}/${multilib}/${f}" ]; then
                ln -sf "${multilib}/${f}" "${base}/${f}"
            fi
        done
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0; do
            if [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ]; then
                ln -sf "${multilib}/${f}" "${base}/${f}"
            fi
        done
    done
    # Fix absolute symlinks that break inside Docker (mounted at /sysroot).
    # Ubuntu packages use absolute paths like /lib/riscv64-linux-gnu/libm.so.6
    # which resolve to the HOST filesystem, not the sysroot.
    info "Fixing absolute symlinks in sysroot..."
    local sl_dir="${sysroot}/usr/lib/riscv64-linux-gnu"
    for link in "${sl_dir}"/*.so; do
        [ -L "$link" ] || continue
        local target
        target=$(readlink "$link")
        if [[ "$target" == /* ]]; then
            # Convert absolute /lib/... to relative ../../lib/...
            local rel_target="../../${target#/}"
            local link_name
            link_name=$(basename "$link")
            ln -sf "$rel_target" "$link"
            echo "  Fixed: ${link_name} -> ${target} => ${rel_target}"
        fi
    done

    find "${sysroot}" -name "libm.a" -delete 2>/dev/null || true
    info "Sysroot extracted to ${sysroot}"
}

# --- Build Docker image ---
build_image() {
    info "Building cross-compilation Docker image..."
    DOCKER_BUILDKIT=1 docker build \
        -t rvfuse-xcompile-builder \
        -f "${SCRIPT_DIR}/Dockerfile" \
        --progress=plain \
        "${SCRIPT_DIR}"
    info "Docker image built: rvfuse-xcompile-builder"
}

# --- Cross-compile ONNX Runtime ---
cross_compile_onnxrt() {
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}/onnxruntime"

    if [[ "${FORCE}" != "true" && -d "${ort_install}/lib" ]]; then
        info "ORT already built at ${ort_install}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling ONNX Runtime (full build, target=${TARGET})..."
    rm -rf "${ort_build}" 2>/dev/null || { warn "Build directory owned by root, using sudo to clean up..."; sudo rm -rf "${ort_build}"; } || true
    mkdir -p "${ort_build}" "${ort_install}"

    # Determine march based on target
    local march="rv64gcv"
    if [[ "${TARGET}" == "c920-inference" ]]; then
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
        -e GCC_BIN_DIR=/riscv64-gcc-bin \
        -w /build \
        rvfuse-xcompile-builder \
        bash -c "
            cmake /onnxruntime/cmake \
                -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake \
                -DCMAKE_INSTALL_PREFIX=/install \
                -DCMAKE_BUILD_TYPE=Release \
                -Donnxruntime_BUILD_SHARED_LIB=ON \
                -Donnxruntime_BUILD_UNIT_TESTS=OFF \
                -Donnxruntime_DISABLE_CONTRIB_OPS=ON \
                -Donnxruntime_DISABLE_RTTI=ON \
                -DFETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
                -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option -march=${march}' \
                -DCMAKE_C_FLAGS='-march=${march}' \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -G Ninja \
            && ninja -j${JOBS} \
            && ninja install/strip
        "

    info "ONNX Runtime cross-compiled to ${ort_install}"
}

# --- Build YOLO runner ---
build_yolo_runner() {
    local ort_install="${OUTPUT_DIR}/onnxruntime"
    local runner_out="${OUTPUT_DIR}/yolo_inference"

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "YOLO runner already built. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling YOLO runner..."

    local march="rv64gcv"
    if [[ "${TARGET}" == "c920-inference" ]]; then
        march="rv64gcv_xtheadba_xtheadbb_xtheadbs_xtheadcondmov_xtheadcmo_xtheadfmidx_xtheadmac_xtheadmemidx_xtheadmempair_xtheadsync_xtheadvdot"
    fi

    docker run --rm \
        -v "${LLVM_INSTALL}:/llvm-install:ro" \
        -v "${ort_install}:/onnxruntime-install:ro" \
        -v "${PROJECT_ROOT}/tools/yolo_runner:/runner:ro" \
        -v "${OUTPUT_DIR}/sysroot:/sysroot:ro" \
        -v "$(dirname "${runner_out}"):/out" \
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
                -Wl,-rpath,'\$ORIGIN/../onnxruntime/lib'
        "

    info "YOLO runner built: ${runner_out}"
}

# --- Setup sysroot symlinks ---
setup_sysroot() {
    info "Setting up sysroot symlinks..."
    local sysroot="${OUTPUT_DIR}/sysroot"
    local ort_install="${OUTPUT_DIR}/onnxruntime"

    # Create ld-linux symlink if not present
    if [ ! -e "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1" ]; then
        ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
            "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"
    fi

    # Symlink libonnxruntime.so into sysroot for runtime linking
    local ort_lib="${ort_install}/lib"
    if [ -d "${ort_lib}" ]; then
        mkdir -p "${sysroot}/usr/local/lib"
        for f in "${ort_lib}"/libonnxruntime.so*; do
            if [ -e "${f}" ]; then
                cp -a "${f}" "${sysroot}/usr/local/lib/"
            fi
        done
    fi

    info "Sysroot setup complete."
}

# --- Main flow ---
main() {
    info "RVFuse x86_64 cross-compilation build (ONNXRT v1.23.2)"
    echo "  Target:   ${TARGET}"
    echo "  Jobs:     ${JOBS}"
    echo "  LLVM:     ${LLVM_INSTALL}"
    echo "  Output:   ${OUTPUT_DIR}"
    echo ""

    mkdir -p "${OUTPUT_DIR}"

    # Step 1: Clone dependencies on host
    clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"
    clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${VENDOR_DIR}/eigen"
    fetch_ort_submodules

    # Step 2: Extract sysroot
    extract_sysroot

    # Step 3: Build Docker image
    build_image

    # Step 4: Cross-compile ONNX Runtime
    cross_compile_onnxrt

    # Step 5: Build YOLO runner
    build_yolo_runner

    # Step 6: Setup sysroot symlinks
    setup_sysroot

    info "All done!"
    echo ""
    echo "Artifacts:"
    echo "  ORT:       ${OUTPUT_DIR}/onnxruntime/"
    echo "  Runner:    ${OUTPUT_DIR}/yolo_inference"
    echo "  Sysroot:   ${OUTPUT_DIR}/sysroot/"
    echo ""
    file "${OUTPUT_DIR}/yolo_inference" || true
}

main
