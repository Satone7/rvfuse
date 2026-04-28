#!/usr/bin/env bash
# build.sh — Cross-compile ONNX Runtime + ViT-Base/32 runner for RISC-V (rv64gcv)
# Modeled after applications/onnxrt/vit-base-16/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-vit-base-32"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
ORT_SOURCE="${VENDOR_DIR}/onnxruntime"
EIGEN_SOURCE="${VENDOR_DIR}/eigen"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
RUNNER_DIR="${SCRIPT_DIR}/runner"

# Reuse existing ORT build if available
EXISTING_ORT="${PROJECT_ROOT}/output/cross-ort"

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
REUSE_ORT=true   # Default: reuse existing ORT build
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --skip-sysroot)  SKIP_SYSROOT=true; shift ;;
        --skip-source)   SKIP_SOURCE=true; shift ;;
        --no-reuse-ort)  REUSE_ORT=false; shift ;;
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

# --- Step 1: Set up ORT (reuse or build) ---
setup_ort() {
    if [[ "${REUSE_ORT}" == "true" && -d "${EXISTING_ORT}/lib" && -f "${EXISTING_ORT}/lib/libonnxruntime.so" ]]; then
        info "Reusing existing ONNX Runtime build at ${EXISTING_ORT}"
        mkdir -p "${OUTPUT_DIR}"
        # Symlink ORT artifacts
        ln -sfn "${EXISTING_ORT}/lib" "${OUTPUT_DIR}/lib"
        ln -sfn "${EXISTING_ORT}/include" "${OUTPUT_DIR}/include"
        return 0
    fi

    info "Building ONNX Runtime from source (no existing build found or --no-reuse-ort)..."

    # Clone sources
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

    if [[ "${SKIP_SOURCE}" != "true" ]]; then
        clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${ORT_SOURCE}"
        clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${EIGEN_SOURCE}"

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
    fi

    # Extract sysroot
    local sysroot="${OUTPUT_DIR}/sysroot"
    if [[ "${SKIP_SYSROOT}" != "true" ]]; then
        if [[ "${FORCE}" == "true" || ! -d "${sysroot}/usr" ]]; then
            info "Extracting riscv64 sysroot from riscv64/ubuntu:24.04..."
            command -v docker &>/dev/null || error "Docker not found."
            rm -rf "${sysroot}"
            mkdir -p "${sysroot}"
            local tmp_container="rvfuse-vit32-sysroot-$$"
            docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:24.04 tail -f /dev/null > /dev/null
            trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN
            docker exec "${tmp_container}" apt-get update -qq
            docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
                libc6-dev libstdc++-12-dev libgcc-12-dev > /dev/null
            mkdir -p "${sysroot}/usr"
            docker cp "${tmp_container}:/usr/lib" "${sysroot}/usr_lib_tmp"
            docker cp "${tmp_container}:/usr/include" "${sysroot}/usr_include_tmp"
            mv "${sysroot}/usr_lib_tmp" "${sysroot}/usr/lib"
            mv "${sysroot}/usr_include_tmp" "${sysroot}/usr/include"
            mkdir -p "${sysroot}/lib/riscv64-linux-gnu"
            docker cp "${tmp_container}:/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
                "${sysroot}/lib/riscv64-linux-gnu/" || error "Failed to copy dynamic linker."
            docker rm -f "${tmp_container}" > /dev/null
            trap - RETURN
            ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" "${sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

            # Create symlinks for C runtime and shared libs
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
        else
            info "Sysroot already exists at ${sysroot}. Use --force to re-extract."
        fi
    fi

    # Cross-compile ORT
    local ort_build="${OUTPUT_DIR}/.build"
    local ort_install="${OUTPUT_DIR}"

    [ -d "${ORT_SOURCE}/cmake" ] || error "ORT source not found at ${ORT_SOURCE}."
    [ -d "${sysroot}/usr" ] || error "Sysroot not found at ${sysroot}."

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${sysroot}"

    if [[ "${FORCE}" == "true" ]]; then
        rm -rf "${ort_build}"
    fi

    if [ ! -d "${ort_build}" ]; then
        info "Cross-compiling ONNX Runtime v${ONNXRUNTIME_VERSION}..."
        mkdir -p "${ort_build}" "${ort_install}"
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
    fi

    info "Building ORT (ninja -j${JOBS})..."
    ninja -C "${ort_build}" -j"${JOBS}"
    ninja -C "${ort_build}" install
    unset SYSROOT
}

setup_ort

# --- Step 2: Install ORT into sysroot ---
install_ort_to_sysroot() {
    local ort_install="${OUTPUT_DIR}"
    local sysroot="${OUTPUT_DIR}/sysroot"

    # If reusing ORT, use the existing sysroot
    if [[ "${REUSE_ORT}" == "true" && ! -d "${sysroot}/usr" ]]; then
        local existing_sysroot="${EXISTING_ORT}/sysroot"
        if [ -d "${existing_sysroot}/usr" ]; then
            info "Reusing existing sysroot at ${existing_sysroot}"
            ln -sfn "${existing_sysroot}" "${sysroot}"
            return 0
        fi
    fi

    [ -d "${sysroot}/usr" ] || { warn "Sysroot not found, skipping ORT installation to sysroot."; return 0; }

    local ort_so="${ort_install}/lib/libonnxruntime.so.1.24.4"
    if [ ! -f "${ort_so}" ]; then
        # Try the symlink
        ort_so="${ort_install}/lib/libonnxruntime.so"
        if [ ! -L "${ort_so}" ]; then
            warn "libonnxruntime.so not found at ${ort_install}/lib/, skipping sysroot install."
            return 0
        fi
        # Follow the symlink to get the real file
        ort_so=$(readlink -f "${ort_so}")
    fi

    info "Installing libonnxruntime.so into sysroot..."
    local sysroot_lib="${sysroot}/usr/lib/riscv64-linux-gnu"
    if [ -d "${sysroot_lib}" ]; then
        cp "${ort_so}" "${sysroot_lib}/"
        ln -sf libonnxruntime.so.1.24.4 "${sysroot_lib}/libonnxruntime.so.1" 2>/dev/null || true
        ln -sf libonnxruntime.so.1 "${sysroot_lib}/libonnxruntime.so" 2>/dev/null || true

        local top_lib="${sysroot}/usr/lib"
        [ -e "${top_lib}/libonnxruntime.so.1.24.4" ] || \
            ln -sf "riscv64-linux-gnu/libonnxruntime.so.1.24.4" "${top_lib}/libonnxruntime.so.1.24.4" 2>/dev/null || true
        [ -e "${top_lib}/libonnxruntime.so.1" ] || \
            ln -sf "riscv64-linux-gnu/libonnxruntime.so.1" "${top_lib}/libonnxruntime.so.1" 2>/dev/null || true
        [ -e "${top_lib}/libonnxruntime.so" ] || \
            ln -sf "riscv64-linux-gnu/libonnxruntime.so" "${top_lib}/libonnxruntime.so" 2>/dev/null || true

        info "libonnxruntime.so installed to sysroot."
    fi
}

install_ort_to_sysroot

# --- Step 3: Cross-compile ViT runner ---
build_runner() {
    local ort_install="${OUTPUT_DIR}"
    local sysroot="${OUTPUT_DIR}/sysroot"
    local runner_build="${OUTPUT_DIR}/runner_build"

    [ -d "${ort_install}/include/onnxruntime" ] || error "ORT headers not found at ${ort_install}/include"
    [ -d "${ort_install}/lib" ] || error "ORT lib not found at ${ort_install}/lib"

    info "Cross-compiling ViT-Base/32 runner..."

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
    cp "${runner_build}/vit_inference" "${OUTPUT_DIR}/"
    chmod +x "${OUTPUT_DIR}/vit_inference"

    unset SYSROOT

    info "ViT-Base/32 runner built: ${OUTPUT_DIR}/vit_inference"
    file "${OUTPUT_DIR}/vit_inference" || true
}

build_runner

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  ORT:         ${OUTPUT_DIR}/lib/libonnxruntime.so"
echo "  Runner:      ${OUTPUT_DIR}/vit_inference"
echo "  Sysroot:     ${OUTPUT_DIR}/sysroot/"
echo "  Model:       ${SCRIPT_DIR}/model/vit_base_patch32_384.onnx"
echo ""
file "${OUTPUT_DIR}/vit_inference" || true
