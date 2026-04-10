#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.24.4"
# Eigen commit from ORT deps.txt (eigen-mirror repo, commit 1d8b82b)
# This matches the archive: https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33
EIGEN_REPO="https://github.com/eigen-mirror/eigen.git"
EIGEN_COMMIT="1d8b82b0740839c0de7f1242a3585e3390ff5f33"

clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${version} (shallow, no submodules) ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}"
    fi
}

clone_eigen_commit() {
    local repo_url="$1" commit="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${commit} (shallow) ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --depth=1 --no-recurse-submodules "${repo_url}" "${dest}"
        cd "${dest}"
        # Fetch the specific commit if not in shallow clone
        if ! git rev-parse --verify "${commit}" >/dev/null 2>&1; then
            git fetch --depth=1 origin "${commit}"
        fi
        git checkout "${commit}"
        cd - > /dev/null
    fi
}

# Pre-clone dependencies on the host so Docker doesn't need network access.
clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"

# Fetch onnxruntime submodules with retries (these are much smaller than the main repo)
# Uses --recursive to get ALL nested submodules (e.g. onnx -> benchmark, pybind11)
if [ -d "${VENDOR_DIR}/onnxruntime" ]; then
    echo "=== Fetching onnxruntime submodules (recursive) ==="
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

# Clone Eigen at the specific commit ORT requires
clone_eigen_commit "${EIGEN_REPO}" "${EIGEN_COMMIT}" "${VENDOR_DIR}/eigen"

echo "=== Building ONNX Runtime v1.24.4 for RISC-V (LLVM 18, riscv64gcv) ==="
echo "This is a FULL build (not minimal). Expected time: 6-12 hours under QEMU."

DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    --network=host \
    -t rvfuse-onnxrt-1.24.4 \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"

echo "=== Build complete ==="
echo "Docker image: rvfuse-onnxrt-1.24.4"
docker images rvfuse-onnxrt-1.24.4

# Verify build output exists in the image
echo "=== Verifying build output ==="
docker run --rm rvfuse-onnxrt-1.24.4 \
    ls -la /onnxruntime/build-output/

echo "=== Checking libonnxruntime.so architecture ==="
docker run --rm rvfuse-onnxrt-1.24.4 \
    file /onnxruntime/build-output/libonnxruntime.so