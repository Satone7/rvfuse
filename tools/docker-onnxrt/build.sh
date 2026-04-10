#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
ONNXRUNTIME_VERSION="v1.17.3"
EIGEN_REPO="https://gitlab.com/libeigen/eigen.git"
EIGEN_VERSION="3.4.0"

clone_if_missing() {
    local repo_url="$1" version="$2" dest="$3"
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${version} (shallow, no submodules) ==="
        mkdir -p "$(dirname "${dest}")"
        # Shallow clone without submodules to reduce download size.
        # Submodules are fetched separately below with retries.
        git clone --depth=1 --branch "${version}" --no-recurse-submodules "${repo_url}" "${dest}"
    fi
}

# Pre-clone dependencies on the host so Docker doesn't need network access.
clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"
clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${VENDOR_DIR}/eigen"

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

mkdir -p "${OUTPUT_DIR}"

echo "=== Building ONNX Runtime + YOLO runner for RISC-V ==="
echo "This may take 2-6 hours under QEMU emulation."

DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    --network=host \
    -t rvfuse-yolo-builder \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"

CONTAINER_ID=$(docker create rvfuse-yolo-builder)
docker cp "${CONTAINER_ID}:/yolo_inference" "${OUTPUT_DIR}/yolo_inference"
docker cp "${CONTAINER_ID}:/yolo_preprocess" "${OUTPUT_DIR}/yolo_preprocess"
docker cp "${CONTAINER_ID}:/yolo_postprocess" "${OUTPUT_DIR}/yolo_postprocess"

# Extract RISC-V sysroot for QEMU user-mode emulation (-L flag).
# Copy ALL shared libraries from the container so no transitive deps are missed.
SYSROOT="${OUTPUT_DIR}/sysroot"
mkdir -p "${SYSROOT}/lib/riscv64-linux-gnu"

# Copy entire library directories from the stopped container.
# This ensures all transitive deps (libx264, libswresample, etc.) are included.
docker cp "${CONTAINER_ID}:/usr/lib/riscv64-linux-gnu/." "${SYSROOT}/lib/riscv64-linux-gnu/"
docker cp "${CONTAINER_ID}:/lib/riscv64-linux-gnu/." "${SYSROOT}/lib/riscv64-linux-gnu/" 2>/dev/null || true

# ld-linux symlink (QEMU default path)
ln -sf riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1 "${SYSROOT}/lib/ld-linux-riscv64-lp64d.so.1"

docker rm "${CONTAINER_ID}" > /dev/null
docker rmi rvfuse-yolo-builder > /dev/null 2>&1 || true

echo "=== Build complete ==="
file "${OUTPUT_DIR}/yolo_inference"
file "${OUTPUT_DIR}/yolo_preprocess"
file "${OUTPUT_DIR}/yolo_postprocess"
echo "=== Sysroot ==="
ls -la "${SYSROOT}/lib/riscv64-linux-gnu/"
