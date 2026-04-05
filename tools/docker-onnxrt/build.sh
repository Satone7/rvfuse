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
    if [ -d "${dest}" ]; then
        echo "=== Skipping ${dest} (already exists) ==="
    else
        echo "=== Cloning ${repo_url} @ ${version} ==="
        mkdir -p "$(dirname "${dest}")"
        git clone --recursive --branch "${version}" "${repo_url}" "${dest}"
    fi
}

# Pre-clone dependencies on the host so Docker doesn't need network access.
clone_if_missing "${ONNXRUNTIME_REPO}" "${ONNXRUNTIME_VERSION}" "${VENDOR_DIR}/onnxruntime"
clone_if_missing "${EIGEN_REPO}" "${EIGEN_VERSION}" "${VENDOR_DIR}/eigen"

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
docker rm "${CONTAINER_ID}" > /dev/null
docker rmi rvfuse-yolo-builder > /dev/null 2>&1 || true

echo "=== Build complete ==="
file "${OUTPUT_DIR}/yolo_inference"
