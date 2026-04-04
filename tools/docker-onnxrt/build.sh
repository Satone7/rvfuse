#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"

mkdir -p "${OUTPUT_DIR}"

echo "=== Building ONNX Runtime + YOLO runner for RISC-V ==="
echo "This may take 2-6 hours under QEMU emulation."

DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
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
