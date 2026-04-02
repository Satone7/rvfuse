#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build script for ONNX Runtime + YOLO Runner RISC-V cross-compilation
#
# Uses Docker multi-stage build to cross-compile for RISC-V and extracts
# the resulting artifacts (executable, model, test image) to output/.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${PROJECT_ROOT}/output"
YOLO_RUNNER_DIR="${PROJECT_ROOT}/tools/yolo_runner"

echo "=== Building ONNX Runtime + YOLO Runner for RISC-V ==="
echo ""

# ---------------------------------------------------------------------------
# Prerequisites: verify required source files exist
# ---------------------------------------------------------------------------
for file in yolo_runner.cpp stb_image.h CMakeLists.txt; do
    if [ ! -f "${YOLO_RUNNER_DIR}/${file}" ]; then
        echo "ERROR: Missing required file: ${YOLO_RUNNER_DIR}/${file}" >&2
        exit 1
    fi
done
echo "[OK] Prerequisites verified: yolo_runner.cpp, stb_image.h, CMakeLists.txt"

# ---------------------------------------------------------------------------
# Build Docker image (multi-stage, target: export)
# ---------------------------------------------------------------------------
echo ""
echo "--- Building Docker image ---"
docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag rvfuse-onnxrt-builder:latest \
    --target export \
    "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Extract artifacts from Docker image into output directory
# ---------------------------------------------------------------------------
echo ""
echo "--- Extracting artifacts ---"
mkdir -p "${OUTPUT_DIR}"

CONTAINER_ID=$(docker create rvfuse-onnxrt-builder:latest)

# Copy the three artifacts from the scratch image
docker cp "${CONTAINER_ID}:/yolo_inference" "${OUTPUT_DIR}/yolo_inference"
docker cp "${CONTAINER_ID}:/yolo11n.onnx"   "${OUTPUT_DIR}/yolo11n.onnx"
docker cp "${CONTAINER_ID}:/test_image.jpg"  "${OUTPUT_DIR}/test_image.jpg"

# Remove the temporary container
docker rm "${CONTAINER_ID}" > /dev/null

# ---------------------------------------------------------------------------
# Verify extracted artifacts
# ---------------------------------------------------------------------------
echo ""
echo "--- Verifying artifacts ---"

file "${OUTPUT_DIR}/yolo_inference"
if command -v readelf > /dev/null 2>&1; then
    readelf -h "${OUTPUT_DIR}/yolo_inference" | head -5
else
    echo "[INFO] readelf not available, skipping ELF header inspection"
fi

ls -lh "${OUTPUT_DIR}/yolo_inference" "${OUTPUT_DIR}/yolo11n.onnx" "${OUTPUT_DIR}/test_image.jpg"

# ---------------------------------------------------------------------------
# Completion summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Build complete ==="
echo "Artifacts written to: ${OUTPUT_DIR}/"
echo "  - yolo_inference   (RISC-V ELF binary)"
echo "  - yolo11n.onnx     (YOLO11n model)"
echo "  - test_image.jpg   (test input image)"
