#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

check_or_install() {
    if ! python3 -c "import $1" 2>/dev/null; then
        echo "Installing $1..."
        if pip3 install "$1" 2>&1 | grep -q "externally-managed-environment"; then
            pip3 install --break-system-packages "$1"
        fi
    fi
}

echo "=== RVFuse: Preparing YOLO11n model ==="

mkdir -p "${OUTPUT_DIR}"

check_or_install ultralytics
check_or_install onnx
check_or_install onnxruntime

# --- Step 1: Export ONNX model ---
MODEL_PATH="${OUTPUT_DIR}/yolo11n.onnx"
if [ -f "${MODEL_PATH}" ]; then
    echo "ONNX model already exists: ${MODEL_PATH}"
else
    echo "Exporting YOLO11n to ONNX (opset 12, batch 1)..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', opset=12, batch=1, simplify=True)
import shutil
shutil.move('yolo11n.onnx', '${MODEL_PATH}')
"
    echo "Model exported: ${MODEL_PATH}"
fi

# --- Step 2: Download test image ---
IMAGE_PATH="${OUTPUT_DIR}/test.jpg"
if [ -f "${IMAGE_PATH}" ]; then
    echo "Test image already exists: ${IMAGE_PATH}"
else
    echo "Downloading COCO test image (bus.jpg)..."
    curl -fL -o "${IMAGE_PATH}" \
        "https://ultralytics.com/images/bus.jpg"
    echo "Test image downloaded: ${IMAGE_PATH}"
fi

# --- Step 3: Convert to ORT format ---
# The minimal ONNX Runtime build does not load .onnx files directly;
# it requires the pre-compiled ORT format.
# Using ORT_ENABLE_EXTENDED (not ORT_ENABLE_ALL) avoids x86-specific
# NCHWC layout optimizations that crash on RISC-V.
ORT_PATH="${OUTPUT_DIR}/yolo11n.ort"
if [ -f "${ORT_PATH}" ]; then
    echo "ORT model already exists: ${ORT_PATH}"
else
    echo "Converting ONNX to ORT format (ORT_ENABLE_EXTENDED)..."
    python3 -c "
import onnxruntime as ort

so = ort.SessionOptions()
so.optimized_model_filepath = '${ORT_PATH}'
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort.InferenceSession('${MODEL_PATH}', so)
"
    echo "ORT model saved: ${ORT_PATH}"
fi

echo "=== Done ==="
ls -lh "${OUTPUT_DIR}/yolo11n.onnx" "${OUTPUT_DIR}/yolo11n.ort" "${OUTPUT_DIR}/test.jpg"
