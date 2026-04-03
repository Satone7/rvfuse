#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

check_or_install() {
    if ! python3 -c "import $1" 2>/dev/null; then
        echo "Installing $1..."
        pip3 install "$1"
    fi
}

echo "=== RVFuse: Preparing YOLO11n ONNX model ==="

mkdir -p "${OUTPUT_DIR}"

check_or_install ultralytics
check_or_install onnx

MODEL_PATH="${OUTPUT_DIR}/yolo11n.onnx"
if [ -f "${MODEL_PATH}" ]; then
    echo "Model already exists: ${MODEL_PATH}"
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

IMAGE_PATH="${OUTPUT_DIR}/test.jpg"
if [ -f "${IMAGE_PATH}" ]; then
    echo "Test image already exists: ${IMAGE_PATH}"
else
    echo "Downloading COCO test image (bus.jpg)..."
    curl -fL -o "${IMAGE_PATH}" \
        "https://ultralytics.com/images/bus.jpg"
    echo "Test image downloaded: ${IMAGE_PATH}"
fi

echo "=== Done ==="
ls -lh "${OUTPUT_DIR}/yolo11n.onnx" "${OUTPUT_DIR}/test.jpg"
