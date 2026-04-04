#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="${SCRIPT_DIR}/third_party/qemu"

echo "=== Building QEMU with BBV plugin ==="

if [ ! -d "${QEMU_DIR}/.git" ]; then
    echo "Initializing QEMU submodule..."
    git submodule update --init --depth 1 third_party/qemu
fi

cd "${QEMU_DIR}"
mkdir -p build
cd build

echo "Configuring QEMU (riscv64-linux-user, plugins enabled)..."
../configure \
    --target-list=riscv64-linux-user \
    --disable-werror \
    --enable-plugins

echo "Building QEMU..."
make -j"$(nproc)"

echo "Building BBV plugin..."
make plugins

BINARY="${PWD}/qemu-riscv64"
PLUGIN="${PWD}/contrib/plugins/libbbv.so"

if [ ! -f "${BINARY}" ]; then
    echo "Error: qemu-riscv64 not found at ${BINARY}" >&2
    exit 1
fi
echo "QEMU binary: ${BINARY}"

if [ ! -f "${PLUGIN}" ]; then
    echo "Error: libbbv.so not found at ${PLUGIN}" >&2
    exit 1
fi
echo "BBV plugin: ${PLUGIN}"

echo ""
echo "=== QEMU+BBV build complete ==="
echo "To profile a binary:"
echo "  ${BINARY} -plugin ${PLUGIN},interval=10000,outfile=output/prof.bbv -- ./output/yolo_inference ./output/yolo11n.onnx ./output/test.jpg"
