#!/usr/bin/env bash
# ResNet50 ONNX Runtime inference test demo
# Tests the generic_ort_runner with ResNet50 model on RISC-V (via QEMU or Banana Pi)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-ort"
MODEL_DIR="${PROJECT_ROOT}/output/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
MODE="${1:-qemu}"
ITERATIONS="${2:-10}"

# --- Check prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."

    # Runner
    [ -f "${OUTPUT_DIR}/generic_ort_runner" ] || error "generic_ort_runner not found. Run build first."
    file "${OUTPUT_DIR}/generic_ort_runner"

    # ORT library
    [ -f "${OUTPUT_DIR}/lib/libonnxruntime.so" ] || error "libonnxruntime.so not found."
    ls -lh "${OUTPUT_DIR}/lib/libonnxruntime.so.1.24.4"

    # Sysroot (for QEMU)
    [ -d "${OUTPUT_DIR}/sysroot" ] || error "Sysroot not found."

    # Model
    [ -f "${MODEL_DIR}/resnet50.onnx" ] || {
        warn "ResNet50 model not found. Downloading..."
        wget -q -O "${MODEL_DIR}/resnet50.onnx" \
            "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
    }
    ls -lh "${MODEL_DIR}/resnet50.onnx"

    info "All prerequisites met."
}

# --- QEMU local test ---
run_qemu_test() {
    info "Running ResNet50 inference via QEMU..."

    # QEMU with sysroot
    QEMU_CMD="qemu-riscv64 -L ${OUTPUT_DIR}/sysroot"

    # Set library path
    export QEMU_LD_PREFIX="${OUTPUT_DIR}/sysroot"
    export LD_LIBRARY_PATH="${OUTPUT_DIR}/lib:${OUTPUT_DIR}/sysroot/usr/lib/riscv64-linux-gnu"

    # Run inference
    ${QEMU_CMD} "${OUTPUT_DIR}/generic_ort_runner" "${MODEL_DIR}/resnet50.onnx" "${ITERATIONS}"

    info "QEMU test completed."
}

# --- Banana Pi perf profiling (chroot mode via sshpass) ---
run_perf_profile() {
    info "Running perf profiling on Banana Pi..."

    if [ -z "${RVFUSE_HOST:-}" ]; then
        error "RVFUSE_HOST environment variable not set. Example: RVFUSE_HOST=192.168.1.22"
    fi

    local HOST="${RVFUSE_HOST}"
    local PASS="${RVFUSE_PASSWORD:-bianbu}"
    local REMOTE="/root/resnet50-perf"
    local MODEL="resnet50"
    local MODEL_FILE="${MODEL_DIR}/${MODEL}.onnx"
    local PERF_OUT="${PROJECT_ROOT}/output/perf/${MODEL}"
    local SSH_CMD="sshpass -p '${PASS}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@${HOST}"
    local SCP_CMD="sshpass -p '${PASS}' scp -o StrictHostKeyChecking=no"

    # 1. Check board
    info "Checking remote board..."
    ${SSH_CMD} 'uname -m && perf --version && df -h / | tail -1'
    ${SSH_CMD} 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

    # 2. Upload rootfs + model
    info "Uploading rootfs and model..."
    ${SSH_CMD} "mkdir -p ${REMOTE}"
    ${SCP_CMD} "${OUTPUT_DIR}/rootfs.tar.gz" root@${HOST}:${REMOTE}/
    ${SSH_CMD} "cd ${REMOTE} && tar xzf rootfs.tar.gz"
    ${SCP_CMD} "${MODEL_FILE}" root@${HOST}:${REMOTE}/rootfs/

    # 3. Setup chroot mounts
    info "Setting up chroot mounts..."
    ${SSH_CMD} "mkdir -p ${REMOTE}/rootfs/{proc,dev,sys,tmp}"
    ${SSH_CMD} "mount -t proc proc ${REMOTE}/rootfs/proc"
    ${SSH_CMD} "mount -t sysfs sysfs ${REMOTE}/rootfs/sys"
    ${SSH_CMD} "mount --bind /dev ${REMOTE}/rootfs/dev"

    # 4. Run perf (stat + record + report + annotate)
    info "Running perf stat..."
    local RUN_CMD="chroot ${REMOTE}/rootfs /generic_ort_runner /${MODEL}.onnx ${ITERATIONS}"
    ${SSH_CMD} "perf stat -d -o /tmp/perf_stat.txt -- ${RUN_CMD}"

    info "Running perf record (cpu-clock, ${ITERATIONS} iterations)..."
    ${SSH_CMD} "perf record -e cpu-clock -g -F 999 -o /tmp/perf.data -- ${RUN_CMD}"

    info "Generating perf report..."
    ${SSH_CMD} "perf report --stdio -n --percent-limit 0.5 --symfs ${REMOTE}/rootfs -i /tmp/perf.data > /tmp/perf_report.txt"

    info "Generating perf annotate..."
    ${SSH_CMD} "perf annotate --stdio --symfs ${REMOTE}/rootfs -i /tmp/perf.data > /tmp/perf_annotate.txt"

    # 5. Download results
    info "Downloading results..."
    mkdir -p "${PERF_OUT}"
    ${SCP_CMD} root@${HOST}:/tmp/perf_stat.txt "${PERF_OUT}/"
    ${SCP_CMD} root@${HOST}:/tmp/perf_report.txt "${PERF_OUT}/"
    ${SCP_CMD} root@${HOST}:/tmp/perf_annotate.txt "${PERF_OUT}/"

    # 6. Cleanup
    info "Cleaning up remote..."
    ${SSH_CMD} "rm -f /tmp/perf_stat.txt /tmp/perf_report.txt /tmp/perf_annotate.txt /tmp/perf.data"
    ${SSH_CMD} "umount ${REMOTE}/rootfs/proc; umount ${REMOTE}/rootfs/sys; umount ${REMOTE}/rootfs/dev"

    info "Perf profiling completed. Results in: ${PERF_OUT}"
}

# --- Direct lib mode (simpler setup, no chroot) ---
run_lib_mode() {
    info "Running perf profiling with direct lib upload..."

    if [ -z "${RVFUSE_HOST:-}" ]; then
        error "RVFUSE_HOST environment variable not set."
    fi

    local HOST="${RVFUSE_HOST}"
    local PASS="${RVFUSE_PASSWORD:-bianbu}"
    local REMOTE="/root/resnet50-lib"
    local MODEL="resnet50"
    local MODEL_FILE="${MODEL_DIR}/${MODEL}.onnx"
    local PERF_OUT="${PROJECT_ROOT}/output/perf/${MODEL}-lib"
    local SSH_CMD="sshpass -p '${PASS}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@${HOST}"
    local SCP_CMD="sshpass -p '${PASS}' scp -o StrictHostKeyChecking=no"

    # 1. Check board
    info "Checking remote board..."
    ${SSH_CMD} 'uname -m && perf --version'
    ${SSH_CMD} 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

    # 2. Upload runner + libs + model
    info "Uploading runner, libs, and model..."
    ${SSH_CMD} "mkdir -p ${REMOTE}/lib"
    ${SCP_CMD} "${OUTPUT_DIR}/generic_ort_runner" root@${HOST}:${REMOTE}/
    ${SCP_CMD} "${OUTPUT_DIR}"/lib/*.so* root@${HOST}:${REMOTE}/lib/
    ${SCP_CMD} "${MODEL_FILE}" root@${HOST}:${REMOTE}/

    # 3. Run perf
    local RUN_CMD="LD_LIBRARY_PATH=${REMOTE}/lib ${REMOTE}/generic_ort_runner ${REMOTE}/${MODEL}.onnx ${ITERATIONS}"

    info "Running perf stat..."
    ${SSH_CMD} "perf stat -d -o /tmp/perf_stat.txt -- ${RUN_CMD}"

    info "Running perf record..."
    ${SSH_CMD} "perf record -e cpu-clock -g -F 999 -o /tmp/perf.data -- ${RUN_CMD}"

    info "Generating perf report..."
    ${SSH_CMD} "perf report --stdio -n --percent-limit 0.5 -i /tmp/perf.data > /tmp/perf_report.txt"

    info "Generating perf annotate..."
    ${SSH_CMD} "perf annotate --stdio -i /tmp/perf.data > /tmp/perf_annotate.txt"

    # 4. Download results
    mkdir -p "${PERF_OUT}"
    ${SCP_CMD} root@${HOST}:/tmp/perf_stat.txt "${PERF_OUT}/"
    ${SCP_CMD} root@${HOST}:/tmp/perf_report.txt "${PERF_OUT}/"
    ${SCP_CMD} root@${HOST}:/tmp/perf_annotate.txt "${PERF_OUT}/"

    # 5. Cleanup
    ${SSH_CMD} "rm -f /tmp/perf_stat.txt /tmp/perf_report.txt /tmp/perf_annotate.txt /tmp/perf.data"

    info "Lib mode profiling completed. Results in: ${PERF_OUT}"
}

# --- Main ---
check_prerequisites

echo ""
echo "Available test modes:"
echo "  qemu     - Local QEMU simulation (default)"
echo "  perf     - Banana Pi perf profiling (requires RVFUSE_HOST)"
echo "  lib      - Direct lib upload mode (simpler)"
echo ""
echo "Running mode: ${MODE}"
echo "Iterations: ${ITERATIONS}"
echo ""

case "${MODE}" in
    qemu)   run_qemu_test ;;
    perf)   run_perf_profile ;;
    lib)    run_lib_mode ;;
    *)      error "Unknown mode: ${MODE}" ;;
esac

echo ""
echo "Test completed successfully!"