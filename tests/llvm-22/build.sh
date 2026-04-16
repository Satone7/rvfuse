#!/usr/bin/env bash
set -euo pipefail

# Build script for LLVM 22 RISC-V backend bug reproduction
# Compiles bug_test.cpp with both buggy and correct march flags,
# runs both under QEMU, and compares output.
#
# Usage:
#   ./build.sh          # Compile and run both variants
#   ./build.sh --clean  # Remove build artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLANG="${PROJECT_ROOT}/third_party/llvm-install/bin/clang++"
QEMU="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
SYSROOT="${PROJECT_ROOT}/output/llama.cpp/sysroot"

MARCH_BUG="rv64gcv_zvl512b_zfh_zvfh"
MARCH_OK="rv64gc"
COMMON_FLAGS="-std=c++17 -O2 --target=riscv64-unknown-linux-gnu --sysroot=${SYSROOT} -mabi=lp64d -fuse-ld=lld"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# Pre-flight checks
for tool in "$CLANG" "$QEMU"; do
    [ -x "$tool" ] || error "Not found or not executable: $tool"
done
[ -d "$SYSROOT" ] || error "Sysroot not found: $SYSROOT"

cd "$SCRIPT_DIR"

if [ "${1:-}" = "--clean" ]; then
    info "Cleaning build artifacts"
    rm -f test_bug test_ok
    exit 0
fi

# Compile buggy variant (rv64gcv_zvl512b)
info "Compiling with -march=${MARCH_BUG} (bug trigger)..."
${CLANG} ${COMMON_FLAGS} -march=${MARCH_BUG} bug_test.cpp -o test_bug -lm

# Compile correct variant (rv64gc)
info "Compiling with -march=${MARCH_OK} (correct)..."
${CLANG} ${COMMON_FLAGS} -march=${MARCH_OK} bug_test.cpp -o test_ok -lm

echo ""
info "Running buggy variant (expect FAIL):"
echo -e "${CYAN}-------------------------------------------${NC}"
${QEMU} -L "${SYSROOT}" ./test_bug || true
echo -e "${CYAN}-------------------------------------------${NC}"

echo ""
info "Running correct variant (expect PASS):"
echo -e "${CYAN}-------------------------------------------${NC}"
${QEMU} -L "${SYSROOT}" ./test_ok
echo -e "${CYAN}-------------------------------------------${NC}"
