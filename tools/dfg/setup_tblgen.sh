#!/usr/bin/env bash
# setup_tblgen.sh — Build llvm-tblgen from LLVM and extract RISC-V instruction JSON.
#
# This is a one-time setup step (or re-run when the llvm-project submodule is updated)
# to produce tools/dfg/riscv_instrs.json, which is the reproducible input for
# gen_isadesc.py.
#
# Usage:
#   ./tools/dfg/setup_tblgen.sh            # default: build + extract
#   ./tools/dfg/setup_tblgen.sh --extract  # skip build, only re-extract JSON
#
# Prerequisites:
#   - cmake, make, a C++ compiler (gcc/clang)
#   - git submodule third_party/llvm-project initialized
#
# After running, regenerate ISA descriptors with:
#   python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext F -o tools/dfg/isadesc/rv64f.py
#   python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext M -o tools/dfg/isadesc/rv64m.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLVM_DIR="$PROJECT_ROOT/third_party/llvm-project"
BUILD_DIR="$LLVM_DIR/build"
TBLGEN="$BUILD_DIR/bin/llvm-tblgen"
OUTPUT="$SCRIPT_DIR/riscv_instrs.json"
TD_FILE="$LLVM_DIR/llvm/lib/Target/RISCV/RISCV.td"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Step 1: Initialize submodule (if needed)
# ---------------------------------------------------------------------------
ensure_submodule() {
    if [ ! -f "$LLVM_DIR/llvm/CMakeLists.txt" ]; then
        info "Initializing llvm-project submodule..."
        cd "$PROJECT_ROOT"
        git submodule update --init third_party/llvm-project
    else
        info "llvm-project submodule already initialized."
    fi
}

# ---------------------------------------------------------------------------
# Step 2: Build llvm-tblgen
# ---------------------------------------------------------------------------
build_tblgen() {
    if [ -x "$TBLGEN" ]; then
        warn "llvm-tblgen already built at $TBLGEN"
        warn "Re-run with: rm -rf $BUILD_DIR && $0"
        return 0
    fi

    info "Building llvm-tblgen (RISCV target only, Release)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake \
        -DLLVM_TARGETS_TO_BUILD=RISCV \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="" \
        -GNinja \
        ../llvm

    ninja llvm-tblgen

    info "llvm-tblgen built: $TBLGEN"
}

# ---------------------------------------------------------------------------
# Step 3: Extract RISC-V instruction JSON
# ---------------------------------------------------------------------------
extract_json() {
    if [ ! -x "$TBLGEN" ]; then
        error "llvm-tblgen not found at $TBLGEN. Run without --extract first."
        exit 1
    fi

    info "Extracting RISC-V instruction definitions..."
    "$TBLGEN" \
        -I "$LLVM_DIR/llvm/include" \
        -I "$LLVM_DIR/llvm/lib/Target/RISCV" \
        "$TD_FILE" \
        --dump-json \
        > "$OUTPUT"

    local size
    size=$(du -h "$OUTPUT" | cut -f1)
    local count
    count=$(python3 -c "import json; print(len(json.load(open('$OUTPUT'))))")

    info "Written $OUTPUT ($size, $count records)"
}

# ---------------------------------------------------------------------------
# Step 4: Verify
# ---------------------------------------------------------------------------
verify() {
    info "Verifying key instructions in JSON..."
    python3 -c "
import json
with open('$OUTPUT') as f:
    data = json.load(f)
missing = []
for name in ['FLW', 'FSW', 'FMADD_S', 'FADD_S', 'FMUL_S', 'FEQ_S', 'MUL', 'DIV']:
    if name not in data:
        missing.append(name)
if missing:
    print(f'MISSING: {missing}')
    raise SystemExit(1)
print(f'All 8 key instructions found in {len(data)} total records')
"
    info "Verification passed."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local skip_build=false
    if [[ "${1:-}" == "--extract" ]]; then
        skip_build=true
    fi

    echo "=== DFG llvm-tblgen setup ==="
    echo "  LLVM dir:   $LLVM_DIR"
    echo "  Build dir:  $BUILD_DIR"
    echo "  Output:     $OUTPUT"
    echo ""

    if [ "$skip_build" = false ]; then
        ensure_submodule
        build_tblgen
    fi

    extract_json
    verify

    echo ""
    info "Done. To regenerate ISA descriptors:"
    echo "  python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext F -o tools/dfg/isadesc/rv64f.py"
    echo "  python3 tools/dfg/gen_isadesc.py tools/dfg/riscv_instrs.json --ext M -o tools/dfg/isadesc/rv64m.py"
}

main "$@"
