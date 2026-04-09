#!/usr/bin/env bash
# build_llvm.sh — Build LLVM as a RISC-V cross-compiler.
#
# Compiles third_party/llvm-project with RISC-V target support including
# the V (vector) extension for auto-vectorization. Produces clang, lld,
# compiler-rt under third_party/llvm-install/.
#
# Usage:
#   ./tools/build_llvm.sh              # build (incremental)
#   ./tools/build_llvm.sh --clean      # wipe build & start fresh
#   ./tools/build_llvm.sh --install    # build + install to third_party/llvm-install
#
# Prerequisites:
#   - cmake, ninja, gcc/g++ (or clang), make
#   - python3 (required by LLVM build)
#   - git submodule third_party/llvm-project initialized

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLVM_SRC="$PROJECT_ROOT/third_party/llvm-project"
BUILD_DIR="$LLVM_SRC/build-native"
INSTALL_DIR="$PROJECT_ROOT/third_party/llvm-install"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${CYAN}==>${NC} $*"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DO_CLEAN=false
DO_INSTALL=false
JOBS="$(nproc 2>/dev/null || echo 4)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build LLVM as a RISC-V cross-compiler with V extension support.

Options:
  --clean       Remove build directory before building
  --install     Install to $INSTALL_DIR after building
  -j N          Number of parallel jobs (default: $JOBS)
  -h, --help    Show this help

Environment:
  CC            C compiler for host build (default: gcc)
  CXX           C++ compiler for host build (default: g++)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)    DO_CLEAN=true; shift ;;
        --install)  DO_INSTALL=true; shift ;;
        -j)         JOBS="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 0: Prerequisites
# ---------------------------------------------------------------------------
check_prereqs() {
    step "Checking prerequisites"
    local missing=()

    for cmd in cmake ninja "${CC:-gcc}" python3; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing prerequisites: ${missing[*]}"
        error "Install with: sudo apt install cmake ninja-build gcc g++ python3"
        exit 1
    fi

    info "All prerequisites met."
}

# ---------------------------------------------------------------------------
# Step 1: Submodule check
# ---------------------------------------------------------------------------
ensure_submodule() {
    step "Checking llvm-project submodule"
    if [ ! -f "$LLVM_SRC/llvm/CMakeLists.txt" ]; then
        info "Initializing llvm-project submodule..."
        cd "$PROJECT_ROOT"
        git submodule update --init third_party/llvm-project
    else
        info "llvm-project submodule ready."
    fi
}

# ---------------------------------------------------------------------------
# Step 2: Clean (optional)
# ---------------------------------------------------------------------------
maybe_clean() {
    if [ "$DO_CLEAN" = true ]; then
        step "Cleaning build directory"
        rm -rf "$BUILD_DIR"
        info "Removed $BUILD_DIR"
    fi
}

# ---------------------------------------------------------------------------
# Step 3: CMake configure
# ---------------------------------------------------------------------------
cmake_configure() {
    if [ -f "$BUILD_DIR/build.ninja" ]; then
        warn "CMake already configured. Use --clean to reconfigure."
        return 0
    fi

    step "Configuring LLVM (CMake)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DLLVM_TARGETS_TO_BUILD=RISCV \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
        -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        ../llvm

    info "CMake configuration complete."
}

# ---------------------------------------------------------------------------
# Step 4: Build
# ---------------------------------------------------------------------------
build() {
    step "Building LLVM (RISCV + V) with $JOBS jobs"
    cd "$BUILD_DIR"

    ninja -j "$JOBS"

    info "Build complete: $BUILD_DIR/bin/"
}

# ---------------------------------------------------------------------------
# Step 5: Install (optional)
# ---------------------------------------------------------------------------
install_toolchain() {
    if [ "$DO_INSTALL" != true ]; then
        return 0
    fi

    step "Installing to $INSTALL_DIR"
    cd "$BUILD_DIR"

    ninja install

    info "Toolchain installed to $INSTALL_DIR"
    info "Add to PATH: export PATH=\"$INSTALL_DIR/bin:\$PATH\""
}

# ---------------------------------------------------------------------------
# Step 6: Verification
# ---------------------------------------------------------------------------
verify() {
    local BIN_DIR
    if [ "$DO_INSTALL" = true ] && [ -d "$INSTALL_DIR/bin" ]; then
        BIN_DIR="$INSTALL_DIR/bin"
    else
        BIN_DIR="$BUILD_DIR/bin"
    fi

    local CLANG="$BIN_DIR/clang"

    if [ ! -x "$CLANG" ]; then
        error "clang not found at $CLANG"
        exit 1
    fi

    step "Verifying toolchain"

    # Version check
    info "Clang version:"
    "$CLANG" --version | head -1

    # Verify RISCV target is supported
    if "$CLANG" --print-targets | grep -q "riscv64"; then
        info "RISC-V 64 target: OK"
    else
        error "RISC-V 64 target not found!"
        exit 1
    fi

    # Compile test — basic RISC-V target
    local TEST_SRC
    TEST_SRC=$(mktemp /tmp/rvfuse_llvm_test_XXXXXX.c)
    local TEST_OBJ
    TEST_OBJ=$(mktemp /tmp/rvfuse_llvm_test_XXXXXX.o)

    cat > "$TEST_SRC" <<'TESTEOF'
int add(int a, int b) { return a + b; }
TESTEOF

    if "$CLANG" --target=riscv64-unknown-linux-gnu \
         -march=rv64gc \
         -c "$TEST_SRC" -o "$TEST_OBJ" 2>/dev/null; then
        info "Compile test (rv64gc): OK"

        # Verify it's a RISC-V ELF
        if "$BIN_DIR/llvm-readobj" -h "$TEST_OBJ" 2>/dev/null | grep -q "RISC-V"; then
            info "ELF format (RISC-V): OK"
        else
            warn "Could not verify ELF format (llvm-readelf may not be built yet)"
        fi
    else
        error "Compile test failed!"
        rm -f "$TEST_SRC" "$TEST_OBJ"
        exit 1
    fi

    # Compile test — RVV (V extension for auto-vectorization)
    if "$CLANG" --target=riscv64-unknown-linux-gnu \
         -march=rv64gcv \
         -c "$TEST_SRC" -o "$TEST_OBJ" 2>/dev/null; then
        info "Compile test (rv64gcv / V extension): OK"
    else
        error "Compile test (rv64gcv) failed!"
        rm -f "$TEST_SRC" "$TEST_OBJ"
        exit 1
    fi

    rm -f "$TEST_SRC" "$TEST_OBJ"

    echo ""
    info "All verifications passed!"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary() {
    echo ""
    echo "============================================="
    info "LLVM RISC-V Cross-Compiler Ready"
    echo "============================================="
    echo ""
    echo "  Build dir:   $BUILD_DIR/bin/"
    if [ "$DO_INSTALL" = true ]; then
        echo "  Install dir: $INSTALL_DIR/bin/"
    fi
    echo ""
    echo "  Supported march options:"
    echo "    rv64gc    RV64IMAFDC (base + float + compressed)"
    echo "    rv64gcv   RV64IMAFDCV (with vector extension for auto-vectorization)"
    echo ""
    echo "  Usage:"
    echo "    $BUILD_DIR/bin/clang --target=riscv64 -march=rv64gcv -c foo.c"
    echo ""
    if [ "$DO_INSTALL" = true ]; then
        echo "  Or use wrapper scripts:"
        echo "    ./tools/local-llvm/riscv64-clang -c foo.c"
        echo ""
        echo "  Add to PATH for direct access:"
        echo "    export PATH=\"$INSTALL_DIR/bin:\$PATH\""
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "=== LLVM RISC-V Cross-Compiler Build ==="
    echo "  Source:     $LLVM_SRC"
    echo "  Build dir:  $BUILD_DIR"
    echo "  Install:    $INSTALL_DIR"
    echo "  Jobs:       $JOBS"
    echo "  Clean:      $DO_CLEAN"
    echo "  Install:    $DO_INSTALL"
    echo ""

    check_prereqs
    ensure_submodule
    maybe_clean
    cmake_configure
    build
    install_toolchain
    verify
    summary
}

main "$@"
