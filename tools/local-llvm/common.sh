#!/bin/bash
# common.sh — Shared functions for local LLVM toolchain wrappers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Install dir takes priority; fall back to build dir
INSTALL_DIR="$PROJECT_ROOT/third_party/llvm-install"
BUILD_DIR="$PROJECT_ROOT/third_party/llvm-project/build-native"

if [ -d "$INSTALL_DIR/bin" ]; then
    BIN_DIR="$INSTALL_DIR/bin"
elif [ -d "$BUILD_DIR/bin" ]; then
    BIN_DIR="$BUILD_DIR/bin"
else
    BIN_DIR=""
fi

# ANSI colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

error_msg() {
    echo -e "${RED}Error: $1${NC}" >&2
}

warn_msg() {
    echo -e "${YELLOW}Warning: $1${NC}" >&2
}

# Resolve a tool binary path
resolve_tool() {
    local tool="$1"

    if [ -z "$BIN_DIR" ]; then
        error_msg "Local LLVM toolchain not found."
        echo "Build it first: ./tools/build_llvm.sh --install" >&2
        echo "Or without install: ./tools/build_llvm.sh" >&2
        exit 1
    fi

    local path="$BIN_DIR/$tool"
    if [ ! -x "$path" ]; then
        error_msg "Tool not found: $path"
        echo "Run ./tools/build_llvm.sh to build the toolchain." >&2
        exit 1
    fi

    echo "$path"
}
