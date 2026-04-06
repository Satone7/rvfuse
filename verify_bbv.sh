#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# QEMU BBV Plugin Verification Script
# ==============================================================================

FORCE_REBUILD=false
if [ "${1:-}" = "--force-rebuild" ] || [ "${1:-}" = "-f" ]; then
    FORCE_REBUILD=true
fi

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="${WORKSPACE}/third_party/qemu"
DEMO_SRC="${WORKSPACE}/demo.c"
DEMO_ELF="${WORKSPACE}/demo.elf"
BBV_OUT="${WORKSPACE}/bbv.out"
QEMU_BIN="${QEMU_DIR}/build/qemu-riscv64"
PLUGIN_SO="${QEMU_DIR}/build/contrib/plugins/libbbv.so"

echo "========================================"
echo "1. Verify Demo test program"
echo "========================================"
if [ ! -f "${DEMO_SRC}" ]; then
    echo "demo.c not found, creating..."
    cat << 'EOF' > "${DEMO_SRC}"
void _start() {
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        if (i % 2 == 0) {
            sum += i;
        } else {
            sum -= i;
        }
    }

    // exit syscall
    asm volatile(
        "li a7, 93\n\t" // sys_exit
        "li a0, 0\n\t"  // status 0
        "ecall\n\t"
    );
}
EOF
fi

if [ -f "${DEMO_ELF}" ]; then
    echo "[OK] Demo compiled: ${DEMO_ELF}"
else
    echo "[SKIP] Demo binary not found at ${DEMO_ELF}"
    echo "       Compile demo.c with a RISC-V toolchain to verify BBV output, e.g.:"
    echo "       riscv64-unknown-clang -nostdlib -mno-relax -o demo.elf demo.c"
    DEMO_AVAILABLE=false
fi
echo ""

echo "========================================"
echo "2. Build QEMU and BBV plugin"
echo "========================================"
if [ ! -d "${QEMU_DIR}/.git" ]; then
    echo "Initializing QEMU submodule..."
    git submodule update --init --depth 1 third_party/qemu
fi

cd "${QEMU_DIR}"

if [ ! -f "${QEMU_BIN}" ] || [ ! -f "${PLUGIN_SO}" ] || [ "${FORCE_REBUILD}" = true ]; then
    if [ "${FORCE_REBUILD}" = true ]; then
        echo "Force rebuild mode (--force-rebuild)"
    fi
    echo "Configuring QEMU (riscv64-linux-user, plugins enabled)..."
    mkdir -p build
    cd build
    ../configure --target-list=riscv64-linux-user --disable-werror --enable-plugins

    echo "Building QEMU..."
    make -j"$(nproc)"

    echo "Building BBV plugin..."
    make plugins
    cd "${QEMU_DIR}"
else
    echo "QEMU and plugin already built, skipping."
    echo "Use -f or --force-rebuild to rebuild."
fi

if [ -f "${QEMU_BIN}" ] && [ -f "${PLUGIN_SO}" ]; then
    echo "[OK] QEMU and BBV plugin built successfully."
else
    echo "[ERROR] QEMU or plugin build failed!"
    exit 1
fi
echo ""

: "${DEMO_AVAILABLE:=true}"

echo "========================================"
echo "3. Run BBV plugin verification"
echo "========================================"
cd "${WORKSPACE}"

if [ "${DEMO_AVAILABLE}" = false ]; then
    echo "[SKIP] Demo binary unavailable — skipping BBV verification."
    echo "       QEMU and BBV plugin build was verified successfully."
    echo "       Verification complete!"
else
    # Clean old output files
    rm -f "${BBV_OUT}"*

    echo "Running:"
    echo "${QEMU_BIN} -plugin ${PLUGIN_SO},interval=10000,outfile=${BBV_OUT} ${DEMO_ELF}"
    ${QEMU_BIN} -plugin "${PLUGIN_SO}",interval=10000,outfile="${BBV_OUT}" "${DEMO_ELF}"

    if [ -f "${BBV_OUT}.0.bb" ]; then
        echo "[OK] BBV output generated: ${BBV_OUT}.0.bb"
        echo "----------------------------------------"
        echo "First 5 lines:"
        head -n 5 "${BBV_OUT}.0.bb"
        echo "----------------------------------------"
        echo "Verification complete!"
    else
        echo "[ERROR] BBV output not generated!"
        exit 1
    fi
fi
