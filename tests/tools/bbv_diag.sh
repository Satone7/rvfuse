#!/usr/bin/env bash
# bbv_diag.sh — Diagnose why ONNX Runtime BBs are missing from BBV profiling
# Usage: bash tests/tools/bbv_diag.sh [A|B|C|all]   (default: all)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_BINDIR="${PROJECT_ROOT}/third_party/llvm-install/bin"
SYSROOT="${PROJECT_ROOT}/output/cross-ort/sysroot"
QEMU="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
PLUGIN="${PROJECT_ROOT}/tools/bbv/libbbv.so"
ANALYZE="${PROJECT_ROOT}/tools/analyze_bbv.py"
ORT_SO="${PROJECT_ROOT}/output/cross-ort/lib/libonnxruntime.so.1.24.4"
YOLO_BIN="${PROJECT_ROOT}/output/cross-ort/yolo_inference"
MODEL="${PROJECT_ROOT}/output/yolo11n.ort"
IMAGE="${PROJECT_ROOT}/output/test.jpg"
BB_FILE="${PROJECT_ROOT}/output/yolo.bbv.0.bb"
DISAS_FILE="${PROJECT_ROOT}/output/yolo.bbv.disas"

run="${1:-all}"

# ============================================================================
# Test A: Shared library capture and resolution verification
# ============================================================================
test_a() {
    echo "============================================"
    echo "Test A: Shared library capture verification"
    echo "============================================"

    # Prerequisites
    for f in "${LLVM_BINDIR}/clang" "${QEMU}" "${PLUGIN}"; do
        if [ ! -f "${f}" ]; then echo "[SKIP] Missing: ${f}"; return 0; fi
    done

    local T=$(mktemp -d)
    trap "rm -rf ${T}" RETURN

    # Step 1: Build a shared library with a hot loop
    cat > "${T}/mylib.c" << 'CEOF'
int mylib_heavy(int n) {
    volatile int sum = 0;
    for (int i = 0; i < n; i++) {
        if (i & 1) sum += i; else sum -= i;
    }
    return sum;
}
CEOF

    "${LLVM_BINDIR}/clang" \
        --target=riscv64-unknown-linux-gnu \
        --sysroot="${SYSROOT}" \
        -shared -fPIC -O2 -fuse-ld=lld \
        -o "${T}/libmylib.so" "${T}/mylib.c"

    # Step 2: Build main program
    cat > "${T}/main.c" << 'CEOF'
#include <stdio.h>
extern int mylib_heavy(int n);
int main() {
    int r = mylib_heavy(50000000);
    printf("result=%d\n", r);
    return 0;
}
CEOF

    "${LLVM_BINDIR}/clang" \
        --target=riscv64-unknown-linux-gnu \
        --sysroot="${SYSROOT}" \
        -O2 -fuse-ld=lld \
        -L"${T}" -lmylib \
        -Wl,-rpath,"${T}" \
        -o "${T}/test_shlib" "${T}/main.c"

    # Step 3: Copy libmylib.so into sysroot for symbol resolution
    local MINI_SYSROOT="${T}/sysroot"
    mkdir -p "${MINI_SYSROOT}/lib"
    cp "${T}/libmylib.so" "${MINI_SYSROOT}/lib/"

    # Step 4: Run under QEMU+BBV
    local BBV_OUT="${T}/bbv.out"
    rm -f "${BBV_OUT}"*

    "${QEMU}" -L "${SYSROOT}" \
        -plugin "${PLUGIN}",interval=10000,outfile="${BBV_OUT}" \
        "${T}/test_shlib" 2>/dev/null || true

    local DISAS="${BBV_OUT}.disas"
    if [ ! -f "${DISAS}" ]; then
        echo "[FAIL] No .disas file generated"
        return 1
    fi

    local TOTAL_BBS
    TOTAL_BBS=$(grep -c "^BB " "${DISAS}")
    echo "  Total unique BBs: ${TOTAL_BBS}"

    # Check address ranges — main binary (low PIE) + libmylib.so (high mmap)
    local ADDRS
    ADDRS=$(grep -oP 'vaddr:\s+\K0x[0-9a-fA-F]+' "${DISAS}" | sort -u)
    local MIN_ADDR MAX_ADDR
    MIN_ADDR=$(echo "${ADDRS}" | head -1)
    MAX_ADDR=$(echo "${ADDRS}" | tail -1)
    echo "  Address range: ${MIN_ADDR} — ${MAX_ADDR}"

    # Step 5: Run analyze_bbv.py
    local BB_FILE_A
    BB_FILE_A=$(ls "${BBV_OUT}".*.bb 2>/dev/null | head -1)
    if [ -z "${BB_FILE_A}" ]; then
        echo "[FAIL] No .bb file generated"
        return 1
    fi

    local REPORT="${T}/report.json"
    python3 "${ANALYZE}" \
        --bbv "${BB_FILE_A}" \
        --elf "${T}/test_shlib" \
        --sysroot "${MINI_SYSROOT}" \
        --json-output "${REPORT}" \
        --top 10 2>&1 | tail -15

    # Step 6: Check resolution results
    if [ -f "${REPORT}" ]; then
        local HAS_MYLIB
        HAS_MYLIB=$(python3 -c "
import json
data = json.load(open('${REPORT}'))
mylib = [b for b in data['blocks'] if 'mylib' in b['location'].lower()]
print(len(mylib))
" 2>/dev/null)
        echo "  Blocks attributed to libmylib.so: ${HAS_MYLIB}"

        if [ "${HAS_MYLIB}" -gt 0 ]; then
            echo "[PASS] Test A: Shared library blocks captured and resolved"
        else
            echo "[FAIL] Test A: No libmylib.so blocks found in report"
            # Show top locations for debugging
            python3 -c "
import json
data = json.load(open('${REPORT}'))
for b in data['blocks'][:5]:
    print(f\"  {b['pct']:>5.1f}%  {b['location'][:80]}\")
" 2>/dev/null
        fi
    else
        echo "[FAIL] Test A: analyze_bbv.py produced no JSON report"
    fi
}

# ============================================================================
# Test B: Raw address analysis — does ORT data exist in the BBV capture?
# ============================================================================
test_b() {
    echo ""
    echo "============================================"
    echo "Test B: Raw BBV address analysis for ORT blocks"
    echo "============================================"

    if [ ! -f "${DISAS_FILE}" ]; then
        echo "[SKIP] Missing: ${DISAS_FILE}"
        return 0
    fi

    # ORT text LOAD segment: vaddr=0x453c40, memsz=0xa9ad50
    python3 << 'PYEOF'
import re, sys
from collections import Counter

DISAS = "/home/pren/wsp/rvfuse/output/yolo.bbv.disas"
ORT_TEXT_VADDR = 0x453c40
ORT_TEXT_SIZE  = 0xa9ad50
ORT_TEXT_END   = ORT_TEXT_VADDR + ORT_TEXT_SIZE

# Parse all BB addresses from .disas
addrs = []
with open(DISAS) as f:
    for line in f:
        m = re.match(r'BB\s+\d+\s+\(vaddr:\s+(0x[0-9a-fA-F]+)', line.strip())
        if m:
            addrs.append(int(m.group(1), 16))

addrs.sort()
total = len(addrs)
print(f"  Total unique BBs in .disas: {total}")
print(f"  Address range: 0x{addrs[0]:x} — 0x{addrs[-1]:x}")
print(f"  ORT text segment: vaddr=0x{ORT_TEXT_VADDR:x}, size=0x{ORT_TEXT_SIZE:x}")

# Strategy: For any runtime base B, BB addr X is in ORT text iff
#   ORT_TEXT_VADDR <= X - B < ORT_TEXT_END
# => B in (X - ORT_TEXT_END, X - ORT_TEXT_VADDR]
#
# Use random sampling to find the intersection of possible base ranges.
# If a common base exists, many addresses will cluster in ORT text.
import random
random.seed(42)
sample = random.sample(addrs, min(300, len(addrs)))

# Compute intersection of all base ranges
base_lo = max(a - ORT_TEXT_END for a in sample)
base_hi = min(a - ORT_TEXT_VADDR for a in sample)

print(f"\n  Base range intersection (from {len(sample)} samples):")
print(f"    [0x{base_lo:x}, 0x{base_hi:x}]")
print(f"    width: {base_hi - base_lo:#x} ({(base_hi - base_lo) / (1024*1024):.1f} MB)")

if base_lo <= base_hi:
    # Try page-aligned base at start of intersection
    base = base_lo & ~0xFFF
    matches = [a for a in addrs if ORT_TEXT_VADDR <= a - base < ORT_TEXT_END]
    print(f"\n  Trying base=0x{base:x}:")
    print(f"    BBs matching ORT text range: {len(matches)} / {total}")

    if len(matches) > 50:
        print(f"\n  *** FOUND {len(matches)} potential ORT blocks! (H1 likely) ***")
        print("  ORT code WAS executed but wrongly attributed to other libraries.")
        print("\n  Sample matched BBs:")
        for a in matches[:5]:
            offset = a - base
            print(f"    runtime 0x{a:x} -> file offset 0x{offset:x}")
    else:
        # Also try other aligned bases in the intersection
        best_base = base
        best_count = len(matches)
        for offset_mb in range(0, (base_hi - base_lo) // (1024*1024) + 1, 1):
            trial = ((base_lo + offset_mb * 1024 * 1024) & ~0xFFF)
            if trial <= 0:
                continue
            cnt = sum(1 for a in addrs if ORT_TEXT_VADDR <= a - trial < ORT_TEXT_END)
            if cnt > best_count:
                best_count = cnt
                best_base = trial

        if best_count > 50:
            print(f"\n  *** FOUND {best_count} potential ORT blocks at base=0x{best_base:x}! (H1 likely) ***")
        else:
            print(f"\n  No significant ORT block cluster found (best={best_count}).")
            print("  Confirms H2: ORT code was never meaningfully executed.")
else:
    print("\n  No common base found — addresses cannot all be from ORT text.")
    print("  Confirms H2: ORT code was never meaningfully executed.")

# Additional: Check how many blocks are in the typical mmap region for large .so files
mmap_blocks = sum(1 for a in addrs if a > 0x7d0000000000)
print(f"\n  Blocks in mmap region (>0x7d0000000000): {mmap_blocks} ({mmap_blocks/total*100:.1f}%)")

# Check address density — large gaps suggest a missing library
if len(addrs) > 1:
    gaps = []
    for i in range(1, min(len(addrs), 50000)):
        gaps.append(addrs[i] - addrs[i-1])
    gaps.sort(reverse=True)
    print(f"  Top 5 address gaps between consecutive BBs:")
    for g in gaps[:5]:
        print(f"    {g:#x} ({g / (1024*1024):.1f} MB)")
PYEOF
}

# ============================================================================
# Test C: yolo_inference actual execution verification
# ============================================================================
test_c() {
    echo ""
    echo "============================================"
    echo "Test C: yolo_inference execution verification"
    echo "============================================"

    for f in "${QEMU}" "${YOLO_BIN}"; do
        if [ ! -f "${f}" ]; then echo "[SKIP] Missing: ${f}"; return 0; fi
    done

    local T=$(mktemp -d)
    trap "rm -rf ${T}" RETURN

    # C1: Basic execution check
    echo "--- C1: Basic execution ---"
    if [ ! -f "${MODEL}" ]; then
        echo "[SKIP] Model not found: ${MODEL}"
    else
        set +e
        timeout 120 "${QEMU}" -L "${SYSROOT}" \
            "${YOLO_BIN}" "${MODEL}" "${IMAGE}" 1 \
            > "${T}/stdout.log" 2> "${T}/stderr.log"
        local RC=$?
        set -e
        echo "  Exit code: ${RC}"
        echo "  stdout (last 15 lines):"
        tail -15 "${T}/stdout.log" 2>/dev/null | sed 's/^/    /'
        echo "  stderr (last 5 lines):"
        tail -5 "${T}/stderr.log" 2>/dev/null | sed 's/^/    /'

        # Check markers from yolo_runner.cpp
        for marker in "Model:" "Loading model" "Preprocessing image" "Running inference" "Done."; do
            if grep -q "${marker}" "${T}/stdout.log" 2>/dev/null; then
                echo "  [OK] \"${marker}\" found"
            else
                echo "  [MISS] \"${marker}\" not found"
            fi
        done

        if grep -qi "error\|cannot\|fatal\|segfault\|assert\|no such" "${T}/stderr.log" 2>/dev/null; then
            echo "  [WARN] Error keywords in stderr:"
            grep -i "error\|cannot\|fatal\|segfault\|assert\|no such" "${T}/stderr.log" 2>/dev/null | head -5 | sed 's/^/    /'
        fi
    fi

    # C2: Dynamic linker trace
    echo ""
    echo "--- C2: Dynamic linker trace (QEMU_LD_DEBUG=libs) ---"
    set +e
    timeout 30 "${QEMU}" -L "${SYSROOT}" \
        -E QEMU_LD_DEBUG=libs \
        "${YOLO_BIN}" "${MODEL}" "${IMAGE}" 1 \
        > "${T}/ld_debug.log" 2>&1
    set -e

    echo "  Libraries found in linker trace:"
    if grep -qi "calling init:" "${T}/ld_debug.log" 2>/dev/null; then
        grep "calling init:" "${T}/ld_debug.log" 2>/dev/null | sed 's/.*calling init: /  /' | sort -u
    else
        echo "    (no 'calling init:' lines — showing onnx-related lines or first 20 lines)"
        grep -i "onnx\|ort\|cannot open\|error\|not found" "${T}/ld_debug.log" 2>/dev/null | head -20 | sed 's/^/    /' || \
            head -20 "${T}/ld_debug.log" | sed 's/^/    /'
    fi

    if grep -qi "onnxruntime" "${T}/ld_debug.log" 2>/dev/null; then
        echo "  [OK] libonnxruntime.so appears in linker trace"
    else
        echo "  [FAIL] libonnxruntime.so NOT found in linker trace"
        echo "         -> ORT was never loaded — this explains missing BBs"
    fi

    # Check for unexpected libraries
    echo ""
    echo "  Unexpected library loads:"
    for lib in libcrypto libssl libapt libgcrypt libgnutls libsystemd libdb; do
        if grep -qi "${lib}" "${T}/ld_debug.log" 2>/dev/null; then
            echo "    [WARN] ${lib} loaded (should not be needed by yolo_inference)"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================
case "${run}" in
    A|a) test_a ;;
    B|b) test_b ;;
    C|c) test_c ;;
    all)  test_a; test_b; test_c ;;
    *)   echo "Usage: $0 [A|B|C|all]"; exit 1 ;;
esac
