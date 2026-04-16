# LLVM 22 RISC-V Backend Bug: Scalar Code Corruption with RVV march

## Summary

LLVM 22.1.3 RISC-V backend produces incorrect code when compiling scalar C++ code with RVV-enabled `-march`. Array elements receive garbage values despite explicit initialization.

## Quick Start

```bash
./build.sh            # Compile both variants and run under QEMU
```

## Affected Versions

- LLVM 22.1.3 (commit e9846648fd6183ee6d8cbdb4502213fcf902a211)
- Tested on: 2026-04-16

## Files

| File | Purpose |
|------|---------|
| `README.md` | This document |
| `bug_test.cpp` | Minimal reproduction test case (~150 lines, no external deps) |
| `build.sh` | Build script: compiles both variants and runs comparison |

## Reproduction

### Using build.sh (recommended)

```bash
./build.sh
```

This will:
1. Compile with `-march=rv64gcv` (bug trigger) → `test_bug`
2. Compile with `-march=rv64gc` (correct) → `test_ok`
3. Run both under QEMU and compare output

### Manual commands

**Compile with bug trigger:**
```bash
clang++ -std=c++17 -O2 \
  --target=riscv64-unknown-linux-gnu \
  --sysroot=<sysroot> \
  -march=rv64gcv_zvl512b_zfh_zvfh \
  -mabi=lp64d \
  -fuse-ld=lld \
  bug_test.cpp -o test_bug -lm
```

**Compile correctly (workaround):**
```bash
clang++ -std=c++17 -O2 \
  --target=riscv64-unknown-linux-gnu \
  --sysroot=<sysroot> \
  -march=rv64gc \
  -mabi=lp64d \
  -fuse-ld=lld \
  bug_test.cpp -o test_ok -lm
```

**Run under QEMU:**
```bash
qemu-riscv64 -L <sysroot> ./test_bug   # Bug: garbage values
qemu-riscv64 -L <sysroot> ./test_ok    # Correct output
```

### Expected vs Actual Output

**Correct output (rv64gc):**
```
delta[0] = 16384.00 (scaled_lo=16384.00, scaled_hi=0.00)
delta[1] = 16384.00 (scaled_lo=16384.00, scaled_hi=0.00)
...
Final acc_f32[0] = 655360.00 655360.00 589824.00 589824.00
Final acc_f32[1] = 524288.00 524288.00 458752.00 458752.00
PASS: All values are consistent and within expected range.
```

**Bug output (rv64gcv):**
```
delta[0] = 16384.00 (scaled_lo=16384.00, scaled_hi=0.00)
delta[1] = 0.00 (scaled_lo=0.00, scaled_hi=0.00)
delta[2] = 64.00 (scaled_lo=64.00, scaled_hi=0.00)
delta[3] = 272922816.00 (scaled_lo=272922816.00, scaled_hi=0.00)
...
Final acc_f32[0] = 655360.00 0.00 590848.00 4366765568.00
Final acc_f32[1] = 14681070632960.00 4365693440.00 4366380544.00 11776.00
FAIL: output[3] = 4366765568.00 (garbage value > 1e9)
FAIL: output[4] = 14681070632960.00 (garbage value > 1e9)
```

## Workaround

Compile without RVV extensions:

```bash
-march=rv64gc  # Works correctly
```

## Root Cause Analysis

The bug appears in LLVM's RISC-V backend optimizer when RVV extensions are enabled. Even with:
- `-fno-vectorize`
- `-fno-slp-vectorize`
- `-mllvm -riscv-v-vector-bits-max=0`

The bug persists. This suggests a deeper issue in the backend's register allocation or instruction scheduling for VLEN-enabled targets.

## Key Observations

1. **No RVV intrinsics used**: Test code is pure scalar C++
2. **Explicit array initialization**: All arrays have `{0}` initialization
3. **Optimization level sensitive**: Bug appears at `-O1`, `-O2`, `-O3`
4. **-O0 works**: Unoptimized code produces correct results
5. **VLEN-specific**: Bug triggered by `zvl512b` extension

## Impact

This bug blocks development of RVV-optimized code in llama.cpp and similar projects, as:
1. Tests cannot be verified under QEMU
2. Scalar reference implementations are corrupted
3. Makes it impossible to compare RVV vs scalar correctness

## Test Matrix

| Optimization | march | Result |
|--------------|-------|--------|
| -O0 | rv64gcv_zvl512b | PASS |
| -O1 | rv64gcv_zvl512b | FAIL |
| -O2 | rv64gcv_zvl512b | FAIL |
| -O2 | rv64gc | PASS |
| -O2 + -fno-vectorize | rv64gcv_zvl512b | FAIL |
| -O2 + vectorizer disabled | rv64gcv_zvl512b | FAIL |

## Proposed LLVM Bug Report Title

"RISC-V backend corrupts scalar array values with -march=rv64gcv_zvl512b at O1+"