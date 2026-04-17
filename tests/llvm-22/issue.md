# [RISC-V] Register allocator reuses address register for data value with -march=rv64gcv_zvl512b at -O2

## Summary

LLVM 22 RISC-V backend produces incorrect code when compiling scalar C++ code with `-march=rv64gcv_zvl512b` at optimization level `-O2` or higher. The register allocator incorrectly reuses a callee-saved register (s8/s4) that was holding an array base address for a data value passed to `printf`, causing subsequent memory accesses to use garbage addresses and produce NaN or extremely large values.

## Reproduction

### Minimal Test Case

```cpp
// File: min_repro_final.cpp
#include <stdint.h>
#include <stdio.h>

static void trigger_bug_pattern(float *output) {
    float acc_f32[2][4] = {{0}};
    int32_t acc_lo[4][4] = {{0}};
    int32_t acc_hi[4][4] = {{0}};
    int16_t scales[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    float sb_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Dual-variable loop - both i and p used as array indices
    for (int i = 0, p = 0; p < 4; i++, p += 2) {
        for (int k = 0; k < 4; k++) {
            for (int n = 0; n < 4; n++) {
                acc_lo[p][k] += 8 * 4;
                acc_hi[p][k] += 8 * 4;
                acc_lo[p+1][k] += 8 * 4;
                acc_hi[p+1][k] += 8 * 4;
            }

            int32_t sum_lo[4] = {0};
            int32_t sum_hi[4] = {0};
            sum_lo[0] = acc_lo[p][0] + acc_lo[p][1];
            sum_lo[1] = acc_lo[p][2] + acc_lo[p][3];
            sum_lo[2] = acc_lo[p+1][0] + acc_lo[p+1][1];
            sum_lo[3] = acc_lo[p+1][2] + acc_lo[p+1][3];
            sum_hi[0] = acc_hi[p][0] + acc_hi[p][1];
            sum_hi[1] = acc_hi[p][2] + acc_hi[p][3];
            sum_hi[2] = acc_hi[p+1][0] + acc_hi[p+1][1];
            sum_hi[3] = acc_hi[p+1][2] + acc_hi[p+1][3];

            // Conditional indexing based on p==0
            int idx_lo = (p == 0) ? k : k + 4;
            int idx_hi = (p == 0) ? k + 4 : k;

            float scaled_lo = (float)(scales[idx_lo] * sum_lo[k]);
            float scaled_hi = (float)(scales[idx_hi] * sum_hi[k]);
            float delta = sb_scale[k] * (scaled_lo + scaled_hi);

            // printf in inner loop breaks caller-saved registers (a0-a7)
            if (i == 0) {
                printf("  delta[%d] = %.2f\n", k, delta);
            }

            // BUG: acc_f32[i][k] uses address that was corrupted
            acc_f32[i][k] += delta;
        }
    }

    for (int k = 0; k < 4; k++) {
        output[k] = acc_f32[0][k];
        output[k + 4] = acc_f32[1][k];
    }
}

int main() {
    float output[8] = {0};
    trigger_bug_pattern(output);

    printf("\nOutput:\n");
    for (int i = 0; i < 8; i++) {
        printf("  output[%d] = %.2f\n", i, output[i]);
    }

    int failures = 0;
    for (int i = 0; i < 8; i++) {
        if (output[i] < -1e9f || output[i] > 1e9f) {
            printf("FAIL: output[%d] garbage value\n", i);
            failures++;
        }
    }
    return failures;
}
```

### Build Commands

```bash
# Bug-triggering compilation
clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu \
    --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b_zfh_zvfh \
    -mabi=lp64d \
    -fuse-ld=lld \
    min_repro_final.cpp -o bug_version -lm

# Correct compilation (workaround)
clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu \
    --sysroot=<sysroot> \
    -march=rv64gc \
    -mabi=lp64d \
    -fuse-ld=lld \
    min_repro_final.cpp -o ok_version -lm

# Generate assembly for comparison
clang++ -O2 --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b_zfh_zvfh -mabi=lp64d -S min_repro_final.cpp -o bug.s

clang++ -O2 --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gc -mabi=lp64d -S min_repro_final.cpp -o ok.s
```

### Run Under QEMU

```bash
qemu-riscv64 -L <sysroot> ./bug_version
qemu-riscv64 -L <sysroot> ./ok_version
```

### Expected vs Actual Output

**Bug version (incorrect):**
```
Output:
  output[0] = 384.00
  output[1] = 0.00
  output[2] = 107659392.00       ← garbage
  output[3] = 97680.00
  output[4] = -959892362019182527582872065277952.00  ← huge garbage
  output[5] = nan                ← NaN
  output[6] = 116171520.00       ← garbage
  output[7] = 97680.00
FAIL: output[4] garbage value
```

**Correct version (expected):**
```
Output:
  output[0] = 384.00
  output[1] = 0.00
  output[2] = 768.00
  output[3] = 768.00
  output[4] = 384.00
  output[5] = 0.00
  output[6] = 768.00
  output[7] = 768.00
```

## Disassembly Analysis

### Bug Version Assembly (key excerpt)

The bug manifests in how register `s8` is used:

```asm
# s8 initialized as acc_f32 base address
addi    s8, sp, 88              # s8 = sp + 88 (acc_f32[0] base)

# Later, s8 is updated for second iteration
addi    s8, sp, 104             # s8 = sp + 104 (acc_f32[1] base)

# BUG: s8 reused for data value passed to printf!
fmv.x.d s8, fa5                 # s8 now holds delta value (e.g., 768.0)
mv      a2, s8                  # Pass s8 (data) to printf as argument

# Subsequent memory access uses corrupted s8 as address
flw     fa5, 0(s8)              # Load from address 768 (garbage!)
                                # Instead of loading from sp+104
```

### Correct Version Assembly

In the correct version, `s10` is dedicated for the address:

```asm
# s10 dedicated for acc_f32 base address
addi    s10, sp, 72             # s10 = acc_f32[0] base
addi    s10, sp, 88             # s10 = acc_f32[1] base (updated correctly)

# Data value uses different register (s8)
fmv.x.d s8, fa5                 # s8 holds delta value (data)
mv      a2, s8                  # Pass to printf

# Address register s10 remains intact
flw     fa5, 0(s10)             # Correct load from sp+88
```

### Register Allocation Comparison

| Register | Bug Version | Correct Version |
|----------|-------------|-----------------|
| s8 | Address → **Data** (corrupted) | Data only |
| s10 | Not used for address | **Address** (preserved) |

The bug occurs because the register allocator, under pressure from:
1. Multiple 2D arrays requiring many base addresses
2. `printf` call breaking caller-saved registers (a0-a7)
3. Dual-loop variables (i, p) both used as indices
4. Conditional expressions requiring extra registers

incorrectly decides to reuse `s8` (which held `acc_f32` address) for passing `delta` to `printf`. After `printf`, subsequent `acc_f32[i][k]` accesses use `s8` as address, but it now contains a data value (e.g., 768), causing memory access to address 768 instead of the stack location.

## Trigger Conditions

This bug requires a specific combination of conditions:

| Condition | Pattern | Effect |
|-----------|---------|--------|
| Dual-variable loop | `for (i=0,p=0; p<4; i++,p+=2)` | Two loop vars tracked |
| Multiple 2D arrays | `arr[2][4]`, `arr[4][4]` | Need >6 base addr regs |
| Conditional indexing | `scales[(p==0)?k:k+4]` | Need condition reg |
| printf in inner loop | Call breaks a0-a7 | Only s0-s11 stable |
| Register pressure >12 | Inner loop needs >12 values | Forces reg reuse |

## Affected Versions

- LLVM 22.1.3 (commit e9846648fd6183ee6d8cbdb4502213fcf902a211)
- Tested: 2026-04-17

## Workaround

Compile without RVV extension:
```bash
-march=rv64gc    # Works correctly
```

Or use `-O0` (not practical for production):
```bash
-O0              # Works but no optimization
```

## Related Issues

- #57939 - Frame pointer corruption with RVV stack temporaries (similar register allocation issue)
- #82616 - Miscompilation due to incorrect scheduling

## Attachments

Minimal test case available at: `tests/llvm-22/min_repro_final.cpp` in the reproduction repository.

---

**Labels:** `backend:RISC-V`, `miscompilation`