// LLVM 22 RISC-V Backend Bug - Minimal Reproduction
//
// Bug: Register allocator incorrectly reuses address register (s8/s4) for data value
//      when compiling with -march=rv64gcv_zvl512b at -O2
//
// Trigger conditions:
// 1. Dual-variable loop (i=0,p=0; p<4; i++,p+=2) - complex tracking
// 2. Multiple 2D arrays - need many base address registers
// 3. Conditional indexing (p==0 ? k : k+4) - needs condition register
// 4. printf in inner loop - breaks a0-a7, only s0-s11 stable
// 5. Register pressure > 12 active values - forces reuse
//
// Result: s8 (acc_f32 base addr) reused as data value for printf,
//         subsequent acc[i][k] access uses garbage address.
//
// Bug compile:
//   clang++ -O2 -march=rv64gcv_zvl512b_zfh_zvfh -mabi=lp64d min_repro_final.cpp
//
// Correct compile:
//   clang++ -O2 -march=rv64gc -mabi=lp64d min_repro_final.cpp

#include <stdint.h>
#include <stdio.h>

static void trigger_bug_pattern(float *output) {
    // CRITICAL: Multiple 2D arrays create register pressure
    float acc_f32[2][4] = {{0}};
    int32_t acc_lo[4][4] = {{0}};
    int32_t acc_hi[4][4] = {{0}};
    int16_t scales[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    float sb_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // CRITICAL: Dual-variable loop pattern
    // Both i and p are used as array indices simultaneously
    for (int i = 0, p = 0; p < 4; i++, p += 2) {
        for (int k = 0; k < 4; k++) {
            // Build values in acc_lo/acc_hi using p and p+1
            for (int n = 0; n < 4; n++) {
                acc_lo[p][k] += 8 * 4;
                acc_hi[p][k] += 8 * 4;
                acc_lo[p+1][k] += 8 * 4;
                acc_hi[p+1][k] += 8 * 4;
            }

            int32_t sum_lo[4] = {0};
            int32_t sum_hi[4] = {0};

            // Compute sums from acc_lo/acc_hi
            sum_lo[0] = acc_lo[p][0] + acc_lo[p][1];
            sum_lo[1] = acc_lo[p][2] + acc_lo[p][3];
            sum_lo[2] = acc_lo[p+1][0] + acc_lo[p+1][1];
            sum_lo[3] = acc_lo[p+1][2] + acc_lo[p+1][3];

            sum_hi[0] = acc_hi[p][0] + acc_hi[p][1];
            sum_hi[1] = acc_hi[p][2] + acc_hi[p][3];
            sum_hi[2] = acc_hi[p+1][0] + acc_hi[p+1][1];
            sum_hi[3] = acc_hi[p+1][2] + acc_hi[p+1][3];

            // CRITICAL: Conditional indexing based on p==0
            int idx_lo = (p == 0) ? k : k + 4;
            int idx_hi = (p == 0) ? k + 4 : k;

            float scaled_lo = (float)(scales[idx_lo] * sum_lo[k]);
            float scaled_hi = (float)(scales[idx_hi] * sum_hi[k]);
            float delta = sb_scale[k] * (scaled_lo + scaled_hi);

            // CRITICAL: printf in inner loop breaks caller-saved registers
            // Forces use of s0-s11, but we have >12 active values
            if (i == 0) {
                printf("  delta[%d] = %.2f (s_lo=%.2f, s_hi=%.2f)\n",
                       k, delta, scaled_lo, scaled_hi);
            }

            // BUG manifests here: acc_f32[i][k] uses i-based address
            // but s8 (address register) was reused for data value
            acc_f32[i][k] += delta;
        }
    }

    // Copy to output
    for (int k = 0; k < 4; k++) {
        output[k] = acc_f32[0][k];
        output[k + 4] = acc_f32[1][k];
    }
}

int main() {
    printf("=== LLVM RISC-V Backend Bug Test ===\n");

    float output[8] = {0};
    trigger_bug_pattern(output);

    printf("\nFinal output:\n");
    for (int i = 0; i < 8; i++) {
        printf("  output[%d] = %.2f\n", i, output[i]);
    }

    // Verify
    printf("\n=== Verification ===\n");
    int failures = 0;
    for (int i = 0; i < 8; i++) {
        // Expected values are small (< 10000)
        // Any value > 1e9 or NaN indicates register corruption bug
        if (output[i] < -1e9f || output[i] > 1e9f) {
            printf("FAIL: output[%d] = %.2f (garbage > 1e9)\n", i, output[i]);
            failures++;
        }
    }

    if (failures) {
        printf("\n*** LLVM RISC-V backend bug detected! ***\n");
        printf("Register allocator reused address register for data value.\n");
    } else {
        printf("PASS: All values within expected range.\n");
    }

    return failures;
}