// llvm_riscv_bug_test.cpp — Minimal reproduction of LLVM 22 RISC-V backend bug
//
// Bug: Array elements receive garbage values when compiled with -O1+ and
//      -march=rv64gcv_zvl512b_zfh_zvfh, even with explicit initialization.
//
// The bug manifests as:
// 1. NaN values in float arrays
// 2. Large garbage values (e.g., -536550016.00 instead of expected values)
// 3. Values appearing to be from uninitialized stack memory
//
// Compile (bug triggers):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//     --sysroot=<sysroot> -march=rv64gcv_zvl512b_zfh_zvfh -mabi=lp64d \
//     -fuse-ld=lld llvm_riscv_bug_test.cpp -o test_bug -lm
//
// Compile (works):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//     --sysroot=<sysroot> -march=rv64gc -mabi=lp64d \
//     -fuse-ld=lld llvm_riscv_bug_test.cpp -o test_ok -lm
//
// Run under QEMU:
//   qemu-riscv64 -L <sysroot> ./test_bug

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Trigger pattern: 4-level nested loops with inner accumulator arrays
// This matches the structure of the original gemv algorithm
static void trigger_bug_pattern(float * output) {
    // Explicitly initialized arrays
    float acc_f32[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    int32_t acc_lo[4][4] = {{0}};
    int32_t acc_hi[4][4] = {{0}};
    int16_t scales[8] = {16, 16, 16, 16, 0, 0, 0, 0};
    float sb_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Outer loop (mimics block iteration)
    for (int b = 0; b < 1; b++) {
        // Subblock loop
        for (int sb = 0; sb < 4; sb++) {
            // Column pair loop - this level seems critical
            for (int cp = 0; cp < 4; cp++) {
                // Inner vector loop
                for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                    // Sum index loop
                    for (int sum_idx = 0; sum_idx < 4; sum_idx++) {
                        int32_t sum_lo = 0, sum_hi = 0;
                        // Innermost computation
                        for (int n = 0; n < 4; n++) {
                            sum_lo += 8 * 4;  // nibble=8, q8=4
                            sum_hi += 8 * 4;
                        }
                        acc_lo[cp][sum_idx] += sum_lo;
                        acc_hi[cp][sum_idx] += sum_hi;
                    }
                }

                // Pairwise add and scale - this is where bug appears
                for (int i = 0, p = 0; p < 4; i++, p += 2) {
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

                    // Apply scales - BUG: acc_f32[i][2] and [3] get garbage
                    for (int k = 0; k < 4; k++) {
                        float scaled_lo = (float)(scales[p == 0 ? k : k + 4] * sum_lo[k]);
                        float scaled_hi = (float)(scales[p == 0 ? k + 4 : k] * sum_hi[k]);
                        float delta = sb_scale[k] * (scaled_lo + scaled_hi);

                        // Debug print to show where bug occurs
                        if (sb == 0 && i == 0) {
                            printf("  delta[%d] = %.2f (scaled_lo=%.2f, scaled_hi=%.2f)\n",
                                   k, delta, scaled_lo, scaled_hi);
                        }

                        acc_f32[i][k] += delta;
                    }
                }
            }

            // Print accumulator state after each subblock
            if (sb < 4) {
                printf("After sb=%d: acc_f32[0] = %.2f %.2f %.2f %.2f\n",
                       sb, acc_f32[0][0], acc_f32[0][1], acc_f32[0][2], acc_f32[0][3]);
            }
        }
    }

    printf("\nFinal acc_f32[0] = %.2f %.2f %.2f %.2f\n",
           acc_f32[0][0], acc_f32[0][1], acc_f32[0][2], acc_f32[0][3]);
    printf("Final acc_f32[1] = %.2f %.2f %.2f %.2f\n",
           acc_f32[1][0], acc_f32[1][1], acc_f32[1][2], acc_f32[1][3]);

    // Copy to output
    for (int k = 0; k < 4; k++) {
        output[k] = acc_f32[0][k];
        output[k + 4] = acc_f32[1][k];
    }
}

int main() {
    printf("=== LLVM RISC-V Backend Bug Test ===\n");
    printf("Testing nested loop pattern with local array accumulation.\n\n");

    float output[8] = {0};

    printf("Running trigger_bug_pattern():\n");
    trigger_bug_pattern(output);

    printf("\nOutput array:\n");
    printf("  output = %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
           output[0], output[1], output[2], output[3],
           output[4], output[5], output[6], output[7]);

    // Verification
    printf("\n=== Verification ===\n");
    printf("Expected: all output values should be < 1e7 (reasonable computation)\n");
    printf("          no NaN, no huge values like 1e10+\n");

    bool bug_detected = false;

    // Check for NaN (definite uninitialized memory)
    for (int i = 0; i < 8; i++) {
        if (std::isnan(output[i])) {
            printf("FAIL: output[%d] = NaN\n", i);
            bug_detected = true;
        }
    }

    // Check for garbage values (values > 1e9 are from uninitialized memory)
    // The computation uses small values (scales 0-63, nibbles 0-15, quants -128..127)
    // so any result > ~1e7 is impossible
    for (int i = 0; i < 8; i++) {
        if (fabs(output[i]) > 1e9f) {
            printf("FAIL: output[%d] = %.2f (garbage value > 1e9)\n", i, output[i]);
            bug_detected = true;
        }
    }

    if (bug_detected) {
        printf("\n*** LLVM RISC-V backend bug detected! ***\n");
        printf("The optimizer corrupts array values when RVV march is enabled.\n");
        return 1;
    } else {
        printf("PASS: All values are consistent and within expected range.\n");
        return 0;
    }
}