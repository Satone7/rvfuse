// test_quantize_q8_0_4x4.cpp — correctness test for RVV ggml_quantize_mat_q8_0_4x4
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -o test_quantize_q8_0_4x4 test_quantize_q8_0_4x4.cpp -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <math.h>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal ggml type definitions (copied from ggml-common.h)
// ---------------------------------------------------------------------------
typedef uint16_t ggml_half;

#define QK8_0 32

#define GGML_RESTRICT
#define GGML_UNUSED(x) (void)(x)

static inline float ggml_half_to_fp32(ggml_half h) {
    union { uint32_t u; float f; } u32f;
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { u32f.u = sign << 31; return u32f.f; }
        while (!(frac & 0x400)) frac <<= 1;
        frac &= 0x3FF;
        u32f.u = (sign << 31) | ((exp + 127 - 14) << 23) | ((frac << 13) & 0x7FFFFF);
    } else if (exp == 31) {
        u32f.u = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        u32f.u = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    return u32f.f;
}

static inline ggml_half ggml_fp32_to_half(float f) {
    union { uint32_t u; float f; } u32f;
    u32f.f = f;
    uint32_t u = u32f.u;
    uint32_t sign = (u >> 31) & 1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t frac = u & 0x7FFFFF;

    if (exp == 0) {
        return (ggml_half)(sign << 15);
    } else if (exp == 255) {
        return (ggml_half)((sign << 15) | 0x7C00 | (frac >> 13));
    } else {
        int new_exp = exp - 127 + 15;
        if (new_exp <= 0) {
            // Underflow to zero
            return (ggml_half)(sign << 15);
        } else if (new_exp >= 31) {
            // Overflow to infinity
            return (ggml_half)((sign << 15) | 0x7C00);
        } else {
            return (ggml_half)((sign << 15) | (new_exp << 10) | (frac >> 13));
        }
    }
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) ggml_fp32_to_half(x)

// ---------------------------------------------------------------------------
// Data structures for Q8_0 4x4 interleaved format
// ---------------------------------------------------------------------------
struct block_q8_0x4 {
    ggml_half d[4];            // 4 scales (one per row)
    int8_t qs[QK8_0 * 4];      // 4*32 = 128 quantized values, interleaved
};

static_assert(sizeof(block_q8_0x4) == 4 * sizeof(ggml_half) + QK8_0 * 4,
              "wrong q8_0x4 block size");

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_quantize_mat_q8_0_4x4_generic(
    const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k)
{
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    for (int i = 0; i < nb; i++) {
        for (int r = 0; r < 4; r++) {
            const float * src = x + i * QK8_0 + r * k;

            // Find max absolute value
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                float v = fabsf(src[j]);
                if (v > amax) amax = v;
            }

            float d = amax / 127.0f;
            y[i].d[r] = GGML_CPU_FP32_TO_FP16(d);

            float id = d ? 1.0f / d : 0.0f;

            // Quantize each value
            for (int j = 0; j < QK8_0; j++) {
                float v = src[j] * id;
                int vi = (int)roundf(v);
                // Clamp to [-127, 127]
                if (vi < -127) vi = -127;
                if (vi > 127) vi = 127;

                // Store in interleaved format matching RVV vsseg4e32 layout.
                // vsseg4e32 stores 4 segments with stride=4:
                //   segment_r elements at int32 positions r, r+4, r+8, ...
                // Each int32 contains 4 consecutive elements from same row.
                // For row r, element j: int32_index = r + (j/4)*4, byte = j%4
                // pos = int32_index * 4 + byte_offset = 4*r + (j/4)*16 + (j%4)
                int pos = 4 * r + (j / 4) * 16 + (j % 4);
                y[i].qs[pos] = (int8_t)vi;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_quantize_q8_0_4x4.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static float fp32_from_bits(uint32_t u) {
    float f; memcpy(&f, &u, 4); return f;
}

static float rng_float() {
    // Generate floats in range [-1.0, 1.0] with various magnitudes
    uint32_t sign = rng_next() & 1;
    uint32_t exp = (rng_next() % 8) + 120;  // exp from 120-127 gives ~0.5 to ~2.0 magnitude
    uint32_t frac = rng_next() | (rng_next() << 15);
    uint32_t u = (sign << 31) | (exp << 23) | (frac & 0x7FFFFF);
    return fp32_from_bits(u);
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(int n_blocks, const char * label) {
    const int k = n_blocks * QK8_0;  // Total elements per row

    // Source: 4 rows of FP32 data
    std::vector<float> src(4 * k);
    for (int i = 0; i < 4 * k; i++) {
        src[i] = rng_float();
    }

    // Output buffers
    std::vector<block_q8_0x4> out_generic(n_blocks);
    std::vector<block_q8_0x4> out_rvv(n_blocks);

    // Run generic (scalar) reference
    ggml_quantize_mat_q8_0_4x4_generic(src.data(), out_generic.data(), k);

#if defined(__riscv_v_intrinsic)
    // Run RVV
    ggml_quantize_mat_q8_0_4x4_rvv(src.data(), out_rvv.data(), k);
#else
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    // Compare outputs
    int n_mismatch = 0;
    float max_scale_diff = 0.0f;
    int max_q_diff = 0;

    for (int b = 0; b < n_blocks; b++) {
        for (int r = 0; r < 4; r++) {
            // Compare scales
            float d_gen = GGML_CPU_FP16_TO_FP32(out_generic[b].d[r]);
            float d_rvv = GGML_CPU_FP16_TO_FP32(out_rvv[b].d[r]);
            float scale_diff = fabsf(d_gen - d_rvv);
            if (scale_diff > max_scale_diff) max_scale_diff = scale_diff;

            // Compare quantized values
            for (int j = 0; j < QK8_0; j++) {
                // Use same indexing formula as vsseg4e32 layout
                int pos = 4 * r + (j / 4) * 16 + (j % 4);
                int q_gen = (int)out_generic[b].qs[pos];
                int q_rvv = (int)out_rvv[b].qs[pos];
                int q_diff = abs(q_gen - q_rvv);
                if (q_diff > max_q_diff) max_q_diff = q_diff;

                // Allow small rounding differences (max 1)
                if (q_diff > 1) {
                    n_mismatch++;
                    if (n_mismatch <= 5) {
                        printf("  MISMATCH [%s] block=%d row=%d idx=%d: gen=%d rvv=%d\n",
                               label, b, r, j, q_gen, q_rvv);
                    }
                }
            }
        }
    }

    if (n_mismatch == 0) {
        printf("  [PASS] %s — %d blocks, max_scale_diff=%.2e, max_q_diff=%d [VLEN=%d]\n",
               label, n_blocks, max_scale_diff, max_q_diff,
#if defined(__riscv_v_fixed_vlen)
               __riscv_v_fixed_vlen
#else
               -1
#endif
               );
        return 0;
    } else {
        printf("  [FAIL] %s — %d value mismatches, max_q_diff=%d\n",
               label, n_mismatch, max_q_diff);
        return 1;
    }
}

int main() {
    printf("=== ggml_quantize_mat_q8_0_4x4: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Test 1: minimum viable
    failures += run_test(1, "min-1block");

    // Test 2: typical
    failures += run_test(4, "typical-4blocks");

    // Test 3: stress
    failures += run_test(16, "stress-16blocks");

    // Test 4: large
    failures += run_test(64, "large-64blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}