// test_vec_dot_q6_K_q8_K.cpp — correctness test for RVV vec_dot_q6_K_q8_K
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -o test test.cpp -lm
//
// Build (scalar reference, x86 or arm):
//   g++ -std=c++17 -O2 -o test_scalar test.cpp -lm

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
#define QK_K 256

// ggml_half: 16-bit IEEE float (simple implementation for testing)
typedef uint16_t ggml_half;

static inline float ggml_half_to_fp32(ggml_half h) {
    // IEEE 754 half-precision to float conversion
    union { uint32_t u; float f; } u;
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            u.u = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            u.u = (sign << 31) | ((127 - 15 - exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u.u = (sign << 31) | (255 << 23) | (mant << 13);
    } else {
        u.u = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return u.f;
}

static inline ggml_half ggml_fp32_to_half(float f) {
    // Float to IEEE 754 half-precision conversion (truncate rounding)
    union { uint32_t u; float f; } u;
    u.f = f;
    uint32_t sign = (u.u >> 31) & 1;
    uint32_t exp = (u.u >> 23) & 0xFF;
    uint32_t mant = u.u & 0x007FFFFF;
    uint16_t h;
    if (exp <= 127 - 15) {
        // Underflow to zero
        h = sign << 15;
    } else if (exp >= 127 + 16) {
        // Overflow to infinity
        h = (sign << 15) | 0x7C00;
    } else {
        int new_exp = exp - 127 + 15;
        h = (sign << 15) | (new_exp << 10) | (mant >> 13);
    }
    return h;
}

// Data structures
struct block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    ggml_half d;             // super-block scale
};

struct block_q8_K {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants (256 bytes)
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16 (16 elements)
};

#define GGML_RESTRICT
#define GGML_UNUSED(x) ((void)(x))

#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_fp32(x)

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
// ---------------------------------------------------------------------------
// This is a direct copy of ggml_vec_dot_q6_K_q8_K_generic from llama.cpp
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                            const void * GGML_RESTRICT vx, size_t bx,
                                            const void * GGML_RESTRICT vy, size_t by,
                                            int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#include "rvv_vec_dot_q6_K_q8_K.inl"
#endif

// ---------------------------------------------------------------------------
// Wrapper function for testing
// ---------------------------------------------------------------------------
static void ggml_vec_dot_q6_K_q8_K_test(int n, float * GGML_RESTRICT s, size_t bs,
                                         const void * GGML_RESTRICT vx, size_t bx,
                                         const void * GGML_RESTRICT vy, size_t by,
                                         int nrc) {
#if defined(__riscv_v_intrinsic)
    ggml_vec_dot_q6_K_q8_K_rvv(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static int8_t rng_int8() {
    return (int8_t)(rng_next() % 256 - 128);
}

static uint8_t rng_uint8() {
    return (uint8_t)(rng_next() % 256);
}

static float rng_float(float lo, float hi) {
    float f = (float)rng_next() / 32767.0f;
    return lo + f * (hi - lo);
}

static void fill_block_q6_K(block_q6_K * block, float d_scale) {
    // Fill ql with valid 4-bit values (0-15)
    for (int i = 0; i < QK_K/2; ++i) {
        block->ql[i] = rng_uint8();  // Full byte, we use nibbles
    }
    // Fill qh with valid 2-bit values (0-3 in each 2-bit position)
    for (int i = 0; i < QK_K/4; ++i) {
        block->qh[i] = rng_uint8();  // Full byte, we use pairs
    }
    // Fill scales
    for (int i = 0; i < QK_K/16; ++i) {
        block->scales[i] = rng_int8();  // Can be positive or negative
    }
    // Fill d (scale)
    block->d = ggml_fp32_to_half(rng_float(0.01f, d_scale));
}

static void fill_block_q8_K(block_q8_K * block, float d_scale) {
    // Fill qs
    for (int i = 0; i < QK_K; ++i) {
        block->qs[i] = rng_int8();
    }
    // Fill bsums (sum of qs in groups of 16)
    for (int g = 0; g < QK_K/16; ++g) {
        int16_t sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += block->qs[g * 16 + i];
        }
        block->bsums[g] = sum;
    }
    // Fill d (scale)
    block->d = rng_float(0.01f, d_scale);
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(int n_blocks, const char * label) {
    printf("Test: %s (n_blocks=%d, n=%d elements)\n", label, n_blocks, n_blocks * QK_K);

    // Generate random test data
    std::vector<block_q6_K> x(n_blocks);
    std::vector<block_q8_K> y(n_blocks);

    rng_state = 42 + (uint32_t)n_blocks;  // Unique seed per test

    for (int i = 0; i < n_blocks; ++i) {
        fill_block_q6_K(&x[i], 1.0f);
        fill_block_q8_K(&y[i], 1.0f);
    }

    // Compute scalar reference
    float s_generic = 0.0f;
    ggml_vec_dot_q6_K_q8_K_generic(n_blocks * QK_K, &s_generic, 0,
                                    x.data(), 0, y.data(), 0, 1);

    // Compute RVV result
    float s_rvv = 0.0f;
    ggml_vec_dot_q6_K_q8_K_test(n_blocks * QK_K, &s_rvv, 0,
                                 x.data(), 0, y.data(), 0, 1);

    // Compare results
    float abs_diff = fabs(s_generic - s_rvv);
    float rel_diff = (s_generic != 0.0f) ? abs_diff / fabs(s_generic) : abs_diff;

    printf("  Generic: %.6f\n", s_generic);
    printf("  RVV:     %.6f\n", s_rvv);
    printf("  Abs diff: %.6e\n", abs_diff);
    printf("  Rel diff: %.6e\n", rel_diff);

    // Tolerance: allow small floating point differences
    // The integer math should be exact, only the final float multiply may differ slightly
    const float tol_abs = 1e-3f;
    const float tol_rel = 1e-4f;

    int failed = 0;
    if (abs_diff > tol_abs && rel_diff > tol_rel) {
        printf("  FAILED: diff exceeds tolerance\n");
        failed = 1;
    } else {
        printf("  PASSED\n");
    }

    return failed;
}

int main() {
    printf("=== ggml_vec_dot_q6_K_q8_K: RVV vs Scalar correctness test ===\n\n");

#if defined(__riscv_v_intrinsic)
    int vlen = __riscv_vlenb() * 8;
    printf("RVV detected: VLEN=%d\n\n", vlen);
#else
    printf("RVV not detected: running scalar reference only\n\n");
#endif

    int failures = 0;

    // Test various sizes
    failures += run_test(1,    "single-block");
    failures += run_test(2,    "two-blocks");
    failures += run_test(4,    "four-blocks");
    failures += run_test(8,    "eight-blocks");
    failures += run_test(16,   "sixteen-blocks");
    failures += run_test(32,   "thirty-two-blocks");
    failures += run_test(64,   "sixty-four-blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}