// test.cpp — correctness test for RVV ggml_vec_dot_q5_0_q8_0
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu \
//       --sysroot=<sysroot> \
//       -march=rv64gcv_zvl512b_zfh_zvfh \
//       -mabi=lp64d \
//       -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
//       -I. \
//       -o test test.cpp -lm
//
// Run (under QEMU):
//   qemu-riscv64 -L <sysroot> ./test

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
// Minimal ggml type definitions (from ggml-common.h)
// ---------------------------------------------------------------------------
#define QK5_0 32
#define QK8_0 32
#define GGML_RESTRICT
#define GGML_UNUSED(x) (void)(x)

typedef uint16_t ggml_half;

// IEEE 754 half-precision to single-precision conversion
static inline float ggml_half_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t h_exp = (h >> 10) & 0x1F;
    uint32_t h_mant = h & 0x3FF;
    uint32_t f_exp, f_mant;
    if (h_exp == 0) {
        if (h_mant == 0) return sign ? -0.0f : 0.0f;
        int shift = 0;
        while ((h_mant & (1 << 9)) == 0) { h_mant <<= 1; shift++; }
        h_mant &= 0x3FF;
        f_exp = 127 - 15 - shift + 1;
        f_mant = h_mant << 13;
    } else if (h_exp == 31) {
        if (h_mant == 0) return sign ? -INFINITY : INFINITY;
        uint32_t nan_bits = (sign << 31) | (255 << 23) | (h_mant << 13);
        return *(float*)&nan_bits;
    } else {
        f_exp = h_exp - 15 + 127;
        f_mant = h_mant << 13;
    }
    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mant;
    return *(float*)&f_bits;
}

static inline uint16_t ggml_fp32_to_fp16(float f) {
    if (f == 0.0f) return 0;
    if (f < 0.0f) return ggml_fp32_to_fp16(-f) | 0x8000;
    int exp = 15;
    while (f >= 2.0f && exp < 30) { f /= 2.0f; exp++; }
    while (f < 1.0f && exp > 0) { f *= 2.0f; exp--; }
    uint32_t mant = (uint32_t)((f - 1.0f) * 1024.0f);
    return (exp << 10) | (mant & 0x3FF);
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)

// block_q5_0: 32 quantized values stored as nibbles + 5th-bit mask
typedef struct {
    ggml_half d;           // delta (scale factor)
    uint8_t   qh[4];       // 5-th bit of quants (32 bits, one per value)
    uint8_t   qs[QK5_0 / 2]; // nibbles / quants (16 bytes, 2 values per byte)
} block_q5_0;

// block_q8_0: 32 quantized int8 values
typedef struct {
    ggml_half d;       // delta (scale factor)
    int8_t   qs[QK8_0]; // quants (32 signed 8-bit values)
} block_q8_0;

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation (from llama.cpp quants.c)
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Disable optimization to ensure correct scalar behavior
#endif
static void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                             const void * GGML_RESTRICT vx, size_t bx,
                                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = (const block_q5_0 * GGML_RESTRICT)vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 * GGML_RESTRICT)vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_vec_dot_q5_0_q8_0.inl"

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
    return rng_next() % 256;
}

static float rng_float() {
    return (float)rng_next() / 32768.0f;
}

// ---------------------------------------------------------------------------
// Test data initialization
// ---------------------------------------------------------------------------
static void init_test_block(block_q5_0 * q5, block_q8_0 * q8, uint32_t seed) {
    rng_state = seed;

    // Initialize scale factor with positive value
    q5->d = ggml_fp32_to_fp16(0.1f + rng_float() * 0.9f);

    // Initialize qh[4] with random 32-bit mask
    for (int i = 0; i < 4; i++) {
        q5->qh[i] = rng_uint8();
    }

    // Initialize qs[16] with random packed nibbles
    for (int i = 0; i < QK5_0 / 2; i++) {
        q5->qs[i] = rng_uint8();
    }

    // Initialize Q8_0 block
    q8->d = ggml_fp32_to_fp16(0.1f + rng_float() * 0.9f);

    for (int i = 0; i < QK8_0; i++) {
        q8->qs[i] = rng_int8();
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Disable optimization to avoid LLVM issues
#endif
static int run_test(int n_blocks, uint32_t seed, float rel_tolerance, const char * label) {
    printf("Test %s: n_blocks=%d, seed=%u, rel_tol=%.6e\n",
           label, n_blocks, seed, rel_tolerance);

    std::vector<block_q5_0> q5(n_blocks);
    std::vector<block_q8_0> q8(n_blocks);
    float scalar_out = 0;
    float rvv_out = 0;

    for (int b = 0; b < n_blocks; b++) {
        init_test_block(&q5[b], &q8[b], seed + b);
    }

    // Run scalar (generic) reference
    float temp_scalar = 0;
    ggml_vec_dot_q5_0_q8_0_generic(QK8_0 * n_blocks, &temp_scalar, 0,
                                     q5.data(), 0, q8.data(), 0, 1);
    scalar_out = temp_scalar;

#if defined(__riscv_v_intrinsic)
    // Run RVV implementation
    float temp_rvv = 0;
    ggml_vec_dot_q5_0_q8_0_rvv(QK8_0 * n_blocks, &temp_rvv, 0,
                                 q5.data(), 0, q8.data(), 0, 1);
    rvv_out = temp_rvv;
#else
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    printf("  Scalar: %.8f\n", scalar_out);
    printf("  RVV:    %.8f\n", rvv_out);

    // Compare with relative tolerance
    int failures = 0;
    float abs_diff = fabsf(scalar_out - rvv_out);
    float magnitude = fabsf(scalar_out) + fabsf(rvv_out);
    float rel_diff = (magnitude > 1e-6f) ? abs_diff / magnitude : abs_diff;

    if (rel_diff > rel_tolerance) {
        failures++;
        printf("  FAIL: scalar=%.8f, rvv=%.8f, rel_diff=%.6e\n",
               scalar_out, rvv_out, rel_diff);
    }

    printf("  Abs diff: %.6e, rel diff: %.6e\n", abs_diff, rel_diff);
    printf("  %s\n\n", failures ? "FAIL" : "PASS");

    return failures;
}

int main() {
    printf("=== ggml_vec_dot_q5_0_q8_0: RVV-512 vs Scalar correctness test ===\n\n");

    int total_failures = 0;

#if defined(__riscv_v_fixed_vlen)
    printf("VLEN = %d bits\n\n", __riscv_v_fixed_vlen);
#else
    printf("VLEN detection: runtime (__riscv_vlenb())\n\n");
#endif

    // Basic tests: single block with various seeds
    total_failures += run_test(1, 42,    1e-6, "seed-42-1-block");
    total_failures += run_test(1, 123,   1e-6, "seed-123-1-block");
    total_failures += run_test(1, 456,   1e-6, "seed-456-1-block");
    total_failures += run_test(1, 999,   1e-6, "seed-999-1-block");
    total_failures += run_test(1, 0xDEAD, 1e-6, "seed-dead-1-block");

    // Multi-block tests
    total_failures += run_test(2, 42,    1e-6, "seed-42-2-blocks");
    total_failures += run_test(4, 42,    1e-6, "seed-42-4-blocks");
    total_failures += run_test(8, 42,    1e-6, "seed-42-8-blocks");
    total_failures += run_test(16, 42,   1e-6, "seed-42-16-blocks");
    total_failures += run_test(32, 42,   1e-6, "seed-42-32-blocks");

    // Edge case: all zeros
    {
        printf("Test edge-all-zeros: n_blocks=1, seed=0, rel_tol=1e-6\n");
        block_q5_0 q5 = {};
        block_q8_0 q8 = {};
        q5.d = ggml_fp32_to_fp16(1.0f);
        q8.d = ggml_fp32_to_fp16(1.0f);

        float scalar_out = 0, rvv_out = 0;
        ggml_vec_dot_q5_0_q8_0_generic(QK8_0, &scalar_out, 0, &q5, 0, &q8, 0, 1);
#if defined(__riscv_v_intrinsic)
        ggml_vec_dot_q5_0_q8_0_rvv(QK8_0, &rvv_out, 0, &q5, 0, &q8, 0, 1);
#else
        rvv_out = scalar_out;
#endif

        printf("  Scalar: %.8f\n", scalar_out);
        printf("  RVV:    %.8f\n", rvv_out);

        float abs_diff = fabsf(scalar_out - rvv_out);
        if (abs_diff > 1e-6f) {
            total_failures++;
            printf("  FAIL: abs_diff=%.6e\n", abs_diff);
        } else {
            printf("  PASS\n\n");
        }
    }

    // Edge case: all qh bits set (all values positive)
    {
        printf("Test edge-all-positive: n_blocks=1, seed=0, rel_tol=1e-6\n");
        block_q5_0 q5 = {};
        block_q8_0 q8 = {};
        q5.d = ggml_fp32_to_fp16(1.0f);
        q8.d = ggml_fp32_to_fp16(1.0f);
        memset(q5.qh, 0xFF, 4);  // All bits set = all values positive
        for (int i = 0; i < QK5_0 / 2; i++) {
            q5.qs[i] = 0x11;  // nibbles: 1, 1 → values: 1, 1 (positive)
        }
        for (int i = 0; i < QK8_0; i++) {
            q8.qs[i] = 1;  // All ones
        }

        float scalar_out = 0, rvv_out = 0;
        ggml_vec_dot_q5_0_q8_0_generic(QK8_0, &scalar_out, 0, &q5, 0, &q8, 0, 1);
#if defined(__riscv_v_intrinsic)
        ggml_vec_dot_q5_0_q8_0_rvv(QK8_0, &rvv_out, 0, &q5, 0, &q8, 0, 1);
#else
        rvv_out = scalar_out;
#endif

        printf("  Scalar: %.8f\n", scalar_out);
        printf("  RVV:    %.8f\n", rvv_out);

        float abs_diff = fabsf(scalar_out - rvv_out);
        if (abs_diff > 1e-6f) {
            total_failures++;
            printf("  FAIL: abs_diff=%.6e\n", abs_diff);
        } else {
            printf("  PASS\n\n");
        }
    }

    // Edge case: no qh bits set (all values negative)
    {
        printf("Test edge-all-negative: n_blocks=1, seed=0, rel_tol=1e-6\n");
        block_q5_0 q5 = {};
        block_q8_0 q8 = {};
        q5.d = ggml_fp32_to_fp16(1.0f);
        q8.d = ggml_fp32_to_fp16(1.0f);
        memset(q5.qh, 0x00, 4);  // No bits set = all values negative
        for (int i = 0; i < QK5_0 / 2; i++) {
            q5.qs[i] = 0x11;  // nibbles: 1, 1 → values: -15, -15 (negative)
        }
        for (int i = 0; i < QK8_0; i++) {
            q8.qs[i] = 1;  // All ones
        }

        float scalar_out = 0, rvv_out = 0;
        ggml_vec_dot_q5_0_q8_0_generic(QK8_0, &scalar_out, 0, &q5, 0, &q8, 0, 1);
#if defined(__riscv_v_intrinsic)
        ggml_vec_dot_q5_0_q8_0_rvv(QK8_0, &rvv_out, 0, &q5, 0, &q8, 0, 1);
#else
        rvv_out = scalar_out;
#endif

        printf("  Scalar: %.8f\n", scalar_out);
        printf("  RVV:    %.8f\n", rvv_out);

        float abs_diff = fabsf(scalar_out - rvv_out);
        if (abs_diff > 1e-6f) {
            total_failures++;
            printf("  FAIL: abs_diff=%.6e\n", abs_diff);
        } else {
            printf("  PASS\n\n");
        }
    }

    printf("=== Summary: %d test(s) failed ===\n", total_failures);
    return total_failures;
}
