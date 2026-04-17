// test.cpp -- correctness test for RVV ggml_gemv_q5_0_q8_0
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//       -march=rv64gcv_zvl512b_zvfh -mabi=lp64d \
//       -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
//       -I. test.cpp -o test -lm
//
// Build (native, scalar-only for debugging):
//   g++ -std=c++17 -O2 -I. test.cpp -o test -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal ggml type definitions (copied from ggml-common.h)
// ---------------------------------------------------------------------------
typedef uint16_t ggml_half;
#define GGML_RESTRICT
#define GGML_UNUSED(x) (void)(x)

// Redefine GGML_CPU_FP16_TO_FP32 to use our local conversion
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_fp32(x)

#define QK5_0 32
#define QK8_0 32

struct block_q5_0 {
    ggml_half d;           // delta (scale)
    uint8_t qh[4];         // 5th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants (16 bytes)
};

struct block_q8_0 {
    ggml_half d;       // delta (scale)
    int8_t qs[QK8_0]; // quants
};

// ---------------------------------------------------------------------------
// FP16 conversion helpers (IEEE 754 half-precision <-> single-precision)
// ---------------------------------------------------------------------------
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    float result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = 0.0f;
        } else {
            result = ldexpf(mantissa / 1024.0f, -14);
        }
    } else if (exponent == 31) {
        result = mantissa ? NAN : INFINITY;
        if (sign) result = -result;
    } else {
        result = ldexpf(1.0f + mantissa / 1024.0f, (int)exponent - 15);
    }

    return sign ? -result : result;
}

static inline uint16_t fp32_to_fp16(float f) {
    if (f != f) return 0x7E00; // NaN
    if (f == INFINITY) return 0x7C00;
    if (f == -INFINITY) return 0xFC00;

    int32_t sign = f < 0 ? 1 : 0;
    if (sign) f = -f;

    int32_t exp;
    frexpf(f, &exp);

    if (f == 0.0f) return sign ? 0x8000 : 0;

    float normal = ldexpf(f, -exp);
    if (normal < 0.5f) { normal *= 2; exp--; }
    int32_t exponent = exp + 14;
    if (exponent > 30) return sign ? 0xFC00 : 0x7C00;
    if (exponent < 0) return sign ? 0x8000 : 0;

    uint32_t mantissa = (uint32_t)((normal - 0.5f) * 1024 + 0.5f);
    if (mantissa >= 1024) { mantissa = 0; exponent++; }
    if (exponent > 30) return sign ? 0xFC00 : 0x7C00;

    return (uint16_t)((sign << 15) | (exponent << 10) | mantissa);
}

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
// ---------------------------------------------------------------------------
// This always uses the scalar path regardless of build target.
// We define it before including the .inl to provide the _generic symbol.
// The .inl file only defines _generic when NOT building for RVV+Zvfh,
// so we provide it here unconditionally.
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_gemv_q5_0_q8_0_ref(int n, float * GGML_RESTRICT s, size_t bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nr == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    const block_q5_0 * b_ptr_base = (const block_q5_0 *) vx;
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;

    for (int col = 0; col < nc; col++) {
        float sumf = 0.0f;

        for (int l = 0; l < nb; l++) {
            const block_q5_0 * b_ptr = b_ptr_base + col * nb + l;
            const block_q8_0 * y = a_ptr + l;

            uint32_t qh;
            memcpy(&qh, b_ptr->qh, sizeof(qh));

            int32_t sumi = 0;
            for (int j = 0; j < qk / 2; j++) {
                const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
                const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));
                const int32_t x0 = (int8_t)(((b_ptr->qs[j] & 0x0F) | xh_0) - 16);
                const int32_t x1 = (int8_t)(((b_ptr->qs[j] >>   4) | xh_1) - 16);
                sumi += x0 * y->qs[j];
                sumi += x1 * y->qs[j + qk / 2];
            }

            sumf += (GGML_CPU_FP16_TO_FP32(b_ptr->d) * GGML_CPU_FP16_TO_FP32(y->d)) * sumi;
        }

        s[col] = sumf;
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation -- single source of truth
// ---------------------------------------------------------------------------
// The .inl defines:
//   - q5_0_q8_0_dot_block()       -- scalar block dot helper (always)
//   - q5_0_q8_0_dot_block_rvv()   -- RVV block dot (when __riscv_v_intrinsic)
//   - ggml_gemv_q5_0_q8_0_rvv()   -- full RVV GEMV (when __riscv_v_intrinsic + __riscv_zvfh)
//   - ggml_gemv_q5_0_q8_0()       -- wrapper (dispatches RVV or generic)
//   - ggml_gemv_q5_0_q8_0_generic() -- scalar fallback (when NOT __riscv_v_intrinsic)
//
// For test purposes, we need the _generic symbol on RVV targets too.
// We define it as a wrapper around _ref before including the .inl.
#if defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
// On RVV+Zvfh targets, the .inl does NOT define _generic.
// Provide it by aliasing to our _ref implementation.
static void ggml_gemv_q5_0_q8_0_generic(int n, float * s, size_t bs,
                                         const void * vx, const void * vy,
                                         int nr, int nc) {
    ggml_gemv_q5_0_q8_0_ref(n, s, bs, vx, vy, nr, nc);
}
#endif

#include "rvv_gemv_q5_0_q8_0.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static float rng_float() {
    return (rng_next() % 10000) / 10000.0f;
}

static int8_t rng_i8() {
    return (int8_t)(rng_next() % 256 - 128);
}

static uint8_t rng_u8() {
    return (uint8_t)(rng_next() % 256);
}

// ---------------------------------------------------------------------------
// Generate random Q5_0 block
// ---------------------------------------------------------------------------
static void random_block_q5_0(block_q5_0 * blk) {
    float d = 0.1f + rng_float() * 1.0f;
    blk->d = fp32_to_fp16(d);

    for (int i = 0; i < 4; i++) {
        blk->qh[i] = rng_u8();
    }

    for (int i = 0; i < QK5_0 / 2; i++) {
        blk->qs[i] = rng_u8();
    }
}

// ---------------------------------------------------------------------------
// Generate random Q8_0 block from float values
// ---------------------------------------------------------------------------
static void quantize_to_q8_0(block_q8_0 * blk, const float * values) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_0; i++) {
        float abs_v = fabsf(values[i]);
        if (abs_v > amax) amax = abs_v;
    }

    float d = amax / 127.0f;
    if (d == 0.0f) d = 1.0f;
    blk->d = fp32_to_fp16(d);

    float id = 1.0f / d;
    for (int i = 0; i < QK8_0; i++) {
        blk->qs[i] = (int8_t)(roundf(values[i] * id));
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(int n_cols, int n_rows, const char * label) {
    const int nc = n_cols;
    const int n = n_rows;
    const int nr = 1;
    const int nb = n / QK8_0;

    printf("  Test: %-30s  n=%d  nc=%d  nb=%d\n", label, n, nc, nb);

    // Allocate input data
    std::vector<block_q5_0> weight_data(nc * nb);
    std::vector<block_q8_0> act_data(nb);
    std::vector<float> act_float(n);

    // Generate random weights
    for (size_t i = 0; i < (size_t)(nc * nb); i++) {
        random_block_q5_0(&weight_data[i]);
    }

    // Generate random activations and quantize
    for (int i = 0; i < n; i++) {
        act_float[i] = (rng_float() - 0.5f) * 4.0f;
    }
    for (int i = 0; i < nb; i++) {
        quantize_to_q8_0(&act_data[i], &act_float[i * QK8_0]);
    }

    // Compute results using both implementations
    std::vector<float> s_ref(nc, 0.0f);
    std::vector<float> s_impl(nc, 0.0f);

    // Run scalar reference (always _ref, not affected by __riscv_v)
    ggml_gemv_q5_0_q8_0_ref(n, s_ref.data(), sizeof(float),
                              weight_data.data(), act_data.data(), nr, nc);

    // Run the implementation under test (may be RVV or generic)
    ggml_gemv_q5_0_q8_0(n, s_impl.data(), sizeof(float),
                          weight_data.data(), act_data.data(), nr, nc);

    // Compare results
    int failures = 0;
    float max_abs_diff = 0.0f;

    for (int col = 0; col < nc; col++) {
        float expected = s_ref[col];
        float actual = s_impl[col];

        float abs_diff = fabsf(expected - actual);
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;

        // Q5_0 has 5-bit precision. The integer dot product is exact.
        // Only FP scale multiplication introduces error.
        // Tolerance: allow 1e-3 relative error or 0.01 absolute error.
        float rel_diff = (fabsf(expected) > 1e-6f) ? abs_diff / fabsf(expected) : abs_diff;
        float tol = (fabsf(expected) > 1.0f) ? fabsf(expected) * 1e-4f : 0.01f;

        if (abs_diff > tol) {
            if (failures < 5) {
                printf("    FAIL: col=%d  expected=%.6f  got=%.6f  diff=%.6f  rel=%.2e\n",
                       col, expected, actual, abs_diff, rel_diff);
            }
            failures++;
        }
    }

    if (failures == 0) {
        printf("    PASS (max_abs_diff=%.6e)\n", max_abs_diff);
    } else {
        printf("    FAIL: %d column(s) mismatched\n", failures);
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== ggml_gemv_q5_0_q8_0: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Test 1: Single column, single block (minimal)
    rng_state = 42;
    failures += run_test(1, 32, "1x32 (1 col, 1 block)");

    // Test 2: Multiple columns, single block
    rng_state = 42;
    failures += run_test(4, 32, "4x32 (4 cols, 1 block)");

    // Test 3: Exactly 16 columns (one tile), 2 blocks
    rng_state = 42;
    failures += run_test(16, 64, "16x64 (16 cols, 2 blocks)");

    // Test 4: 32 columns (two tiles), 2 blocks
    rng_state = 42;
    failures += run_test(32, 64, "32x64 (32 cols, 2 blocks)");

    // Test 5: 48 columns (three tiles), 4 blocks
    rng_state = 42;
    failures += run_test(48, 128, "48x128 (48 cols, 4 blocks)");

    // Test 6: Single column, many blocks
    rng_state = 42;
    failures += run_test(1, 128, "1x128 (1 col, 4 blocks)");

    // Test 7: 16 columns, many blocks (larger matrix)
    rng_state = 42;
    failures += run_test(16, 256, "16x256 (16 cols, 8 blocks)");

    // Test 8: 64 columns (four tiles), 4 blocks
    rng_state = 42;
    failures += run_test(64, 128, "64x128 (64 cols, 4 blocks)");

    // Test 9: Non-tile-aligned column count (edge case)
    rng_state = 42;
    failures += run_test(17, 64, "17x64 (17 cols, 2 blocks)");

    // Test 10: Single column, single block (edge: minimum viable test)
    rng_state = 42;
    failures += run_test(1, 32, "1x32-repeat (1 col, 1 block)");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}