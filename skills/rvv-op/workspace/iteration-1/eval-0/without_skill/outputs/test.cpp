// test.cpp — correctness test for RVV ggml_gemv_q5_0_q8_0
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 -o test test.cpp -lm
//
// Build (scalar, for reference):
//   g++ -std=c++17 -O2 -o test_scalar test.cpp -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Q8_0 block structure (from ggml-common.h)
// ---------------------------------------------------------------------------
typedef struct {
    ggml_half d;       // delta (scale factor)
    int8_t  qs[QK8_0]; // quantized values
} block_q8_0;

// ---------------------------------------------------------------------------
// Q5_0 single block structure (from ggml-common.h)
// ---------------------------------------------------------------------------
typedef struct {
    ggml_half d;           // delta (scale factor)
    uint8_t  qh[4];       // 5th bit of quants (32 bits for 32 elements)
    uint8_t  qs[QK5_0/2]; // nibbles / quants (16 bytes)
} block_q5_0;

// ---------------------------------------------------------------------------
// Q5_0 interleaved block structure for 8-column GEMV
//
// Custom structure (Q5_0 has 5-bit packing + qh bitmask, not in template<>):
//   d[8]:    8 half-precision scale factors (16 bytes)
//   qh[8][4]: 8 x 4-byte qh bitmasks (32 bytes)
//   qs[128]: interleaved nibble-packed quants (128 bytes)
//
// qs interleaving: qs[k * 8 * 8 + j * 8 + i] gives the i-th byte of
// the j-th column's k-th chunk. Each byte contains two nibbles (low + high).
// ---------------------------------------------------------------------------
struct block_q5_0x8 {
    ggml_half d[8];       // 8 scale factors (16 bytes)
    uint8_t qh[8][4];     // 8 x 4-byte qh bitmasks (32 bytes)
    uint8_t qs[128];      // interleaved nibble-packed quants (128 bytes)
};
// Total: 16 + 32 + 128 = 176 bytes
static_assert(sizeof(block_q5_0x8) == 176, "block_q5_0x8 should be 176 bytes");

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
//
// Based on ggml_vec_dot_q5_0_q8_0_generic from quants.c,
// extended to the GEMV interleaved pattern from ggml_gemv_q4_0_8x8_q8_0_generic.
//
// For each group of 8 columns and each block:
//   For each chunk k (0..1, each 8 elements):
//     For each column j (0..7):
//       sumi = 0
//       For each element i (0..7):
//         Extract 5-bit value from qs (nibble + qh bit)
//         sumi += q5_value * q8_0_value
//       sumf[j] += sumi * d_x[j] * d_y[l]
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Disable optimization to avoid LLVM bug
#endif
static void ggml_gemv_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                          const void * GGML_RESTRICT vx,
                                          const void * GGML_RESTRICT vy,
                                          int nr, int nc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_0x8 * b_ptr = (const block_q5_0x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        // Index into interleaved qs
                        const int qs_idx = k * ncols_interleaved * blocklen
                                         + j * blocklen + i;

                        // Low nibble: element at position (k * blocklen + i)
                        // within the Q5_0 block (indices 0..15)
                        const int elem_lo = k * blocklen + i;
                        const int qh_byte_lo = elem_lo / 8;
                        const int qh_bit_lo  = elem_lo % 8;
                        const int xh_0 = (b_ptr[l].qh[j][qh_byte_lo] >> qh_bit_lo) & 1;
                        const int v0 = (int8_t)(((b_ptr[l].qs[qs_idx] & 0x0F) | (xh_0 << 4)) - 16);

                        // High nibble: element at position (k * blocklen + i + qk/2)
                        // within the Q5_0 block (indices 16..31)
                        const int elem_hi = k * blocklen + i + qk / 2;
                        const int qh_byte_hi = elem_hi / 8;
                        const int qh_bit_hi  = elem_hi % 8;
                        const int xh_1 = (b_ptr[l].qh[j][qh_byte_hi] >> qh_bit_hi) & 1;
                        const int v1 = (int8_t)(((b_ptr[l].qs[qs_idx] >> 4) | (xh_1 << 4)) - 16);

                        // Dot product with Q8_0 activation
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i]
                              + v1 * a_ptr[l].qs[k * blocklen + i + qk / 2];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j])
                                    * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_gemv_q5_0_q8_0.inl"

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
static void init_test_block(block_q5_0x8 * q5, block_q8_0 * q8, uint32_t seed) {
    rng_state = seed;

    // Initialize d[8] with positive values
    for (int j = 0; j < 8; j++) {
        q5->d[j] = ggml_fp32_to_fp16(0.1f + rng_float() * 0.9f);
    }

    // Initialize qh[8][4] with random bitmasks
    for (int j = 0; j < 8; j++) {
        for (int b = 0; b < 4; b++) {
            q5->qh[j][b] = rng_uint8();
        }
    }

    // Initialize qs[128] with random nibble values
    for (int i = 0; i < 128; i++) {
        q5->qs[i] = rng_uint8();
    }

    // Initialize Q8_0 block
    q8->d = ggml_fp32_to_fp16(0.5f + rng_float() * 0.5f);
    for (int i = 0; i < QK8_0; i++) {
        q8->qs[i] = rng_int8();
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Disable optimization to avoid LLVM bug in array comparisons
#endif
static int run_test(int n_blocks, uint32_t seed, float rel_tolerance, const char * label) {
    printf("Test %s: n_blocks=%d, seed=%u, rel_tol=%.6e\n",
           label, n_blocks, seed, rel_tolerance);

    const int n = n_blocks * QK8_0;
    const int nc = 8;

    std::vector<block_q5_0x8> q5(n_blocks);
    std::vector<block_q8_0>   q8(n_blocks);
    std::vector<float> scalar_out(nc, 0.0f);
    std::vector<float> rvv_out(nc, 0.0f);

    for (int b = 0; b < n_blocks; b++) {
        init_test_block(&q5[b], &q8[b], seed + b);
    }

    // Run scalar (generic) reference
    ggml_gemv_q5_0_q8_0_generic(n, scalar_out.data(), 0, q5.data(), q8.data(), 1, nc);

#if defined(__riscv_v_intrinsic)
    // Run RVV implementation
    ggml_gemv_q5_0_q8_0_rvv(n, rvv_out.data(), 0, q5.data(), q8.data(), 1, nc);
#else
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    printf("  Scalar: ");
    for (int j = 0; j < nc; j++) printf("%.2f ", scalar_out[j]);
    printf("\n");
    printf("  RVV:    ");
    for (int j = 0; j < nc; j++) printf("%.2f ", rvv_out[j]);
    printf("\n");

    // Compare with relative tolerance
    int failures = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int j = 0; j < nc; j++) {
        float abs_diff = fabsf(scalar_out[j] - rvv_out[j]);
        float magnitude = fabsf(scalar_out[j]) + fabsf(rvv_out[j]);
        float rel_diff = (magnitude > 1e-6f) ? abs_diff / magnitude : abs_diff;

        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        if (rel_diff > rel_tolerance) {
            failures++;
            printf("  FAIL col %d: scalar=%.6f, rvv=%.6f, rel_diff=%.6e\n",
                   j, scalar_out[j], rvv_out[j], rel_diff);
        }
    }

    printf("  Max abs diff: %.6e, max rel diff: %.6e\n", max_abs_diff, max_rel_diff);
    printf("  %s\n\n", failures ? "FAIL" : "PASS");

    return failures;
}

int main() {
    printf("=== ggml_gemv_q5_0_q8_0: RVV-512 vs Scalar correctness test ===\n\n");

    int total_failures = 0;

#if defined(__riscv_v_fixed_vlen)
    printf("VLEN = %d bits\n", __riscv_v_fixed_vlen);
#else
    printf("VLEN detection: runtime (__riscv_vlenb())\n");
#endif
    printf("block_q5_0x8 size: %zu bytes (expected 176)\n\n", sizeof(block_q5_0x8));

    // Test cases with various seeds and block counts
    total_failures += run_test(1,  42,  1e-4, "seed-42-1-block");
    total_failures += run_test(1,  123, 1e-4, "seed-123-1-block");
    total_failures += run_test(1,  456, 1e-4, "seed-456-1-block");
    total_failures += run_test(4,  42,  1e-4, "seed-42-4-blocks");
    total_failures += run_test(8,  42,  1e-4, "seed-42-8-blocks");
    total_failures += run_test(16, 999, 1e-4, "seed-999-16-blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", total_failures);
    return total_failures;
}
