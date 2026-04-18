// test.cpp — correctness test for RVV ggml_gemv_q4_K_8x8_q8_K
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
#include <math.h>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal ggml type definitions (from ggml-common.h)
// ---------------------------------------------------------------------------
#define QK_K 256
#define K_SCALE_SIZE 12
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

// Q8_K block structure
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

// Q4_Kx8 interleaved block structure
struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4-bit quants (256 × 4 bits × 8 columns)
};

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation (from llama.cpp repack.cpp)
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Disable optimization to avoid LLVM bug
#endif
static void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                            const void * GGML_RESTRICT vx,
                                            const void * GGML_RESTRICT vy,
                                            int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1, sumi2, sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_gemv_q4_K_8x8_q8_K.inl"

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

static uint8_t rng_uint6() {
    return rng_next() % 64;
}

static float rng_float() {
    return (float)rng_next() / 32768.0f;
}

// ---------------------------------------------------------------------------
// Test data initialization
// ---------------------------------------------------------------------------
static void init_test_block(block_q4_Kx8 * q4, block_q8_K * q8, uint32_t seed) {
    rng_state = seed;

    // Initialize d[8] and dmin[8] with positive values
    for (int j = 0; j < 8; j++) {
        q4->d[j] = ggml_fp32_to_fp16(0.1f + rng_float() * 0.9f);
        q4->dmin[j] = ggml_fp32_to_fp16(0.01f + rng_float() * 0.09f);
    }

    // Initialize scales[96] with random 6-bit packed values
    for (int i = 0; i < 96; i++) {
        q4->scales[i] = rng_uint8();
    }

    // Initialize qs[1024] with random 4-bit values (packed as bytes)
    for (int i = 0; i < 1024; i++) {
        q4->qs[i] = rng_uint8();
    }

    // Initialize Q8_K block
    q8->d = 0.5f + rng_float() * 0.5f;

    for (int i = 0; i < QK_K; i++) {
        q8->qs[i] = rng_int8();
    }

    // Compute bsums: sum of quants in groups of 16
    for (int i = 0; i < QK_K / 16; i++) {
        int16_t sum = 0;
        for (int j = 0; j < 16; j++) {
            sum += q8->qs[i * 16 + j];
        }
        q8->bsums[i] = sum;
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

    std::vector<block_q4_Kx8> q4(n_blocks);
    std::vector<block_q8_K> q8(n_blocks);
    float scalar_out[8] = {0};
    float rvv_out[8] = {0};

    for (int b = 0; b < n_blocks; b++) {
        init_test_block(&q4[b], &q8[b], seed + b);
    }

    // Run scalar (generic) reference
    float temp_scalar[8] = {0};
    for (int b = 0; b < n_blocks; b++) {
        ggml_gemv_q4_K_8x8_q8_K_generic(QK_K, temp_scalar, 0, &q4[b], &q8[b], 1, 8);
        for (int j = 0; j < 8; j++) scalar_out[j] += temp_scalar[j];
    }

#if defined(__riscv_v_intrinsic)
    // Run RVV implementation
    float temp_rvv[8] = {0};
    for (int b = 0; b < n_blocks; b++) {
        ggml_gemv_q4_K_8x8_q8_K_rvv(QK_K, temp_rvv, 0, &q4[b], &q8[b], 1, 8);
        for (int j = 0; j < 8; j++) rvv_out[j] += temp_rvv[j];
    }
#else
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    printf("  Scalar: ");
    for (int j = 0; j < 8; j++) printf("%.2f ", scalar_out[j]);
    printf("\n");
    printf("  RVV:    ");
    for (int j = 0; j < 8; j++) printf("%.2f ", rvv_out[j]);
    printf("\n");

    // Compare with relative tolerance
    int failures = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int j = 0; j < 8; j++) {
        float abs_diff = fabs(scalar_out[j] - rvv_out[j]);
        float magnitude = fabs(scalar_out[j]) + fabs(rvv_out[j]);
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
    printf("=== ggml_gemv_q4_K_8x8_q8_K: RVV-512 vs Scalar correctness test ===\n\n");

    int total_failures = 0;

#if defined(__riscv_v_fixed_vlen)
    printf("VLEN = %d bits\n", __riscv_v_fixed_vlen);
#else
    printf("VLEN detection: runtime (__riscv_vlenb())\n");
#endif
    printf("\n");

    // Test cases with various seeds and block counts
    total_failures += run_test(1, 42, 1e-4, "seed-42-1-block");
    total_failures += run_test(1, 123, 1e-4, "seed-123-1-block");
    total_failures += run_test(1, 456, 1e-4, "seed-456-1-block");
    total_failures += run_test(4, 42, 1e-4, "seed-42-4-blocks");
    total_failures += run_test(8, 42, 1e-4, "seed-42-8-blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", total_failures);
    return total_failures;
}