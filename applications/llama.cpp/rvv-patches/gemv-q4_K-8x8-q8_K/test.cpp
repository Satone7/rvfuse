// test.cpp — correctness test for RVV ggml_gemv_q4_K_8x8_q8_K
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -o test test.cpp -lm
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
// 6-bit scales/mins encoder for Q4_Kx8 layout
// ---------------------------------------------------------------------------
static inline void encode_q_Kx8_6bit_scales(const uint8_t * scales_in,  // 8 values (0-63)
                                            const uint8_t * mins_in,    // 8 values (0-63)
                                            uint8_t * packed_out) {    // 12 bytes output
    uint32_t sm[3] = {0};

    // sm[0]: scales 0-3 in bits 0-5, scales 4-7 upper 2 bits in bits 6-7
    for (int i = 0; i < 4; i++) {
        ((uint8_t*)&sm[0])[i] = (scales_in[i] & 0x3F) | ((scales_in[4+i] >> 4) << 6);
    }

    // sm[1]: mins 0-3 in bits 0-5, mins 4-7 upper 2 bits in bits 6-7
    for (int i = 0; i < 4; i++) {
        ((uint8_t*)&sm[1])[i] = (mins_in[i] & 0x3F) | ((mins_in[4+i] >> 4) << 6);
    }

    // sm[2]: scales 4-7 lower 4 bits in bits 0-3, mins 4-7 lower 4 bits in bits 4-7
    for (int i = 0; i < 4; i++) {
        ((uint8_t*)&sm[2])[i] = (scales_in[4+i] & 0x0F) | ((mins_in[4+i] & 0x0F) << 4);
    }

    memcpy(packed_out, sm, 12);
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

    // Generate raw scales and mins (6-bit values) for each subblock
    // Layout: 4 subblocks × 24 bytes each = 96 bytes
    // Each subblock: 12 bytes for low nibble scales/mins + 12 bytes for high nibble
    for (int sb = 0; sb < 4; sb++) {
        uint8_t raw_scales_lo[8], raw_mins_lo[8];
        uint8_t raw_scales_hi[8], raw_mins_hi[8];
        for (int i = 0; i < 8; i++) {
            raw_scales_lo[i] = rng_uint6();
            raw_mins_lo[i] = rng_uint6();
            raw_scales_hi[i] = rng_uint6();
            raw_mins_hi[i] = rng_uint6();
        }
        encode_q_Kx8_6bit_scales(raw_scales_lo, raw_mins_lo, q4->scales + sb * 24);
        encode_q_Kx8_6bit_scales(raw_scales_hi, raw_mins_hi, q4->scales + sb * 24 + 12);
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