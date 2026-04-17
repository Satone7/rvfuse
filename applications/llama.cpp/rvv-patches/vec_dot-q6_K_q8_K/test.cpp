// test.cpp — correctness test for RVV ggml_vec_dot_q6_K_q8_K (VL512)
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-clang++ -std=c++17 -O2 -march=rv64gcv_zvl512b_zfh_zvfh -mabi=lp64d \
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

// Q6_K block structure (6-bit quantization, 256 elements per super-block)
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    ggml_half d;             // super-block scale (2 bytes)
} block_q6_K;

// Q8_K block structure (8-bit quantization for dot products)
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation (from llama.cpp quants.c)
// ---------------------------------------------------------------------------
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
    float   sums[8];
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
#include "rvv_vec_dot_q6_K_q8_K.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static int8_t rng_i8(void) {
    return (int8_t)(rng_next() & 0xFF);
}

static uint8_t rng_u8(void) {
    return (uint8_t)(rng_next() & 0xFF);
}

static float rng_float(void) {
    // Generate float in range [0.5, 2.0]
    return 0.5f + (rng_next() % 1000) / 1000.0f;
}

// Fill a block_q6_K with pseudo-random data
static void fill_block_q6_K(block_q6_K & blk) {
    for (int i = 0; i < QK_K/2; ++i) blk.ql[i] = rng_u8();
    for (int i = 0; i < QK_K/4; ++i) blk.qh[i] = rng_u8() & 0xFF;
    for (int i = 0; i < QK_K/16; ++i) blk.scales[i] = rng_i8();
    blk.d = GGML_CPU_FP32_TO_FP16(rng_float());
}

// Fill a block_q8_K with pseudo-random data
static void fill_block_q8_K(block_q8_K & blk) {
    blk.d = rng_float();
    for (int i = 0; i < QK_K; ++i) blk.qs[i] = rng_i8();
    for (int i = 0; i < QK_K/16; ++i) blk.bsums[i] = (int16_t)(rng_next() & 0xFFFF);
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(int num_blocks, const char * label) {
    const int n = num_blocks * QK_K;

    std::vector<block_q6_K> x(num_blocks);
    std::vector<block_q8_K> y(num_blocks);

    // Fill with deterministic pseudo-random data
    rng_state = 42;  // Reset RNG for reproducibility
    for (int i = 0; i < num_blocks; ++i) {
        fill_block_q6_K(x[i]);
        fill_block_q8_K(y[i]);
    }

    // Compute with scalar reference
    float s_ref = 0.0f;
    rng_state = 42;  // Reset RNG for reproducibility
    for (int i = 0; i < num_blocks; ++i) {
        fill_block_q6_K(x[i]);  // Re-fill to ensure identical data
        fill_block_q8_K(y[i]);
    }
    ggml_vec_dot_q6_K_q8_K_generic(n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);

    // Compute with RVV implementation (if available)
    float s_rvv = 0.0f;
#if defined(__riscv_v_intrinsic)
    ggml_vec_dot_q6_K_q8_K_vl512(n, &s_rvv, 0, x.data(), 0, y.data(), 0, 1);
#else
    GGML_UNUSED(s_rvv);
    printf("  [SKIP] RVV intrinsics not available (scalar-only build)\n");
    return 0;
#endif

    // Compare results
    float diff = fabsf(s_ref - s_rvv);
    float rel_err = (fabsf(s_ref) > 1e-6f) ? diff / fabsf(s_ref) : diff;
    bool pass = rel_err < 1e-5f;

    printf("  %s: ref=%.6f rvv=%.6f diff=%.2e rel=%.2e %s\n",
           label, s_ref, s_rvv, diff, rel_err, pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}

int main() {
    printf("=== ggml_vec_dot_q6_K_q8_K: RVV VL512 vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Test various block counts (powers of 2 and odd numbers)
    failures += run_test(1,    "1-block");
    failures += run_test(2,    "2-blocks");
    failures += run_test(4,    "4-blocks");
    failures += run_test(8,    "8-blocks");
    failures += run_test(16,   "16-blocks");
    failures += run_test(32,   "32-blocks");
    failures += run_test(3,    "3-blocks");
    failures += run_test(7,    "7-blocks");
    failures += run_test(13,   "13-blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
