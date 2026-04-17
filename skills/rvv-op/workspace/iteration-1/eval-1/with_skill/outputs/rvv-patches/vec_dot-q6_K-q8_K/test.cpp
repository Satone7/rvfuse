// test.cpp — correctness test for RVV ggml_vec_dot_q6_K_q8_K
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//       --sysroot=/path/to/sysroot \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
//       -fuse-ld=lld test.cpp -o test_vec_dot_q6_K_q8_K -lm
//
// Run:
//   qemu-riscv64 -L /path/to/sysroot ./test_vec_dot_q6_K_q8_K
//
// Scalar build (for reference comparison on x86):
//   g++ -std=c++17 -O2 test.cpp -o test_vec_dot_q6_K_q8_K_scalar -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>

// =============================================================================
// Minimal ggml type definitions (copied from ggml-common.h)
// =============================================================================

#ifndef GGML_COMMON_DECL
#define GGML_COMMON_DECL

#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

static inline float ggml_half_to_float(uint16_t h) {
    // IEEE 754 half-precision to float
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    float result;
    if (exp == 0) {
        if (frac == 0) {
            result = 0.0f;
        } else {
            // subnormal
            result = ldexpf((float)frac / 1024.0f, -14);
        }
    } else if (exp == 31) {
        result = (frac == 0) ? INFINITY : NAN;
    } else {
        result = ldexpf(1.0f + (float)frac / 1024.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

static inline uint16_t ggml_float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp  = (bits >> 23) & 0xff;
    uint32_t frac = bits & 0x7fffff;
    uint16_t result;
    if (exp == 0) {
        result = 0;
    } else if (exp == 255) {
        result = 0x7c00 | (frac ? 0x0200 : 0);
    } else {
        int new_exp = (int)exp - 127 + 15;
        if (new_exp >= 31) { result = 0x7c00; }
        else if (new_exp <= 0) {
            // subnormal
            frac |= 0x800000;
            int shift = 14 - new_exp;
            result = (uint16_t)(frac >> shift);
        } else {
            result = (uint16_t)((new_exp << 10) | (frac >> 13));
        }
    }
    return (uint16_t)(result | (sign << 15));
}

#define GGML_FP16_TO_FP32(x) ggml_half_to_float(x)
#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_float(x)

#endif // GGML_COMMON_DECL

#ifndef GGML_RESTRICT
#define GGML_RESTRICT __restrict
#endif

#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)
#endif

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

// =============================================================================
// Data structures from ggml-common.h
// =============================================================================

// block_q6_K: 6-bit quantization, 256 elements per super-block
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, 8-bit (16 bytes)
    ggml_half d;             // super-block scale (2 bytes)
} block_q6_K;

// block_q8_K: intermediate 8-bit quantization for dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants (256 bytes)
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16 (32 bytes)
} block_q8_K;

// Compile-time size assertions
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4,
              "wrong q6_K block size");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "wrong q8_K block size");

// =============================================================================
// Scalar (generic) reference implementation
// =============================================================================
// Extracted from ggml/src/ggml-cpu/quants.c ggml_vec_dot_q6_K_q8_K_generic
// This is the reference implementation from llama.cpp.

#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                             const void * GGML_RESTRICT vx, size_t bx,
                                             const void * GGML_RESTRICT vy, size_t by,
                                             int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

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

// =============================================================================
// RVV vectorized implementation — single source of truth
// =============================================================================
#include "rvv_vec_dot_q6_K_q8_K.inl"

// =============================================================================
// Pseudo-random data generator (LCG, deterministic)
// =============================================================================

static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static uint8_t rng_u8() {
    return (uint8_t)(rng_next() & 0xFF);
}

static int8_t rng_i8() {
    return (int8_t)(rng_next() & 0xFF);
}

static float rng_f32() {
    // Generate float in range [-1.0, 1.0]
    return ((float)(rng_next() & 0xFFFF) / 32768.0f) - 1.0f;
}

static ggml_half rng_f16() {
    float f = rng_f32();
    return ggml_float_to_half(f);
}

// =============================================================================
// Test data generation
// =============================================================================

static void generate_test_data(int n_blocks,
                                std::vector<block_q6_K>& x_blocks,
                                std::vector<block_q8_K>& y_blocks) {
    rng_state = 42; // Reset for reproducibility

    x_blocks.resize(n_blocks);
    y_blocks.resize(n_blocks);

    for (int i = 0; i < n_blocks; ++i) {
        // Fill block_q6_K
        // ql: lower 4 bits of 6-bit quants (packed 2 per byte)
        for (int j = 0; j < QK_K / 2; ++j) {
            // Each byte has two nibbles: [3:0] = elem(2j), [7:4] = elem(2j+1)
            uint8_t lo = rng_u8() & 0x0F;
            uint8_t hi = rng_u8() & 0x0F;
            x_blocks[i].ql[j] = (hi << 4) | lo;
        }

        // qh: upper 2 bits of 6-bit quants (packed 4 per byte)
        for (int j = 0; j < QK_K / 4; ++j) {
            uint8_t b0 = rng_u8() & 0x03;
            uint8_t b1 = rng_u8() & 0x03;
            uint8_t b2 = rng_u8() & 0x03;
            uint8_t b3 = rng_u8() & 0x03;
            x_blocks[i].qh[j] = (b3 << 6) | (b2 << 4) | (b1 << 2) | b0;
        }

        // scales: 8-bit per-block scales
        for (int j = 0; j < QK_K / 16; ++j) {
            x_blocks[i].scales[j] = rng_i8();
        }

        // d: FP16 super-block scale (avoid zero)
        float d;
        do { d = rng_f32(); } while (d == 0.0f);
        x_blocks[i].d = ggml_float_to_half(d);

        // Fill block_q8_K
        // d: FP32 delta (avoid zero)
        do { y_blocks[i].d = rng_f32(); } while (y_blocks[i].d == 0.0f);

        // qs: int8 quantized values
        for (int j = 0; j < QK_K; ++j) {
            y_blocks[i].qs[j] = rng_i8();
        }

        // bsums: int16 block sums (sum of qs in groups of 16)
        for (int j = 0; j < QK_K / 16; ++j) {
            int16_t sum = 0;
            for (int k = 0; k < 16; ++k) {
                sum += y_blocks[i].qs[j * 16 + k];
            }
            y_blocks[i].bsums[j] = sum;
        }
    }
}

// =============================================================================
// Test runner
// =============================================================================

static int run_test(int n_blocks, const char * label) {
    std::vector<block_q6_K> x_blocks;
    std::vector<block_q8_K> y_blocks;
    generate_test_data(n_blocks, x_blocks, y_blocks);

    int n = n_blocks * QK_K;
    float result_scalar = 0.0f;
    float result_rvv = 0.0f;

    // Run scalar reference
    memset(&result_scalar, 0, sizeof(float));
    ggml_vec_dot_q6_K_q8_K_generic(n, &result_scalar, sizeof(float),
                                     x_blocks.data(), sizeof(block_q6_K),
                                     y_blocks.data(), sizeof(block_q8_K),
                                     1);

    // Run RVV implementation
#if defined(__riscv_v_intrinsic)
    memset(&result_rvv, 0, sizeof(float));
    ggml_vec_dot_q6_K_q8_K_rvv(n, &result_rvv, sizeof(float),
                                 x_blocks.data(), sizeof(block_q6_K),
                                 y_blocks.data(), sizeof(block_q8_K),
                                 1);
#else
    result_rvv = result_scalar; // No RVV available, trivially pass
#endif

    // Compare results
    // Tolerance: int8 quantization can have accumulation differences
    // For Q6_K × Q8_K with n blocks, the accumulation involves ~n*256 multiply-adds.
    // Relative tolerance of 1e-4 should be sufficient for FP32 accumulation.
    float abs_diff = fabsf(result_scalar - result_rvv);
    float rel_diff = (fabsf(result_scalar) > 1e-10f)
                     ? abs_diff / fabsf(result_scalar)
                     : abs_diff;
    bool pass = (rel_diff < 1e-4f);

    printf("  [%s] n_blocks=%d, n=%d: scalar=%.6f, rvv=%.6f, rel_diff=%.2e %s\n",
           label, n_blocks, n, result_scalar, result_rvv, rel_diff,
           pass ? "PASS" : "FAIL");

    if (!pass) {
        printf("    FAIL: abs_diff=%.6f exceeds tolerance\n", abs_diff);
    }

    return pass ? 0 : 1;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== ggml_vec_dot_q6_K_q8_K: RVV vs Scalar correctness test ===\n");
    printf("    QK_K=%d, sizeof(block_q6_K)=%zu, sizeof(block_q8_K)=%zu\n",
           QK_K, sizeof(block_q6_K), sizeof(block_q8_K));

#if defined(__riscv_v_intrinsic)
    printf("    RVV: enabled (VLEN=%d)\n", __riscv_vlenb() * 8);
#else
    printf("    RVV: not available (scalar-only test)\n");
#endif

    printf("\n");

    int failures = 0;

    // Test 1: Single block (minimum size)
    failures += run_test(1, "single-block");

    // Test 2: Two blocks (covers QK_K/128=2 inner iterations × 2 blocks)
    failures += run_test(2, "two-blocks");

    // Test 3: Four blocks (typical small workload)
    failures += run_test(4, "four-blocks");

    // Test 4: Eight blocks (medium workload)
    failures += run_test(8, "eight-blocks");

    // Test 5: Sixteen blocks (larger workload for accumulation stress)
    failures += run_test(16, "sixteen-blocks");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
