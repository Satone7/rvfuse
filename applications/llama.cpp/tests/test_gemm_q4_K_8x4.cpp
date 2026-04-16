// test_gemm_q4_K_8x4.cpp — correctness test for RVV ggml_gemm_q4_K_8x4_q8_K
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
//       -DGGML_USE_RISCV_V -o test_gemm_q4_K_8x4 test_gemm_q4_K_8x4.cpp -lm
//
// Run:
//   ./test_gemm_q4_K_8x4
//
// Cross-compile (from x86 host):
//   /path/to/riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b \
//       -mabi=lp64d -DGGML_USE_RISCV_V -static -o test_gemm_q4_K_8x4 \
//       test_gemm_q4_K_8x4.cpp -lm

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

#define QK_K 256
#define K_SCALE_SIZE 12

#define GGML_RESTRICT
#define GGML_UNUSED(x) (void)(x)

// FP16 -> FP32 via software conversion (portable, no <arm_fp16.h> needed)
static inline float ggml_half_to_fp32(ggml_half h) {
    union { uint32_t u; float f; } u32f;
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { u32f.u = sign << 31; return u32f.f; }
        // subnormal: normalize
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

#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_fp32(x)

// ---------------------------------------------------------------------------
// Data structures (copied from ggml-cpu/repack.h)
// ---------------------------------------------------------------------------
struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4-bit quants
};

struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(sizeof(block_q4_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4,
              "wrong q4_K block size");
static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t),
              "wrong q8_K block size");

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
// ---------------------------------------------------------------------------
static void ggml_gemm_q4_K_8x4_q8_K_scalar(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    GGML_UNUSED(nb);
    GGML_UNUSED(ncols_interleaved);
    GGML_UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
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
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            int sumi1 = 0, sumi2 = 0, sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
        }
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// File: include/rvv_gemm_q4_K_8x4.inl (included by both this test and the
// production patch to prevent copy divergence)
// ---------------------------------------------------------------------------
#include "../include/rvv_gemm_q4_K_8x4.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static uint8_t  rng_u8()  { return (uint8_t)(rng_next() & 0xFF); }
static int8_t   rng_i8()  { return (int8_t)(rng_next() & 0xFF); }
static uint16_t rng_u16() { return (uint16_t)(rng_next() | (rng_next() << 15)); }
static int16_t  rng_i16() { return (int16_t)(rng_next() | (rng_next() << 15)); }

static float fp32_from_bits(uint32_t u) {
    float f; memcpy(&f, &u, 4); return f;
}

static ggml_half rng_f16() {
    // Generate a reasonable FP16 value: small magnitude, positive or negative
    uint32_t sign = rng_next() & 1;
    uint32_t exp  = (rng_next() % 20) + 15; // 15..34, avoiding subnormals/inf most of the time
    uint32_t frac = rng_next() & 0x3FF;
    return (ggml_half)((sign << 15) | (exp << 10) | frac);
}

static float rng_float_scale() {
    // Generate float in range [0.1, 2.0]
    uint32_t u = 0x3DCCCCCD + (rng_next() & 0x00FFFFF); // ~0.1 .. ~1.1 in float
    float f = fp32_from_bits(u);
    return f > 0 ? f : 0.1f;
}

// ---------------------------------------------------------------------------
// Fill block data with deterministic pseudo-random values
// ---------------------------------------------------------------------------
static void fill_q4_Kx8(block_q4_Kx8 & blk) {
    for (int i = 0; i < 8; i++) blk.d[i] = rng_f16();
    for (int i = 0; i < 8; i++) blk.dmin[i] = rng_f16();
    for (int i = 0; i < 96; i++) blk.scales[i] = rng_u8();
    for (int i = 0; i < 1024; i++) blk.qs[i] = rng_u8();
}

static void fill_q8_Kx4(block_q8_Kx4 & blk) {
    for (int i = 0; i < 4; i++) blk.d[i] = rng_float_scale();
    for (int i = 0; i < QK_K * 4; i++) blk.qs[i] = rng_i8();
    // Compute bsums: sum of each group of 16 consecutive q8 values
    // bsums layout is interleaved (same as the kernel expects)
    // For now, compute correct bsums for the interleaved layout
    memset(blk.bsums, 0, sizeof(blk.bsums));
    // The bsums in the interleaved layout: for row m, sub-block sb,
    // bsum at index (sb*8) + (m*4) - ((sb%2)*6), values bp[0]+bp[1]
    // This is complex, so just compute raw sums per 16-element group
    // and store in a format compatible with how the kernel reads them.
    // The bsums array has QK_K/4 = 64 int16_t entries.
    // Interpretation: for row m, the bsums are at offsets that depend on
    // the interleaved repack layout. We compute them from the actual q8 data.
    for (int m = 0; m < 4; m++) {
        for (int sb = 0; sb < 8; sb++) {
            int16_t * bp = blk.bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
            int sum0 = 0, sum1 = 0;
            for (int i = 0; i < 16; i++)
                sum0 += blk.qs[m * QK_K + sb * 32 + i];
            for (int i = 0; i < 16; i++)
                sum1 += blk.qs[m * QK_K + sb * 32 + 16 + i];
            bp[0] = (int16_t)sum0;
            bp[1] = (int16_t)sum1;
        }
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(int n_blocks, int n_col_groups, int n_row_groups,
                    const char * label) {
    const int n  = n_blocks * QK_K;
    const int nr = n_row_groups * 4;
    const int nc = n_col_groups * 8;
    const size_t bs = nc; // stride = nc (row-major output)

    // Allocate output buffers
    std::vector<float> out_scalar(nr * nc, 0.0f);
    std::vector<float> out_rvv(nr * nc, 0.0f);

    // Allocate input blocks
    std::vector<block_q4_Kx8> q4_blocks(n_blocks * n_col_groups);
    std::vector<block_q8_Kx4> q8_blocks(n_blocks * n_row_groups);

    // Fill with deterministic data
    for (int i = 0; i < (int)q4_blocks.size(); i++) fill_q4_Kx8(q4_blocks[i]);
    for (int i = 0; i < (int)q8_blocks.size(); i++) fill_q8_Kx4(q8_blocks[i]);

    // Run scalar reference
    ggml_gemm_q4_K_8x4_q8_K_scalar(n, out_scalar.data(), bs,
                                     q4_blocks.data(), q8_blocks.data(), nr, nc);

#if defined(__riscv_v_intrinsic)
    // Run RVV
    ggml_gemm_q4_K_8x4_q8_K_rvv(n, out_rvv.data(), bs,
                                   q4_blocks.data(), q8_blocks.data(), nr, nc);
#else
    // On non-RVV platforms, just skip comparison
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    // Compare outputs
    int max_diff_idx = -1;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int n_mismatch = 0;

    for (int i = 0; i < nr * nc; i++) {
        float diff = out_scalar[i] - out_rvv[i];
        float abs_diff = fabsf(diff);
        float ref = fabsf(out_scalar[i]);
        float rel_diff = ref > 1.0f ? abs_diff / ref : abs_diff;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_rel_diff = rel_diff;
            max_diff_idx = i;
        }

        // Tolerance: absolute 0.5 (half-ULP of int32->float at typical magnitudes)
        // or relative 1e-4 for large values
        if (abs_diff > 0.5f && rel_diff > 1e-4f) {
            n_mismatch++;
            if (n_mismatch <= 5) {
                int row = i / nc;
                int col = i % nc;
                printf("  MISMATCH [%s] [%d,%d]: scalar=%.6f rvv=%.6f diff=%.6e\n",
                       label, row, col, out_scalar[i], out_rvv[i], diff);
            }
        }
    }

    if (n_mismatch == 0) {
        printf("  [PASS] %s — %d elements, max_abs_diff=%.2e (at [%d,%d])\n",
               label, nr * nc, max_abs_diff,
               max_diff_idx >= 0 ? max_diff_idx / nc : 0,
               max_diff_idx >= 0 ? max_diff_idx % nc : 0);
        return 0;
    } else {
        printf("  [FAIL] %s — %d/%d mismatches, max_abs_diff=%.2e\n",
               label, n_mismatch, nr * nc, max_abs_diff);
        return 1;
    }
}

int main() {
    printf("=== ggml_gemm_q4_K_8x4_q8_K: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Test 1: minimum viable size (1 block, 1 col group, 1 row group)
    failures += run_test(1, 1, 1, "min-1x1x1");

    // Test 2: typical tile (1 block, 1 col group, 2 row groups)
    failures += run_test(1, 1, 2, "typical-1x1x2");

    // Test 3: multiple blocks (4 blocks, 2 col groups, 2 row groups)
    failures += run_test(4, 2, 2, "multi-4x2x2");

    // Test 4: stress test with many blocks
    failures += run_test(8, 2, 2, "stress-8x2x2");

    // Test 5: single col, many rows
    failures += run_test(2, 1, 4, "tall-2x1x4");

    // Test 6: many cols, single row group
    failures += run_test(2, 4, 1, "wide-2x4x1");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
