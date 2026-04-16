// test_gemm_q4_K_8x4.cpp — correctness test for RVV ggml_gemm_q4_K_8x4_q8_K
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// VLEN variants:
//   - VLEN >= 512: dual-tile 4x16 output (ncols_interleaved=16)
//   - VLEN < 512:  single-tile 4x8 output (ncols_interleaved=8)
//
// Build (rv64gcv, VLEN=512):
//   riscv64-linux-gnu-g++ -std=c++17 -O2 -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DGGML_USE_RISCV_V -o test_gemm_q4_K_8x4 test_gemm_q4_K_8x4.cpp -lm

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

#define QK_K 256
#define K_SCALE_SIZE 12

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

#define GGML_CPU_FP16_TO_FP32(x) ggml_half_to_fp32(x)

// ---------------------------------------------------------------------------
// Data structures (copied from ggml-cpu/repack.h)
// ---------------------------------------------------------------------------
struct block_q4_Kx8 {
    ggml_half d[8];
    ggml_half dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
};

struct block_q8_Kx4 {
    float d[4];
    int8_t qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
};

static_assert(sizeof(block_q4_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4,
              "wrong q4_K block size");
static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t),
              "wrong q8_K block size");

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation (always 8-column interleaved)
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_gemm_q4_K_8x4_q8_K_scalar(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;  // Scalar reference: fixed at 8
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

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

static float fp32_from_bits(uint32_t u) {
    float f; memcpy(&f, &u, 4); return f;
}

static ggml_half rng_f16() {
    uint32_t sign = rng_next() & 1;
    uint32_t exp  = (rng_next() % 29) + 1;
    uint32_t frac = rng_next() & 0x3FF;
    return (ggml_half)((sign << 15) | (exp << 10) | frac);
}

static float rng_float_scale() {
    uint32_t u = 0x3DCCCCCD + (rng_next() & 0x00FFFFF);
    float f = fp32_from_bits(u);
    return f > 0 ? f : 0.1f;
}

// ---------------------------------------------------------------------------
// Fill block data with deterministic pseudo-random values
// ---------------------------------------------------------------------------
static void fill_q4_kx8(block_q4_Kx8 & blk) {
    for (int i = 0; i < 8; i++) blk.d[i] = rng_f16();
    for (int i = 0; i < 8; i++) blk.dmin[i] = rng_f16();
    for (int i = 0; i < 96; i++) blk.scales[i] = rng_u8();
    for (int i = 0; i < 1024; i++) blk.qs[i] = rng_u8();
}

static void fill_q8_kx4(block_q8_Kx4 & blk) {
    for (int i = 0; i < 4; i++) blk.d[i] = rng_float_scale();
    for (int i = 0; i < QK_K * 4; i++) blk.qs[i] = rng_i8();
    memset(blk.bsums, 0, sizeof(blk.bsums));
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
// Test runner (adapts nc for VLEN variant)
// ---------------------------------------------------------------------------
static int run_test(int n_blocks, int n_col_groups_8, int n_row_groups,
                    const char * label) {
    // Determine nc based on VLEN variant
    // VLEN >= 512: ncols_interleaved = 16 (processes 2 blocks per x-iteration)
    // VLEN < 512:  ncols_interleaved = 8
#if defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 512
    const int ncols_interleaved = 16;
    const int n_col_groups = n_col_groups_8 / 2;  // Each RVV iteration processes 2 scalar groups
#else
    const int ncols_interleaved = 8;
    const int n_col_groups = n_col_groups_8;
#endif

    const int n  = n_blocks * QK_K;
    const int nr = n_row_groups * 4;
    const int nc = n_col_groups_8 * 8;  // Scalar reference always uses 8-column groups
    const size_t bs = nc;

    std::vector<float> out_scalar(nr * nc, 0.0f);
    std::vector<float> out_rvv(nr * nc, 0.0f);

    // Scalar reference: n_col_groups_8 blocks of 8 columns each
    std::vector<block_q4_Kx8> q4_blocks(n_blocks * n_col_groups_8);
    std::vector<block_q8_Kx4> q8_blocks(n_blocks * n_row_groups);

    for (int i = 0; i < (int)q4_blocks.size(); i++) fill_q4_kx8(q4_blocks[i]);
    for (int i = 0; i < (int)q8_blocks.size(); i++) fill_q8_kx4(q8_blocks[i]);

    // Run scalar reference (always 8-column interleaved)
    ggml_gemm_q4_K_8x4_q8_K_scalar(n, out_scalar.data(), bs,
                                   q4_blocks.data(), q8_blocks.data(), nr, nc);

#if defined(__riscv_v_intrinsic)
    // Run RVV (ncols_interleaved depends on VLEN)
    ggml_gemm_q4_K_8x4_q8_K_rvv(n, out_rvv.data(), bs,
                                 q4_blocks.data(), q8_blocks.data(), nr, nc);
#else
    printf("  [SKIP] %s — RVV intrinsics not available\n", label);
    return 0;
#endif

    // Compare outputs
    int max_diff_idx = -1;
    float max_abs_diff = 0.0f;
    int n_mismatch = 0;

    for (int i = 0; i < nr * nc; i++) {
        float diff = out_scalar[i] - out_rvv[i];
        float abs_diff = fabsf(diff);
        float ref = fabsf(out_scalar[i]);
        float rel_diff = ref > 1.0f ? abs_diff / ref : abs_diff;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_diff_idx = i;
        }

        if ((abs_diff > 0.5f && rel_diff > 1e-4f) || isnan(out_scalar[i]) || isnan(out_rvv[i])) {
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
        printf("  [PASS] %s — %d elements, max_abs_diff=%.2e (at [%d,%d]) [VLEN=%d]\n",
               label, nr * nc, max_abs_diff,
               max_diff_idx >= 0 ? max_diff_idx / nc : 0,
               max_diff_idx >= 0 ? max_diff_idx % nc : 0,
#if defined(__riscv_v_fixed_vlen)
               __riscv_v_fixed_vlen
#else
               -1
#endif
               );
        return 0;
    } else {
        printf("  [FAIL] %s — %d/%d mismatches, max_abs_diff=%.2e\n",
               label, n_mismatch, nr * nc, max_abs_diff);
        return 1;
    }
}

int main() {
#if defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 512
    printf("=== ggml_gemm_q4_K_8x4_q8_K: RVV-512 (4x16) vs Scalar correctness test ===\n\n");
#else
    printf("=== ggml_gemm_q4_K_8x4_q8_K: RVV-256 (4x8) vs Scalar correctness test ===\n\n");
#endif

    int failures = 0;

    // Test 1: minimum viable (nc must be multiple of 16 for VLEN=512)
    // Use n_col_groups_8=2 to give nc=16
    failures += run_test(1, 2, 1, "min-1x2x1");

    // Test 2: typical (nc=16 for VLEN=512)
    failures += run_test(1, 2, 2, "typical-1x2x2");

    // Test 3: multiple blocks (nc=32)
    failures += run_test(4, 4, 2, "multi-4x4x2");

    // Test 4: stress (nc=32)
    failures += run_test(8, 4, 2, "stress-8x4x2");

    // Test 5: tall (nc=16)
    failures += run_test(2, 2, 4, "tall-2x2x4");

    // Test 6: wide (nc=64)
    failures += run_test(2, 8, 1, "wide-2x8x1");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}