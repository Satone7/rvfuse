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
// RVV vectorized implementation (copy of the fixed patch)
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)

#include <riscv_vector.h>

static void ggml_gemm_q4_K_8x4_q8_K_rvv(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    GGML_UNUSED(nb);
    GGML_UNUSED(ncols_interleaved);
    GGML_UNUSED(blocklen);

    const size_t vl = __riscv_vsetvl_e32m2(ncols_interleaved);

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    auto decode_scales = [&](const uint8_t * raw, int8_t * out_scales, int16_t * out_mins) {
        uint32_t sm[3];
        memcpy(sm, raw, 12);
        const uint32_t m03 = sm[1] & kmask1;
        const uint32_t m47 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
        uint32_t sc[2];
        sc[0] = sm[0] & kmask1;
        sc[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
        memcpy(out_scales, sc, 8);
        for (int j = 0; j < 4; j++) { int v = (m03 >> (j*8)) & 0xFF; out_mins[j]   = (int16_t)(int8_t)(v > 31 ? v - 64 : v); }
        for (int j = 0; j < 4; j++) { int v = (m47 >> (j*8)) & 0xFF; out_mins[4+j] = (int16_t)(int8_t)(v > 31 ? v - 64 : v); }
    };

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * GGML_RESTRICT q8_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

            vfloat32m2_t acc_f32_0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            for (int b = 0; b < nb; b++) {
                const vfloat32m2_t q4_d = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vle16_v_f16m1((const _Float16 *)q4_ptr[b].d, vl), vl);
                const vfloat32m2_t q4_dmin = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vle16_v_f16m1((const _Float16 *)q4_ptr[b].dmin, vl), vl);

                int16_t bsums_arr[8][4];
                for (int sb = 0; sb < 8; sb++) {
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bp = q8_ptr[b].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        bsums_arr[sb][m] = bp[0] + bp[1];
                    }
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int8_t  scales_lo[8], scales_hi[8];
                    int16_t mins_lo[8],   mins_hi[8];
                    decode_scales(&q4_ptr[b].scales[sb * 24],      scales_lo, mins_lo);
                    decode_scales(&q4_ptr[b].scales[sb * 24 + 12], scales_hi, mins_hi);

                    const vint16m1_t v_sc_lo = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf4(scales_lo, vl), 0, vl);
                    const vint16m1_t v_sc_hi = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf4(scales_hi, vl), 0, vl);

                    vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0, vl);

                    for (int half = 0; half < 2; half++) {
                        vint16m1_t acc_lo_0 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_1 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_2 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_3 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_0 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_1 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_2 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_3 = __riscv_vmv_v_x_i16m1(0, vl);

                        for (int k = half * 4; k < half * 4 + 4; k++) {
                            const int q8_base = sb * 256 + k * 16;
                            for (int i = 0; i < blocklen; i++) {
                                const vuint8mf2_t q4_packed = __riscv_vlse8_v_u8mf2(
                                    (const uint8_t *)&q4_ptr[b].qs[sb * 256 + k * 32 + i],
                                    (ptrdiff_t)4, vl);
                                const vint8mf2_t q4_lo = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vand_vx_u8mf2(q4_packed, 0xF, vl));
                                const vint8mf2_t q4_hi = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vsrl_vx_u8mf2(q4_packed, 4, vl));

                                const int8_t q8v0_lo = q8_ptr[b].qs[q8_base + 0 * 4 + i];
                                const int8_t q8v1_lo = q8_ptr[b].qs[q8_base + 1 * 4 + i];
                                const int8_t q8v2_lo = q8_ptr[b].qs[q8_base + 2 * 4 + i];
                                const int8_t q8v3_lo = q8_ptr[b].qs[q8_base + 3 * 4 + i];
                                const int8_t q8v0_hi = q8_ptr[b].qs[q8_base + 128 + 0 * 4 + i];
                                const int8_t q8v1_hi = q8_ptr[b].qs[q8_base + 128 + 1 * 4 + i];
                                const int8_t q8v2_hi = q8_ptr[b].qs[q8_base + 128 + 2 * 4 + i];
                                const int8_t q8v3_hi = q8_ptr[b].qs[q8_base + 128 + 3 * 4 + i];

                                acc_lo_0 = __riscv_vwmacc_vx_i16m1(acc_lo_0, q8v0_lo, q4_lo, vl);
                                acc_lo_1 = __riscv_vwmacc_vx_i16m1(acc_lo_1, q8v1_lo, q4_lo, vl);
                                acc_lo_2 = __riscv_vwmacc_vx_i16m1(acc_lo_2, q8v2_lo, q4_lo, vl);
                                acc_lo_3 = __riscv_vwmacc_vx_i16m1(acc_lo_3, q8v3_lo, q4_lo, vl);
                                acc_hi_0 = __riscv_vwmacc_vx_i16m1(acc_hi_0, q8v0_hi, q4_hi, vl);
                                acc_hi_1 = __riscv_vwmacc_vx_i16m1(acc_hi_1, q8v1_hi, q4_hi, vl);
                                acc_hi_2 = __riscv_vwmacc_vx_i16m1(acc_hi_2, q8v2_hi, q4_hi, vl);
                                acc_hi_3 = __riscv_vwmacc_vx_i16m1(acc_hi_3, q8v3_hi, q4_hi, vl);
                            }
                        }

                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0, v_sc_lo, acc_lo_0, vl);
                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0, v_sc_hi, acc_hi_0, vl);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1, v_sc_lo, acc_lo_1, vl);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1, v_sc_hi, acc_hi_1, vl);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2, v_sc_lo, acc_lo_2, vl);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2, v_sc_hi, acc_hi_2, vl);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3, v_sc_lo, acc_lo_3, vl);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3, v_sc_hi, acc_hi_3, vl);
                    }

                    const vint16m1_t v_min_lo = __riscv_vle16_v_i16m1(mins_lo, vl);
                    const vint16m1_t v_min_hi = __riscv_vle16_v_i16m1(mins_hi, vl);

                    {
                        const vfloat32m2_t sbd_scale = __riscv_vfmul_vf_f32m2(q4_d, q8_ptr[b].d[0], vl);
                        const vfloat32m2_t sbd_min   = __riscv_vfmul_vf_f32m2(q4_dmin, q8_ptr[b].d[0], vl);
                        acc_f32_0 = __riscv_vfmacc_vv_f32m2(acc_f32_0,
                            __riscv_vfcvt_f_x_v_f32m2(sumi_0, vl), sbd_scale, vl);
                        vint32m2_t bias_inc = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl),
                                bsums_arr[sb * 2][0], v_min_lo, vl),
                            bsums_arr[sb * 2 + 1][0], v_min_hi, vl);
                        acc_f32_0 = __riscv_vfnmsac_vv_f32m2(acc_f32_0,
                            __riscv_vfcvt_f_x_v_f32m2(bias_inc, vl), sbd_min, vl);
                    }
                    {
                        const vfloat32m2_t sbd_scale = __riscv_vfmul_vf_f32m2(q4_d, q8_ptr[b].d[1], vl);
                        const vfloat32m2_t sbd_min   = __riscv_vfmul_vf_f32m2(q4_dmin, q8_ptr[b].d[1], vl);
                        acc_f32_1 = __riscv_vfmacc_vv_f32m2(acc_f32_1,
                            __riscv_vfcvt_f_x_v_f32m2(sumi_1, vl), sbd_scale, vl);
                        vint32m2_t bias_inc = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl),
                                bsums_arr[sb * 2][1], v_min_lo, vl),
                            bsums_arr[sb * 2 + 1][1], v_min_hi, vl);
                        acc_f32_1 = __riscv_vfnmsac_vv_f32m2(acc_f32_1,
                            __riscv_vfcvt_f_x_v_f32m2(bias_inc, vl), sbd_min, vl);
                    }
                    {
                        const vfloat32m2_t sbd_scale = __riscv_vfmul_vf_f32m2(q4_d, q8_ptr[b].d[2], vl);
                        const vfloat32m2_t sbd_min   = __riscv_vfmul_vf_f32m2(q4_dmin, q8_ptr[b].d[2], vl);
                        acc_f32_2 = __riscv_vfmacc_vv_f32m2(acc_f32_2,
                            __riscv_vfcvt_f_x_v_f32m2(sumi_2, vl), sbd_scale, vl);
                        vint32m2_t bias_inc = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl),
                                bsums_arr[sb * 2][2], v_min_lo, vl),
                            bsums_arr[sb * 2 + 1][2], v_min_hi, vl);
                        acc_f32_2 = __riscv_vfnmsac_vv_f32m2(acc_f32_2,
                            __riscv_vfcvt_f_x_v_f32m2(bias_inc, vl), sbd_min, vl);
                    }
                    {
                        const vfloat32m2_t sbd_scale = __riscv_vfmul_vf_f32m2(q4_d, q8_ptr[b].d[3], vl);
                        const vfloat32m2_t sbd_min   = __riscv_vfmul_vf_f32m2(q4_dmin, q8_ptr[b].d[3], vl);
                        acc_f32_3 = __riscv_vfmacc_vv_f32m2(acc_f32_3,
                            __riscv_vfcvt_f_x_v_f32m2(sumi_3, vl), sbd_scale, vl);
                        vint32m2_t bias_inc = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl),
                                bsums_arr[sb * 2][3], v_min_lo, vl),
                            bsums_arr[sb * 2 + 1][3], v_min_hi, vl);
                        acc_f32_3 = __riscv_vfnmsac_vv_f32m2(acc_f32_3,
                            __riscv_vfcvt_f_x_v_f32m2(bias_inc, vl), sbd_min, vl);
                    }
                }
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * ncols_interleaved, acc_f32_0, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * ncols_interleaved, acc_f32_1, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * ncols_interleaved, acc_f32_2, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * ncols_interleaved, acc_f32_3, vl);
        }
    }
}
#endif // __riscv_v_intrinsic

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
