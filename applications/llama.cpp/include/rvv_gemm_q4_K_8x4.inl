// rvv_gemm_q4_K_8x4.inl — RVV implementation of ggml_gemm_q4_K_8x4_q8_K
//
// Single source of truth for the RVV vectorized kernel. Included by:
//   - arch/riscv/repack.cpp  (via patch, production build)
//   - tests/test_gemm_q4_K_8x4.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK_K, block_q4_Kx8, block_q8_Kx4 must be defined
//   - GGML_RESTRICT must be defined
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead

#include <cstring>
#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

static void ggml_gemm_q4_K_8x4_q8_K_rvv(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    GGML_UNUSED(nb);
    GGML_UNUSED(ncols_interleaved);
    GGML_UNUSED(blocklen);

#if defined(__riscv_v_intrinsic)
    const size_t vl = __riscv_vsetvl_e32m2(ncols_interleaved);

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    // Helper: decode 12 raw bytes into 8 int8_t scales and 8 int16_t mins (sign-extended 6-bit).
    // Same algorithm as ARM NEON decode_q_Kx8_6bit_scales.
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

            // 4 rows x 8 columns float accumulators (named — Clang rejects RVV arrays in some contexts)
            vfloat32m2_t acc_f32_0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t acc_f32_3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            for (int b = 0; b < nb; b++) {
                // Load q4 d[0..7] (FP16 -> FP32) and dmin[0..7]
                const vfloat32m2_t q4_d = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vle16_v_f16m1((const _Float16 *)q4_ptr[b].d, vl), vl);
                const vfloat32m2_t q4_dmin = __riscv_vfwcvt_f_f_v_f32m2(
                    __riscv_vle16_v_f16m1((const _Float16 *)q4_ptr[b].dmin, vl), vl);

                // Precompute pairwise bsums (interleaved layout, same as generic code)
                int16_t bsums_arr[8][4]; // [sub_block][row]
                for (int sb = 0; sb < 8; sb++) {
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bp = q8_ptr[b].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        bsums_arr[sb][m] = bp[0] + bp[1];
                    }
                }

                // Process 4 sub-block groups (each covers 64 elements = 2 sub-blocks)
                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int8_t  scales_lo[8], scales_hi[8];
                    int16_t mins_lo[8],   mins_hi[8];
                    decode_scales(&q4_ptr[b].scales[sb * 24],      scales_lo, mins_lo);
                    decode_scales(&q4_ptr[b].scales[sb * 24 + 12], scales_hi, mins_hi);

                    // Load scales: 8 int8_t -> widen to 8 int16_t (sign-extend each byte)
                    // i8mf2 because vwadd_vx_i16m1 expects i8mf2 input (LMUL=0.5 for e8)
                    const vint16m1_t v_sc_lo = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales_lo, vl), 0, vl);
                    const vint16m1_t v_sc_hi = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales_hi, vl), 0, vl);

                    // i32 scale accumulators (persist across both inner-loop halves)
                    vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0, vl);
                    vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0, vl);

                    // Split inner loop into 2 halves to avoid i16 accumulator overflow.
                    // Each half does 4 k-iterations x 4 blocklen = 16 i8*i8 widen-MACs.
                    // Max per half: 16 * 127 * 15 = 30240 < 32767 (i16 max). Safe.
                    for (int half = 0; half < 2; half++) {
                        vint16m1_t acc_lo_0 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_1 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_2 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_lo_3 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_0 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_1 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_2 = __riscv_vmv_v_x_i16m1(0, vl);
                        vint16m1_t acc_hi_3 = __riscv_vmv_v_x_i16m1(0, vl);

                        // 4 reads per half
                        for (int k = half * 4; k < half * 4 + 4; k++) {
                            const int q8_base = sb * 256 + k * 16;

                            for (int i = 0; i < blocklen; i++) {
                                // Load 8 q4 bytes at stride 4 (8 columns, position i)
                                const vuint8mf2_t q4_packed = __riscv_vlse8_v_u8mf2(
                                    (const uint8_t *)&q4_ptr[b].qs[sb * 256 + k * 32 + i],
                                    (ptrdiff_t)4, vl);

                                // Extract low/high nibbles
                                const vint8mf2_t q4_lo = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vand_vx_u8mf2(q4_packed, 0xF, vl));
                                const vint8mf2_t q4_hi = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vsrl_vx_u8mf2(q4_packed, 4, vl));

                                // lo nibble -> q8 WITHOUT +128, hi nibble -> q8 WITH +128
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

                        // Widen-MAC: sumi += (i32)scale * (i32)acc, i16*i16 -> i32
                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0, v_sc_lo, acc_lo_0, vl);
                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0, v_sc_hi, acc_hi_0, vl);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1, v_sc_lo, acc_lo_1, vl);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1, v_sc_hi, acc_hi_1, vl);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2, v_sc_lo, acc_lo_2, vl);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2, v_sc_hi, acc_hi_2, vl);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3, v_sc_lo, acc_lo_3, vl);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3, v_sc_hi, acc_hi_3, vl);
                    }  // for half

                    // Load mins as i16 vectors (int16_t[8] = 16 bytes, correct for vl=8)
                    const vint16m1_t v_min_lo = __riscv_vle16_v_i16m1(mins_lo, vl);
                    const vint16m1_t v_min_hi = __riscv_vle16_v_i16m1(mins_hi, vl);

                    // Float-accumulate and bias-correct per row (row-by-row to limit register pressure).
                    // sbd_scale and sbd_min are computed on-the-fly per row to save register space.
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
                }  // for sb
            }  // for b

            // Store 4 rows x 8 columns
            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * ncols_interleaved, acc_f32_0, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * ncols_interleaved, acc_f32_1, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * ncols_interleaved, acc_f32_2, vl);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * ncols_interleaved, acc_f32_3, vl);
        }  // for x
    }  // for y
    return;
#endif  // defined(__riscv_v_intrinsic)
    // Non-RVV: function is unused, caller should fall through to generic
    GGML_UNUSED(s); GGML_UNUSED(bs); GGML_UNUSED(vx); GGML_UNUSED(vy);
    GGML_UNUSED(nr); GGML_UNUSED(nc);
}
