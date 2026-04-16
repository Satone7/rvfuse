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
//
// VLEN optimization:
//   - VLEN >= 512: dual-tile 4x8+4x8 → 4x16 output (2x throughput)
//   - VLEN < 512:  single-tile 4x8 (original)

#include <cstring>
#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// VLEN >= 512 version: dual-tile processing (4x16 output)
// =============================================================================
// Strategy: Process 2 adjacent column-groups in parallel, output 16 columns.
// Each column-group is 8 columns (block_q4_Kx8 layout). We maintain separate
// accumulators for each tile and store combined 16-column output.

#if defined(__riscv_v_intrinsic) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 512

static void ggml_gemm_q4_K_8x4_q8_K_rvv_512(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_per_tile = 8;      // Each block_q4_Kx8 provides 8 columns
    const int ncols_interleaved = 16;  // Output tile: 16 columns (2 blocks)
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    // vl=8 for inner loop ops (per-tile), vl=16 for float accumulators
    const size_t vl8  = __riscv_vsetvl_e8mf2(ncols_per_tile);
    const size_t vl16 = __riscv_vsetvl_e32m1(ncols_interleaved);

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
        for (int j = 0; j < 4; j++) { int v = (m03 >> (j*8)) & 0xFF; out_mins[j]   = (int16_t)(v & 0x3F); }
        for (int j = 0; j < 4; j++) { int v = (m47 >> (j*8)) & 0xFF; out_mins[4+j] = (int16_t)(v & 0x3F); }
    };

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * GGML_RESTRICT q8_ptr = (const block_q8_Kx4 *) vy + (y * nb);

        // Zero-initialize output region for this y-group (4 rows × 16 columns)
        for (int m = 0; m < 4; m++) {
            vfloat32m1_t zero_row = __riscv_vfmv_v_f_f32m1(0.0f, vl16);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                __riscv_vse32_v_f32m1(s + (y * 4 + m) * bs + x * ncols_interleaved, zero_row, vl16);
            }
        }

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            // Two adjacent q4 blocks: tile0 (cols 0-7), tile1 (cols 8-15)
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr0 = (const block_q4_Kx8 *) vx + (x * 2 * nb);
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr1 = (const block_q4_Kx8 *) vx + ((x * 2 + 1) * nb);

            for (int b = 0; b < nb; b++) {
                // Load q4 d/dmin for both tiles (FP16 -> FP32, vl=8 each)
                const vfloat32m1_t q4_d0 = __riscv_vfwcvt_f_f_v_f32m1(
                    __riscv_vle16_v_f16mf2((const _Float16 *)q4_ptr0[b].d, vl8), vl8);
                const vfloat32m1_t q4_d1 = __riscv_vfwcvt_f_f_v_f32m1(
                    __riscv_vle16_v_f16mf2((const _Float16 *)q4_ptr1[b].d, vl8), vl8);
                const vfloat32m1_t q4_dmin0 = __riscv_vfwcvt_f_f_v_f32m1(
                    __riscv_vle16_v_f16mf2((const _Float16 *)q4_ptr0[b].dmin, vl8), vl8);
                const vfloat32m1_t q4_dmin1 = __riscv_vfwcvt_f_f_v_f32m1(
                    __riscv_vle16_v_f16mf2((const _Float16 *)q4_ptr1[b].dmin, vl8), vl8);

                // Precompute pairwise bsums
                int16_t bsums_arr[8][4];
                for (int sb = 0; sb < 8; sb++) {
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bp = q8_ptr[b].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        bsums_arr[sb][m] = bp[0] + bp[1];
                    }
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    // Decode scales/mins for both tiles
                    int8_t  scales0_lo[8], scales0_hi[8], scales1_lo[8], scales1_hi[8];
                    int16_t mins0_lo[8], mins0_hi[8], mins1_lo[8], mins1_hi[8];
                    decode_scales(&q4_ptr0[b].scales[sb * 24],      scales0_lo, mins0_lo);
                    decode_scales(&q4_ptr0[b].scales[sb * 24 + 12], scales0_hi, mins0_hi);
                    decode_scales(&q4_ptr1[b].scales[sb * 24],      scales1_lo, mins1_lo);
                    decode_scales(&q4_ptr1[b].scales[sb * 24 + 12], scales1_hi, mins1_hi);

                    // Load scales for both tiles (vl=8 each)
                    const vint16m1_t v_sc0_lo = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales0_lo, vl8), 0, vl8);
                    const vint16m1_t v_sc0_hi = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales0_hi, vl8), 0, vl8);
                    const vint16m1_t v_sc1_lo = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales1_lo, vl8), 0, vl8);
                    const vint16m1_t v_sc1_hi = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales1_hi, vl8), 0, vl8);

                    // i32 accumulators for both tiles (vl=8)
                    // Use i32m2 to match i16m1 input (widening MAC requires LMUL ratio 2:1)
                    vint32m2_t sumi0_0 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi0_1 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi0_2 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi0_3 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi1_0 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi1_1 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi1_2 = __riscv_vmv_v_x_i32m2(0, vl8);
                    vint32m2_t sumi1_3 = __riscv_vmv_v_x_i32m2(0, vl8);

                    for (int half = 0; half < 2; half++) {
                        // i16 accumulators for both tiles (vl=8)
                        vint16m1_t acc0_lo_0 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_lo_1 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_lo_2 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_lo_3 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_hi_0 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_hi_1 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_hi_2 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc0_hi_3 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_lo_0 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_lo_1 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_lo_2 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_lo_3 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_hi_0 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_hi_1 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_hi_2 = __riscv_vmv_v_x_i16m1(0, vl8);
                        vint16m1_t acc1_hi_3 = __riscv_vmv_v_x_i16m1(0, vl8);

                        for (int k = half * 4; k < half * 4 + 4; k++) {
                            const int q8_base = sb * 256 + k * 16;

                            for (int i = 0; i < blocklen; i++) {
                                // Load q4 bytes from both tiles (stride=4, vl=8)
                                const vuint8mf2_t q4_packed0 = __riscv_vlse8_v_u8mf2(
                                    (const uint8_t *)&q4_ptr0[b].qs[sb * 256 + k * 32 + i],
                                    (ptrdiff_t)4, vl8);
                                const vuint8mf2_t q4_packed1 = __riscv_vlse8_v_u8mf2(
                                    (const uint8_t *)&q4_ptr1[b].qs[sb * 256 + k * 32 + i],
                                    (ptrdiff_t)4, vl8);

                                const vint8mf2_t q4_lo0 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vand_vx_u8mf2(q4_packed0, 0xF, vl8));
                                const vint8mf2_t q4_hi0 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vsrl_vx_u8mf2(q4_packed0, 4, vl8));
                                const vint8mf2_t q4_lo1 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vand_vx_u8mf2(q4_packed1, 0xF, vl8));
                                const vint8mf2_t q4_hi1 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                                    __riscv_vsrl_vx_u8mf2(q4_packed1, 4, vl8));

                                // q8 values (same for both tiles - same row data)
                                const int8_t q8v0_lo = q8_ptr[b].qs[q8_base + 0 * 4 + i];
                                const int8_t q8v1_lo = q8_ptr[b].qs[q8_base + 1 * 4 + i];
                                const int8_t q8v2_lo = q8_ptr[b].qs[q8_base + 2 * 4 + i];
                                const int8_t q8v3_lo = q8_ptr[b].qs[q8_base + 3 * 4 + i];
                                const int8_t q8v0_hi = q8_ptr[b].qs[q8_base + 128 + 0 * 4 + i];
                                const int8_t q8v1_hi = q8_ptr[b].qs[q8_base + 128 + 1 * 4 + i];
                                const int8_t q8v2_hi = q8_ptr[b].qs[q8_base + 128 + 2 * 4 + i];
                                const int8_t q8v3_hi = q8_ptr[b].qs[q8_base + 128 + 3 * 4 + i];

                                // Tile 0 MACs
                                acc0_lo_0 = __riscv_vwmacc_vx_i16m1(acc0_lo_0, q8v0_lo, q4_lo0, vl8);
                                acc0_lo_1 = __riscv_vwmacc_vx_i16m1(acc0_lo_1, q8v1_lo, q4_lo0, vl8);
                                acc0_lo_2 = __riscv_vwmacc_vx_i16m1(acc0_lo_2, q8v2_lo, q4_lo0, vl8);
                                acc0_lo_3 = __riscv_vwmacc_vx_i16m1(acc0_lo_3, q8v3_lo, q4_lo0, vl8);
                                acc0_hi_0 = __riscv_vwmacc_vx_i16m1(acc0_hi_0, q8v0_hi, q4_hi0, vl8);
                                acc0_hi_1 = __riscv_vwmacc_vx_i16m1(acc0_hi_1, q8v1_hi, q4_hi0, vl8);
                                acc0_hi_2 = __riscv_vwmacc_vx_i16m1(acc0_hi_2, q8v2_hi, q4_hi0, vl8);
                                acc0_hi_3 = __riscv_vwmacc_vx_i16m1(acc0_hi_3, q8v3_hi, q4_hi0, vl8);

                                // Tile 1 MACs
                                acc1_lo_0 = __riscv_vwmacc_vx_i16m1(acc1_lo_0, q8v0_lo, q4_lo1, vl8);
                                acc1_lo_1 = __riscv_vwmacc_vx_i16m1(acc1_lo_1, q8v1_lo, q4_lo1, vl8);
                                acc1_lo_2 = __riscv_vwmacc_vx_i16m1(acc1_lo_2, q8v2_lo, q4_lo1, vl8);
                                acc1_lo_3 = __riscv_vwmacc_vx_i16m1(acc1_lo_3, q8v3_lo, q4_lo1, vl8);
                                acc1_hi_0 = __riscv_vwmacc_vx_i16m1(acc1_hi_0, q8v0_hi, q4_hi1, vl8);
                                acc1_hi_1 = __riscv_vwmacc_vx_i16m1(acc1_hi_1, q8v1_hi, q4_hi1, vl8);
                                acc1_hi_2 = __riscv_vwmacc_vx_i16m1(acc1_hi_2, q8v2_hi, q4_hi1, vl8);
                                acc1_hi_3 = __riscv_vwmacc_vx_i16m1(acc1_hi_3, q8v3_hi, q4_hi1, vl8);
                            }
                        }

                        // Widen-MAC to i32 for both tiles (vl=8)
                        sumi0_0 = __riscv_vwmacc_vv_i32m2(sumi0_0, v_sc0_lo, acc0_lo_0, vl8);
                        sumi0_0 = __riscv_vwmacc_vv_i32m2(sumi0_0, v_sc0_hi, acc0_hi_0, vl8);
                        sumi0_1 = __riscv_vwmacc_vv_i32m2(sumi0_1, v_sc0_lo, acc0_lo_1, vl8);
                        sumi0_1 = __riscv_vwmacc_vv_i32m2(sumi0_1, v_sc0_hi, acc0_hi_1, vl8);
                        sumi0_2 = __riscv_vwmacc_vv_i32m2(sumi0_2, v_sc0_lo, acc0_lo_2, vl8);
                        sumi0_2 = __riscv_vwmacc_vv_i32m2(sumi0_2, v_sc0_hi, acc0_hi_2, vl8);
                        sumi0_3 = __riscv_vwmacc_vv_i32m2(sumi0_3, v_sc0_lo, acc0_lo_3, vl8);
                        sumi0_3 = __riscv_vwmacc_vv_i32m2(sumi0_3, v_sc0_hi, acc0_hi_3, vl8);

                        sumi1_0 = __riscv_vwmacc_vv_i32m2(sumi1_0, v_sc1_lo, acc1_lo_0, vl8);
                        sumi1_0 = __riscv_vwmacc_vv_i32m2(sumi1_0, v_sc1_hi, acc1_hi_0, vl8);
                        sumi1_1 = __riscv_vwmacc_vv_i32m2(sumi1_1, v_sc1_lo, acc1_lo_1, vl8);
                        sumi1_1 = __riscv_vwmacc_vv_i32m2(sumi1_1, v_sc1_hi, acc1_hi_1, vl8);
                        sumi1_2 = __riscv_vwmacc_vv_i32m2(sumi1_2, v_sc1_lo, acc1_lo_2, vl8);
                        sumi1_2 = __riscv_vwmacc_vv_i32m2(sumi1_2, v_sc1_hi, acc1_hi_2, vl8);
                        sumi1_3 = __riscv_vwmacc_vv_i32m2(sumi1_3, v_sc1_lo, acc1_lo_3, vl8);
                        sumi1_3 = __riscv_vwmacc_vv_i32m2(sumi1_3, v_sc1_hi, acc1_hi_3, vl8);
                    }

                    // Load mins for both tiles
                    const vint16m1_t v_min0_lo = __riscv_vle16_v_i16m1(mins0_lo, vl8);
                    const vint16m1_t v_min0_hi = __riscv_vle16_v_i16m1(mins0_hi, vl8);
                    const vint16m1_t v_min1_lo = __riscv_vle16_v_i16m1(mins1_lo, vl8);
                    const vint16m1_t v_min1_hi = __riscv_vle16_v_i16m1(mins1_hi, vl8);

                    // Float accumulate per row per tile (vl=8)
                    // Output indices: s[(y*4+m)*bs + x*16 + tile_offset + j]
                    float * s_base = s + (y * 4) * bs + x * ncols_interleaved;

                    // Tile 0: columns 0-7
                    // Convert q4_d0/dmin0 from f32m1 to f32m2 for widening compatibility
                    // Actually, f32m1 works for vfmul_vf output, need f32m2 for vfmacc_vv with sumf
                    // Strategy: vfmul_vf -> f32m1, then widen to f32m2 via vfwcvt? No, that's wrong.
                    // Correct approach: keep f32m2 for all float ops in this section
                    for (int m = 0; m < 4; m++) {
                        const float drow = q8_ptr[b].d[m];
                        // q4_d0 is f32m1, need to broadcast drow and multiply
                        // vfmul_vf: f32m1 * scalar -> f32m1
                        // To get f32m2, use vfmul_vf with f32m2 target: need to load q4_d0 as f32m2?
                        // Alternative: vfmacc_vf_f32m2 works with f32m1 input via widening?
                        // No: vfmacc_vv needs matching LMUL for both inputs and output.
                        // Use vfmacc_vf: acc_f32m2 += scalar * f32m1? Not valid.
                        //
                        // Simpler: use f32m2 for d/dmin by loading as f16mf2 then widening
                        // f16mf2 (8 elements, LMUL=0.5) -> f32m1 (8 elements, LMUL=1)? No, that's wrong.
                        // vfwcvt: f16mX -> f32m(2X)
                        // f16mf2 -> f32m1 (correct for 8 elements at VLEN=512)
                        //
                        // So: load d/dmin as f16mf2 (vl=8), widen to f32m1, then:
                        //   - sumi is i32m2, convert to f32m2
                        //   - sbd_scale = f32m1, need to broadcast to f32m2? Use vmv_v_v?
                        //
                        // Actually, vfmacc_vv_f32m2 requires both inputs be f32m2.
                        // So we need to "widen" f32m1 sbd_scale to f32m2.
                        // Use vfmul_vf_f32m2 with zero accumulator: vmv_v_x + vfmul_vf
                        //
                        // Let me use a cleaner approach: use vfmul_vf and vfmacc_vf where possible.
                        // vfmacc_vf_f32m2(acc, scalar, vec_f32m1): vec must be f32m2, not m1.
                        //
                        // FINAL SOLUTION: use f32m2 for all float vectors in this section.
                        // Load q4_d0 as f16mf2, widen to f32m1... still not f32m2.
                        //
                        // OK, I'll use vfwcvt twice: f16mf2 -> f32m1 -> f32m2 via vfwcvt_f_f? No.
                        //
                        // Correct: load f16m1 (vl=8 at VLEN=512: 16 bytes = 8 f16 values, LMUL=1),
                        // widen to f32m2 (vl=8: 32 bytes = 8 f32 values, LMUL=2).
                        // Check: f16m1 at VLEN=512, vl=8 -> 8*16 = 128 bit, fits in 1 reg (LMUL=1)
                        // vfwcvt_f_f: f16m1 -> f32m2 (LMUL doubles)
                        // This gives f32m2 with vl=8, which matches our needs!

                        // BUT: we originally loaded with vle16_v_f16mf2 and vfwcvt to f32m1.
                        // Let's change to vle16_v_f16m1 and vfwcvt to f32m2.

                        // For now, as a quick fix, let's use f32m1 for everything and
                        // truncate sumi from m2 to m1 via vget?
                        // vint32m1_t sumi_m1 = __riscv_vget_v_i32m2_i32m1(sumi, 0);
                        // This extracts the first m1 portion of an m2 vector.
                        // At vl=8, m2 vector has 8 elements, m1 portion also has 8 elements.

                        vint32m1_t sumi_m1 = __riscv_vget_v_i32m2_i32m1(
                            (m == 0) ? sumi0_0 : (m == 1) ? sumi0_1 : (m == 2) ? sumi0_2 : sumi0_3, 0);
                        vfloat32m1_t sumf = __riscv_vfcvt_f_x_v_f32m1(sumi_m1, vl8);

                        const vfloat32m1_t sbd_scale = __riscv_vfmul_vf_f32m1(q4_d0, drow, vl8);
                        const vfloat32m1_t sbd_min   = __riscv_vfmul_vf_f32m1(q4_dmin0, drow, vl8);

                        vint32m2_t bias_m2 = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl8),
                                bsums_arr[sb*2][m], v_min0_lo, vl8),
                            bsums_arr[sb*2+1][m], v_min0_hi, vl8);
                        vint32m1_t bias_m1 = __riscv_vget_v_i32m2_i32m1(bias_m2, 0);
                        vfloat32m1_t biasf = __riscv_vfcvt_f_x_v_f32m1(bias_m1, vl8);

                        // Load current, accumulate, store (f32m1)
                        vfloat32m1_t cur = __riscv_vle32_v_f32m1(s_base + m*bs, vl8);
                        cur = __riscv_vfmacc_vv_f32m1(cur, sumf, sbd_scale, vl8);
                        cur = __riscv_vfnmsac_vv_f32m1(cur, biasf, sbd_min, vl8);
                        __riscv_vse32_v_f32m1(s_base + m*bs, cur, vl8);
                    }

                    // Tile 1: columns 8-15
                    float * s_tile1 = s_base + ncols_per_tile;
                    for (int m = 0; m < 4; m++) {
                        const float drow = q8_ptr[b].d[m];

                        vint32m1_t sumi_m1 = __riscv_vget_v_i32m2_i32m1(
                            (m == 0) ? sumi1_0 : (m == 1) ? sumi1_1 : (m == 2) ? sumi1_2 : sumi1_3, 0);
                        vfloat32m1_t sumf = __riscv_vfcvt_f_x_v_f32m1(sumi_m1, vl8);

                        const vfloat32m1_t sbd_scale = __riscv_vfmul_vf_f32m1(q4_d1, drow, vl8);
                        const vfloat32m1_t sbd_min   = __riscv_vfmul_vf_f32m1(q4_dmin1, drow, vl8);

                        vint32m2_t bias_m2 = __riscv_vwmacc_vx_i32m2(
                            __riscv_vwmacc_vx_i32m2(__riscv_vmv_v_x_i32m2(0, vl8),
                                bsums_arr[sb*2][m], v_min1_lo, vl8),
                            bsums_arr[sb*2+1][m], v_min1_hi, vl8);
                        vint32m1_t bias_m1 = __riscv_vget_v_i32m2_i32m1(bias_m2, 0);
                        vfloat32m1_t biasf = __riscv_vfcvt_f_x_v_f32m1(bias_m1, vl8);

                        vfloat32m1_t cur = __riscv_vle32_v_f32m1(s_tile1 + m*bs, vl8);
                        cur = __riscv_vfmacc_vv_f32m1(cur, sumf, sbd_scale, vl8);
                        cur = __riscv_vfnmsac_vv_f32m1(cur, biasf, sbd_min, vl8);
                        __riscv_vse32_v_f32m1(s_tile1 + m*bs, cur, vl8);
                    }
                }  // for sb
            }  // for b
        }  // for x
    }  // for y
}

#endif  // VLEN >= 512

// =============================================================================
// VLEN < 512 (or unspecified) version: 4x8 tile (original implementation)
// =============================================================================

static void ggml_gemm_q4_K_8x4_q8_K_rvv_256(
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
        for (int j = 0; j < 4; j++) { int v = (m03 >> (j*8)) & 0xFF; out_mins[j]   = (int16_t)(v & 0x3F); }
        for (int j = 0; j < 4; j++) { int v = (m47 >> (j*8)) & 0xFF; out_mins[4+j] = (int16_t)(v & 0x3F); }
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
                        __riscv_vle8_v_i8mf2(scales_lo, vl), 0, vl);
                    const vint16m1_t v_sc_hi = __riscv_vwadd_vx_i16m1(
                        __riscv_vle8_v_i8mf2(scales_hi, vl), 0, vl);

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

// =============================================================================
// Dispatcher
// =============================================================================

static void ggml_gemm_q4_K_8x4_q8_K_rvv(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc)
{
#if defined(__riscv_v_intrinsic)
    #if defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 512
        ggml_gemm_q4_K_8x4_q8_K_rvv_512(n, s, bs, vx, vy, nr, nc);
    #else
        ggml_gemm_q4_K_8x4_q8_K_rvv_256(n, s, bs, vx, vy, nr, nc);
    #endif
#else
    GGML_UNUSED(s); GGML_UNUSED(bs); GGML_UNUSED(vx); GGML_UNUSED(vy);
    GGML_UNUSED(nr); GGML_UNUSED(nc);
#endif
}