// rvv_gemv_q5_0_q8_0.inl — RVV implementation of Q5_0 x Q8_0 GEMV (8x8 interleaved)
//
// Computes: s[col] = sum_l( d_x[col] * d_y[l] * sum_k( dequant_q5_0(x, col, k) * q8_0_y[k] ) )
// for 8 interleaved columns across multiple blocks.
//
// Data layout (block_q5_0x8):
//   - d[8]:              8 half-precision scale factors (one per column)
//   - qh[32]:            5th-bit arrays for 8 columns (4 bytes per column)
//   - qs[128]:           nibble-packed quants (16 bytes per column, interleaved)
//     qs layout: [col0_blk0 | col1_blk0 | ... | col7_blk0 | col0_blk1 | ... ]
//     Within each column's 16 bytes: [nib_lo_e0..e15 | nib_hi_e0..e15]
//
// Data layout (block_q8_0):
//   - d:                 half-precision scale factor
//   - qs[32]:            int8 quantized values
//
// Algorithm (based on ARM NEON vec_dot_q5_0_q8_0 + RVV gemv_q4_0_8x8_q8_0):
//   Per block l, per chunk k (0..1), per column group (0..3 pairs):
//   1. Load 32 bytes of interleaved nibbles (low nibbles of 4 columns)
//   2. Load 32 bytes of interleaved nibbles (another offset for high nibbles)
//   3. Split nibbles: AND with 0x0F (low) / SHR by 4 (high)
//   4. Apply qh bitmask: masked subtract 0x10 where qh bit = 0
//   5. Widening multiply with Q8_0 activations
//   6. Horizontal reduction of i16 accumulator -> 4 scalar int32 sums
//   7. Scale by d_x[j] * d_y[l] and accumulate to float
//
// VLEN target: 512-bit (vlenb = 64)
//   - 32 bytes of qs fits in mf2 (half-length vector)
//   - 32 int8 elements fits in m1
//   - Can process 4 columns of 8 elements simultaneously
//   - Two passes: low nibbles (elements 0..15) + high nibbles (elements 16..31)
//
// Based on:
//   - ARM NEON: ggml_vec_dot_q5_0_q8_0 (arch/arm/quants.c)
//   - x86 AVX2: ggml_vec_dot_q5_0_q8_0 (arch/x86/quants.c)
//   - RISC-V RVV: ggml_vec_dot_q5_0_q8_0_rvv (arch/riscv/rvv_vec_dot_q5_0_q8_0.inl)
//   - RISC-V RVV: ggml_gemv_q4_0_8x8_q8_0 (arch/riscv/repack.cpp)
//
// Prerequisites (satisfied by repack.cpp or test.cpp):
//   - QK5_0, QK8_0 must be defined (both 32)
//   - block_q5_0x8, block_q8_0 structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED, GGML_CPU_FP16_TO_FP32 must be defined
//   - assert() must be available
//
// This file provides only the RVV-optimized function (_rvv suffix).
// The caller is responsible for selecting RVV vs generic fallback.

#include <cassert>
#include <cstring>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation — 512-bit target (vlenb >= 32)
// =============================================================================
#if defined(__riscv_v_intrinsic)
static void ggml_gemv_q5_0_q8_0_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                      const void * GGML_RESTRICT vx,
                                      const void * GGML_RESTRICT vy,
                                      int nr, int nc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const size_t vlenb = __riscv_vlenb();

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    GGML_UNUSED(nr);
    GGML_UNUSED(bs);

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;

    if (vlenb >= 32) {
        // =====================================================================
        // VLEN >= 256 (optimized path, target: 512-bit vlenb=64)
        //
        // Strategy: Process 8 columns per outer iteration, using the
        // interleaved qs layout to load 32 bytes (4 columns x 8 elements)
        // at a time via m1 or mf2 vector registers.
        //
        // Per block l:
        //   For chunk k in [0, 1]:
        //     Load a_ptr[l].qs[k*8 .. k*8+7]   -> a_lo (low half of q8_0)
        //     Load a_ptr[l].qs[k*8+16 .. k*8+23] -> a_hi (high half of q8_0)
        //
        //     For column pair groups cp in [0, 1, 2, 3]:
        //       Load 4 x 8 bytes from b_ptr[l].qs at interleaved offsets
        //       Extract nibbles, apply qh, compute dot product
        //
        //   Multiply sumi[j] by d_x[j] * d_y[l], accumulate to sumf[j]
        // =====================================================================

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_0x8 * b_ptr = (const block_q5_0x8 *) vx + (x * nb);

            // Float accumulators for 8 output columns
            float sumf[8] = {0.0f};

            for (int l = 0; l < nb; l++) {
                const float a_scale = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);

                // ----------------------------------------------------------
                // Process 2 chunks of 8 elements each
                // chunk 0: elements 0-7 (low nibbles) + elements 16-23 (high nibbles)
                // chunk 1: elements 8-15 (low nibbles) + elements 24-31 (high nibbles)
                // ----------------------------------------------------------
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    // Q8_0 activation: 8 elements from low half and 8 from high half
                    // a_lo: a_ptr[l].qs[k*8 .. k*8+7]
                    // a_hi: a_ptr[l].qs[k*8+16 .. k*8+23]
                    const size_t vl_8 = 8;

                    // Load Q8_0 low-half activations (8 elements)
                    const vint8m1_t a_lo = __riscv_vle8_v_i8m1(
                        &a_ptr[l].qs[k * blocklen], vl_8);
                    // Load Q8_0 high-half activations (8 elements)
                    const vint8m1_t a_hi = __riscv_vle8_v_i8m1(
                        &a_ptr[l].qs[k * blocklen + qk / 2], vl_8);

                    // ------------------------------------------------------
                    // Process 2 column-pair groups, 4 columns per group
                    // Group 0: columns 0,1,2,3
                    // Group 1: columns 4,5,6,7
                    // ------------------------------------------------------
                    for (int cp = 0; cp < 2; cp++) {
                        const int col_base = cp * 4;

                        // Load 32 bytes of interleaved qs for 4 columns
                        // Layout: [col0_bytes | col1_bytes | col2_bytes | col3_bytes]
                        // Each column contributes 8 bytes (blocklen=8)
                        // Offset: k * ncols_interleaved * blocklen + col_base * blocklen
                        const int qs_offset = k * ncols_interleaved * blocklen
                                            + col_base * blocklen;

                        const vuint8m1_t raw = __riscv_vle8_v_u8m1(
                            &b_ptr[l].qs[qs_offset], vl_8 * 4);

                        // Split into lower nibbles [0..7] per column
                        const vint8m1_t nib_lo = __riscv_vreinterpret_v_u8m1_i8m1(
                            __riscv_vand_vx_u8m1(raw, 0x0F, vl_8 * 4));
                        // Split into upper nibbles [16..23] per column
                        const vint8m1_t nib_hi = __riscv_vreinterpret_v_u8m1_i8m1(
                            __riscv_vsrl_vx_u8m1(raw, 4, vl_8 * 4));

                        // --------------------------------------------------
                        // Apply qh bitmask for sign extension
                        // For each column j (0..3):
                        //   qh[j*4 + byte_idx] bit (k*8+i) controls element i
                        //   If bit = 1: value has 5th bit set (already >= 16)
                        //   If bit = 0: subtract 0x10 to center around 0
                        //
                        // Implementation: load qh bytes, create mask,
                        // masked-subtract 0x10 where qh bit = 0
                        // --------------------------------------------------

                        // For nib_lo (elements k*8+0 .. k*8+7 for each column):
                        // Element index within block = k*blocklen + i (i=0..7)
                        // qh byte index = (k*blocklen + i) / 8 = k (since blocklen=8)
                        // qh bit index  = (k*blocklen + i) % 8 = i
                        // So qh[j*4 + k] bit i controls the 5th bit
                        uint8_t qh_bytes_lo[4];
                        uint8_t qh_bytes_hi[4];
                        for (int j = 0; j < 4; j++) {
                            // Low nibble elements: indices k*8+0 .. k*8+7
                            qh_bytes_lo[j] = b_ptr[l].qh[col_base + j].qh[k];
                            // High nibble elements: indices k*8+16 .. k*8+23
                            qh_bytes_hi[j] = b_ptr[l].qh[col_base + j].qh[(k * 8 + 16) / 8];
                        }

                        // Load qh as bitmask and invert: we want mask=1 where qh=0
                        // (those elements need subtract 0x10)
                        // Load 4 bytes as b8 mask (32 bits for 32 elements)
                        vbool8_t qh_mask_lo = __riscv_vlm_v_b8(qh_bytes_lo, vl_8 * 4);
                        qh_mask_lo = __riscv_vmnand_mm_b8(qh_mask_lo, qh_mask_lo, vl_8 * 4);

                        vbool8_t qh_mask_hi = __riscv_vlm_v_b8(qh_bytes_hi, vl_8 * 4);
                        qh_mask_hi = __riscv_vmnand_mm_b8(qh_mask_hi, qh_mask_hi, vl_8 * 4);

                        // Apply sign extension: subtract 0x10 where qh bit was 0
                        vint8m1_t v0_lo = __riscv_vsub_vx_i8m1_mu(
                            qh_mask_lo, nib_lo, nib_lo, 0x10, vl_8 * 4);
                        vint8m1_t v0_hi = __riscv_vsub_vx_i8m1_mu(
                            qh_mask_hi, nib_hi, nib_hi, 0x10, vl_8 * 4);

                        // --------------------------------------------------
                        // Dot product with Q8_0 activations
                        // We have 32 int8 values (4 columns x 8 elements)
                        // and 8 int8 activation values.
                        // Need to replicate activations across 4 columns.
                        //
                        // Strategy: widening multiply + horizontal reduction
                        // Per-column dot: sum(v0_lo[j*8..j*8+7] * a_lo) +
                        //                 sum(v0_hi[j*8..j*8+7] * a_hi)
                        // --------------------------------------------------

                        // Replicate a_lo to 32 elements (4x repetition for 4 columns)
                        vint8m1_t a_lo_rep = __riscv_vmv_v_x_i8m1(0, vl_8 * 4);
                        a_lo_rep = __riscv_vslideup_vx_i8m1(a_lo_rep, a_lo, 0, vl_8);
                        a_lo_rep = __riscv_vslideup_vx_i8m1(a_lo_rep, a_lo, 8, vl_8 * 2);
                        a_lo_rep = __riscv_vslideup_vx_i8m1(a_lo_rep, a_lo, 16, vl_8 * 4);

                        vint8m1_t a_hi_rep = __riscv_vmv_v_x_i8m1(0, vl_8 * 4);
                        a_hi_rep = __riscv_vslideup_vx_i8m1(a_hi_rep, a_hi, 0, vl_8);
                        a_hi_rep = __riscv_vslideup_vx_i8m1(a_hi_rep, a_hi, 8, vl_8 * 2);
                        a_hi_rep = __riscv_vslideup_vx_i8m1(a_hi_rep, a_hi, 16, vl_8 * 4);

                        // Widening multiply: i8 * i8 -> i16
                        vint16m2_t mul_lo = __riscv_vwmul_vv_i16m2(v0_lo, a_lo_rep, vl_8 * 4);
                        vint16m2_t mul_hi = __riscv_vwmul_vv_i16m2(v0_hi, a_hi_rep, vl_8 * 4);

                        // Add: mul_total = mul_lo + mul_hi
                        vint16m2_t mul_total = __riscv_vadd_vv_i16m2(mul_lo, mul_hi, vl_8 * 4);

                        // Horizontal reduction: i16 -> i32
                        // We need 4 separate sums, one per column group of 8 elements
                        // Use narrowing shift to pack adjacent pairs, then widen-add
                        // Approach: reinterpret i16m2 as pairs, shift-and-add
                        //
                        // mul_total has 32 i16 values:
                        //   [col0_e0, col0_e1, ..., col0_e7, col1_e0, ..., col3_e7]
                        //
                        // Step 1: reduce each group of 8 i16 -> i32
                        //   Use vredsum to get a single i32, but we need 4 separate
                        //   Approach: vnsrl to narrow i16->i32 with shift
                        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl_8);

                        // Reduction for each column: extract and sum 8 i16 values
                        // Use vget to extract groups, then vwredsum
                        vint16m1_t col0 = __riscv_vget_v_i16m2_i16m1(mul_total, 0);
                        vint16m1_t col1 = __riscv_vget_v_i16m2_i16m1(mul_total, 1);
                        vint16m1_t col2 = __riscv_vget_v_i16m2_i16m1(mul_total, 2);
                        vint16m1_t col3 = __riscv_vget_v_i16m2_i16m1(mul_total, 3);

                        int32_t sumi_0 = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vwredsum_vs_i16m1_i32m1(col0, zero, vl_8));
                        int32_t sumi_1 = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vwredsum_vs_i16m1_i32m1(col1, zero, vl_8));
                        int32_t sumi_2 = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vwredsum_vs_i16m1_i32m1(col2, zero, vl_8));
                        int32_t sumi_3 = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vwredsum_vs_i16m1_i32m1(col3, zero, vl_8));

                        // Scale and accumulate
                        sumf[col_base + 0] += (float)sumi_0
                            * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[col_base + 0]) * a_scale;
                        sumf[col_base + 1] += (float)sumi_1
                            * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[col_base + 1]) * a_scale;
                        sumf[col_base + 2] += (float)sumi_2
                            * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[col_base + 2]) * a_scale;
                        sumf[col_base + 3] += (float)sumi_3
                            * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[col_base + 3]) * a_scale;
                    }
                }
            }

            for (int j = 0; j < ncols_interleaved; j++) {
                s[x * ncols_interleaved + j] = sumf[j];
            }
        }
    } else {
        // =====================================================================
        // VLEN < 256: fall back to generic scalar implementation
        // =====================================================================
        ggml_gemv_q5_0_q8_0_generic(n, s, bs, vx, vy, nr, nc);
    }
}
#endif // __riscv_v_intrinsic
