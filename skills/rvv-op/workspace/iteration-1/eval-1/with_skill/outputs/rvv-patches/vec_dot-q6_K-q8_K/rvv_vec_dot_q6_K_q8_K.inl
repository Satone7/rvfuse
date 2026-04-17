// rvv_vec_dot_q6_K_q8_K.inl — RVV implementation of ggml_vec_dot_q6_K_q8_K (VLEN=512)
//
// Single source of truth. Included by:
//   - ggml/src/ggml-cpu/arch/riscv/quants.c  (via patch, production build)
//   - test.cpp                                (correctness test)
//
// Prerequisites before including:
//   - QK_K (256) must be defined
//   - block_q6_K and block_q8_K structures must be defined
//   - GGML_RESTRICT, UNUSED must be defined
//   - GGML_CPU_FP16_TO_FP32 must be defined
//   - ggml_vec_dot_q6_K_q8_K_generic must be declared
//   - On RVV: <riscv_vector.h> must be included
//
// VLEN=512 Strategy:
//   VLEN=512 gives us 64 bytes per vector register. The Q6_K super-block
//   processes 128 elements per inner iteration (QK_K/128 = 2 iterations).
//   With VLEN=512:
//     - 64 bytes = 64 uint8 elements (enough for qh[32] in e8m1)
//     - Can process the full q6[64] (two 32-byte halves) and qh[32] together
//     - Can process all 128 q8 elements in a single load (e8m2 with vl=128)
//     - Can compute all 8 scale products in parallel
//
// This fills the gap: the existing RISC-V implementation in arch/riscv/quants.c
// has VLEN=256 and VLEN=128 paths but NO VLEN=512 path, causing it to fall
// back to the generic scalar implementation for VLEN=512 hardware.

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation: ggml_vec_dot_q6_K_q8_K (VLEN=512)
// =============================================================================
// Algorithm (matching x86 AVX2 and ARM NEON strategies):
//
// 1. For each super-block (QK_K=256 elements):
//    a. Compute d = x[i].d * y[i].d (block-level scale)
//    b. For each half-block (QK_K/128 = 2 iterations):
//       - Load qh (32 bytes): upper 2 bits for 128 elements
//       - Load q6 low (64 bytes): lower 4 bits for 128 elements
//       - Decode Q6_K 6-bit values:
//         * q6_low  = q6 & 0x0F         (lower nibble)
//         * q6_high = q6 >> 4           (upper nibble)
//         * Extract 2-bit qh fields and shift into position
//         * Combine: q6_byte = q6_low | (qh_field << 4)
//         * Subtract 32: q6_signed = q6_byte - 32
//       - Load q8 (128 bytes): int8 activation values
//       - Compute widening multiply: vwmul(q6_signed, q8) → int16
//       - Apply per-block scales (8 scales per half-block):
//         * Multiply each 16-element chunk's sum by its scale
//         * Reduce to int32 accumulators
//    c. Final: sumf += d * sum_t
//
// Data layout:
//   block_q6_K:
//     ql[128]    - lower 4 bits of each 6-bit quant (packed 2 per byte)
//     qh[64]     - upper 2 bits of each 6-bit quant (packed 4 per byte)
//     scales[16] - 8-bit per-block scales (16 blocks of 16 elements)
//     d          - FP16 super-block scale
//
//   block_q8_K:
//     qs[256]    - int8 quantized values
//     d          - FP32 block scale
//     bsums[16]  - int16 block sums (unused here; used in AVX path for bias)

#if defined(__riscv_v_intrinsic)
static void ggml_vec_dot_q6_K_q8_K_rvv(int n, float * GGML_RESTRICT s, size_t bs,
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

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        // Block-level scale: combines x's FP16 scale and y's FP32 scale
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT q6 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;

        const int8_t * GGML_RESTRICT scale = x[i].scales;

        int sum_t = 0;
        int is = 0;

        // Each Q6_K super-block has QK_K=256 elements.
        // We process QK_K/128 = 2 half-blocks, each with 128 elements.
        // Each half-block: 64 bytes of q6 (ql), 32 bytes of qh, 128 bytes of q8.
        for (int j = 0; j < QK_K / 128; ++j) {

            size_t vl;

            // ----------------------------------------------------------------
            // Step 1: Load and decode qh (32 bytes = 128 × 2-bit fields)
            // ----------------------------------------------------------------
            // qh byte layout: each byte holds 4 × 2-bit fields
            // qh[l] bits: [1:0]=elem(4l+0), [3:2]=elem(4l+1),
            //             [5:4]=elem(4l+2), [7:6]=elem(4l+3)
            vl = 32;
            vuint8m1_t qh_v = __riscv_vle8_v_u8m1(qh, vl);

            // Extract 2-bit fields from qh and position them for 4 groups:
            //   Group 0: bits [1:0] of each byte → shift left by 4
            //   Group 1: bits [3:2] of each byte → shift left by 4
            //   Group 2: bits [5:4] of each byte → shift left by 4
            //   Group 3: bits [7:6] of each byte → shift left by 4
            vuint8m1_t qh_0 = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(qh_v, 0x03, vl), 4, vl);
            vuint8m1_t qh_1 = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 2, vl), 0x03, vl), 4, vl);
            vuint8m1_t qh_2 = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 4, vl), 0x03, vl), 4, vl);
            vuint8m1_t qh_3 = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 6, vl), 0x03, vl), 4, vl);

            // ----------------------------------------------------------------
            // Step 2: Load q6 (lower 4 bits from 64 bytes)
            // ----------------------------------------------------------------
            // ql byte layout: each byte holds 2 nibbles
            // ql[l]: [3:0]=elem(2l+0), [7:4]=elem(2l+1)
            vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);       // first 32 bytes
            vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6 + 32, vl);  // second 32 bytes

            // Extract lower nibbles from q6_0 and q6_1
            vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
            vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
            // Extract upper nibbles from q6_0 and q6_1
            vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 4, vl);
            vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 4, vl);

            // ----------------------------------------------------------------
            // Step 3: Combine q6 nibbles with qh fields to get 6-bit values
            // ----------------------------------------------------------------
            // Result layout (each group has 32 elements from 32 bytes):
            //   qhi_0 = lower nibble of q6_0 | upper bits from qh bits[1:0]
            //   qhi_1 = lower nibble of q6_1 | upper bits from qh bits[3:2]
            //   qhi_2 = upper nibble of q6_0 | upper bits from qh bits[5:4]
            //   qhi_3 = upper nibble of q6_1 | upper bits from qh bits[7:6]
            vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, qh_0, vl);
            vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, qh_1, vl);
            vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, qh_2, vl);
            vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, qh_3, vl);

            // ----------------------------------------------------------------
            // Step 4: Convert to signed by subtracting 32 (Q6_K offset)
            // ----------------------------------------------------------------
            // Q6_K values are in [0, 63], subtract 32 to get [-32, 31]
            vint8m1_t a_0 = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
            vint8m1_t a_1 = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
            vint8m1_t a_2 = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
            vint8m1_t a_3 = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

            // ----------------------------------------------------------------
            // Step 5: Load q8 and compute widening multiply-accumulate
            // ----------------------------------------------------------------
            // q8 has 128 int8 elements for this half-block
            // With VLEN=512, we can load all 128 elements in a single e8m2 load
            // Group 0 (a_0, 32 elems): q8[0..31]
            // Group 1 (a_1, 32 elems): q8[32..63]
            // Group 2 (a_2, 32 elems): q8[64..95]
            // Group 3 (a_3, 32 elems): q8[96..127]
            vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
            vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

            // Widening multiply: int8 × int8 → int16 (each result is 32 int16s)
            vint16m2_t prod_0 = __riscv_vwmul_vv_i16m2(a_0, q8_0, vl);
            vint16m2_t prod_1 = __riscv_vwmul_vv_i16m2(a_1, q8_1, vl);
            vint16m2_t prod_2 = __riscv_vwmul_vv_i16m2(a_2, q8_2, vl);
            vint16m2_t prod_3 = __riscv_vwmul_vv_i16m2(a_3, q8_3, vl);

            // ----------------------------------------------------------------
            // Step 6: Apply per-block scales (8 blocks per half-block)
            // ----------------------------------------------------------------
            // Each half-block has 128 elements = 8 blocks of 16 elements each
            // Each block has its own scale from x[i].scales[]
            // The 32 int16 products from each vwmul split into:
            //   lower 16 (prod_lo) × scale[2k]   and
            //   upper 16 (prod_hi) × scale[2k+1]
            // This gives 8 int32 partial sums
            vl = 16;

            // Get lower and upper halves of each int16m2 group
            vint16m1_t prod_0_lo = __riscv_vget_v_i16m2_i16m1(prod_0, 0);
            vint16m1_t prod_0_hi = __riscv_vget_v_i16m2_i16m1(prod_0, 1);
            vint16m1_t prod_1_lo = __riscv_vget_v_i16m2_i16m1(prod_1, 0);
            vint16m1_t prod_1_hi = __riscv_vget_v_i16m2_i16m1(prod_1, 1);
            vint16m1_t prod_2_lo = __riscv_vget_v_i16m2_i16m1(prod_2, 0);
            vint16m1_t prod_2_hi = __riscv_vget_v_i16m2_i16m1(prod_2, 1);
            vint16m1_t prod_3_lo = __riscv_vget_v_i16m2_i16m1(prod_3, 0);
            vint16m1_t prod_3_hi = __riscv_vget_v_i16m2_i16m1(prod_3, 1);

            // Widening multiply: int16 × int8 scale → int32
            // Then reduce each group to a single int32 sum
            vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(prod_0_lo, scale[is + 0], vl);
            vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(prod_0_hi, scale[is + 1], vl);
            vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(prod_1_lo, scale[is + 2], vl);
            vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(prod_1_hi, scale[is + 3], vl);
            vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(prod_2_lo, scale[is + 4], vl);
            vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(prod_2_hi, scale[is + 5], vl);
            vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(prod_3_lo, scale[is + 6], vl);
            vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(prod_3_hi, scale[is + 7], vl);

            // Reduce: add pairs of int32 groups, then horizontal sum
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

            sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

            // Advance pointers for next half-block
            q6 += 64;    // 64 bytes of ql consumed
            qh += 32;    // 32 bytes of qh consumed
            q8 += 128;   // 128 int8 values consumed
            is += 8;     // 8 scales consumed
        }

        sumf += d * sum_t;
    }

    *s = sumf;
}
#endif // __riscv_v_intrinsic

// =============================================================================
// Wrapper: selects RVV (VLEN=512) or generic at compile time
// =============================================================================
// Note: This wrapper is provided for test.cpp. In the production build,
// the wrapper logic is in arch/riscv/quants.c which dispatches based on
// __riscv_vlenb() * 8.
inline void ggml_vec_dot_q6_K_q8_K_rvv_wrapper(int n, float * GGML_RESTRICT s, size_t bs,
                                                const void * GGML_RESTRICT vx, size_t bx,
                                                const void * GGML_RESTRICT vy, size_t by,
                                                int nrc) {
#if defined(__riscv_v_intrinsic)
    ggml_vec_dot_q6_K_q8_K_rvv(n, s, bs, vx, bx, vy, by, nrc);
#else
    UNUSED(n); UNUSED(s); UNUSED(bs); UNUSED(vx); UNUSED(bx); UNUSED(vy); UNUSED(by); UNUSED(nrc);
    // Fallback not available in standalone test without the generic impl linked
    assert(false && "RVV not available");
#endif
}
