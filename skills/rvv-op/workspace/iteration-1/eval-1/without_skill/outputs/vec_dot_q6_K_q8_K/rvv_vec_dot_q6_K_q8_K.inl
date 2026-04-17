// rvv_vec_dot_q6_K_q8_K.inl — RVV implementation of ggml_vec_dot_q6_K_q8_K
//
// Single source of truth for the RVV vectorized kernel. Included by:
//   - arch/riscv/quants.c  (via patch, production build)
//   - tests/test_vec_dot_q6_K_q8_K.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK_K (256) must be defined
//   - block_q6_K, block_q8_K structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED must be defined
//   - GGML_CPU_FP16_TO_FP32 must be defined
//   - On RVV targets: <riscv_vector.h> must be included
//
// Target: VLEN=512 (rv64gcv_zvl512b)
//
// Data format (block_q6_K):
//   - ql[128]:  lower 4 bits of 6-bit quantized values (2 values packed per byte)
//   - qh[64]:   upper 2 bits of 6-bit quantized values (4 values packed per byte)
//   - scales[16]: per-subblock scales (8-bit, one per 16-element group)
//   - d:        super-block scale (fp16)
//
// Data format (block_q8_K):
//   - qs[256]:  int8 quantized activations
//   - d:        delta scale (fp32)
//   - bsums[16]: sum of quants in groups of 16
//
// Algorithm (per block of QK_K=256 elements):
//   1. Decode 6-bit values: (ql[n/2] & 0xF) | ((qh[n/4] >> (n%4)*2) & 3) << 4) - 32
//      Two groups per byte in ql: low nibble and high nibble
//      Two bits from qh at positions depending on element index
//   2. Multiply decoded int8 values with Q8_K activations
//   3. Apply per-16-element-group scales
//   4. Accumulate into 8 partial sums (one per 32-element group)
//   5. Final: sumf += d_x * d_y * total_integer_sum

#include <cassert>
#include <cstring>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation for VLEN=512
// =============================================================================
// With VLEN=512 bytes, we can process:
//   - e8mf2: 64 elements per register (VL/8 = 512/8 = 64)
//   - e8m1:  128 elements per register (VL/8 = 512/8 = 64, but SEW=8, LMUL=1 → 64 elements; LMUL=2 → 128)
//   - e16m4: 128 elements (VL/16 = 32, LMUL=4 → 128)
//   - e32m8: 128 elements (VL/32 = 16, LMUL=8 → 128)
//
// Processing strategy for one 128-element subblock:
//   - We need to process 32 qh bytes, 64 ql bytes, 128 q8 bytes
//   - Decode into 4 quadrants of 32 int8 values each
//   - Multiply each quadrant by corresponding q8 segment
//   - Apply 8 scales (one per 16-element group)
//   - Reduce to 4 int32 partial sums
//
// Two subblocks per QK_K block → 8 partial sums total → sumf

#if defined(__riscv_v_intrinsic)
static void ggml_vec_dot_q6_K_q8_K_rvv(int n, float * GGML_RESTRICT s, size_t bs,
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

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT q6 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        const int8_t  * GGML_RESTRICT scale = x[i].scales;

        // Zero accumulator for sequential reduction
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        int sum_t = 0;
        int is = 0;

        // Process 2 subblocks of 128 elements each (QK_K/128 = 2)
        for (int j = 0; j < QK_K/128; ++j) {

            // =================================================================
            // Load and decode Q6 values
            // =================================================================
            // Layout: 128 elements per subblock
            //   ql[0..63] packs elements 0-127:
            //     ql[n] low nibble = element 2n, ql[n] high nibble = element 2n+1
            //   qh[0..31] packs elements 0-127:
            //     qh[n] bits [1:0] = elements 4n,4n+1
            //     qh[n] bits [3:2] = elements 4n+2,4n+3
            //     qh[n] bits [5:4] = elements 4n+4,4n+5
            //     qh[n] bits [7:6] = elements 4n+6,4n+7
            //
            // Decoded value = (ql_nibble | (qh_bits << 4)) - 32

            size_t vl32 = __riscv_vsetvl_e8m1(32);

            // Load 32 bytes of qh (covers elements 0-127)
            vuint8m1_t qh_v = __riscv_vle8_v_u8m1(qh, vl32);

            // Load 64 bytes of ql in two halves
            // ql[0..31] covers elements 0-63 (low nibble=even, high=odd)
            // ql[32..63] covers elements 64-127
            vuint8m1_t q6_lo = __riscv_vle8_v_u8m1(q6, vl32);
            vuint8m1_t q6_hi = __riscv_vle8_v_u8m1(q6 + 32, vl32);

            // Extract 4-bit nibbles from ql
            vuint8m1_t q6a_lo = __riscv_vand_vx_u8m1(q6_lo, 0x0F, vl32);  // elements 0,2,4,...,62
            vuint8m1_t q6a_hi = __riscv_vand_vx_u8m1(q6_hi, 0x0F, vl32);  // elements 64,66,...,126
            vuint8m1_t q6s_lo = __riscv_vsrl_vx_u8m1(q6_lo, 0x04, vl32);  // elements 1,3,5,...,63
            vuint8m1_t q6s_hi = __riscv_vsrl_vx_u8m1(q6_hi, 0x04, vl32);  // elements 65,67,...,127

            // Extract 2-bit high parts from qh
            // qh bits [1:0] → shift 0 → elements 0,2,4,6,...
            // qh bits [3:2] → shift 2 → elements 1,3,5,7,...
            // qh bits [5:4] → shift 4 → elements 64,66,68,70,...
            // qh bits [7:6] → shift 6 → elements 65,67,69,71,...
            vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_v,                   0x03, vl32);
            vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 0x02, vl32), 0x03, vl32);
            vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 0x04, vl32), 0x03, vl32);
            vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_v, 0x06, vl32), 0x03, vl32);

            // Combine: 6-bit value = (4-bit nibble | (2-bit << 4))
            vuint8m1_t qi_0 = __riscv_vor_vv_u8m1(q6a_lo, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl32), vl32);
            vuint8m1_t qi_1 = __riscv_vor_vv_u8m1(q6a_hi, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl32), vl32);
            vuint8m1_t qi_2 = __riscv_vor_vv_u8m1(q6s_lo, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl32), vl32);
            vuint8m1_t qi_3 = __riscv_vor_vv_u8m1(q6s_hi, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl32), vl32);

            // Subtract 32 to convert unsigned 6-bit (0-63) to signed int8 (-32 to +31)
            vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qi_0), 32, vl32);
            vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qi_1), 32, vl32);
            vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qi_2), 32, vl32);
            vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qi_3), 32, vl32);

            // =================================================================
            // Load Q8 values and compute dot products
            // =================================================================
            // Q8 layout: 256 contiguous int8 values, one per element
            // q8[0..127] corresponds to subblock j
            // q8[128..255] corresponds to subblock (j+1)
            //
            // Each quadrant (a_0..a_3) has 32 elements, mapping to q8 segments:
            //   a_0 → q8[0..31]    (elements 0,2,4,...,62)
            //   a_1 → q8[64..95]   (elements 64,66,...,126)
            //   a_2 → q8[32..63]   (elements 1,3,...,63)
            //   a_3 → q8[96..127]  (elements 65,67,...,127)

            vint16m2_t p_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8,       vl32), vl32);
            vint16m2_t p_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8 + 64,  vl32), vl32);
            vint16m2_t p_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8 + 32,  vl32), vl32);
            vint16m2_t p_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8 + 96,  vl32), vl32);

            // =================================================================
            // Apply scales and reduce to int32
            // =================================================================
            // 8 scales per subblock, one per 16-element group
            // Each quadrant has 32 elements = 2 scale groups of 16
            // So for p_0 (32 elements):
            //   lower 16 elements (half 0) use scale[is+0], upper 16 (half 1) use scale[is+1]
            //
            // On VLEN=512, each vint16m2 has 32 elements.
            // Split into two vint16m1 halves, each with 16 elements.

            size_t vl16 = __riscv_vsetvl_e16m1(16);

            // Get lower and upper halves of each product vector
            vint16m1_t p_0_lo = __riscv_vget_v_i16m2_i16m1(p_0, 0);
            vint16m1_t p_0_hi = __riscv_vget_v_i16m2_i16m1(p_0, 1);
            vint16m1_t p_1_lo = __riscv_vget_v_i16m2_i16m1(p_1, 0);
            vint16m1_t p_1_hi = __riscv_vget_v_i16m2_i16m1(p_1, 1);
            vint16m1_t p_2_lo = __riscv_vget_v_i16m2_i16m1(p_2, 0);
            vint16m1_t p_2_hi = __riscv_vget_v_i16m2_i16m1(p_2, 1);
            vint16m1_t p_3_lo = __riscv_vget_v_i16m2_i16m1(p_3, 0);
            vint16m1_t p_3_hi = __riscv_vget_v_i16m2_i16m1(p_3, 1);

            // Multiply each half by its corresponding scale, widening to int32
            vint32m2_t aux_0 = __riscv_vwmul_vx_i32m2(p_0_lo, scale[is + 0], vl16);
            vint32m2_t aux_1 = __riscv_vwmul_vx_i32m2(p_0_hi, scale[is + 1], vl16);
            vint32m2_t aux_2 = __riscv_vwmul_vx_i32m2(p_1_lo, scale[is + 2], vl16);
            vint32m2_t aux_3 = __riscv_vwmul_vx_i32m2(p_1_hi, scale[is + 3], vl16);
            vint32m2_t aux_4 = __riscv_vwmul_vx_i32m2(p_2_lo, scale[is + 4], vl16);
            vint32m2_t aux_5 = __riscv_vwmul_vx_i32m2(p_2_hi, scale[is + 5], vl16);
            vint32m2_t aux_6 = __riscv_vwmul_vx_i32m2(p_3_lo, scale[is + 6], vl16);
            vint32m2_t aux_7 = __riscv_vwmul_vx_i32m2(p_3_hi, scale[is + 7], vl16);

            // Reduce: pair-wise add then horizontal sum
            // Chain the reduction through isum0 → isum1 → isum2 → isum3
            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(aux_0, aux_1, vl16), vzero, vl16);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(aux_2, aux_3, vl16), isum0, vl16);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(aux_4, aux_5, vl16), isum1, vl16);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(aux_6, aux_7, vl16), isum2, vl16);

            sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

            // Advance pointers to next subblock
            q6   += 64;
            qh   += 32;
            q8   += 128;
            is   += 8;
        }

        sumf += d * sum_t;
    }

    *s = sumf;
}
#endif // __riscv_v_intrinsic
