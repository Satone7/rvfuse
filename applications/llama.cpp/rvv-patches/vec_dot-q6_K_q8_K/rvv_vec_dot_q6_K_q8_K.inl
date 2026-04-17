// rvv_vec_dot_q6_K_q8_K.inl — RVV VL512 implementation of vec_dot_q6_K_q8_K
//
// Single source of truth for the RVV VL512 vectorized kernel. Included by:
//   - arch/riscv/quants.c  (via patch, production build)
//   - tests/test.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK_K must be defined (256)
//   - block_q6_K and block_q8_K structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED, GGML_CPU_FP16_TO_FP32 must be defined
//   - On RVV targets: <riscv_vector.h> must be included and __riscv_v_intrinsic defined

#include <assert.h>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV VL512 implementation: ggml_vec_dot_q6_K_q8_K
// =============================================================================
// Computes dot product of Q6_K (6-bit quantized) and Q8_K (8-bit quantized)
// vectors. Each super-block contains QK_K=256 elements, split into 2 sub-blocks
// of 128 elements each.
//
// block_q6_K layout (QK_K = 256 elements):
//   ql[128]:    packed 4-bit values (2 elements per byte)
//   qh[64]:     packed 2-bit values (4 elements per byte)
//   scales[16]: per-16-element-group int8 scales
//   d:          super-block FP16 scale
//
// 6-bit reconstruction per sub-block (128 elements from 32 qh + 64 ql bytes):
//   a[0..31]   = (ql[0..31]  & 0xF) | ((qh[l] >> 0) & 3) << 4)  - 32
//   a[32..63]  = (ql[32..63] & 0xF) | ((qh[l] >> 2) & 3) << 4)  - 32
//   a[64..95]  = (ql[0..31]  >> 4)  | ((qh[l] >> 4) & 3) << 4)  - 32
//   a[96..127] = (ql[32..63] >> 4)  | ((qh[l] >> 6) & 3) << 4)  - 32
//
// Algorithm (per super-block):
//   1. d_sb = x[i].d * y[i].d
//   2. For each sub-block (128 elements):
//      a. Unpack 128 x 6-bit values from ql + qh
//      b. Multiply with 128 x q8_K values  (i8*i8 -> i16 widening)
//      c. Apply 8 per-16-element scales      (i16*i8 -> i32 widening)
//      d. Chain-reduce to scalar
//   3. sumf += d_sb * sub_total
//
// VL512 register capacity (64 bytes per m1):
//   e8,m1 = 64 elements | e8,m2 = 128 elements | e16,m1 = 32 elements
//   e16,m2 = 64 elements | e32,m1 = 16 elements  | e32,m2 = 32 elements
//
// VL512 vs VL256: Same algorithm, wider registers allow the vwmul to process
// 32 elements per group (vs 32 in VL256 with same intrinsic). The scale
// application also benefits from wider e32,m2 registers (16 vs 16 elements).

#if defined(__riscv_v_intrinsic)

static void ggml_vec_dot_q6_K_q8_K_vl512(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        __builtin_prefetch(&x[i + 1].d, 0, 1);

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT q6    = x[i].ql;
        const uint8_t * GGML_RESTRICT qh    = x[i].qh;
        const  int8_t * GGML_RESTRICT q8    = y[i].qs;
        const  int8_t * GGML_RESTRICT scale = x[i].scales;

        size_t vl;
        const vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        int sum_t = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            // ---------------------------------------------------------------
            // Step 1: Load qh (32 bytes = 32 elements)
            // ---------------------------------------------------------------
            vl = 32;
            vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

            // ---------------------------------------------------------------
            // Step 2: Load ql (2 x 32 bytes)
            // ---------------------------------------------------------------
            vuint8m1_t vql_0 = __riscv_vle8_v_u8m1(q6,      vl);
            vuint8m1_t vql_1 = __riscv_vle8_v_u8m1(q6 + 32, vl);

            // ---------------------------------------------------------------
            // Step 3: Extract nibbles from ql
            // ---------------------------------------------------------------
            vuint8m1_t vqla_0 = __riscv_vand_vx_u8m1(vql_0, 0x0F, vl);  // lower nibble
            vuint8m1_t vqla_1 = __riscv_vand_vx_u8m1(vql_1, 0x0F, vl);
            vuint8m1_t vqls_0 = __riscv_vsrl_vx_u8m1(vql_0, 4, vl);    // upper nibble
            vuint8m1_t vqls_1 = __riscv_vsrl_vx_u8m1(vql_1, 4, vl);

            // ---------------------------------------------------------------
            // Step 4: Extract 2-bit groups from qh
            // ---------------------------------------------------------------
            vuint8m1_t vqh_0 = __riscv_vand_vx_u8m1(vqh, 0x03, vl);
            vuint8m1_t vqh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh, 2, vl), 0x03, vl);
            vuint8m1_t vqh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh, 4, vl), 0x03, vl);
            vuint8m1_t vqh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh, 6, vl), 0x03, vl);

            // ---------------------------------------------------------------
            // Step 5: Combine nibbles + qh bits -> 6-bit unsigned values
            // ---------------------------------------------------------------
            vuint8m1_t vhi_0 = __riscv_vor_vv_u8m1(vqla_0, __riscv_vsll_vx_u8m1(vqh_0, 4, vl), vl);
            vuint8m1_t vhi_1 = __riscv_vor_vv_u8m1(vqla_1, __riscv_vsll_vx_u8m1(vqh_1, 4, vl), vl);
            vuint8m1_t vhi_2 = __riscv_vor_vv_u8m1(vqls_0, __riscv_vsll_vx_u8m1(vqh_2, 4, vl), vl);
            vuint8m1_t vhi_3 = __riscv_vor_vv_u8m1(vqls_1, __riscv_vsll_vx_u8m1(vqh_3, 4, vl), vl);

            // ---------------------------------------------------------------
            // Step 6: Subtract 32 -> signed int8 (range: -32..+31)
            // ---------------------------------------------------------------
            vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vhi_0), 32, vl);
            vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vhi_1), 32, vl);
            vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vhi_2), 32, vl);
            vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vhi_3), 32, vl);

            // ---------------------------------------------------------------
            // Step 7: Load q8 (4 x 32 bytes, matching unpacked groups)
            //   Note: With VL512, e8,m2 can hold 128 elements but vget only
            //   splits into 2 x 64-element groups. Use 4 m1 loads instead
            //   for clean 32-element alignment with the a_0..a_3 groups.
            // ---------------------------------------------------------------
            vint8m1_t vq8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t vq8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
            vint8m1_t vq8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
            vint8m1_t vq8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

            // ---------------------------------------------------------------
            // Step 8: Multiply a[k]*q8[k] -> int16 (widening, 32 per group)
            // ---------------------------------------------------------------
            vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, vq8_0, vl);
            vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, vq8_1, vl);
            vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, vq8_2, vl);
            vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, vq8_3, vl);

            // ---------------------------------------------------------------
            // Step 9: Apply per-16-element scale (int16*scale -> int32)
            //   Each 32-element product splits into two 16-element halves,
            //   each multiplied by its corresponding scale.
            //   4 products x 2 halves = 8 scale groups per sub-block.
            // ---------------------------------------------------------------
            vl = 16;
            vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[0], vl);
            vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[1], vl);
            vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[2], vl);
            vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[3], vl);
            vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[4], vl);
            vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[5], vl);
            vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[6], vl);
            vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[7], vl);

            // ---------------------------------------------------------------
            // Step 10: Chain-reduce (add pairs, then horizontal reduce)
            // ---------------------------------------------------------------
            vint32m1_t isum = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
            isum = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum, vl);
            isum = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum, vl);
            isum = __riscv_vredsum_vs_i32m2_i32m1(
                __riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum, vl);

            sum_t += __riscv_vmv_x_s_i32m1_i32(isum);

            q6    += 64;
            qh    += 32;
            q8    += 128;
            scale += 8;
        }

        sumf += d * sum_t;
    }

    *s = sumf;
}

#endif // defined(__riscv_v_intrinsic)
