// rvv_vec_dot_q5_0_q8_0.inl — RVV implementation of Q5_0 x Q8_0 dot product
//
// Dual-path implementation:
//   - VLEN >= 256 (m1-based): optimized, lower register pressure
//   - VLEN = 128  (m2-based): standard, matches original llama.cpp code
//
// Based on x86 AVX2 and ARM NEON implementations from llama.cpp.
//
// Algorithm:
//   1. Unpack 16 packed nibble bytes into 32 int8 values (lower + upper nibbles)
//   2. Apply sign extension using qh bitmask (subtract 0x10 where bit=0)
//   3. Compute dot product with Q8_0 block via widening multiply + reduction
//   4. Scale by combined delta factors and accumulate
//
// Prerequisites (satisfied by quants.c or test.cpp):
//   - QK5_0, QK8_0 must be defined (both 32)
//   - block_q5_0, block_q8_0 structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED, GGML_CPU_FP16_TO_FP32 must be defined
//   - assert() must be available
//
// This file provides only the RVV-optimized function (_rvv suffix).
// The caller is responsible for selecting RVV vs generic fallback.

#include <assert.h>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation — runtime VLEN detection
// =============================================================================
#if defined(__riscv_v_intrinsic)
static void ggml_vec_dot_q5_0_q8_0_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                        const void * GGML_RESTRICT vx, size_t bx,
                                        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk;
    const size_t vlenb = __riscv_vlenb();

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = (const block_q5_0 * GGML_RESTRICT)vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 * GGML_RESTRICT)vy;

    float sumf = 0;

    if (vlenb >= 32) {
        // =================================================================
        // VLEN >= 256: optimized m1-based path
        // Uses mf2 for 16-byte loads, m1 for 32-element ops, b8 mask
        // =================================================================
        const size_t vl = qk;       // 32 elements
        const size_t vl_half = qk / 2;  // 16 elements

        for (int ib = 0; ib < nb; ++ib) {
            // Load 16 bytes of packed nibbles using mf2
            vuint8mf2_t raw = __riscv_vle8_v_u8mf2(x[ib].qs, vl_half);

            // Split into lower nibbles [0,15] and upper nibbles [0,15]
            vint8mf2_t v0l = __riscv_vreinterpret_v_u8mf2_i8mf2(
                __riscv_vand_vx_u8mf2(raw, 0x0F, vl_half));
            vint8mf2_t v0h = __riscv_vreinterpret_v_u8mf2_i8mf2(
                __riscv_vsrl_vx_u8mf2(raw, 4, vl_half));

            // Interleave into m1: [low0..low15, high0..high15]
            vint8m1_t v0c = __riscv_vlmul_ext_v_i8mf2_i8m1(v0l);
            v0c = __riscv_vslideup_vx_i8m1(v0c,
                __riscv_vlmul_ext_v_i8mf2_i8m1(v0h), qk / 2, vl);

            // Sign extension: load qh as b8 mask, invert, masked subtract 0x10
            vbool8_t qh = __riscv_vlm_v_b8(x[ib].qh, vl);
            qh = __riscv_vmnand_mm_b8(qh, qh, vl);
            vint8m1_t v0f = __riscv_vsub_vx_i8m1_mu(qh, v0c, v0c, 0x10, vl);

            // Dot product
            vint8m1_t v1 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
            vint16m2_t mul = __riscv_vwmul_vv_i16m2(v0f, v1, vl);

            // Reduction: sum i16 -> i32
            vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(mul, zero, vl);
            int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

            sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
        }
    } else {
        // =================================================================
        // VLEN = 128: standard m2-based path (matches original llama.cpp)
        // Uses m1 for 16-byte loads, m2 for 32-element ops, b4 mask
        // =================================================================
        for (int ib = 0; ib < nb; ++ib) {
            size_t vl = qk / 2;
            vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
            vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
            vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
            vint8m2_t v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);

            vl = qk;
            vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
            qh = __riscv_vmnand_mm_b4(qh, qh, vl);
            vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
            vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
            vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
            vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
            int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

            sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
        }
    }

    *s = sumf;
}
#endif // __riscv_v_intrinsic
