// rvv_quantize_q8_0_4x4.inl — RVV implementation of ggml_quantize_mat_q8_0_4x4
//
// Single source of truth for the RVV vectorized kernel. Included by:
//   - arch/riscv/repack.cpp  (via patch, production build)
//   - tests/test_quantize_q8_0_4x4.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK8_0, block_q8_0x4 must be defined
//   - GGML_RESTRICT must be defined
//   - GGML_CPU_FP32_TO_FP16 must be defined
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation: ggml_quantize_mat_q8_0_4x4
// =============================================================================
// Quantizes 4 rows of FP32 activations to Q8_0 format with 4x4 interleaving.
// Each block has 32 elements (QK8_0=32). Output is block_q8_0x4 structure.
//
// Algorithm:
//   1. Load 32 FP32 values from each of 4 rows
//   2. Compute max absolute value for each row (for scale)
//   3. Scale to int8 range [-127, 127] with rounding
//   4. Interleave 4 rows using segment store (vsseg4e32)

inline void ggml_quantize_mat_q8_0_4x4_rvv(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

#if defined(__riscv_v_intrinsic)
    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
    const size_t vl_calc = __riscv_vsetvl_e32m8(QK8_0);
    const size_t vl_save = __riscv_vsetvl_e32m2(8);
    vfloat32m1_t v_scalar_zero = __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

    for (int i = 0; i < nb; i++) {
        const float *x_block_base = x + i * QK8_0;
        vint8m2_t q_r0, q_r1, q_r2, q_r3;
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 0 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[0] = GGML_CPU_FP32_TO_FP16(d);

            float id = d ? 1.0f / d : 0.0f;
            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r0 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");

        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 1 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[1] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r1 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 2 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[2] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r2 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 3 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[3] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r3 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }

        // Interleave 4 rows with block_size=4 using segment store.
        // Reinterpret int8 data as int32 (each 32-bit element = 4 consecutive int8 values).
        // vsseg4e32 with vl=8 stores 4 segments × 8 elements × 4 bytes = 128 bytes.
        // Memory layout: r0[0:4], r1[0:4], r2[0:4], r3[0:4], r0[4:8], r1[4:8], ...
        vint32m2_t v_q32_r0 = __riscv_vreinterpret_v_i8m2_i32m2(q_r0);
        vint32m2_t v_q32_r1 = __riscv_vreinterpret_v_i8m2_i32m2(q_r1);
        vint32m2_t v_q32_r2 = __riscv_vreinterpret_v_i8m2_i32m2(q_r2);
        vint32m2_t v_q32_r3 = __riscv_vreinterpret_v_i8m2_i32m2(q_r3);
        vint32m2x4_t v_quant_tuple = __riscv_vcreate_v_i32m2x4(v_q32_r0, v_q32_r1, v_q32_r2, v_q32_r3);
        __riscv_vsseg4e32_v_i32m2x4((int32_t*)y[i].qs, v_quant_tuple, vl_save);
    }
#else
    // Fallback: call generic implementation
    UNUSED(nb);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

// Wrapper function that selects RVV or generic based on compile-time detection
inline void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
#if defined(__riscv_v_intrinsic)
    ggml_quantize_mat_q8_0_4x4_rvv(x, vy, k);
#else
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}