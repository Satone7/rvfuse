// rvv_l2_normalize_f32.inl — RVV implementation of L2 Normalize for float32.
//
// L2Normalize(x) = x / ||x||_2  where ||x||_2 = sqrt(sum(x_i^2))
//
// Key shape for SuperPoint: 256-dim descriptors (256 = 16*16 — perfectly
// aligned with VL=16 for VLEN=512!).
//
// Copyright (c) RVFuse Project. Licensed under the MIT License.

#ifndef RVV_L2_NORMALIZE_F32_INL
#define RVV_L2_NORMALIZE_F32_INL

#include <riscv_vector.h>
#include <cstddef>
#include <cmath>

// L2 normalize a single vector in-place.
// buffer: [N] input/output (modified in place)
// N: number of elements
static inline void MlasL2NormalizeF32_rvv(float* buffer, size_t N)
{
    // Step 1: Compute sum of squares using RVV
    float sumSq = 0.0f;
    size_t avl = N;
    const float* ptr = buffer;

    while (avl > 0) {
        size_t vl = vsetvl_e32m1(avl);
        vfloat32m1_t v_val = vle32_v_f32m1(ptr, vl);
        vfloat32m1_t v_sq = vfmul_vv_f32m1(v_val, v_val, vl);  // square

        // Ordered sum reduction
        vfloat32m1_t v_zero = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_red = vfredusum_vs_f32m1_f32(v_zero, v_sq, v_zero, vl);
        sumSq += vfmv_f_s_f32m1_f32(v_red);

        ptr += vl;
        avl -= vl;
    }

    // Step 2: Compute inverse norm
    float norm = sqrtf(sumSq);
    float invNorm = 1.0f / (norm > 1e-6f ? norm : 1.0f);

    // Step 3: Scale all elements by inverse norm
    avl = N;
    float* out_ptr = buffer;

    while (avl > 0) {
        size_t vl = vsetvl_e32m1(avl);
        vfloat32m1_t v_val = vle32_v_f32m1(out_ptr, vl);
        vfloat32m1_t v_inv = vfmv_v_f_f32m1(invNorm, vl);
        vfloat32m1_t v_result = vfmul_vv_f32m1(v_val, v_inv, vl);
        vse32_v_f32m1(out_ptr, v_result, vl);
        out_ptr += vl;
        avl -= vl;
    }
}

// L2 normalize with separate input and output buffers.
static inline void MlasL2NormalizeF32_rvv(
    const float* input, float* output, size_t N)
{
    // Step 1: Compute sum of squares
    float sumSq = 0.0f;
    size_t avl = N;
    const float* ptr = input;

    while (avl > 0) {
        size_t vl = vsetvl_e32m1(avl);
        vfloat32m1_t v_val = vle32_v_f32m1(ptr, vl);
        vfloat32m1_t v_sq = vfmul_vv_f32m1(v_val, v_val, vl);
        vfloat32m1_t v_zero = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_red = vfredusum_vs_f32m1_f32(v_zero, v_sq, v_zero, vl);
        sumSq += vfmv_f_s_f32m1_f32(v_red);
        ptr += vl;
        avl -= vl;
    }

    // Step 2: Scale by inverse norm
    float invNorm = 1.0f / (sqrtf(sumSq) > 1e-6f ? sqrtf(sumSq) : 1.0f);

    avl = N;
    const float* in_ptr = input;
    float* out_ptr = output;

    while (avl > 0) {
        size_t vl = vsetvl_e32m1(avl);
        vfloat32m1_t v_val = vle32_v_f32m1(in_ptr, vl);
        vfloat32m1_t v_inv = vfmv_v_f_f32m1(invNorm, vl);
        vfloat32m1_t v_result = vfmul_vv_f32m1(v_val, v_inv, vl);
        vse32_v_f32m1(out_ptr, v_result, vl);
        in_ptr += vl;
        out_ptr += vl;
        avl -= vl;
    }
}

#endif // RVV_L2_NORMALIZE_F32_INL
