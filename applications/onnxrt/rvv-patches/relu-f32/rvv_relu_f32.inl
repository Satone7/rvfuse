// rvv_relu_f32.inl — RVV implementation of ReLU activation for float32.
//
// ReLU(x) = max(0, x)
//
// Uses RISC-V Vector Extension to process VL elements per iteration.
// Handles tail elements correctly via setvl.
//
// Copyright (c) RVFuse Project. Licensed under the MIT License.

#ifndef RVV_RELU_F32_INL
#define RVV_RELU_F32_INL

#include <riscv_vector.h>
#include <cstddef>

static inline void MlasReLuF32_rvv(const float* input, float* output, size_t N)
{
    size_t vl;
    const float* in_ptr = input;
    float* out_ptr = output;
    vfloat32m1_t v_zero = vfmv_v_f_f32m1(0.0f, 1);  // broadcast zero

    while (N > 0) {
        vl = vsetvl_e32m1(N);
        vfloat32m1_t v_in = vle32_v_f32m1(in_ptr, vl);
        vfloat32m1_t v_out = vfmax_vf_f32m1(v_in, 0.0f, vl);
        vse32_v_f32m1(out_ptr, v_out, vl);
        in_ptr += vl;
        out_ptr += vl;
        N -= vl;
    }
}

// In-place ReLU: modifies buffer in place
static inline void MlasReLuF32Inplace_rvv(float* buffer, size_t N)
{
    size_t vl;
    float* ptr = buffer;

    while (N > 0) {
        vl = vsetvl_e32m1(N);
        vfloat32m1_t v_val = vle32_v_f32m1(ptr, vl);
        vfloat32m1_t v_result = vfmax_vf_f32m1(v_val, 0.0f, vl);
        vse32_v_f32m1(ptr, v_result, vl);
        ptr += vl;
        N -= vl;
    }
}

#endif // RVV_RELU_F32_INL
