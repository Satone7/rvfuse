// rvv_softmax_channel_f32.inl — RVV implementation of channel-wise Softmax for float32.
//
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))
//
// Optimized for channel-wise softmax where the reduction dimension (channels)
// may not be aligned to VL. Handles tails via setvl.
//
// Key shape for SuperPoint: 65 channels (65 = 4*16 + 1 — tail issue at VL=16)
//
// NOTE: Uses scalar expf() for the exponent step since RISC-V does not have
// a vector exp instruction in the base V extension. The RVV-accelerated parts
// are: (1) max reduction, (2) subtract-max, (3) sum reduction, (4) division.
// expf() calls are vectorized per-element but benefit from VL-based batching.
//
// Copyright (c) RVFuse Project. Licensed under the MIT License.

#ifndef RVV_SOFTMAX_CHANNEL_F32_INL
#define RVV_SOFTMAX_CHANNEL_F32_INL

#include <riscv_vector.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

// Vectorized exp using scalar expf per element — no Zvfbfmin dependency.
// This is the portable fallback; hardware with vector exp can replace this.
static inline void rvv_expf_f32(const float* input, float* output, size_t N)
{
    size_t vl;
    const float* in_ptr = input;
    float* out_ptr = output;

    while (N > 0) {
        vl = vsetvl_e32m1(N);
        // Load, compute exp per element using scalar, store
        // NOTE: On compilers that auto-vectorize expf, this will be vectorized.
        // On others, it degrades gracefully to scalar loops.
        for (size_t i = 0; i < vl; i++) {
            out_ptr[i] = expf(in_ptr[i]);
        }
        in_ptr += vl;
        out_ptr += vl;
        N -= vl;
    }
}

// Channel-wise softmax for contiguous channel data.
// input/output: [spatial, channels] with stride between spatial positions
static inline void MlasSoftmaxChannelF32_rvv(
    const float* input, float* output,
    int channels, int spatial, int stride)
{
    for (int s = 0; s < spatial; s++) {
        const float* in_ptr = input + s * stride;
        float* out_ptr = output + s * stride;

        // Step 1: Vectorized max reduction
        float maxVal;
        {
            size_t avl = channels;
            size_t vl = vsetvl_e32m1(avl);
            vfloat32m1_t v_in = vle32_v_f32m1(in_ptr, vl);
            vfloat32m1_t v_max = v_in;
            const float* ptr = in_ptr + vl;
            avl -= vl;

            while (avl > 0) {
                vl = vsetvl_e32m1(avl);
                v_in = vle32_v_f32m1(ptr, vl);
                v_max = vfmax_vv_f32m1(v_max, v_in, vl);
                ptr += vl;
                avl -= vl;
            }

            // Final reduction: reduce all elements of v_max to scalar
            vl = vsetvl_e32m1(channels);
            vfloat32m1_t v_zero = vfmv_v_f_f32m1(-INFINITY, vl);
            vfloat32m1_t v_red = vfredmax_vs_f32m1_f32(v_zero, v_max, v_zero, vl);
            maxVal = vfmv_f_s_f32m1_f32(v_red);
        }

        // Step 2: Subtract max, compute exp, and sum
        float sum = 0.0f;
        {
            size_t avl = channels;
            const float* in_p = in_ptr;
            float* out_p = out_ptr;

            while (avl > 0) {
                size_t vl = vsetvl_e32m1(avl);
                vfloat32m1_t v_max_bc = vfmv_v_f_f32m1(maxVal, vl);
                vfloat32m1_t v_in = vle32_v_f32m1(in_p, vl);
                vfloat32m1_t v_sub = vfsub_vv_f32m1(v_in, v_max_bc, vl);

                // Store subtracted values temporarily
                vse32_v_f32m1(out_p, v_sub, vl);

                // Compute exp (scalar per element — no vector exp in base V extension)
                for (size_t i = 0; i < vl; i++) {
                    out_p[i] = expf(out_p[i]);
                }

                // Reload exp results and reduce sum
                vfloat32m1_t v_exp = vle32_v_f32m1(out_p, vl);
                vfloat32m1_t v_zero = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_red = vfredusum_vs_f32m1_f32(v_zero, v_exp, v_zero, vl);
                sum += vfmv_f_s_f32m1_f32(v_red);

                in_p += vl;
                out_p += vl;
                avl -= vl;
            }
        }

        // Step 3: Normalize by sum (multiply by 1/sum)
        {
            float inv_sum = 1.0f / sum;
            size_t avl = channels;
            float* out_p = out_ptr;

            while (avl > 0) {
                size_t vl = vsetvl_e32m1(avl);
                vfloat32m1_t v_exp = vle32_v_f32m1(out_p, vl);
                vfloat32m1_t v_inv = vfmv_v_f_f32m1(inv_sum, vl);
                vfloat32m1_t v_result = vfmul_vv_f32m1(v_exp, v_inv, vl);
                vse32_v_f32m1(out_p, v_result, vl);
                out_p += vl;
                avl -= vl;
            }
        }
    }
}

#endif // RVV_SOFTMAX_CHANNEL_F32_INL
