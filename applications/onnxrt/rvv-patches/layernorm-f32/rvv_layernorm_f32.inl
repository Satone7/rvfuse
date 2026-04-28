// rvv_layernorm_f32.inl — RVV implementation of LayerNorm for float32.
//
// LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + eps) * gamma + beta
//
// Target shapes (SuperGlue): (N, 256) where N ∈ [1, 1024] keypoints
// Reduces along last dimension (D=256), broadcasts back.
// 256 = 16 * 16 — perfectly aligned with VL=16 at VLEN=512.
//
// Copyright (c) RVFuse Project. Licensed under the MIT License.

#ifndef RVV_LAYERNORM_F32_INL
#define RVV_LAYERNORM_F32_INL

#include <riscv_vector.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

// Layer normalization: reduce over last dimension D, broadcast back.
// Input shape: (N, D), output: same shape.
// gamma and beta are optional (passed as nullptr or valid D-element arrays).
// eps: small constant for numerical stability.
static inline void MlasLayerNormF32_rvv(
    const float* input,      // (N, D) row-major
    float* output,            // (N, D) row-major
    const float* gamma,       // (D,) or nullptr (use 1.0)
    const float* beta,        // (D,) or nullptr (use 0.0)
    size_t N,                 // Number of rows (sequence length)
    size_t D,                 // Feature dimension
    float eps)                // Epsilon for numerical stability
{
    for (size_t n = 0; n < N; n++) {
        const float* in_ptr = input + n * D;
        float* out_ptr = output + n * D;

        // Step 1: Compute mean via vectorized reduction sum
        float mean;
        {
            size_t avl = D;
            const float* ptr = in_ptr;
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t v_sum = v_zero;

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);
                vfloat32m1_t v_in = __riscv_vle32_v_f32m1(ptr, vl);
                v_sum = __riscv_vfredusum_vs_f32m1_f32(v_sum, v_in, v_sum, vl);
                ptr += vl;
                avl -= vl;
            }
            float total = __riscv_vfmv_f_s_f32m1_f32(v_sum);
            mean = total / static_cast<float>(D);
        }

        // Step 2: Compute variance = mean((x - mean)^2)
        float variance;
        {
            size_t avl = D;
            const float* ptr = in_ptr;
            vfloat32m1_t v_mean = __riscv_vfmv_v_f_f32m1(mean, __riscv_vsetvl_e32m1(D));
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t v_sum_sq = v_zero;

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);
                vfloat32m1_t v_in = __riscv_vle32_v_f32m1(ptr, vl);
                vfloat32m1_t v_diff = __riscv_vfsub_vv_f32m1(v_in, v_mean, vl);
                vfloat32m1_t v_sq = __riscv_vfmul_vv_f32m1(v_diff, v_diff, vl);
                v_sum_sq = __riscv_vfredusum_vs_f32m1_f32(v_sum_sq, v_sq, v_sum_sq, vl);
                ptr += vl;
                avl -= vl;
            }
            float total_sq = __riscv_vfmv_f_s_f32m1_f32(v_sum_sq);
            variance = total_sq / static_cast<float>(D);
        }

        // Step 3: Normalize, scale, shift
        float inv_std = 1.0f / std::sqrt(variance + eps);
        {
            size_t avl = D;
            const float* in_p = in_ptr;
            float* out_p = out_ptr;
            vfloat32m1_t v_mean = __riscv_vfmv_v_f_f32m1(mean, __riscv_vsetvl_e32m1(D));
            vfloat32m1_t v_inv_std = __riscv_vfmv_v_f_f32m1(inv_std, __riscv_vsetvl_e32m1(D));

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);
                vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_p, vl);
                vfloat32m1_t v_norm = __riscv_vfsub_vv_f32m1(v_in, v_mean, vl);
                v_norm = __riscv_vfmul_vv_f32m1(v_norm, v_inv_std, vl);

                // Scale by gamma if provided
                if (gamma) {
                    vfloat32m1_t v_gamma = __riscv_vle32_v_f32m1(gamma + (in_p - in_ptr), vl);
                    v_norm = __riscv_vfmul_vv_f32m1(v_norm, v_gamma, vl);
                }

                // Shift by beta if provided
                if (beta) {
                    vfloat32m1_t v_beta = __riscv_vle32_v_f32m1(beta + (in_p - in_ptr), vl);
                    v_norm = __riscv_vfadd_vv_f32m1(v_norm, v_beta, vl);
                }

                __riscv_vse32_v_f32m1(out_p, v_norm, vl);
                in_p += vl;
                out_p += vl;
                avl -= vl;
            }
        }
    }
}

#endif // RVV_LAYERNORM_F32_INL
