// rvv_reduce_minmax_f32.inl — RVV implementation of MlasReduceMinimumMaximumF32Kernel
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/compute.cpp  (via patch, production build)
//   - test.cpp                                (correctness test)
//
// Prerequisites before including this file:
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead

#include <cfloat>
#include <cstddef>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

//
// RVV implementation of min/max reduction
//
// Algorithm (VLEN=512, VL=16):
// 1. Main loop: process 64 elements at once (4 × VL=16 accumulators)
// 2. Secondary loop: process VL elements at once
// 3. Tail: process remaining elements one by one
// 4. Reduce vectors to scalar using vfredmin/vfredmax
//

#if defined(__riscv_v_intrinsic)

inline void
MlasReduceMinimumMaximumF32KernelRVV(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
)
{
    float tmp_min = FLT_MAX;
    float tmp_max = -FLT_MAX;

    if (N >= 16) {
        // Initialize vectors with extreme values
        vfloat32m1_t v_min = __riscv_vfmv_v_f_f32m1(tmp_min, 16);
        vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(tmp_max, 16);

        if (N >= 64) {
            // Use 4 accumulators for better throughput (similar to AVX pattern)
            vfloat32m1_t v_min0 = v_min;
            vfloat32m1_t v_min1 = v_min;
            vfloat32m1_t v_min2 = v_min;
            vfloat32m1_t v_min3 = v_min;

            vfloat32m1_t v_max0 = v_max;
            vfloat32m1_t v_max1 = v_max;
            vfloat32m1_t v_max2 = v_max;
            vfloat32m1_t v_max3 = v_max;

            while (N >= 64) {
                // Load 4 vectors (64 elements)
                vfloat32m1_t v_in0 = __riscv_vle32_v_f32m1(Input, 16);
                vfloat32m1_t v_in1 = __riscv_vle32_v_f32m1(Input + 16, 16);
                vfloat32m1_t v_in2 = __riscv_vle32_v_f32m1(Input + 32, 16);
                vfloat32m1_t v_in3 = __riscv_vle32_v_f32m1(Input + 48, 16);

                // Update min/max accumulators
                v_min0 = __riscv_vfmin_vv_f32m1(v_min0, v_in0, 16);
                v_min1 = __riscv_vfmin_vv_f32m1(v_min1, v_in1, 16);
                v_min2 = __riscv_vfmin_vv_f32m1(v_min2, v_in2, 16);
                v_min3 = __riscv_vfmin_vv_f32m1(v_min3, v_in3, 16);

                v_max0 = __riscv_vfmax_vv_f32m1(v_max0, v_in0, 16);
                v_max1 = __riscv_vfmax_vv_f32m1(v_max1, v_in1, 16);
                v_max2 = __riscv_vfmax_vv_f32m1(v_max2, v_in2, 16);
                v_max3 = __riscv_vfmax_vv_f32m1(v_max3, v_in3, 16);

                Input += 64;
                N -= 64;
            }

            // Merge 4 accumulators into 2
            v_min0 = __riscv_vfmin_vv_f32m1(v_min0, v_min1, 16);
            v_min2 = __riscv_vfmin_vv_f32m1(v_min2, v_min3, 16);

            v_max0 = __riscv_vfmax_vv_f32m1(v_max0, v_max1, 16);
            v_max2 = __riscv_vfmax_vv_f32m1(v_max2, v_max3, 16);

            // Merge 2 accumulators into 1
            v_min = __riscv_vfmin_vv_f32m1(v_min0, v_min2, 16);
            v_max = __riscv_vfmax_vv_f32m1(v_max0, v_max2, 16);
        }

        // Process remaining elements in VL-sized chunks
        while (N >= 16) {
            size_t vl = __riscv_vsetvl_e32m1(N);
            vfloat32m1_t v_in = __riscv_vle32_v_f32m1(Input, vl);
            v_min = __riscv_vfmin_vv_f32m1(v_min, v_in, vl);
            v_max = __riscv_vfmax_vv_f32m1(v_max, v_in, vl);
            Input += vl;
            N -= vl;
        }

        // Reduce vectors to scalar using ordered reduction
        // vfredmin/vfredmax: reduce to single element (ordered, deterministic)
        vfloat32m1_t v_min_scalar = __riscv_vfmv_v_f_f32m1(tmp_min, 1);
        vfloat32m1_t v_max_scalar = __riscv_vfmv_v_f_f32m1(tmp_max, 1);

        vfloat32m1_t v_min_reduced = __riscv_vfredmin_vs_f32m1_f32m1(v_min, v_min_scalar, 16);
        vfloat32m1_t v_max_reduced = __riscv_vfredmax_vs_f32m1_f32m1(v_max, v_max_scalar, 16);

        tmp_min = __riscv_vfmv_f_s_f32m1_f32(v_min_reduced);
        tmp_max = __riscv_vfmv_f_s_f32m1_f32(v_max_reduced);
    }

    // Process remaining tail elements
    while (N > 0) {
        tmp_min = (Input[0] < tmp_min) ? Input[0] : tmp_min;
        tmp_max = (Input[0] > tmp_max) ? Input[0] : tmp_max;
        Input++;
        N--;
    }

    *Min = tmp_min;
    *Max = tmp_max;
}

#endif // defined(__riscv_v_intrinsic)