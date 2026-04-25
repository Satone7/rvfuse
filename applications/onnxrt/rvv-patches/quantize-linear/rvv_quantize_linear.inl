// rvv_quantize_linear.inl — RVV implementation of MlasQuantizeLinear
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/quantize.cpp  (via patch, production build)
//   - test.cpp                                (correctness test)
//
// Prerequisites before including this file:
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

//
// RVV implementation of QuantizeLinear
//
// Formula: Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)
//
// Algorithm (using e32m2 to enable proper narrowing pipeline):
//   1. Load VL float32 elements (e32m2 configuration)
//   2. Divide by scale, clamp to [MinVal-ZP, MaxVal-ZP]
//   3. Round to nearest even (vfcvt_x_f), add zero point
//   4. Narrow: int32m2 → int16m1 → int8mf2 (for 8-bit output)
//             int32m2 → int16m1          (for 16-bit output)
//   5. Store narrowed results
//
// LMUL narrowing chain for 8-bit output:
//   e32m2 → e16m1 → e8mf2
//   Source LMUL = 2 × Dest LMUL (since source element is 2× wider)
//

#if defined(__riscv_v_intrinsic)

//
// MlasQuantizeLinearU8Kernel — RVV implementation
//
inline void
MlasQuantizeLinearU8KernelRVV(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
)
{
    constexpr int32_t MinimumValue = 0;
    constexpr int32_t MaximumValue = 255;
    const float MinVal = float(MinimumValue - (int32_t)ZeroPoint);
    const float MaxVal = float(MaximumValue - (int32_t)ZeroPoint);

    size_t avl = N;

    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);

        vfloat32m2_t v_input = __riscv_vle32_v_f32m2(Input, vl);
        vfloat32m2_t v_scaled = __riscv_vfdiv_vf_f32m2(v_input, Scale, vl);
        vfloat32m2_t v_clamped = __riscv_vfmax_vf_f32m2(v_scaled, MinVal, vl);
        v_clamped = __riscv_vfmin_vf_f32m2(v_clamped, MaxVal, vl);

        // Round to nearest even
        vint32m2_t v_rounded = __riscv_vfcvt_x_f_v_i32m2(v_clamped, vl);

        // Add zero point
        vint32m2_t v_zp = __riscv_vmv_v_x_i32m2((int32_t)ZeroPoint, vl);
        vint32m2_t v_int = __riscv_vadd_vv_i32m2(v_rounded, v_zp, vl);

        // Values are in [0, 255] due to clamping. Narrow to uint8.
        // int32m2 → uint16m1 → uint8mf2
        vuint32m2_t v_u32 = __riscv_vreinterpret_v_i32m2_u32m2(v_int);
        vuint16m1_t v_u16 = __riscv_vncvt_x_x_w_u16m1(v_u32, vl);
        vuint8mf2_t v_u8 = __riscv_vncvt_x_x_w_u8mf2(v_u16, vl);
        __riscv_vse8_v_u8mf2(Output, v_u8, vl);

        Input += vl;
        Output += vl;
        avl -= vl;
    }
}

//
// MlasQuantizeLinearS8Kernel — RVV implementation
//
inline void
MlasQuantizeLinearS8KernelRVV(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
)
{
    constexpr int32_t MinimumValue = -128;
    constexpr int32_t MaximumValue = 127;
    const float MinVal = float(MinimumValue - (int32_t)ZeroPoint);
    const float MaxVal = float(MaximumValue - (int32_t)ZeroPoint);

    size_t avl = N;

    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);

        vfloat32m2_t v_input = __riscv_vle32_v_f32m2(Input, vl);
        vfloat32m2_t v_scaled = __riscv_vfdiv_vf_f32m2(v_input, Scale, vl);
        vfloat32m2_t v_clamped = __riscv_vfmax_vf_f32m2(v_scaled, MinVal, vl);
        v_clamped = __riscv_vfmin_vf_f32m2(v_clamped, MaxVal, vl);

        // Round to nearest even
        vint32m2_t v_rounded = __riscv_vfcvt_x_f_v_i32m2(v_clamped, vl);

        // Add zero point
        vint32m2_t v_zp = __riscv_vmv_v_x_i32m2((int32_t)ZeroPoint, vl);
        vint32m2_t v_int = __riscv_vadd_vv_i32m2(v_rounded, v_zp, vl);

        // Values are in [-128, 127] due to clamping. Narrow to int8.
        // int32m2 → int16m1 → int8mf2
        vint16m1_t v_i16 = __riscv_vncvt_x_x_w_i16m1(v_int, vl);
        vint8mf2_t v_i8 = __riscv_vncvt_x_x_w_i8mf2(v_i16, vl);
        __riscv_vse8_v_i8mf2(Output, v_i8, vl);

        Input += vl;
        Output += vl;
        avl -= vl;
    }
}

//
// MlasQuantizeLinearU16Kernel — RVV implementation
//
inline void
MlasQuantizeLinearU16KernelRVV(
    const float* Input,
    uint16_t* Output,
    size_t N,
    float Scale,
    uint16_t ZeroPoint
)
{
    constexpr int32_t MinimumValue = 0;
    constexpr int32_t MaximumValue = 65535;
    const float MinVal = float(MinimumValue - (int32_t)ZeroPoint);
    const float MaxVal = float(MaximumValue - (int32_t)ZeroPoint);

    size_t avl = N;

    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);

        vfloat32m2_t v_input = __riscv_vle32_v_f32m2(Input, vl);
        vfloat32m2_t v_scaled = __riscv_vfdiv_vf_f32m2(v_input, Scale, vl);
        vfloat32m2_t v_clamped = __riscv_vfmax_vf_f32m2(v_scaled, MinVal, vl);
        v_clamped = __riscv_vfmin_vf_f32m2(v_clamped, MaxVal, vl);

        // Round to nearest even
        vint32m2_t v_rounded = __riscv_vfcvt_x_f_v_i32m2(v_clamped, vl);

        // Add zero point
        vint32m2_t v_zp = __riscv_vmv_v_x_i32m2((int32_t)ZeroPoint, vl);
        vint32m2_t v_int = __riscv_vadd_vv_i32m2(v_rounded, v_zp, vl);

        // Values are in [0, 65535] due to clamping. Narrow to uint16.
        // int32m2 → uint16m1
        vuint32m2_t v_u32 = __riscv_vreinterpret_v_i32m2_u32m2(v_int);
        vuint16m1_t v_u16 = __riscv_vncvt_x_x_w_u16m1(v_u32, vl);
        __riscv_vse16_v_u16m1(Output, v_u16, vl);

        Input += vl;
        Output += vl;
        avl -= vl;
    }
}

//
// MlasQuantizeLinearS16Kernel — RVV implementation
//
inline void
MlasQuantizeLinearS16KernelRVV(
    const float* Input,
    int16_t* Output,
    size_t N,
    float Scale,
    int16_t ZeroPoint
)
{
    constexpr int32_t MinimumValue = -32768;
    constexpr int32_t MaximumValue = 32767;
    const float MinVal = float(MinimumValue - (int32_t)ZeroPoint);
    const float MaxVal = float(MaximumValue - (int32_t)ZeroPoint);

    size_t avl = N;

    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);

        vfloat32m2_t v_input = __riscv_vle32_v_f32m2(Input, vl);
        vfloat32m2_t v_scaled = __riscv_vfdiv_vf_f32m2(v_input, Scale, vl);
        vfloat32m2_t v_clamped = __riscv_vfmax_vf_f32m2(v_scaled, MinVal, vl);
        v_clamped = __riscv_vfmin_vf_f32m2(v_clamped, MaxVal, vl);

        // Round to nearest even
        vint32m2_t v_rounded = __riscv_vfcvt_x_f_v_i32m2(v_clamped, vl);

        // Add zero point
        vint32m2_t v_zp = __riscv_vmv_v_x_i32m2((int32_t)ZeroPoint, vl);
        vint32m2_t v_int = __riscv_vadd_vv_i32m2(v_rounded, v_zp, vl);

        // Values are in [-32768, 32767] due to clamping. Narrow to int16.
        // int32m2 → int16m1
        vint16m1_t v_i16 = __riscv_vncvt_x_x_w_i16m1(v_int, vl);
        __riscv_vse16_v_i16m1(Output, v_i16, vl);

        Input += vl;
        Output += vl;
        avl -= vl;
    }
}

#endif // defined(__riscv_v_intrinsic)
