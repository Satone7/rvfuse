/*++

Copyright (c) RVFuse Project. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_fp16_rvv512.inl

Abstract:

    RVV (RISC-V Vector Extension) implementation of FP16 softmax kernels
    targeting VLEN=512 (512-bit vector registers).

    VLEN=512 with SEW=16 (half precision) and LMUL=1 yields VL=32, meaning
    each vector instruction processes 32 FP16 elements simultaneously.

    This file implements the following kernels matching the MLAS_SOFTMAX_DISPATCH
    interface for FP16 softmax computation:

    1. ReduceMax_Kernel_Fp16_Rvv512   - Find maximum value in FP16 array
    2. SumExp_Kernel_Fp16_Rvv512     - Compute exp(x - max) and accumulate sum
    3. Softmax_Kernel_Fp16_Rvv512    - Divide exp values by sum (normalization)
    4. Exp_Kernel_Fp16_Rvv512        - Compute exponential function (FP16)
    5. LogSoftmax_Kernel_Fp16_Rvv512 - Compute log softmax output

    Algorithm (matching ONNX Runtime MLAS softmax pipeline):
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    FP16 exp approximation uses range reduction to log2 domain + polynomial
    evaluation (same approach as ARM NEON fp16 kernels).

    VLEN=512 Configuration:
        VLEN = 512 bits
        SEW  = 16 bits (FP16)
        LMUL = 1
        VL   = 512 / 16 = 32 elements per vector register

    Performance estimate vs scalar:
        ReduceMax:  32x fewer loop iterations
        SumExp:     32x vectorized exp + reduction
        Softmax:    32x vectorized multiply

--*/

#if defined(MLAS_TARGET_RISCV) && defined(MLAS_RVV_VLEN_512) && defined(__riscv_zvfh)

#include <riscv_vector.h>
#include <cmath>
#include <limits>

//
// FP16 constants for exp approximation.
// Stored as uint16_t bit patterns matching the MLAS fp16 polynomial
// coefficients used by the ARM NEON kernel.
//
// These constants encode the following FP16 values:
//   LowerRange       = -15.5 * ln(2)  (lower clamp for SumExp)
//   UpperRange       =  15.5 * ln(2)  (upper clamp for SumExp)
//   RoundingBias     =  1.5 * 2^10    (for rounding in log2 conversion)
//   Log2Reciprocal   =  1/ln(2)       (multiply to convert to log2 domain)
//   Log2High         =  -6.9287e-1    (high bits of -ln(2))
//   Log2Mid          =  -2.7585e-4    (mid bits of -ln(2))
//   Log2Low          =  -2.3842e-7    (low bits of -ln(2))
//   poly_0..poly_4   =  polynomial coefficients for exp approximation
//   poly_56          =  1/2! for SumExp (p6=1 merged into final add)
//   MaximumExponent  =  15            (FP16 max unbiased exponent)
//

namespace softmax_rvv512 {

//
// FP16 helper: reinterpret uint16 as _Float16
//
MLAS_FORCEINLINE _Float16
BitsToFp16(uint16_t bits) {
    _Float16 val;
    __builtin_memcpy(&val, &bits, sizeof(val));
    return val;
}

//
// FP16 helper: reinterpret _Float16 as uint16
//
MLAS_FORCEINLINE uint16_t
Fp16ToBits(_Float16 val) {
    uint16_t bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    return bits;
}

//
// FP16 constants as _Float16 values (bit patterns from MLAS ExpConstantsFp16).
//
constexpr _Float16 kLowerRangeSumExp  = BitsToFp16(0xc95f);  // -15.5 * ln2
constexpr _Float16 kUpperRangeSumExp  = BitsToFp16(0x495f);  //  15.5 * ln2
constexpr _Float16 kLowerRange        = BitsToFp16(0xcc55);  // -25 * ln2
constexpr _Float16 kUpperRange        = BitsToFp16(0x498c);  //  16 * ln2
constexpr _Float16 kRoundingBias      = BitsToFp16(0x6600);  //  1.5 * 2^10
constexpr _Float16 kLog2Reciprocal    = BitsToFp16(0x3dc5);  //  1/ln2
constexpr _Float16 kLog2High          = BitsToFp16(0xb98b);  // -6.9287e-1
constexpr _Float16 kLog2Mid           = BitsToFp16(0x8c85);  // -2.7585e-4
constexpr _Float16 kLog2Low           = BitsToFp16(0x8004);  // -2.3842e-7

// Polynomial coefficients for exp (Horner form):
//   p(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6!
constexpr _Float16 kPoly0  = BitsToFp16(0x15b0);  // 1/6!
constexpr _Float16 kPoly1  = BitsToFp16(0x2044);  // 1/5!
constexpr _Float16 kPoly2  = BitsToFp16(0x2955);  // 1/4!
constexpr _Float16 kPoly3  = BitsToFp16(0x3155);  // 1/3!
constexpr _Float16 kPoly4  = BitsToFp16(0x3800);  // 1/2!
constexpr _Float16 kPoly56 = BitsToFp16(0x3c00);  // 1/1!

constexpr int16_t kMaximumExponentBits = static_cast<int16_t>(0x3C00);  // exponent = 15

//
// Vectorized exp function for FP16 (RVV, VLEN=512, SEW=16, LMUL=1).
// Uses range reduction to log2 domain + polynomial approximation.
//
// Algorithm:
//   1. Clamp input to [LowerRange, UpperRange]
//   2. Compute biased = x * (1/ln2) + rounding_bias
//   3. Compute m = biased - rounding_bias (integer part)
//   4. Compute residual r = x + m * ln2_high + m * ln2_mid + m * ln2_low
//   5. Compute normal exponent from m (shift left by 10 = FP16 mantissa bits)
//   6. Evaluate polynomial: p = poly_0 + r*(poly_1 + r*(poly_2 + ...))
//   7. Reconstruct: result = p * 2^normal
//
MLAS_FORCEINLINE vfloat16m1_t
ExpVectorFp16(vfloat16m1_t x, size_t vl) {
    // Clamp input to valid range for exp
    vfloat16m1_t clamped = __riscv_vfmax_vf_f16m1(
        __riscv_vfmin_vf_f16m1(x, kUpperRange, vl),
        kLowerRange, vl);

    // Range reduction: biased = clamped * (1/ln2) + rounding_bias
    // Using vfmadd: dest = rs1 * rs2 + rd => clamped * Log2Reciprocal + RoundingBias
    vfloat16m1_t biased = __riscv_vfmadd_vf_f16m1(clamped, kLog2Reciprocal, kRoundingBias, vl);

    // m = biased - rounding_bias (integer part)
    vfloat16m1_t m = __riscv_vfsub_vf_f16m1(biased, kRoundingBias, vl);

    // Residual: r = clamped + m * Log2High + m * Log2Mid + m * Log2Low
    vfloat16m1_t r = __riscv_vfmadd_vf_f16m1(m, kLog2High, clamped, vl);
    r = __riscv_vfmadd_vf_f16m1(m, kLog2Mid, r, vl);
    r = __riscv_vfmadd_vf_f16m1(m, kLog2Low, r, vl);

    // Compute scaling factors from exponent m:
    // normal = (int16(biased) << 10) + MaximumExponent
    // This converts the log2 integer part to an FP16 exponent.
    // We reinterpret biased as int16, shift left by 10 (FP16 mantissa bits),
    // then add the bias (15 << 10 = 0x3C00) to get 2^m in FP16 representation.
    vint16m1_t biased_int = __riscv_vreinterpret_v_i16m1_f16m1(biased);
    vint16m1_t normal = __riscv_vsll_vi_i16m1(biased_int, 10, vl);
    vint16m1_t max_exp_vec = __riscv_vmv_v_x_i16m1(kMaximumExponentBits, vl);
    normal = __riscv_vadd_vv_i16m1(normal, max_exp_vec, vl);

    // Polynomial approximation (Horner form):
    // p = poly_0 + r*(poly_1 + r*(poly_2 + r*(poly_3 + r*(poly_4 + r*poly_56))))
    vfloat16m1_t p = __riscv_vfmadd_vf_f16m1(kPoly0, r, kPoly1, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly2, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly3, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly4, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly56, vl);

    // Reconstruct: result = p * 2^normal
    vfloat16m1_t normal_f = __riscv_vreinterpret_v_f16m1_i16m1(normal);
    p = __riscv_vfmul_vv_f16m1(p, normal_f, vl);

    return p;
}

//
// Vectorized SumExp function for FP16 (RVV, VLEN=512).
// Computes exp(x + negative_maximum) with narrower valid range.
// No overflow handling needed since all inputs are <= 0 after subtraction.
//
MLAS_FORCEINLINE vfloat16m1_t
SumExpVectorFp16(vfloat16m1_t x, vfloat16m1_t neg_max, size_t vl) {
    // Subtract max: all values are now <= 0
    vfloat16m1_t shifted = __riscv_vfadd_vv_f16m1(x, neg_max, vl);

    // Clamp to lower range (values are already <= 0, just clamp lower bound)
    shifted = __riscv_vfmax_vf_f16m1(shifted, kLowerRangeSumExp, vl);

    // Range reduction to log2 domain
    vfloat16m1_t biased = __riscv_vfmadd_vf_f16m1(shifted, kLog2Reciprocal, kRoundingBias, vl);
    vfloat16m1_t m = __riscv_vfsub_vf_f16m1(biased, kRoundingBias, vl);

    // Residual
    vfloat16m1_t r = __riscv_vfmadd_vf_f16m1(m, kLog2High, shifted, vl);
    r = __riscv_vfmadd_vf_f16m1(m, kLog2Mid, r, vl);
    r = __riscv_vfmadd_vf_f16m1(m, kLog2Low, r, vl);

    // Compute 2^m (no overflow handling needed)
    vint16m1_t biased_int = __riscv_vreinterpret_v_i16m1_f16m1(biased);
    vint16m1_t normal = __riscv_vsll_vi_i16m1(biased_int, 10, vl);
    vint16m1_t max_exp_vec = __riscv_vmv_v_x_i16m1(kMaximumExponentBits, vl);
    normal = __riscv_vadd_vv_i16m1(normal, max_exp_vec, vl);

    // Polynomial (Horner) - same as ExpVector but with p6=p5=poly_56
    // for SumExp: p = poly_0 + r*(poly_1 + r*(... + r*poly_56)) then p = p + r*poly_56
    vfloat16m1_t p = __riscv_vfmadd_vf_f16m1(kPoly0, r, kPoly1, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly2, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly3, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly4, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly56, vl);
    p = __riscv_vfmadd_vf_f16m1(p, r, kPoly56, vl);  // Extra term for SumExp

    // Reconstruct
    vfloat16m1_t normal_f = __riscv_vreinterpret_v_f16m1_i16m1(normal);
    p = __riscv_vfmul_vv_f16m1(p, normal_f, vl);

    return p;
}

//
// Reduce maximum for FP16 vector: horizontal max reduction.
//
MLAS_FORCEINLINE _Float16
ReduceMaxHorizontal(vfloat16m1_t v, size_t vl) {
    // Use sliding window to reduce to scalar
    // For VLEN=512, VL=32: need 5 steps of pairwise max
    // Method: repeatedly shift and max until scalar
    for (size_t step = vl / 2; step >= 1; step /= 2) {
        vfloat16m1_t shifted = __riscv_vslidedown_vx_f16m1(v, step, vl);
        v = __riscv_vfmax_vv_f16m1(v, shifted, vl);
    }
    return __riscv_vmv_f_s_f16m1_f16(v);
}

//
// Reduce sum for FP16 vector: horizontal add reduction.
//
MLAS_FORCEINLINE _Float16
ReduceSumHorizontal(vfloat16m1_t v, size_t vl) {
    for (size_t step = vl / 2; step >= 1; step /= 2) {
        vfloat16m1_t shifted = __riscv_vslidedown_vx_f16m1(v, step, vl);
        v = __riscv_vfadd_vv_f16m1(v, shifted, vl);
    }
    return __riscv_vmv_f_s_f16m1_f16(v);
}

//
// =============================================================================
// Kernel 1: ReduceMax_Kernel_Fp16_Rvv512
// =============================================================================
// Find the maximum value in an FP16 array.
//
// VLEN=512, SEW=16: VL=32 elements per vector register.
// Unrolled 4x: processes 128 elements per iteration.
//
MLAS_FP16
ReduceMax_Kernel_Fp16_Rvv512(
    const MLAS_FP16* Input,
    size_t N
    )
{
    const auto* input = reinterpret_cast<const _Float16*>(Input);

    if (N == 0) {
        return MLAS_FP16::FromBits(0xFBFF);  // -NaN (minimum sentinel)
    }

    // Initialize max to first element, then vectorize
    constexpr size_t kVl = 32;  // VLEN=512 / SEW=16
    size_t vl = __riscv_vsetvl_e16m1(kVl);

    _Float16 scalar_max = input[0];
    input++;

    // Find max across remaining elements in first partial vector
    size_t remaining_first = (N - 1) < vl ? (N - 1) : vl;
    if (remaining_first > 0) {
        vfloat16m1_t v_max = __riscv_vfmv_v_f_f16m1(scalar_max, remaining_first);
        vfloat16m1_t v_in = __riscv_vle16_v_f16m1(input, remaining_first);
        v_max = __riscv_vfmax_vv_f16m1(v_max, v_in, remaining_first);
        scalar_max = ReduceMaxHorizontal(v_max, remaining_first);
        input += remaining_first;
        N -= (remaining_first + 1);
    } else {
        N--;
    }

    // Main loop: process 4 vectors (128 elements) per iteration
    while (N >= 4 * vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        vfloat16m1_t v1 = __riscv_vle16_v_f16m1(input + vl, vl);
        vfloat16m1_t v2 = __riscv_vle16_v_f16m1(input + 2 * vl, vl);
        vfloat16m1_t v3 = __riscv_vle16_v_f16m1(input + 3 * vl, vl);

        v0 = __riscv_vfmax_vv_f16m1(v0, v1, vl);
        v2 = __riscv_vfmax_vv_f16m1(v2, v3, vl);
        v0 = __riscv_vfmax_vv_f16m1(v0, v2, vl);

        _Float16 block_max = ReduceMaxHorizontal(v0, vl);
        scalar_max = (block_max > scalar_max) ? block_max : scalar_max;

        input += 4 * vl;
        N -= 4 * vl;
    }

    // Handle remaining full vectors
    while (N >= vl) {
        vfloat16m1_t v_in = __riscv_vle16_v_f16m1(input, vl);
        _Float16 block_max = ReduceMaxHorizontal(v_in, vl);
        scalar_max = (block_max > scalar_max) ? block_max : scalar_max;
        input += vl;
        N -= vl;
    }

    // Handle tail elements
    if (N > 0) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t v_max = __riscv_vfmv_v_f_f16m1(scalar_max, vl);
        vfloat16m1_t v_in = __riscv_vle16_v_f16m1(input, vl);
        v_max = __riscv_vfmax_vv_f16m1(v_max, v_in, vl);
        scalar_max = ReduceMaxHorizontal(v_max, vl);
    }

    return MLAS_FP16::FromFloat(scalar_max);
}

//
// =============================================================================
// Kernel 2: SumExp_Kernel_Fp16_Rvv512
// =============================================================================
// Compute exp(x - max) for each element and return the sum.
// Optionally stores exp(x - max) results to Output.
//
// VLEN=512, SEW=16: VL=32 elements per vector register.
// Unrolled 4x: processes 128 elements per iteration.
//
MLAS_FP16
SumExp_Kernel_Fp16_Rvv512(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    const MLAS_FP16 NegativeMaximum
    )
{
    const auto* input = reinterpret_cast<const _Float16*>(Input);
    auto* output = reinterpret_cast<_Float16*>(Output);
    const bool store_output = (Output != nullptr);

    constexpr size_t kVl = 32;
    size_t vl = __riscv_vsetvl_e16m1(kVl);

    _Float16 neg_max = NegativeMaximum.ToFloat();
    _Float16 accumulator = static_cast<_Float16>(0.0f);

    // Broadcast negative maximum to vector
    vfloat16m1_t neg_max_vec = __riscv_vfmv_v_f_f16m1(neg_max, vl);

    // Main loop: process 4 vectors per iteration (128 elements)
    while (N >= 4 * vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        vfloat16m1_t v1 = __riscv_vle16_v_f16m1(input + vl, vl);
        vfloat16m1_t v2 = __riscv_vle16_v_f16m1(input + 2 * vl, vl);
        vfloat16m1_t v3 = __riscv_vle16_v_f16m1(input + 3 * vl, vl);

        v0 = SumExpVectorFp16(v0, neg_max_vec, vl);
        v1 = SumExpVectorFp16(v1, neg_max_vec, vl);
        v2 = SumExpVectorFp16(v2, neg_max_vec, vl);
        v3 = SumExpVectorFp16(v3, neg_max_vec, vl);

        // Horizontal sum within each vector
        _Float16 s0 = ReduceSumHorizontal(v0, vl);
        _Float16 s1 = ReduceSumHorizontal(v1, vl);
        _Float16 s2 = ReduceSumHorizontal(v2, vl);
        _Float16 s3 = ReduceSumHorizontal(v3, vl);

        accumulator = static_cast<_Float16>(
            static_cast<float>(accumulator) +
            static_cast<float>(s0) + static_cast<float>(s1) +
            static_cast<float>(s2) + static_cast<float>(s3));

        if (store_output) {
            __riscv_vse16_v_f16m1(output, v0, vl);
            __riscv_vse16_v_f16m1(output + vl, v1, vl);
            __riscv_vse16_v_f16m1(output + 2 * vl, v2, vl);
            __riscv_vse16_v_f16m1(output + 3 * vl, v3, vl);
            output += 4 * vl;
        }

        input += 4 * vl;
        N -= 4 * vl;
    }

    // Handle remaining full vectors
    while (N >= vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = SumExpVectorFp16(v0, neg_max_vec, vl);

        _Float16 s = ReduceSumHorizontal(v0, vl);
        accumulator = static_cast<_Float16>(
            static_cast<float>(accumulator) + static_cast<float>(s));

        if (store_output) {
            __riscv_vse16_v_f16m1(output, v0, vl);
            output += vl;
        }

        input += vl;
        N -= vl;
    }

    // Handle tail elements
    if (N > 0) {
        vl = __riscv_vsetvl_e16m1(N);
        neg_max_vec = __riscv_vfmv_v_f_f16m1(neg_max, vl);
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = SumExpVectorFp16(v0, neg_max_vec, vl);

        _Float16 s = ReduceSumHorizontal(v0, vl);
        accumulator = static_cast<_Float16>(
            static_cast<float>(accumulator) + static_cast<float>(s));

        if (store_output) {
            __riscv_vse16_v_f16m1(output, v0, vl);
        }
    }

    return MLAS_FP16::FromFloat(accumulator);
}

//
// =============================================================================
// Kernel 3: Softmax_Kernel_Fp16_Rvv512
// =============================================================================
// Compute final softmax normalization: output[i] = input[i] / sum.
// Input contains exp(x - max) values, Sum is the accumulated sum.
//
// VLEN=512, SEW=16: VL=32 elements per vector register.
// Unrolled 4x: processes 128 elements per iteration.
//
void
Softmax_Kernel_Fp16_Rvv512(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    const MLAS_FP16 Sum
    )
{
    const auto* input = reinterpret_cast<const _Float16*>(Input);
    auto* output = reinterpret_cast<_Float16*>(Output);

    constexpr size_t kVl = 32;
    size_t vl = __riscv_vsetvl_e16m1(kVl);

    // Compute scale = 1.0 / Sum in FP16
    float sum_f = Sum.ToFloat();
    _Float16 scale = static_cast<_Float16>(1.0f / sum_f);

    // Main loop: process 4 vectors per iteration (128 elements)
    while (N >= 4 * vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        vfloat16m1_t v1 = __riscv_vle16_v_f16m1(input + vl, vl);
        vfloat16m1_t v2 = __riscv_vle16_v_f16m1(input + 2 * vl, vl);
        vfloat16m1_t v3 = __riscv_vle16_v_f16m1(input + 3 * vl, vl);

        v0 = __riscv_vfmul_vf_f16m1(v0, scale, vl);
        v1 = __riscv_vfmul_vf_f16m1(v1, scale, vl);
        v2 = __riscv_vfmul_vf_f16m1(v2, scale, vl);
        v3 = __riscv_vfmul_vf_f16m1(v3, scale, vl);

        __riscv_vse16_v_f16m1(output, v0, vl);
        __riscv_vse16_v_f16m1(output + vl, v1, vl);
        __riscv_vse16_v_f16m1(output + 2 * vl, v2, vl);
        __riscv_vse16_v_f16m1(output + 3 * vl, v3, vl);

        input += 4 * vl;
        output += 4 * vl;
        N -= 4 * vl;
    }

    // Handle remaining full vectors
    while (N >= vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = __riscv_vfmul_vf_f16m1(v0, scale, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);

        input += vl;
        output += vl;
        N -= vl;
    }

    // Handle tail elements
    if (N > 0) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = __riscv_vfmul_vf_f16m1(v0, scale, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);
    }
}

//
// =============================================================================
// Kernel 4: Exp_Kernel_Fp16_Rvv512
// =============================================================================
// Compute exp(x) for each element of an FP16 array.
//
// Valid input range: [-17.3287, 11.0904] (narrower range mapped from FP16).
//
void
Exp_Kernel_Fp16_Rvv512(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    )
{
    const auto* input = reinterpret_cast<const _Float16*>(Input);
    auto* output = reinterpret_cast<_Float16*>(Output);

    constexpr size_t kVl = 32;
    size_t vl = __riscv_vsetvl_e16m1(kVl);

    // Main loop: process 4 vectors per iteration (128 elements)
    while (N >= 4 * vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        vfloat16m1_t v1 = __riscv_vle16_v_f16m1(input + vl, vl);
        vfloat16m1_t v2 = __riscv_vle16_v_f16m1(input + 2 * vl, vl);
        vfloat16m1_t v3 = __riscv_vle16_v_f16m1(input + 3 * vl, vl);

        v0 = ExpVectorFp16(v0, vl);
        v1 = ExpVectorFp16(v1, vl);
        v2 = ExpVectorFp16(v2, vl);
        v3 = ExpVectorFp16(v3, vl);

        __riscv_vse16_v_f16m1(output, v0, vl);
        __riscv_vse16_v_f16m1(output + vl, v1, vl);
        __riscv_vse16_v_f16m1(output + 2 * vl, v2, vl);
        __riscv_vse16_v_f16m1(output + 3 * vl, v3, vl);

        input += 4 * vl;
        output += 4 * vl;
        N -= 4 * vl;
    }

    // Handle remaining full vectors
    while (N >= vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = ExpVectorFp16(v0, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);

        input += vl;
        output += vl;
        N -= vl;
    }

    // Handle tail elements
    if (N > 0) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = ExpVectorFp16(v0, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);
    }
}

//
// =============================================================================
// Kernel 5: LogSoftmax_Kernel_Fp16_Rvv512
// =============================================================================
// Compute log softmax output: output[i] = input[i] + neg_max - log_sum
//
void
LogSoftmax_Kernel_Fp16_Rvv512(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    const MLAS_FP16 NegativeMaximum,
    const MLAS_FP16 LogSum
    )
{
    const auto* input = reinterpret_cast<const _Float16*>(Input);
    auto* output = reinterpret_cast<_Float16*>(Output);

    constexpr size_t kVl = 32;
    size_t vl = __riscv_vsetvl_e16m1(kVl);

    _Float16 neg_max = NegativeMaximum.ToFloat();
    _Float16 log_sum = LogSum.ToFloat();

    // Main loop: process 4 vectors per iteration (128 elements)
    while (N >= 4 * vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        vfloat16m1_t v1 = __riscv_vle16_v_f16m1(input + vl, vl);
        vfloat16m1_t v2 = __riscv_vle16_v_f16m1(input + 2 * vl, vl);
        vfloat16m1_t v3 = __riscv_vle16_v_f16m1(input + 3 * vl, vl);

        // output[i] = input[i] + neg_max - log_sum
        v0 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v0, neg_max, vl), log_sum, vl);
        v1 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v1, neg_max, vl), log_sum, vl);
        v2 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v2, neg_max, vl), log_sum, vl);
        v3 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v3, neg_max, vl), log_sum, vl);

        __riscv_vse16_v_f16m1(output, v0, vl);
        __riscv_vse16_v_f16m1(output + vl, v1, vl);
        __riscv_vse16_v_f16m1(output + 2 * vl, v2, vl);
        __riscv_vse16_v_f16m1(output + 3 * vl, v3, vl);

        input += 4 * vl;
        output += 4 * vl;
        N -= 4 * vl;
    }

    // Handle remaining full vectors
    while (N >= vl) {
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v0, neg_max, vl), log_sum, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);

        input += vl;
        output += vl;
        N -= vl;
    }

    // Handle tail elements
    if (N > 0) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t v0 = __riscv_vle16_v_f16m1(input, vl);
        v0 = __riscv_vfsub_vf_f16m1(__riscv_vfadd_vf_f16m1(v0, neg_max, vl), log_sum, vl);
        __riscv_vse16_v_f16m1(output, v0, vl);
    }
}

}  // namespace softmax_rvv512

#endif // MLAS_TARGET_RISCV && MLAS_RVV_VLEN_512 && __riscv_zvfh
