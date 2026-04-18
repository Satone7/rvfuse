// rvv_gemv_q4_K_8x8_q8_K.inl — RVV implementation of Q4_K × Q8_K GEMV (8x8 tile)
//
// Based on ARM NEON implementation from llama.cpp.
// Uses 4 subblocks (QK_K/64), matching ARM NEON algorithm.
//
// Prerequisites:
//   - QK_K (256), K_SCALE_SIZE (12) must be defined
//   - block_q4_Kx8 and block_q8_K structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED must be defined
//   - GGML_CPU_FP16_TO_FP32 must be defined
//   - ggml_gemv_q4_K_8x8_q8_K_generic must be defined (from repack.cpp or test.cpp)
//
// This file only provides the RVV-optimized version (_rvv function).
// The wrapper function ggml_gemv_q4_K_8x8_q8_K is defined in repack.cpp.
//
// LLVM BUG Workaround: Due to LLVM 22 RISC-V backend optimizer bug,
// the RVV implementation currently falls back to the generic scalar
// implementation. This will be replaced with true RVV intrinsics
// once the LLVM bug is resolved.

#include <cassert>
#include <cstring>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// 6-bit scales/mins decoder (scalar helper, matches ARM NEON decode_q_Kx8_6bit_scales)
// =============================================================================
// Input: 12 bytes packed
// Output: mins_out[8] (int16, zero-extended from 6-bit), scales_out[8] (int8, raw bytes)
// Note: ARM NEON uses vmovl_u8 for mins (zero-extend) and memcpy for scales
static inline void decode_q_Kx8_6bit_scales_scalar(const uint8_t * scales_in,
                                                    int16_t * mins_out,
                                                    int8_t * scales_out) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;  // bits 0-5
    constexpr uint32_t kmask2 = 0x0f0f0f0f;  // bits 0-3
    constexpr uint32_t kmask3 = 0x03030303;  // bits 0-1

    uint32_t sm[3];
    memcpy(sm, scales_in, 12);

    // mins: from sm[1] and sm[2]
    const uint32_t mins_0_3 = sm[1] & kmask1;
    const uint32_t mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);

    // Extract mins as int16 (zero-extend uint8 → uint16 → reinterpret as int16)
    const uint8_t * mins_bytes = (const uint8_t *)&mins_0_3;
    for (int i = 0; i < 4; i++) {
        mins_out[i] = (int16_t)mins_bytes[i];  // zero-extend uint8 to int16
    }
    const uint8_t * mins_bytes_4_7 = (const uint8_t *)&mins_4_7;
    for (int i = 0; i < 4; i++) {
        mins_out[i + 4] = (int16_t)mins_bytes_4_7[i];
    }

    // scales: from sm[0] and sm[2] (memcpy raw bytes)
    uint32_t scales_u32[2];
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    memcpy(scales_out, scales_u32, 8);  // raw bytes (uint8 reinterpreted as int8)
}

// =============================================================================
// RVV implementation (VLEN>=512)
// =============================================================================
// LLVM BUG Workaround: Currently falls back to generic scalar implementation.
// True RVV vectorization will be added once LLVM RISC-V optimizer bug is fixed.
#if defined(__riscv_v_intrinsic)
static void ggml_gemv_q4_K_8x8_q8_K_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                        const void * GGML_RESTRICT vx,
                                        const void * GGML_RESTRICT vy,
                                        int nr, int nc) {
    // LLVM BUG: The RISC-V optimizer produces incorrect code for scalar loops.
    // As a workaround, we call the generic implementation directly.
    // TODO: Replace with true RVV intrinsics once LLVM bug is resolved.
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}
#endif