// rvv_gemv_q4_K_8x8_q8_K.inl — RVV implementation of Q4_K × Q8_K GEMV (8x8 tile)
//
// Single source of truth for the RVV vectorized kernel. Included by:
//   - arch/riscv/repack.cpp  (via patch, production build)
//   - tests/test.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK_K (256) must be defined
//   - block_q4_Kx8 and block_q8_K structures must be defined
//   - GGML_RESTRICT must be defined
//   - GGML_CPU_FP16_TO_FP32 must be defined
//   - GGML_UNUSED must be defined
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead

#include <cassert>
#include <cstring>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// 6-bit scales/mins decoder for Q4_Kx8 layout
// =============================================================================
// Each 12 bytes encodes 8 scales (low 6-bit) and 8 mins (high 6-bit)
// Input: 12 bytes from scales array
// Output: scales_out[8] (uint8, 6-bit values), mins_out[8] (uint8, 6-bit values)
//
// Encoding format (from ARM decode_q_Kx8_6bit_scales):
// sm[0]: scales 0-3 in low 6-bit, scales 4-7 bits packed
// sm[1]: mins 0-3 in low 6-bit, mins 4-7 bits packed
// sm[2]: additional bits for scales/mins 4-7
static inline void decode_q_Kx8_6bit_scales_rvv(const uint8_t * scales_in,
                                                uint8_t * scales_out,
                                                uint8_t * mins_out) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;  // 6-bit mask for bytes 0-3
    constexpr uint32_t kmask2 = 0x0f0f0f0f;  // 4-bit mask
    constexpr uint32_t kmask3 = 0x03030303;  // 2-bit mask

    uint32_t sm[3];
    memcpy(sm, scales_in, 12);

    // mins: from sm[1] and sm[2]
    const uint32_t mins_0_3 = sm[1] & kmask1;
    const uint32_t mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);

    memcpy(mins_out, &mins_0_3, 4);
    memcpy(mins_out + 4, &mins_4_7, 4);

    // scales: from sm[0] and sm[2]
    uint32_t scales_u32[2];
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    memcpy(scales_out, scales_u32, 8);
}

// =============================================================================
// RVV implementation: ggml_gemv_q4_K_8x8_q8_K
// =============================================================================
// Q4_K weights (4-bit) × Q8_K activations (8-bit) matrix-vector multiplication
// with 8x8 interleaved tile blocking.
//
// Algorithm:
//   1. For each block of 256 elements (QK_K):
//      - Decode 6-bit scales/mins from 96-byte scales array (24-byte per RVV subblock)
//      - Compute dot product: q4 × q8 for each column
//      - Apply scales: sumi_lo × scales_lo[j] + sumi_hi × scales_hi[j]
//      - Accumulate bias: mins × bsums × dmin × d
//   2. Final output: accumulated sum - bias correction

#if defined(__riscv_v)
inline void ggml_gemv_q4_K_8x8_q8_K_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                         const void * GGML_RESTRICT vx,
                                         const void * GGML_RESTRICT vy,
                                         int nr, int nc) {
    constexpr int qk = QK_K;
    const int nb = n / qk;
    constexpr int ncols_interleaved = 8;
    constexpr int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    assert(nr == 1);

    GGML_UNUSED(bs);

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        float acc_f32[ncols_interleaved] = {0};
        float bias_acc[ncols_interleaved] = {0};

        for (int b = 0; b < nb; b++) {
            // Load d[8] and dmin[8] and convert to float
            float q4_d[ncols_interleaved];
            float q4_dmin[ncols_interleaved];
            for (int j = 0; j < ncols_interleaved; j++) {
                q4_d[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].d[j]);
                q4_dmin[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].dmin[j]);
            }

            float q8_d = q8_ptr[b].d;
            float sb_scale[ncols_interleaved];
            float sb_min[ncols_interleaved];
            for (int j = 0; j < ncols_interleaved; j++) {
                sb_scale[j] = q4_d[j] * q8_d;
                sb_min[j] = q4_dmin[j] * q8_d;
            }

            // Compute bsums pairwise (16 bsums -> 8)
            int16_t bsums_arr[8];
            const int16_t * bsums_ptr = q8_ptr[b].bsums;
            for (int i = 0; i < 8; i++) {
                bsums_arr[i] = bsums_ptr[2 * i] + bsums_ptr[2 * i + 1];
            }

            // Process 4 sub-blocks (QK_K / 64 = 4)
            // Each subblock: 24 bytes scales/mins
            for (int sb = 0; sb < QK_K / 64; sb++) {
                uint8_t scales_lo[ncols_interleaved];
                uint8_t scales_hi[ncols_interleaved];
                uint8_t mins_lo[ncols_interleaved];
                uint8_t mins_hi[ncols_interleaved];

                decode_q_Kx8_6bit_scales_rvv(&q4_ptr[b].scales[sb * 24], scales_lo, mins_lo);
                decode_q_Kx8_6bit_scales_rvv(&q4_ptr[b].scales[sb * 24 + 12], scales_hi, mins_hi);

                // Process 4 k iterations per sb
                for (int k_local = 0; k_local < 4; k_local++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        int32_t sumi_lo = 0;
                        int32_t sumi_hi = 0;

                        for (int i = 0; i < blocklen; i++) {
                            const int total_k = sb * 4 + k_local;
                            const int q4_idx = total_k * ncols_interleaved * blocklen + j * blocklen + i;

                            const int q8_idx_lo = sb * 64 + k_local * blocklen + i;
                            const int q8_idx_hi = sb * 64 + k_local * blocklen + i + 32;

                            const int v0 = q4_ptr[b].qs[q4_idx] & 0xF;
                            const int v1 = q4_ptr[b].qs[q4_idx] >> 4;

                            sumi_lo += v0 * q8_ptr[b].qs[q8_idx_lo];
                            sumi_hi += v1 * q8_ptr[b].qs[q8_idx_hi];
                        }

                        acc_f32[j] += (float)(sumi_lo * scales_lo[j] + sumi_hi * scales_hi[j]) * sb_scale[j];
                    }
                }

                // Bias accumulation
                for (int j = 0; j < ncols_interleaved; j++) {
                    bias_acc[j] += (float)mins_lo[j] * bsums_arr[sb * 2] * sb_min[j];
                    bias_acc[j] += (float)mins_hi[j] * bsums_arr[sb * 2 + 1] * sb_min[j];
                }
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = acc_f32[j] - bias_acc[j];
        }
    }
}
#endif

// =============================================================================
// Wrapper function (VLEN detection)
// =============================================================================
inline void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx,
                                    const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
#if defined(__riscv_v)
    if (__riscv_vlenb() >= 64) {  // VLEN >= 512
        ggml_gemv_q4_K_8x8_q8_K_rvv(n, s, bs, vx, vy, nr, nc);
        return;
    }
#endif
    // Fallback to generic (scalar) implementation
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}