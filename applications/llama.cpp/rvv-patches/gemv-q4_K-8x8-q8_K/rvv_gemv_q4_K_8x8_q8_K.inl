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
// Scalar (generic) reference implementation (matching ARM NEON algorithm exactly)
// =============================================================================
static void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                            const void * GGML_RESTRICT vx,
                                            const void * GGML_RESTRICT vy,
                                            int nr, int nc) {
    constexpr int qk = QK_K;
    const int nb = n / qk;
    constexpr int ncols_interleaved = 8;
    constexpr int col_pairs = ncols_interleaved / 2;  // 4

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    // Process 8 columns in parallel
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        // Two float32x4 accumulators: cols 0-3 and cols 4-7
        float acc_f32[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};

        for (int b = 0; b < nb; b++) {
            // Load scales
            float q4_d[8], q4_dmin[8];
            for (int j = 0; j < 8; j++) {
                q4_d[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].d[j]);
                q4_dmin[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].dmin[j]);
            }
            float q8_d = q8_ptr[b].d;

            // Combined scales for cols 0-3 and 4-7
            float sb_scale_0[4], sb_scale_1[4];  // q4_d * q8_d
            float sb_min_0[4], sb_min_1[4];      // q4_dmin * q8_d
            for (int j = 0; j < 4; j++) {
                sb_scale_0[j] = q4_d[j] * q8_d;
                sb_scale_1[j] = q4_d[j + 4] * q8_d;
                sb_min_0[j] = q4_dmin[j] * q8_d;
                sb_min_1[j] = q4_dmin[j + 4] * q8_d;
            }

            // Pairwise add bsums (16 values -> 8 values)
            int16_t bsums_arr[8];
            for (int i = 0; i < 8; i++) {
                bsums_arr[i] = q8_ptr[b].bsums[2*i] + q8_ptr[b].bsums[2*i+1];
            }

            // Bias accumulators: [0] for cols 0-3, [1] for cols 4-7
            int32_t bias_acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

            // Process 4 subblocks (QK_K/64 = 4)
            for (int sb = 0; sb < qk / 64; sb++) {
                // Decode scales/mins for low and high nibbles
                int16_t mins_lo[8], mins_hi[8];
                int8_t scales_lo[8], scales_hi[8];

                decode_q_Kx8_6bit_scales_scalar(&q4_ptr[b].scales[sb * 24], mins_lo, scales_lo);
                decode_q_Kx8_6bit_scales_scalar(&q4_ptr[b].scales[sb * 24 + 12], mins_hi, scales_hi);

                // Initialize per-column-pair accumulators
                int32_t acc_lo[col_pairs][4];  // [cp][sum_idx]
                int32_t acc_hi[col_pairs][4];
                for (int cp = 0; cp < col_pairs; cp++) {
                    for (int k = 0; k < 4; k++) {
                        acc_lo[cp][k] = 0;
                        acc_hi[cp][k] = 0;
                    }
                }

                const uint8_t * q4_base = q4_ptr[b].qs + sb * qk;
                const int8_t * q8_base = q8_ptr[b].qs + sb * 64;

                // Process each column pair
                for (int cp = 0; cp < col_pairs; cp++) {
                    // Load 4 q4 vectors for this column pair
                    // Each vector is 16 bytes containing 32 nibbles (16 low + 16 high)
                    // Layout: q4_base + 16*cp, then +64, +128, +192

                    for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                        const uint8_t * q4_vec = q4_base + 16 * cp + vec_idx * 64;

                        // Compute 4 dot products per vector
                        // Each dot product uses 4 nibbles × 4 q8 values
                        // q8 broadcast: q8_base[vec_idx*8 .. vec_idx*8+7] duplicated

                        for (int sum_idx = 0; sum_idx < 4; sum_idx++) {
                            // Determine nibble and q8 indices
                            // ARM NEON vdotq groups: nibbles[0..3], nibbles[4..7], nibbles[8..11], nibbles[12..15]
                            // q8_qs[vec_idx] broadcast: q8_base[vec_idx*8..vec_idx*8+7] duplicated twice
                            // So q8 indices: sum_idx%2 gives 0→0-3, 1→4-7 (both for original and duplicate)

                            int nibble_base = sum_idx * 4;
                            int q8_half = (sum_idx % 2) * 4;  // 0 for sum_idx=0,2; 4 for sum_idx=1,3

                            // Low nibbles (elements 0-31 of subblock)
                            int32_t sum_lo = 0;
                            for (int n = 0; n < 4; n++) {
                                uint8_t nibble = q4_vec[nibble_base + n] & 0x0F;
                                int8_t q8_val = q8_base[vec_idx * 8 + q8_half + n];
                                sum_lo += nibble * q8_val;
                            }
                            acc_lo[cp][sum_idx] += sum_lo;

                            // High nibbles (elements 32-63 of subblock)
                            int32_t sum_hi = 0;
                            for (int n = 0; n < 4; n++) {
                                uint8_t nibble = q4_vec[nibble_base + n] >> 4;
                                int8_t q8_val = q8_base[vec_idx * 8 + 32 + q8_half + n];
                                sum_hi += nibble * q8_val;
                            }
                            acc_hi[cp][sum_idx] += sum_hi;
                        }
                    }
                }

                // Apply scales and accumulate to float
                // Process column pair groups: p=0 (cp 0,1 for cols 0-3), p=2 (cp 2,3 for cols 4-7)
                for (int i = 0, p = 0; p < col_pairs; i++, p += 2) {
                    // Get scales for this column group
                    // group_scales_lo[k] corresponds to column k (for p=0) or column k+4 (for p=2)
                    int16_t group_scales_lo[4], group_scales_hi[4];
                    float * sb_scale = (p == 0) ? sb_scale_0 : sb_scale_1;

                    // scales_lo/hi have 8 values: indices 0-3 for cols 0-3, 4-7 for cols 4-7
                    // For p=0: use scales_lo[0..3] and scales_hi[0..3]
                    // For p=2: use scales_lo[4..7] and scales_hi[4..7]
                    int scale_offset = (p == 0) ? 0 : 4;
                    for (int k = 0; k < 4; k++) {
                        group_scales_lo[k] = scales_lo[scale_offset + k];
                        group_scales_hi[k] = scales_hi[scale_offset + k];
                    }

                    // Pairwise add: sum_lo[k] = acc_lo[p][k] + acc_lo[p+1][k]
                    int32_t sum_lo[4], sum_hi[4];
                    for (int k = 0; k < 4; k++) {
                        sum_lo[k] = acc_lo[p][k] + acc_lo[p+1][k];
                        sum_hi[k] = acc_hi[p][k] + acc_hi[p+1][k];
                    }

                    // Multiply by scales and accumulate
                    for (int k = 0; k < 4; k++) {
                        float scaled_lo = (float)(group_scales_lo[k] * sum_lo[k]);
                        float scaled_hi = (float)(group_scales_hi[k] * sum_hi[k]);
                        acc_f32[i][k] += sb_scale[k] * (scaled_lo + scaled_hi);
                    }
                }

                // Bias accumulation
                // Each pair of subblocks share the same bsums
                // bsums_arr[2*sb] for low nibbles (cols 0-3), bsums_arr[2*sb+1] for high nibbles (cols 4-7)

                // cols 0-3 bias: mins_lo[0..3] × bsums_arr[2*sb] + mins_hi[0..3] × bsums_arr[2*sb+1]
                for (int k = 0; k < 4; k++) {
                    bias_acc[0][k] += mins_lo[k] * bsums_arr[2*sb];
                    bias_acc[0][k] += mins_hi[k] * bsums_arr[2*sb+1];
                }

                // cols 4-7 bias: mins_lo[4..7] × bsums_arr[2*sb] + mins_hi[4..7] × bsums_arr[2*sb+1]
                for (int k = 0; k < 4; k++) {
                    bias_acc[1][k] += mins_lo[k + 4] * bsums_arr[2*sb];
                    bias_acc[1][k] += mins_hi[k + 4] * bsums_arr[2*sb+1];
                }
            }

            // Final bias subtraction: acc_f32 -= bias_acc * sb_min
            for (int k = 0; k < 4; k++) {
                acc_f32[0][k] -= (float)bias_acc[0][k] * sb_min_0[k];
                acc_f32[1][k] -= (float)bias_acc[1][k] * sb_min_1[k];
            }
        }

        // Store results
        int base = x * ncols_interleaved;
        for (int k = 0; k < 4; k++) {
            s[base + k] = acc_f32[0][k];
            s[base + k + 4] = acc_f32[1][k];
        }
    }
}

// =============================================================================
// RVV implementation (VLEN=512)
// =============================================================================
#if defined(__riscv_v_intrinsic)
inline void ggml_gemv_q4_K_8x8_q8_K_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                         const void * GGML_RESTRICT vx,
                                         const void * GGML_RESTRICT vy,
                                         int nr, int nc) {
    constexpr int qk = QK_K;
    const int nb = n / qk;
    constexpr int ncols_interleaved = 8;
    constexpr int col_pairs = ncols_interleaved / 2;  // 4

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    assert(nr == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    const size_t vl4 = __riscv_vsetvl_e32m1(4);

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    // Process 8 columns in parallel, split into 2 groups of 4
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        float acc_f32[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        float bias_acc_f32[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};

        for (int b = 0; b < nb; b++) {
            // Load scales and convert to float
            float q4_d[ncols_interleaved], q4_dmin[ncols_interleaved];
            for (int j = 0; j < ncols_interleaved; j++) {
                q4_d[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].d[j]);
                q4_dmin[j] = GGML_CPU_FP16_TO_FP32(q4_ptr[b].dmin[j]);
            }
            float q8_d = q8_ptr[b].d;

            // Combined scales
            float sb_scale_0[4], sb_scale_1[4];
            float sb_min_0[4], sb_min_1[4];
            for (int j = 0; j < 4; j++) {
                sb_scale_0[j] = q4_d[j] * q8_d;
                sb_scale_1[j] = q4_d[j + 4] * q8_d;
                sb_min_0[j] = q4_dmin[j] * q8_d;
                sb_min_1[j] = q4_dmin[j + 4] * q8_d;
            }

            // Precompute bsums (pairwise add)
            int16_t bsums_arr[8];
            for (int i = 0; i < 8; i++) {
                bsums_arr[i] = q8_ptr[b].bsums[2*i] + q8_ptr[b].bsums[2*i+1];
            }

            // Process 4 subblocks (QK_K/64 = 4)
            for (int sb = 0; sb < qk / 64; sb++) {
                // Decode scales/mins for low and high nibbles
                int16_t mins_lo[8], mins_hi[8];
                int8_t scales_lo[8], scales_hi[8];

                decode_q_Kx8_6bit_scales_scalar(&q4_ptr[b].scales[sb * 24], mins_lo, scales_lo);
                decode_q_Kx8_6bit_scales_scalar(&q4_ptr[b].scales[sb * 24 + 12], mins_hi, scales_hi);

                // Initialize per-column-pair accumulators (same as scalar)
                int32_t acc_lo[col_pairs][4];
                int32_t acc_hi[col_pairs][4];
                for (int cp = 0; cp < col_pairs; cp++) {
                    for (int k = 0; k < 4; k++) {
                        acc_lo[cp][k] = 0;
                        acc_hi[cp][k] = 0;
                    }
                }

                const uint8_t * q4_base = q4_ptr[b].qs + sb * qk;
                const int8_t * q8_base = q8_ptr[b].qs + sb * 64;

                // Process each column pair (same algorithm as scalar)
                for (int cp = 0; cp < col_pairs; cp++) {
                    for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                        const uint8_t * q4_vec = q4_base + 16 * cp + vec_idx * 64;

                        for (int sum_idx = 0; sum_idx < 4; sum_idx++) {
                            int nibble_base = sum_idx * 4;
                            int q8_half = (sum_idx % 2) * 4;  // 0 for sum_idx=0,2; 4 for sum_idx=1,3

                            int32_t sum_lo = 0, sum_hi = 0;
                            for (int n = 0; n < 4; n++) {
                                uint8_t nibble_lo = q4_vec[nibble_base + n] & 0x0F;
                                uint8_t nibble_hi = q4_vec[nibble_base + n] >> 4;
                                int8_t q8_val_lo = q8_base[vec_idx * 8 + q8_half + n];
                                int8_t q8_val_hi = q8_base[vec_idx * 8 + 32 + q8_half + n];
                                sum_lo += nibble_lo * q8_val_lo;
                                sum_hi += nibble_hi * q8_val_hi;
                            }
                            acc_lo[cp][sum_idx] += sum_lo;
                            acc_hi[cp][sum_idx] += sum_hi;
                        }
                    }
                }

                // Apply scales and accumulate to float (using RVV for final operations)
                for (int i = 0, p = 0; p < col_pairs; i++, p += 2) {
                    int scale_offset = (p == 0) ? 0 : 4;
                    float * sb_scale = (p == 0) ? sb_scale_0 : sb_scale_1;

                    // Pairwise add and scale application
                    for (int k = 0; k < 4; k++) {
                        int32_t sum_lo = acc_lo[p][k] + acc_lo[p+1][k];
                        int32_t sum_hi = acc_hi[p][k] + acc_hi[p+1][k];
                        float scaled_lo = (float)(scales_lo[scale_offset + k] * sum_lo);
                        float scaled_hi = (float)(scales_hi[scale_offset + k] * sum_hi);
                        acc_f32[i][k] += sb_scale[k] * (scaled_lo + scaled_hi);
                    }
                }

                // Bias accumulation
                for (int k = 0; k < 4; k++) {
                    bias_acc_f32[0][k] += (float)(mins_lo[k] * bsums_arr[2*sb] + mins_hi[k] * bsums_arr[2*sb+1]);
                    bias_acc_f32[1][k] += (float)(mins_lo[k + 4] * bsums_arr[2*sb] + mins_hi[k + 4] * bsums_arr[2*sb+1]);
                }
            }

            // Final bias subtraction (using RVV vectors)
            for (int i = 0; i < 2; i++) {
                float * sb_min = (i == 0) ? sb_min_0 : sb_min_1;
                vfloat32m1_t acc_vec = __riscv_vle32_v_f32m1(acc_f32[i], vl4);
                vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_acc_f32[i], vl4);
                vfloat32m1_t sb_min_vec = __riscv_vle32_v_f32m1(sb_min, vl4);

                // acc_f32 = acc_f32 - bias_acc * sb_min
                vfloat32m1_t scaled_bias = __riscv_vfmul_vv_f32m1(bias_vec, sb_min_vec, vl4);
                acc_vec = __riscv_vfsub_vv_f32m1(acc_vec, scaled_bias, vl4);

                __riscv_vse32_v_f32m1(acc_f32[i], acc_vec, vl4);
            }
        }

        // Store results
        int base = x * ncols_interleaved;
        for (int k = 0; k < 4; k++) {
            s[base + k] = acc_f32[0][k];
            s[base + k + 4] = acc_f32[1][k];
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
#if defined(__riscv_v_intrinsic)
    if (__riscv_vlenb() >= 64) {
        ggml_gemv_q4_K_8x8_q8_K_rvv(n, s, bs, vx, vy, nr, nc);
        return;
    }
#endif
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}