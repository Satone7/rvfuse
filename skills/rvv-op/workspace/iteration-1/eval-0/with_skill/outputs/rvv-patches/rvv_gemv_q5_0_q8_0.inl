// rvv_gemv_q5_0_q8_0.inl -- RVV implementation of Q5_0 x Q8_0 GEMV (16x1 tile)
//
// Single source of truth for the RVV vectorized kernel. Included by:
//   - arch/riscv/repack.cpp  (via patch, production build)
//   - tests/test.cpp  (correctness test)
//
// Prerequisites before including this file:
//   - QK5_0, QK8_0 must be defined (both 32)
//   - block_q5_0, block_q8_0 structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED must be defined
//   - GGML_CPU_FP16_TO_FP32 must be defined
//   - On RVV targets: <riscv_vector.h> must be included
//   - On non-RVV targets: __riscv_v_intrinsic will be undefined,
//     so the fallback path runs instead
//
// Algorithm:
//   This is a column-sequential GEMV kernel for Q5_0 weights x Q8_0 activations.
//   Unlike Q4_0 which uses interleaved block formats (block_q4_0x16), Q5_0 has
//   no upstream interleaved format. Each column's weight blocks are stored
//   contiguously.
//
//   Inner loop (per block, per column):
//   1. Load 16 bytes of packed nibbles (qs) from Q5_0 weight block
//   2. Load 4 bytes of high bits (qh) from Q5_0 weight block
//   3. Unpack: lower nibbles [0..15] and upper nibbles [16..31]
//   4. Sign-extend via masked subtract 0x10 (where qh bit is 0)
//   5. Load 32 int8 values from Q8_0 activation block
//   6. Widening MAC: int8 x int8 -> int16, then horizontal sum -> int32
//   7. Scale by combined delta (weight.d * activation.d) and accumulate

#include <cassert>
#include <cstring>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// Q5_0 block unpack + dot product (per-block scalar, used as building block)
// =============================================================================
// Unpacks one Q5_0 block into 32 signed int8 values and computes dot product
// with one Q8_0 block. Returns the integer dot product (before scale).
//
// This scalar helper is used by both the generic and RVV implementations.
// It follows the exact same algorithm as ggml_vec_dot_q5_0_q8_0_generic.

static inline int32_t q5_0_q8_0_dot_block(const block_q5_0 * GGML_RESTRICT b,
                                           const block_q8_0 * GGML_RESTRICT a) {
    const int qk = QK8_0;  // 32
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));

    int32_t sumi = 0;
    for (int j = 0; j < qk / 2; j++) {
        const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
        const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));
        const int32_t x0 = (int8_t)(((b->qs[j] & 0x0F) | xh_0) - 16);
        const int32_t x1 = (int8_t)(((b->qs[j] >>   4) | xh_1) - 16);
        sumi += x0 * a->qs[j];
        sumi += x1 * a->qs[j + qk / 2];
    }
    return sumi;
}

// =============================================================================
// Q5_0 block unpack + dot product (RVV vectorized, single block)
// =============================================================================
// Vectorized version of q5_0_q8_0_dot_block using RVV intrinsics.
// Uses the same algorithm as rvv_vec_dot_q5_0_q8_0.inl but returns int32.
//
// On VLEN >= 256: uses m1-based path (matches rvv_vec_dot_q5_0_q8_0.inl)
// On VLEN = 128: falls back to scalar helper

#if defined(__riscv_v_intrinsic)
static inline int32_t q5_0_q8_0_dot_block_rvv(const block_q5_0 * GGML_RESTRICT b,
                                              const block_q8_0 * GGML_RESTRICT a) {
    const size_t vl = QK8_0;  // 32 elements
    const size_t vl_half = QK8_0 / 2;  // 16 elements
    const size_t vlenb = __riscv_vlenb();

    if (vlenb >= 32) {
        // VLEN >= 256: optimized m1-based path
        // Load 16 bytes of packed nibbles using mf2
        vuint8mf2_t raw = __riscv_vle8_v_u8mf2(b->qs, vl_half);

        // Split into lower nibbles [0,15] and upper nibbles [0,15]
        vint8mf2_t v0l = __riscv_vreinterpret_v_u8mf2_i8mf2(
            __riscv_vand_vx_u8mf2(raw, 0x0F, vl_half));
        vint8mf2_t v0h = __riscv_vreinterpret_v_u8mf2_i8mf2(
            __riscv_vsrl_vx_u8mf2(raw, 4, vl_half));

        // Interleave into m1: [low0..low15, high0..high15]
        vint8m1_t v0c = __riscv_vlmul_ext_v_i8mf2_i8m1(v0l);
        v0c = __riscv_vslideup_vx_i8m1(v0c,
            __riscv_vlmul_ext_v_i8mf2_i8m1(v0h), qk / 2, vl);

        // Sign extension: load qh as b8 mask, invert, masked subtract 0x10
        vbool8_t qh = __riscv_vlm_v_b8(b->qh, vl);
        qh = __riscv_vmnand_mm_b8(qh, qh, vl);
        vint8m1_t v0f = __riscv_vsub_vx_i8m1_mu(qh, v0c, v0c, 0x10, vl);

        // Dot product with Q8_0 activation block
        vint8m1_t v1 = __riscv_vle8_v_i8m1(a->qs, vl);
        vint16m2_t mul = __riscv_vwmul_vv_i16m2(v0f, v1, vl);

        // Horizontal reduction: sum i16 -> i32
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(mul, zero, vl);
        return __riscv_vmv_x_s_i32m1_i32(sum);
    } else {
        // VLEN = 128: scalar fallback
        return q5_0_q8_0_dot_block(b, a);
    }
}
#endif // __riscv_v_intrinsic

// =============================================================================
// RVV implementation: ggml_gemv_q5_0_q8_0 (16x1 tile, VLEN >= 512)
// =============================================================================
// Computes y = W * x where:
//   - W is a (nc x n) matrix of Q5_0 quantized weights (column-major)
//   - x is a (n x 1) vector of Q8_0 quantized activations
//   - y is a (nc x 1) output vector of FP32 values
//
// The 16x1 tile processes 16 output columns per iteration.
// For each column, the inner loop iterates over n/QK8_0 blocks,
// computing the dot product of one Q5_0 weight block with the
// corresponding Q8_0 activation block, scaling by combined deltas.

#if defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
static void ggml_gemv_q5_0_q8_0_rvv(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx,
                                    const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk;
    const int ncols_tile = 16;  // Process 16 columns per tile

    assert(n % qk == 0);
    assert(nr == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    const size_t vlenb = __riscv_vlenb();

    // VLEN >= 512 required for 16-element FP32 vectors (m2, 16 * 4 = 64 bytes)
    if (vlenb < 64) {
        ggml_gemv_q5_0_q8_0_generic(n, s, bs, vx, vy, nr, nc);
        return;
    }

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    const block_q5_0 * b_ptr_base = (const block_q5_0 *) vx;

    // Process groups of 16 columns
    for (int x = 0; x < nc; x += ncols_tile) {
        int cols_this_tile = (nc - x < ncols_tile) ? (nc - x) : ncols_tile;
        const size_t vl_cols = cols_this_tile;

        // 16-column float accumulator (one float per column)
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, vl_cols);

        for (int l = 0; l < nb; l++) {
            // Integer accumulator for 16 columns
            vint32m2_t sumi_vec = __riscv_vmv_v_x_i32m2(0, vl_cols);

            // Compute dot product for each column in this tile
            for (int col = 0; col < cols_this_tile; col++) {
                const block_q5_0 * b_ptr = b_ptr_base + (x + col) * nb + l;

                // Vectorized Q5_0 x Q8_0 dot product for one block
                int32_t dot = q5_0_q8_0_dot_block_rvv(b_ptr, &a_ptr[l]);

                // Insert into vector accumulator at lane 'col'
                // Use masked add to set lane 'col'
                vbool32_t lane_mask = __riscv_vmv_v_x_b32(1, vl_cols);
                lane_mask = __riscv_vslidedown_vx_b32(lane_mask, col, vl_cols);
                sumi_vec = __riscv_vadd_vx_i32m2_mu(lane_mask, sumi_vec, sumi_vec, dot, vl_cols);
            }

            // Convert integer accumulator to float
            vfloat32m2_t facc = __riscv_vfcvt_f_x_v_f32m2(sumi_vec, vl_cols);

            // Apply activation scale (shared across all columns)
            const float a_scale = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);

            // Load weight scales (per-column FP16 deltas)
            ggml_half b_d[16];
            for (int col = 0; col < cols_this_tile; col++) {
                const block_q5_0 * b_ptr = b_ptr_base + (x + col) * nb + l;
                b_d[col] = b_ptr->d;
            }

            // Convert FP16 weight scales to FP32 using Zvfh
            // Load as FP16 and widen to FP32
            vfloat16m1_t b_d_f16 = __riscv_vle16_v_f16m1(b_d, vl_cols);
            vfloat32m2_t b_scales = __riscv_vfwcvt_f_f_v_f32m2(b_d_f16, vl_cols);

            // Combined scale: a_scale * b_scales
            vfloat32m2_t combined_scale = __riscv_vfmul_vf_f32m2(b_scales, a_scale, vl_cols);

            // Multiply dot products by combined scale and accumulate
            sumf = __riscv_vfmacc_vv_f32m2(sumf, facc, combined_scale, vl_cols);
        }

        // Store results for this group of columns
        __riscv_vse32_v_f32m2(s + x, sumf, vl_cols);
    }
}
#endif // __riscv_v_intrinsic && __riscv_zvfh

// =============================================================================
// Wrapper function: selects RVV or generic at compile time
// =============================================================================

inline void ggml_gemv_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                                const void * GGML_RESTRICT vx,
                                const void * GGML_RESTRICT vy,
                                int nr, int nc) {
#if defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    ggml_gemv_q5_0_q8_0_rvv(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_q5_0_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

// =============================================================================
// Generic (scalar) implementation
// =============================================================================
// Column-sequential scalar GEMV. This is the reference implementation
// that matches the algorithm of ggml_vec_dot_q5_0_q8_0_generic but
// extended to process multiple output columns.

#if !defined(__riscv_v_intrinsic) || !defined(__riscv_zvfh)
void ggml_gemv_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx,
                                  const void * GGML_RESTRICT vy,
                                  int nr, int nc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nr == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(nr);

    const block_q5_0 * b_ptr_base = (const block_q5_0 *) vx;
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;

    // Process each output column
    for (int col = 0; col < nc; col++) {
        float sumf = 0.0f;

        for (int l = 0; l < nb; l++) {
            const block_q5_0 * b_ptr = b_ptr_base + col * nb + l;
            const block_q8_0 * y = a_ptr + l;

            // Compute dot product using the scalar helper
            int32_t sumi = q5_0_q8_0_dot_block(b_ptr, y);

            // Apply scale factors
            sumf += (GGML_CPU_FP16_TO_FP32(b_ptr->d) * GGML_CPU_FP16_TO_FP32(y->d)) * sumi;
        }

        s[col] = sumf;
    }
}
#endif // !__riscv_v_intrinsic || !__riscv_zvfh