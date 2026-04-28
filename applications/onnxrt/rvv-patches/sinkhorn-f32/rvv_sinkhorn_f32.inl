// rvv_sinkhorn_f32.inl — RVV-optimized Sinkhorn normalization for float32.
//
// Accelerates the row/column normalization loop in Sinkhorn optimal transport.
// Core operation per iteration: normalize each row, then each column.
//
// Pattern: for each row/col i:
//   sum = Σ_j A[i,j]
//   for j: A[i,j] /= sum
//
// This is a memory-bound operation (reduction + broadcast), but RVV
// accelerates both the sum reduction (vfredusum) and the division (vfdiv).
//
// Target shapes (SuperGlue): (Na+1, Nb+1) where Na, Nb ≤ 1024
// 100 iterations × 2 directions (row, col) = 200 full-pass normalizations
// Each pass: ~(Na+1)*(Nb+1) ≈ 1M elements → 200M total operations
//
// Copyright (c) RVFuse Project. Licensed under the MIT License.

#ifndef RVV_SINKHORN_F32_INL
#define RVV_SINKHORN_F32_INL

#include <riscv_vector.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

// RVV-accelerated row normalization: each row divided by its sum.
// Matrix stored row-major: A[row * stride + col]
static inline void sinkhornNormalizeRows_rvv(
    float* A,                // (rows, cols) row-major, modified in-place
    size_t rows,             // Number of rows
    size_t cols,             // Number of columns
    size_t stride)           // Stride between rows (>= cols)
{
    for (size_t r = 0; r < rows; r++) {
        float* row_ptr = A + r * stride;

        // Step 1: Compute row sum via vectorized reduction
        float row_sum;
        {
            size_t avl = cols;
            const float* ptr = row_ptr;
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t v_sum = v_zero;

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);
                vfloat32m1_t v_row = __riscv_vle32_v_f32m1(ptr, vl);
                v_sum = __riscv_vfredusum_vs_f32m1_f32(v_sum, v_row, v_sum, vl);
                ptr += vl;
                avl -= vl;
            }
            row_sum = __riscv_vfmv_f_s_f32m1_f32(v_sum);
        }

        // Clamp to avoid division by zero
        if (row_sum < 1e-12f) row_sum = 1e-12f;

        // Step 2: Divide each element by row_sum
        float inv_sum = 1.0f / row_sum;
        {
            size_t avl = cols;
            float* ptr = row_ptr;
            vfloat32m1_t v_inv = __riscv_vfmv_v_f_f32m1(inv_sum, __riscv_vsetvl_e32m1(cols));

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);
                vfloat32m1_t v_row = __riscv_vle32_v_f32m1(ptr, vl);
                vfloat32m1_t v_norm = __riscv_vfmul_vv_f32m1(v_row, v_inv, vl);
                __riscv_vse32_v_f32m1(ptr, v_norm, vl);
                ptr += vl;
                avl -= vl;
            }
        }
    }
}

// RVV-accelerated column normalization: each column divided by its sum.
// Matrix stored row-major: A[row * stride + col]
// Column access is strided — less efficient than row normalization,
// but RVV's indexed load/store (vluxei/vsuxei) can help with stride patterns.
static inline void sinkhornNormalizeCols_rvv(
    float* A,                // (rows, cols) row-major, modified in-place
    size_t rows,             // Number of rows
    size_t cols,             // Number of columns
    size_t stride)           // Stride between rows (>= cols)
{
    // For each column, compute sum and normalize
    // Strategy: process multiple columns simultaneously using strided access
    // Since column access is strided, we use strided load pattern

    for (size_t c = 0; c < cols; c++) {
        // Step 1: Compute column sum via strided reduction
        float col_sum = 0.0f;
        {
            // Strided access: load every stride-th element
            // For VL=16 at a time, we process contiguous chunks of columns
            // but within each column, access is strided
            size_t avl = rows;
            size_t r = 0;

            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t v_sum = v_zero;

            while (avl > 0) {
                size_t vl = __riscv_vsetvl_e32m1(avl);

                // Gather strided elements from column c
                // Use indexed load for stride access
                // For simplicity and portability, use scalar accumulation
                // for columns (the primary RVV speedup comes from row ops)
                for (size_t i = 0; i < vl && (r + i) < rows; i++) {
                    col_sum += A[(r + i) * stride + c];
                }
                r += vl;
                avl -= vl;
            }
        }

        if (col_sum < 1e-12f) col_sum = 1e-12f;
        float inv_sum = 1.0f / col_sum;

        // Step 2: Normalize column
        for (size_t r = 0; r < rows; r++) {
            A[r * stride + c] *= inv_sum;
        }
    }
}

// Full Sinkhorn iteration: row normalize + column normalize
// A: (rows, cols) row-major matrix, modified in-place
// This is the inner loop called 100 times.
static inline void sinkhornIterate_rvv(
    float* A,
    size_t rows,
    size_t cols,
    size_t stride)
{
    sinkhornNormalizeRows_rvv(A, rows, cols, stride);
    sinkhornNormalizeCols_rvv(A, rows, cols, stride);
}

#endif // RVV_SINKHORN_F32_INL
