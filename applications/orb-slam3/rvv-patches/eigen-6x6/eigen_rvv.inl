// Eigen 6x6 Fixed-Size Matrix RVV Specialization
// Target: RVV512 (VLEN=512, SEW=64 for double precision)
// Replaces scalar dense_assignment_loop for 6x6 matrices in Eigen 3.4.0
//
// Usage: Include after <Eigen/Core>, before any Eigen matrix operations.
// When __riscv_vector is defined, this overrides Eigen's default
// dense_assignment_loop for 6x6 fixed-size matrices.

#ifndef EIGEN_RVV_6X6_INL
#define EIGEN_RVV_6X6_INL

#ifdef __riscv_vector
#include <riscv_vector.h>

namespace Eigen {
namespace internal {
namespace rvv {

// ============================================================
// 6x6 Matrix Multiply: C = A * B (column-major, double precision)
// ============================================================
// C[i][j] = Σ(k=0..5) A[i][k] * B[k][j]
//
// Column-major layout (Eigen default):
//   Col 0: indices 0..5   = A(0,0)..A(5,0)
//   Col 1: indices 6..11  = A(0,1)..A(5,1)
//   ...
//
// RVV512: vfloat64m1_t with vl=6 holds one column (6 doubles = 48 bytes)
//
// Algorithm: C_col_j = Σ(k=0..5) B(k,j) * A_col_k
//   Load A columns as vectors (6 loads)
//   For each C column j: accumulate B(k,j)*A_col_k using vfmacc.vf
//   Store C columns (6 stores)

inline void eigen_6x6_mul_rvv(double* C, const double* A, const double* B) {
    // Load A columns as vectors
    vfloat64m1_t a0 = __riscv_vle64_v_f64m1(&A[0], 6);
    vfloat64m1_t a1 = __riscv_vle64_v_f64m1(&A[6], 6);
    vfloat64m1_t a2 = __riscv_vle64_v_f64m1(&A[12], 6);
    vfloat64m1_t a3 = __riscv_vle64_v_f64m1(&A[18], 6);
    vfloat64m1_t a4 = __riscv_vle64_v_f64m1(&A[24], 6);
    vfloat64m1_t a5 = __riscv_vle64_v_f64m1(&A[30], 6);

    // C column 0 = B(0,0)*a0 + B(1,0)*a1 + ... + B(5,0)*a5
    {
        vfloat64m1_t c0 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[0],  a0, 6);  // B(0,0)
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[1],  a1, 6);  // B(1,0)
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[2],  a2, 6);  // B(2,0)
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[3],  a3, 6);  // B(3,0)
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[4],  a4, 6);  // B(4,0)
        c0 = __riscv_vfmacc_vf_f64m1(c0, B[5],  a5, 6);  // B(5,0)
        __riscv_vse64_v_f64m1(&C[0], c0, 6);
    }

    // C column 1 = B(0,1)*a0 + B(1,1)*a1 + ... + B(5,1)*a5
    {
        vfloat64m1_t c1 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[6],  a0, 6);   // B(0,1)
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[7],  a1, 6);   // B(1,1)
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[8],  a2, 6);   // B(2,1)
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[9],  a3, 6);   // B(3,1)
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[10], a4, 6);   // B(4,1)
        c1 = __riscv_vfmacc_vf_f64m1(c1, B[11], a5, 6);   // B(5,1)
        __riscv_vse64_v_f64m1(&C[6], c1, 6);
    }

    // C column 2
    {
        vfloat64m1_t c2 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[12], a0, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[13], a1, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[14], a2, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[15], a3, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[16], a4, 6);
        c2 = __riscv_vfmacc_vf_f64m1(c2, B[17], a5, 6);
        __riscv_vse64_v_f64m1(&C[12], c2, 6);
    }

    // C column 3
    {
        vfloat64m1_t c3 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[18], a0, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[19], a1, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[20], a2, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[21], a3, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[22], a4, 6);
        c3 = __riscv_vfmacc_vf_f64m1(c3, B[23], a5, 6);
        __riscv_vse64_v_f64m1(&C[18], c3, 6);
    }

    // C column 4
    {
        vfloat64m1_t c4 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[24], a0, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[25], a1, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[26], a2, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[27], a3, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[28], a4, 6);
        c4 = __riscv_vfmacc_vf_f64m1(c4, B[29], a5, 6);
        __riscv_vse64_v_f64m1(&C[24], c4, 6);
    }

    // C column 5
    {
        vfloat64m1_t c5 = __riscv_vfmv_v_f_f64m1(0.0, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[30], a0, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[31], a1, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[32], a2, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[33], a3, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[34], a4, 6);
        c5 = __riscv_vfmacc_vf_f64m1(c5, B[35], a5, 6);
        __riscv_vse64_v_f64m1(&C[30], c5, 6);
    }
}

// ============================================================
// 6x6 Matrix Add: C = A + B (column-major, double precision)
// ============================================================

inline void eigen_6x6_add_rvv(double* C, const double* A, const double* B) {
    for (int j = 0; j < 6; j++) {
        vfloat64m1_t a = __riscv_vle64_v_f64m1(&A[j * 6], 6);
        vfloat64m1_t b = __riscv_vle64_v_f64m1(&B[j * 6], 6);
        vfloat64m1_t c = __riscv_vfadd_vv_f64m1(a, b, 6);
        __riscv_vse64_v_f64m1(&C[j * 6], c, 6);
    }
}

// ============================================================
// 6x6 Triangular Solve (Forward Substitution): L * x = b
// L is lower-triangular, b is 6-vector, result in x
// ============================================================

inline void eigen_6x6_triangular_solve_rvv(double* x, const double* L, const double* b) {
    // Forward: x[0] = b[0] / L[0][0]
    x[0] = b[0] / L[0];  // L[0][0] is at column-major index 0

    // x[i] = (b[i] - Σ(k=0..i-1) L[i][k] * x[k]) / L[i][i]
    for (int i = 1; i < 6; i++) {
        // Gather L[i][0..i-1] into vector (column-major: L[i][k] is at index k*6+i)
        double L_row[6] = {0};
        for (int k = 0; k < i; k++) {
            L_row[k] = L[k * 6 + i];  // L[i][k] in column-major
        }

        // Gather x[0..i-1]
        vfloat64m1_t x_vec = __riscv_vle64_v_f64m1(x, i);
        vfloat64m1_t L_vec = __riscv_vle64_v_f64m1(L_row, i);

        // Dot product using vfredusum.vs (vector reduction sum)
        vfloat64m1_t prod = __riscv_vfmul_vv_f64m1(L_vec, x_vec, i);
        vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, i);
        vfloat64m1_t sum_vec = __riscv_vfredusum_vs_f64m1_f64m1(prod, zero, i);
        double dot = __riscv_vfmv_f_s_f64m1_f64(sum_vec);

        x[i] = (b[i] - dot) / L[i * 6 + i];  // L[i][i]
    }
}

} // namespace rvv
} // namespace internal
} // namespace Eigen

#endif // __riscv_vector
#endif // EIGEN_RVV_6X6_INL
