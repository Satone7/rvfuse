// Test: Eigen 6x6 RVV vs Scalar Correctness
// Compile: clang++ --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
//   -march=rv64gcv_zvl512b -O3 \
//   -I<eigen_dir> -I<rvv-patches/eigen-6x6> \
//   test.cpp -o eigen6x6_test
// Run: qemu-riscv64 -cpu rv64,v=true,vlen=512 eigen6x6_test

#include <cstdio>
#include <cmath>
#include <cstdlib>

// Define before including Eigen to force scalar path for comparison
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Core>
#undef EIGEN_DONT_VECTORIZE

#ifdef __riscv_vector
// Include the RVV kernels (found via -I path to rvv-patches/eigen-6x6/)
#include "eigen_rvv.inl"
#endif

static bool approx_equal(double a, double b, double eps = 1e-12) {
    return fabs(a - b) < eps;
}

int main() {
    // Test 1: 6x6 Matrix Multiply
    {
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Random();
        Eigen::Matrix<double, 6, 6> B = Eigen::Matrix<double, 6, 6>::Random();
        Eigen::Matrix<double, 6, 6> C_scalar = A * B;  // Eigen scalar path

#ifdef __riscv_vector
        Eigen::Matrix<double, 6, 6> C_rvv;
        // Fill with zeros first
        C_rvv.setZero();
        Eigen::internal::rvv::eigen_6x6_mul_rvv(C_rvv.data(), A.data(), B.data());
#else
        auto C_rvv = C_scalar;  // No RVV available
#endif

        bool pass = true;
        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 6; i++)
                if (!approx_equal(C_scalar(i,j), C_rvv(i,j))) {
                    printf("FAIL: Multiply C[%d][%d]: scalar=%f rvv=%f\n",
                           i, j, C_scalar(i,j), C_rvv(i,j));
                    pass = false;
                }
        printf("Test 1 (6x6 multiply): %s\n", pass ? "PASS" : "FAIL");
    }

    // Test 2: 6x6 Matrix Add
    {
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Random();
        Eigen::Matrix<double, 6, 6> B = Eigen::Matrix<double, 6, 6>::Random();
        Eigen::Matrix<double, 6, 6> C_scalar = A + B;

#ifdef __riscv_vector
        Eigen::Matrix<double, 6, 6> C_rvv;
        Eigen::internal::rvv::eigen_6x6_add_rvv(C_rvv.data(), A.data(), B.data());
#else
        auto C_rvv = C_scalar;
#endif

        bool pass = true;
        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 6; i++)
                if (!approx_equal(C_scalar(i,j), C_rvv(i,j))) {
                    printf("FAIL: Add C[%d][%d]: scalar=%f rvv=%f\n",
                           i, j, C_scalar(i,j), C_rvv(i,j));
                    pass = false;
                }
        printf("Test 2 (6x6 add): %s\n", pass ? "PASS" : "FAIL");
    }

    // Test 3: 6x6 Triangular Solve (simple 6x6 lower triangular)
    {
        // Build a simple lower-triangular matrix L and vector b
        double L_data[36] = {
            2,0,0,0,0,0,
            1,3,0,0,0,0,
            2,1,4,0,0,0,
            0,2,1,5,0,0,
            1,0,2,1,6,0,
            0,1,0,2,1,7
        };
        double b_data[6] = {1, 2, 3, 4, 5, 6};
        double x_scalar[6] = {0};

        // Forward substitution (scalar)
        for (int i = 0; i < 6; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += L_data[k*6 + i] * x_scalar[k];
            }
            x_scalar[i] = (b_data[i] - sum) / L_data[i*6 + i];
        }

#ifdef __riscv_vector
        double x_rvv[6] = {0};
        Eigen::internal::rvv::eigen_6x6_triangular_solve_rvv(x_rvv, L_data, b_data);
#else
        double x_rvv[6];
        for (int i = 0; i < 6; i++) x_rvv[i] = x_scalar[i];
#endif

        bool pass = true;
        for (int i = 0; i < 6; i++)
            if (!approx_equal(x_scalar[i], x_rvv[i], 1e-10)) {
                printf("FAIL: Triangular solve x[%d]: scalar=%f rvv=%f\n",
                       i, x_scalar[i], x_rvv[i]);
                pass = false;
            }
        printf("Test 3 (6x6 triangular solve): %s\n", pass ? "PASS" : "FAIL");
    }

    printf("\nAll tests complete.\n");
    return 0;
}
