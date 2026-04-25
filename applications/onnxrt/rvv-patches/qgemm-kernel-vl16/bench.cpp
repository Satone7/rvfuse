// bench.cpp — BBV profiling harness for RVV INT8 QGEMM kernel (VL=16, 512-bit)
//
// Creates a standalone non-inlined function for the RVV kernel so that
// BBV profiling can isolate it by function offset.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -DVLEN_512 \
//       -I applications/onnxrt/rvv-patches/qgemm-kernel-vl16 \
//       bench.cpp -o output/qgemm_bench -lm -fuse-ld=lld

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// RVV implementation — single source of truth
#include "rvv_qgemm_kernel_vl16.inl"

// Non-inlined wrapper for BBV profiling
__attribute__((noinline))
size_t bench_qgemm_kernel(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountN,
    int32_t RowSum,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode)
{
#if defined(__riscv_v) && defined(VLEN_512)
    return MlasQgemmKernelRvv512Impl(A, B, C, PackedCountK, CountN,
                                      RowSum, ColumnSumBuffer, ZeroPointB, ZeroMode);
#else
    return 1;
#endif
}

int main() {
    // Use a large GEMM to ensure the kernel dominates execution time
    const size_t K = 256;
    const size_t N = 128;  // 8 blocks of 16 columns
    const size_t AlignedCountK = (K + 3) & ~3u;
    const size_t PackedCountK = AlignedCountK / 4;

    // Allocate A (1 row)
    std::vector<uint8_t> A(K, 42);
    std::vector<uint8_t> A_packed(AlignedCountK);
    int32_t RowSum = 0;

#if defined(__riscv_v) && defined(VLEN_512)
    MlasQgemmPackARvv512(A_packed.data(), A.data(), K, 1, K, &RowSum, false);
#else
    for (size_t k = 0; k < K; k++) { A_packed[k] = A[k]; RowSum += A[k]; }
#endif

    // Allocate B (K rows × N cols, row-major)
    std::vector<uint8_t> B_raw(K * N);
    for (size_t i = 0; i < K * N; i++) B_raw[i] = (uint8_t)(i & 0xFF);

    // Pack B in 16-column block format
    const size_t num_blocks = (N + 15) / 16;
    std::vector<uint8_t> B_packed(PackedCountK * 64 * num_blocks);
    std::vector<int32_t> ColSum(N, 0);

#if defined(__riscv_v) && defined(VLEN_512)
    MlasQgemmPackBRvv512(B_packed.data(), B_raw.data(), N, N, K,
                          ColSum.data(), false);
#else
    // Simple packing fallback
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            B_packed[n * AlignedCountK + k] = B_raw[k * N + n];
            ColSum[n] += B_raw[k * N + n];
        }
    }
#endif

    // C output
    std::vector<int32_t> C(N, 0);

    // Run many iterations to get good BBV samples
    const int iterations = 10000;
    for (int i = 0; i < iterations; i++) {
        bench_qgemm_kernel(
            A_packed.data(),
            B_packed.data(),
            C.data(),
            PackedCountK,
            N,
            RowSum,
            ColSum.data(),
            nullptr,
            true);
    }

    // Print a checksum to prevent dead-code elimination
    int32_t checksum = 0;
    for (size_t n = 0; n < N; n++) checksum += C[n];
    printf("QGEMM checksum: %d (iterations=%d K=%zu N=%zu)\n",
           checksum, iterations, K, N);

    return 0;
}
