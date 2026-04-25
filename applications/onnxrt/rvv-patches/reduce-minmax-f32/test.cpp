// test.cpp — correctness test for RVV MlasReduceMinimumMaximumF32Kernel
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
//       --sysroot=/home/pren/wsp/rvfuse/output/cross-ort/sysroot \
//       -I. test.cpp -o test -lm
//
// Run under QEMU:
//   qemu-riscv64 -cpu rv64,v=true,vlen=512 -L <sysroot> ./test

#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// Scalar (generic) reference — matches MLAS fallback implementation
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))  // Prevent LLVM optimizer bugs
#endif
static void
MlasReduceMinimumMaximumF32KernelScalar(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
)
{
    float tmp_min = FLT_MAX;
    float tmp_max = -FLT_MAX;

    while (N > 0) {
        tmp_max = (Input[0] > tmp_max) ? Input[0] : tmp_max;
        tmp_min = (Input[0] < tmp_min) ? Input[0] : tmp_min;
        Input++;
        N--;
    }

    *Min = tmp_min;
    *Max = tmp_max;
}

// RVV implementation — single source of truth
#include "rvv_reduce_minmax_f32.inl"

// Deterministic LCG RNG (seed = 42)
static uint32_t rng_state = 42;
static uint32_t rng_next()
{
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static float rng_float()
{
    // Generate float in range [-100.0, 100.0]
    int32_t val = (int32_t)rng_next() - 16384;  // -16384 to +16383
    return (float)val * (100.0f / 16384.0f);
}

// Test runner
static int run_test(size_t N, const char* label)
{
    std::vector<float> input(N);
    float min_rvv, max_rvv;
    float min_scalar, max_scalar;

    // Generate deterministic input data
    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float();
    }

    // Inject extreme values to test boundary cases
    if (N > 10) {
        input[0] = FLT_MAX;      // Maximum possible value
        input[1] = -FLT_MAX;     // Minimum possible value
        input[N-1] = 0.0f;       // Zero
    }

    // Run RVV version
#if defined(__riscv_v_intrinsic)
    MlasReduceMinimumMaximumF32KernelRVV(input.data(), &min_rvv, &max_rvv, N);
#else
    // No RVV, use scalar directly
    MlasReduceMinimumMaximumF32KernelScalar(input.data(), &min_rvv, &max_rvv, N);
#endif

    // Run scalar reference
    MlasReduceMinimumMaximumF32KernelScalar(input.data(), &min_scalar, &max_scalar, N);

    // Compare results (FP32 tolerance: 1e-6 relative)
    bool min_match = std::fabs(min_rvv - min_scalar) < 1e-6f * std::fabs(min_scalar) ||
                     (min_rvv == min_scalar);  // Exact match for special values
    bool max_match = std::fabs(max_rvv - max_scalar) < 1e-6f * std::fabs(max_scalar) ||
                     (max_rvv == max_scalar);

    int failures = 0;
    if (!min_match) {
        printf("[FAIL] %s: min mismatch (RVV=%.6f, scalar=%.6f, diff=%.6e)\n",
               label, min_rvv, min_scalar, min_rvv - min_scalar);
        failures++;
    }
    if (!max_match) {
        printf("[FAIL] %s: max mismatch (RVV=%.6f, scalar=%.6f, diff=%.6e)\n",
               label, max_rvv, max_scalar, max_rvv - max_scalar);
        failures++;
    }

    if (failures == 0) {
        printf("[PASS] %s: N=%zu, min=%.6f, max=%.6f\n", label, N, min_rvv, max_rvv);
    }

    return failures;
}

int main()
{
    printf("=== MlasReduceMinimumMaximumF32Kernel RVV Test ===\n");
#if defined(__riscv_v_intrinsic)
#if defined(__riscv_v_fixed_vlen)
    printf("RVV intrinsics: enabled (VLEN=%d)\n", __riscv_v_fixed_vlen);
#else
    printf("RVV intrinsics: enabled (VLEN=dynamic)\n");
#endif
#else
    printf("RVV intrinsics: disabled (scalar fallback)\n");
#endif
    printf("\n");

    int total_failures = 0;

    // Test various sizes covering different code paths
    // Small: tail handling
    total_failures += run_test(1, "test-tail-1");
    total_failures += run_test(3, "test-tail-3");
    total_failures += run_test(7, "test-tail-7");
    total_failures += run_test(15, "test-tail-15");

    // Medium: single vector loop
    total_failures += run_test(16, "test-single-vl");
    total_failures += run_test(20, "test-vl-plus-tail");
    total_failures += run_test(31, "test-vl-plus-tail-31");

    // Large: accumulator loop (4×VL)
    total_failures += run_test(64, "test-4vl");
    total_failures += run_test(100, "test-4vl-plus");
    total_failures += run_test(256, "test-large");
    total_failures += run_test(1024, "test-very-large");
    total_failures += run_test(10000, "test-huge");

    // Edge cases
    total_failures += run_test(0, "test-empty");  // Should handle gracefully

    printf("\n=== Summary ===\n");
    printf("Tests failed: %d\n", total_failures);

    return total_failures;
}