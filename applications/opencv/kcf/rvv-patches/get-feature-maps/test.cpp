// test.cpp — correctness test for RVV HOG gradient magnitude computation
//
// Tests the RVV implementations in rvv_get_feature_maps.inl:
//   - computeMagnitudeSimple_rvv: sqrt(dx^2 + dy^2)
//   - computePartialNorm_rvv: sum of squares for normalization
//   - truncateFeatureMap_rvv: clamp values above threshold
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 --target=riscv64-unknown-linux-gnu \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       --sysroot=<sysroot> \
//       -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
//       -I. test.cpp -o test -lm
//
// Run under QEMU:
//   qemu-riscv64 -L <sysroot> -cpu rv64,v=true,vlen=512 ./test

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Define NUM_SECTOR as in fhog.hpp
#define NUM_SECTOR 9

// ---------------------------------------------------------------------------
// Scalar references (matching fhog.cpp implementations)
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void computeMagnitudeSimple_generic(
    const float* dx_data,
    const float* dy_data,
    float* magnitude,
    size_t count)
{
    for (size_t i = 0; i < count; i++) {
        magnitude[i] = std::sqrt(dx_data[i] * dx_data[i] + dy_data[i] * dy_data[i]);
    }
}

#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void computePartialNorm_generic(
    const float* map_data,
    float* partOfNorm,
    int numFeatures,
    int p,
    size_t count)
{
    for (size_t i = 0; i < count; i++) {
        float valOfNorm = 0.0f;
        int pos = i * numFeatures;
        for (int j = 0; j < p; j++) {
            valOfNorm += map_data[pos + j] * map_data[pos + j];
        }
        partOfNorm[i] = valOfNorm;
    }
}

#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void truncateFeatureMap_generic(
    float* data,
    size_t count,
    float threshold)
{
    for (size_t i = 0; i < count; i++) {
        if (data[i] > threshold) {
            data[i] = threshold;
        }
    }
}

// ---------------------------------------------------------------------------
// RVV implementations
// ---------------------------------------------------------------------------
#include "rvv_get_feature_maps.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (deterministic LCG)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;
static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}
static float rng_float() {
    return (float)(rng_next()) / 32767.0f * 2.0f - 1.0f;  // Range: [-1, 1]
}
static float rng_float_positive() {
    return (float)(rng_next()) / 32767.0f + 0.1f;  // Range: [0.1, 1.1]
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------
static int test_magnitude_simple(size_t N, const char* label) {
    std::vector<float> dx(N);
    std::vector<float> dy(N);
    std::vector<float> mag_rvv(N);
    std::vector<float> mag_scalar(N);

    for (size_t i = 0; i < N; i++) {
        dx[i] = rng_float();
        dy[i] = rng_float();
    }

    computeMagnitudeSimple_generic(dx.data(), dy.data(), mag_scalar.data(), N);

#if defined(__riscv_v_intrinsic)
    computeMagnitudeSimple_rvv(dx.data(), dy.data(), mag_rvv.data(), N);
#else
    std::copy(mag_scalar.begin(), mag_scalar.end(), mag_rvv.begin());
#endif

    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-6f;

    for (size_t i = 0; i < N; i++) {
        float expected = mag_scalar[i];
        float actual = mag_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: dx=%.4f dy=%.4f expected=%.6f actual=%.6f diff=%e\n",
                       i, dx[i], dy[i], expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-30s N=%-6zu max_err=%.2e %s\n",
           label, N, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

static int test_partial_norm(size_t count, int numFeatures, int p, const char* label) {
    std::vector<float> map_data(count * numFeatures);
    std::vector<float> norm_rvv(count);
    std::vector<float> norm_scalar(count);

    for (size_t i = 0; i < count * numFeatures; i++) {
        map_data[i] = rng_float_positive();
    }

    computePartialNorm_generic(map_data.data(), norm_scalar.data(), numFeatures, p, count);

#if defined(__riscv_v_intrinsic)
    computePartialNorm_rvv(map_data.data(), norm_rvv.data(), numFeatures, p, count);
#else
    std::copy(norm_scalar.begin(), norm_scalar.end(), norm_rvv.begin());
#endif

    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-5f;

    for (size_t i = 0; i < count; i++) {
        float expected = norm_scalar[i];
        float actual = norm_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: expected=%.6f actual=%.6f diff=%e\n",
                       i, expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-30s count=%-6zu p=%-2d max_err=%.2e %s\n",
           label, count, p, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

static int test_truncate(size_t N, float threshold, const char* label) {
    std::vector<float> data_rvv(N);
    std::vector<float> data_scalar(N);

    for (size_t i = 0; i < N; i++) {
        data_rvv[i] = rng_float_positive() * 0.3f;  // Range: [0.03, 0.33]
        data_scalar[i] = data_rvv[i];
    }

    truncateFeatureMap_generic(data_scalar.data(), N, threshold);

#if defined(__riscv_v_intrinsic)
    truncateFeatureMap_rvv(data_rvv.data(), N, threshold);
#else
    // Already copied from scalar
#endif

    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-7f;

    for (size_t i = 0; i < N; i++) {
        float expected = data_scalar[i];
        float actual = data_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: orig=%.6f expected=%.6f actual=%.6f diff=%e\n",
                       i, data_rvv[i], expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-30s N=%-6zu thresh=%.3f max_err=%.2e %s\n",
           label, N, threshold, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ---------------------------------------------------------------------------
// Main test driver
// ---------------------------------------------------------------------------
int main() {
    printf("=== HOG Gradient Magnitude: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Test 1: Simple magnitude computation
    printf("Test 1: computeMagnitudeSimple (sqrt(dx^2 + dy^2))\n");
    size_t sizes[] = {1, 3, 8, 16, 31, 64, 100, 127, 256, 512, 1000, 4096, 10000};
    for (size_t N : sizes) {
        char label[48];
        snprintf(label, sizeof(label), "magnitude N=%zu", N);
        failures += test_magnitude_simple(N, label);
    }

    // Test 2: Partial norm computation
    printf("\nTest 2: computePartialNorm (sum of squares)\n");
    int numFeatures = 27;  // 3 * NUM_SECTOR
    size_t counts[] = {1, 10, 16, 32, 64, 100, 256};
    int p_values[] = {NUM_SECTOR, 18, 27};
    for (size_t count : counts) {
        for (int p : p_values) {
            char label[48];
            snprintf(label, sizeof(label), "norm count=%zu p=%d", count, p);
            failures += test_partial_norm(count, numFeatures, p, label);
        }
    }

    // Test 3: Truncation
    printf("\nTest 3: truncateFeatureMap (clamp above threshold)\n");
    float thresholds[] = {0.2f, 0.1f, 0.5f};
    for (float thresh : thresholds) {
        for (size_t N : {16, 64, 256, 1024, 4096}) {
            char label[48];
            snprintf(label, sizeof(label), "truncate N=%zu thresh=%.2f", N, thresh);
            failures += test_truncate(N, thresh, label);
        }
    }

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}