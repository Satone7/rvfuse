// test.cpp — correctness test for RVV MlasQuantizeLinear
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

// ---------------------------------------------------------------------------
// Scalar (generic) reference implementation
// ---------------------------------------------------------------------------
template<typename OutputType>
static void
MlasQuantizeLinearScalar(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
)
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::lowest();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    for (size_t n = 0; n < N; n++) {
        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

#include "rvv_quantize_linear.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static float rng_float(float lo, float hi) {
    return lo + (hi - lo) * (rng_next() / 32768.0f);
}

// ---------------------------------------------------------------------------
// Test runner for uint8_t
// ---------------------------------------------------------------------------
static int run_test_u8(
    size_t N,
    float Scale,
    uint8_t ZeroPoint,
    float ValueLo,
    float ValueHi,
    const char* label
)
{
    std::vector<float> input(N);
    std::vector<uint8_t> output_scalar(N);
    std::vector<uint8_t> output_rvv(N);

    rng_state = 42;
    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float(ValueLo, ValueHi);
    }

    // Scalar reference
    MlasQuantizeLinearScalar(input.data(), output_scalar.data(), N, Scale, ZeroPoint);

    // RVV implementation
#if defined(__riscv_v_intrinsic)
    MlasQuantizeLinearU8KernelRVV(input.data(), output_rvv.data(), N, Scale, ZeroPoint);
#else
    memcpy(output_rvv.data(), output_scalar.data(), N);
#endif

    int failures = 0;
    for (size_t i = 0; i < N; i++) {
        if (output_scalar[i] != output_rvv[i]) {
            if (failures < 10) {
                printf("  MISMATCH [%zu]: scalar=%u, rvv=%u, input=%.6f\n",
                       i, (unsigned)output_scalar[i], (unsigned)output_rvv[i], input[i]);
            }
            failures++;
        }
    }

    printf("  %s [%s] %zu elements\n", failures ? "FAIL" : "PASS", label, N);
    return failures > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Test runner for int8_t
// ---------------------------------------------------------------------------
static int run_test_s8(
    size_t N,
    float Scale,
    int8_t ZeroPoint,
    float ValueLo,
    float ValueHi,
    const char* label
)
{
    std::vector<float> input(N);
    std::vector<int8_t> output_scalar(N);
    std::vector<int8_t> output_rvv(N);

    rng_state = 42;
    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float(ValueLo, ValueHi);
    }

    MlasQuantizeLinearScalar(input.data(), output_scalar.data(), N, Scale, ZeroPoint);

#if defined(__riscv_v_intrinsic)
    MlasQuantizeLinearS8KernelRVV(input.data(), output_rvv.data(), N, Scale, ZeroPoint);
#else
    memcpy(output_rvv.data(), output_scalar.data(), N);
#endif

    int failures = 0;
    for (size_t i = 0; i < N; i++) {
        if (output_scalar[i] != output_rvv[i]) {
            if (failures < 10) {
                printf("  MISMATCH [%zu]: scalar=%d, rvv=%d, input=%.6f\n",
                       i, (int)output_scalar[i], (int)output_rvv[i], input[i]);
            }
            failures++;
        }
    }

    printf("  %s [%s] %zu elements\n", failures ? "FAIL" : "PASS", label, N);
    return failures > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Test runner for uint16_t
// ---------------------------------------------------------------------------
static int run_test_u16(
    size_t N,
    float Scale,
    uint16_t ZeroPoint,
    float ValueLo,
    float ValueHi,
    const char* label
)
{
    std::vector<float> input(N);
    std::vector<uint16_t> output_scalar(N);
    std::vector<uint16_t> output_rvv(N);

    rng_state = 42;
    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float(ValueLo, ValueHi);
    }

    MlasQuantizeLinearScalar(input.data(), output_scalar.data(), N, Scale, ZeroPoint);

#if defined(__riscv_v_intrinsic)
    MlasQuantizeLinearU16KernelRVV(input.data(), output_rvv.data(), N, Scale, ZeroPoint);
#else
    memcpy(output_rvv.data(), output_scalar.data(), N * sizeof(uint16_t));
#endif

    int failures = 0;
    for (size_t i = 0; i < N; i++) {
        if (output_scalar[i] != output_rvv[i]) {
            if (failures < 10) {
                printf("  MISMATCH [%zu]: scalar=%u, rvv=%u, input=%.6f\n",
                       i, (unsigned)output_scalar[i], (unsigned)output_rvv[i], input[i]);
            }
            failures++;
        }
    }

    printf("  %s [%s] %zu elements\n", failures ? "FAIL" : "PASS", label, N);
    return failures > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Test runner for int16_t
// ---------------------------------------------------------------------------
static int run_test_s16(
    size_t N,
    float Scale,
    int16_t ZeroPoint,
    float ValueLo,
    float ValueHi,
    const char* label
)
{
    std::vector<float> input(N);
    std::vector<int16_t> output_scalar(N);
    std::vector<int16_t> output_rvv(N);

    rng_state = 42;
    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float(ValueLo, ValueHi);
    }

    MlasQuantizeLinearScalar(input.data(), output_scalar.data(), N, Scale, ZeroPoint);

#if defined(__riscv_v_intrinsic)
    MlasQuantizeLinearS16KernelRVV(input.data(), output_rvv.data(), N, Scale, ZeroPoint);
#else
    memcpy(output_rvv.data(), output_scalar.data(), N * sizeof(int16_t));
#endif

    int failures = 0;
    for (size_t i = 0; i < N; i++) {
        if (output_scalar[i] != output_rvv[i]) {
            if (failures < 10) {
                printf("  MISMATCH [%zu]: scalar=%d, rvv=%d, input=%.6f\n",
                       i, (int)output_scalar[i], (int)output_rvv[i], input[i]);
            }
            failures++;
        }
    }

    printf("  %s [%s] %zu elements\n", failures ? "FAIL" : "PASS", label, N);
    return failures > 0 ? 1 : 0;
}

int main() {
    printf("=== MlasQuantizeLinear: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // U8 tests
    printf("--- uint8_t ---\n");
    failures += run_test_u8(8, 0.1f, 128, -20.0f, 20.0f, "u8/vl8");
    failures += run_test_u8(16, 0.1f, 128, -20.0f, 20.0f, "u8/vl16");
    failures += run_test_u8(64, 0.05f, 0, -15.0f, 15.0f, "u8/vl64/zp0");
    failures += run_test_u8(7, 0.1f, 128, -5.0f, 5.0f, "u8/tail7");
    failures += run_test_u8(1, 0.1f, 128, -5.0f, 5.0f, "u8/single");
    failures += run_test_u8(100, 0.01f, 100, -2.0f, 2.0f, "u8/vl100/small_scale");
    failures += run_test_u8(33, 0.1f, 128, -100.0f, 100.0f, "u8/vl33/clamp");
    failures += run_test_u8(0, 0.1f, 0, 0.0f, 1.0f, "u8/empty");

    // S8 tests
    printf("\n--- int8_t ---\n");
    failures += run_test_s8(8, 0.1f, 0, -20.0f, 20.0f, "s8/vl8");
    failures += run_test_s8(16, 0.1f, 0, -20.0f, 20.0f, "s8/vl16");
    failures += run_test_s8(64, 0.05f, -10, -15.0f, 15.0f, "s8/vl64/neg_zp");
    failures += run_test_s8(7, 0.1f, 0, -5.0f, 5.0f, "s8/tail7");
    failures += run_test_s8(1, 0.1f, 0, -5.0f, 5.0f, "s8/single");
    failures += run_test_s8(50, 0.01f, 5, -2.0f, 2.0f, "s8/vl50/small_scale");
    failures += run_test_s8(0, 0.1f, 0, 0.0f, 1.0f, "s8/empty");

    // U16 tests
    printf("\n--- uint16_t ---\n");
    failures += run_test_u16(8, 0.001f, 32768, -50.0f, 50.0f, "u16/vl8");
    failures += run_test_u16(16, 0.001f, 32768, -50.0f, 50.0f, "u16/vl16");
    failures += run_test_u16(33, 0.001f, 0, -40.0f, 40.0f, "u16/vl33/zp0");
    failures += run_test_u16(7, 0.001f, 100, -5.0f, 5.0f, "u16/tail7");

    // S16 tests
    printf("\n--- int16_t ---\n");
    failures += run_test_s16(8, 0.001f, 0, -50.0f, 50.0f, "s16/vl8");
    failures += run_test_s16(16, 0.001f, 0, -50.0f, 50.0f, "s16/vl16");
    failures += run_test_s16(33, 0.001f, -100, -40.0f, 40.0f, "s16/vl33/neg_zp");
    failures += run_test_s16(7, 0.001f, 0, -5.0f, 5.0f, "s16/tail7");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
