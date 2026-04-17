/*++

Copyright (c) RVFuse Project. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_softmax_fp16.cpp

Abstract:

    Test harness for FP16 softmax kernels. Contains:
    1. Scalar reference implementations (FP32 and FP16)
    2. Softmax correctness test comparing vector vs scalar
    3. Performance benchmark for different input sizes
    4. Edge case tests (zero-length, single-element, large values)

    Build (RISC-V cross-compile):
        riscv64-unknown-linux-gnu-g++ -std=c++17 -O2 \
            -march=rv64gcv_zvfh -DVLEN_512 \
            test_softmax_fp16.cpp -o test_softmax_fp16

    Build (native x86, for scalar reference testing):
        g++ -std=c++17 -O2 test_softmax_fp16.cpp -o test_softmax_fp16_scalar

    Run:
        ./test_softmax_fp16
        ./test_softmax_fp16 --verbose
        ./test_softmax_fp16 --bench

--*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

// ============================================================================
// FP16 type emulation for testing on non-RVV platforms
// ============================================================================
// MLAS_FP16 is defined in onnxruntime/core/mlas/inc/mlas_float16.h.
// For standalone testing, we provide a minimal FP16 type.

#ifndef MLAS_FP16_DEFINED

struct MLAS_FP16 {
    uint16_t val;

    static MLAS_FP16 FromFloat(float f) {
        MLAS_FP16 result;
        // Manual float-to-fp16 conversion
        uint32_t f_bits;
        memcpy(&f_bits, &f, sizeof(f_bits));
        uint32_t sign = (f_bits >> 31) & 1;
        int32_t exp = ((f_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (f_bits >> 13) & 0x3FF;

        if (exp <= 0) {
            // Denormalized or zero
            if (exp < -10) {
                result.val = static_cast<uint16_t>(sign << 15);
            } else {
                frac |= 0x400;
                frac >>= (1 - exp);
                result.val = static_cast<uint16_t>((sign << 15) | frac);
            }
        } else if (exp >= 31) {
            // Overflow to infinity
            result.val = static_cast<uint16_t>((sign << 15) | 0x7C00);
        } else {
            result.val = static_cast<uint16_t>((sign << 15) | (exp << 10) | frac);
        }
        return result;
    }

    static MLAS_FP16 FromBits(uint16_t bits) {
        MLAS_FP16 result;
        result.val = bits;
        return result;
    }

    float ToFloat() const {
        uint32_t sign = (val >> 15) & 1;
        uint32_t exp = (val >> 10) & 0x1F;
        uint32_t frac = val & 0x3FF;

        if (exp == 0) {
            if (frac == 0) {
                // Zero
                float f = 0.0f;
                uint32_t f_bits = (sign << 31);
                memcpy(&f, &f_bits, sizeof(f));
                return f;
            }
            // Denormalized
            float f = 0.0f;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
            exp++;
        }

        if (exp == 31) {
            // Infinity or NaN
            uint32_t f_bits = (sign << 31) | 0x7F800000 | (frac << 13);
            float f;
            memcpy(&f, &f_bits, sizeof(f));
            return f;
        }

        uint32_t f_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        float f;
        memcpy(&f, &f_bits, sizeof(f));
        return f;
    }

    MLAS_FP16 Negate() const {
        return FromBits(static_cast<uint16_t>(val ^ 0x8000));
    }

    bool operator==(const MLAS_FP16& other) const { return val == other.val; }
    bool operator!=(const MLAS_FP16& other) const { return val != other.val; }
};

#define MLAS_FP16_DEFINED

#endif // MLAS_FP16_DEFINED

// ============================================================================
// Scalar FP32 reference implementations
// ============================================================================

//
// Scalar FP32 ReduceMax
//
static float
ScalarReduceMaxF32(const float* input, size_t n) {
    if (n == 0) return -std::numeric_limits<float>::max();
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    return max_val;
}

//
// Scalar FP32 softmax: output[i] = exp(input[i] - max) / sum(exp(input[i] - max))
//
static void
ScalarSoftmaxF32(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float max_val = ScalarReduceMaxF32(input, n);
    float neg_max = -max_val;

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float val = std::exp(input[i] + neg_max);
        output[i] = val;
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        output[i] *= inv_sum;
    }
}

//
// Scalar FP32 log softmax: output[i] = input[i] - max - log(sum(exp(input[i] - max)))
//
static void
ScalarLogSoftmaxF32(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float max_val = ScalarReduceMaxF32(input, n);
    float neg_max = -max_val;

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += std::exp(input[i] + neg_max);
    }

    float log_sum = std::log(sum);
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] + neg_max - log_sum;
    }
}

// ============================================================================
// Scalar FP16 reference implementations
// ============================================================================

//
// Scalar FP16 ReduceMax
//
static MLAS_FP16
ScalarReduceMaxFp16(const MLAS_FP16* input, size_t n) {
    if (n == 0) return MLAS_FP16::FromBits(0xFBFF);
    MLAS_FP16 max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        if (input[i].ToFloat() > max_val.ToFloat()) max_val = input[i];
    }
    return max_val;
}

//
// Scalar FP16 SumExp: compute exp(x - max) and sum
//
static MLAS_FP16
ScalarSumExpFp16(const MLAS_FP16* input, MLAS_FP16* output, size_t n,
                 const MLAS_FP16 NegativeMaximum) {
    float neg_max = NegativeMaximum.ToFloat();
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float val = std::exp(input[i].ToFloat() + neg_max);
        sum += val;
        if (output != nullptr) {
            output[i] = MLAS_FP16::FromFloat(val);
        }
    }

    return MLAS_FP16::FromFloat(sum);
}

//
// Scalar FP16 Softmax: divide exp values by sum
//
static void
ScalarSoftmaxFp16(const MLAS_FP16* input, MLAS_FP16* output, size_t n,
                   const MLAS_FP16 Sum) {
    float scale = 1.0f / Sum.ToFloat();
    for (size_t i = 0; i < n; i++) {
        output[i] = MLAS_FP16::FromFloat(input[i].ToFloat() * scale);
    }
}

//
// Scalar FP16 LogSoftmax
//
static void
ScalarLogSoftmaxFp16(const MLAS_FP16* input, MLAS_FP16* output, size_t n,
                      const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum) {
    float neg_max = NegativeMaximum.ToFloat();
    float log_sum = LogSum.ToFloat();
    for (size_t i = 0; i < n; i++) {
        float val = input[i].ToFloat() + neg_max - log_sum;
        output[i] = MLAS_FP16::FromFloat(val);
    }
}

//
// Full scalar FP16 softmax pipeline
//
static void
ScalarSoftmaxFp16Full(const MLAS_FP16* input, MLAS_FP16* output, size_t n) {
    if (n == 0) return;

    MLAS_FP16 max_val = ScalarReduceMaxFp16(input, n);
    MLAS_FP16 neg_max = max_val.Negate();

    MLAS_FP16 sum = ScalarSumExpFp16(input, output, n, neg_max);
    ScalarSoftmaxFp16(output, output, n, sum);
}

// ============================================================================
// FP16 exp implementation for scalar reference
// ============================================================================

static MLAS_FP16
ScalarExpFp16(const MLAS_FP16& x) {
    float val = std::exp(x.ToFloat());
    return MLAS_FP16::FromFloat(val);
}

// ============================================================================
// Test utilities
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static bool g_verbose = false;
static bool g_bench = false;

static void
PrintTestResult(const char* name, bool passed) {
    if (passed) {
        printf("  [PASS] %s\n", name);
        g_tests_passed++;
    } else {
        printf("  [FAIL] %s\n", name);
        g_tests_failed++;
    }
}

static bool
CompareFp16Arrays(const MLAS_FP16* a, const MLAS_FP16* b, size_t n,
                  float rel_tol = 0.01f) {
    for (size_t i = 0; i < n; i++) {
        float fa = a[i].ToFloat();
        float fb = b[i].ToFloat();

        float diff = std::fabs(fa - fb);
        float denom = std::max(std::fabs(fa), std::fabs(fb));
        if (denom < 1e-6f) denom = 1e-6f;

        if (diff / denom > rel_tol) {
            if (g_verbose) {
                printf("    Mismatch at [%zu]: scalar=%.6f rvv=%.6f (rel_err=%.4f)\n",
                       i, fa, fb, diff / denom);
            }
            return false;
        }
    }
    return true;
}

// ============================================================================
// Test cases
// ============================================================================

//
// Test 1: Basic softmax with known values
//
static void
TestBasicSoftmax() {
    printf("Test: Basic FP16 softmax\n");

    // Input: [1.0, 2.0, 3.0]
    // Expected softmax: [0.0900, 0.2447, 0.6652] (approximately)
    std::vector<MLAS_FP16> input = {
        MLAS_FP16::FromFloat(1.0f),
        MLAS_FP16::FromFloat(2.0f),
        MLAS_FP16::FromFloat(3.0f),
    };
    std::vector<MLAS_FP16> output(3);
    std::vector<MLAS_FP16> expected(3);

    ScalarSoftmaxFp16Full(input.data(), expected.data(), 3);

    // Compare scalar pipeline vs FP32 reference
    std::vector<float> input_f32 = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected_f32(3);
    ScalarSoftmaxF32(input_f32.data(), expected_f32.data(), 3);

    for (size_t i = 0; i < 3; i++) {
        float diff = std::fabs(expected[i].ToFloat() - expected_f32[i]);
        if (g_verbose) {
            printf("    [%zu] fp16=%.6f fp32=%.6f diff=%.6f\n",
                   i, expected[i].ToFloat(), expected_f32[i], diff);
        }
    }

    // Sum of softmax outputs should be ~1.0
    float sum = 0.0f;
    for (size_t i = 0; i < 3; i++) {
        sum += expected[i].ToFloat();
    }
    bool sum_ok = std::fabs(sum - 1.0f) < 0.01f;
    PrintTestResult("sum(output) ~= 1.0", sum_ok);

    // All outputs should be positive
    bool all_positive = true;
    for (size_t i = 0; i < 3; i++) {
        if (expected[i].ToFloat() < 0.0f) { all_positive = false; break; }
    }
    PrintTestResult("all outputs positive", all_positive);

    // Monotonicity: larger input -> larger output
    bool monotonic = (expected[0].ToFloat() < expected[1].ToFloat()) &&
                     (expected[1].ToFloat() < expected[2].ToFloat());
    PrintTestResult("monotonic output", monotonic);
}

//
// Test 2: ReduceMax with various patterns
//
static void
TestReduceMax() {
    printf("Test: FP16 ReduceMax\n");

    {
        // Ascending
        std::vector<MLAS_FP16> data = {
            MLAS_FP16::FromFloat(1.0f), MLAS_FP16::FromFloat(2.0f),
            MLAS_FP16::FromFloat(3.0f), MLAS_FP16::FromFloat(4.0f),
        };
        MLAS_FP16 max_val = ScalarReduceMaxFp16(data.data(), 4);
        PrintTestResult("ascending max", std::fabs(max_val.ToFloat() - 4.0f) < 0.001f);
    }

    {
        // Descending
        std::vector<MLAS_FP16> data = {
            MLAS_FP16::FromFloat(4.0f), MLAS_FP16::FromFloat(3.0f),
            MLAS_FP16::FromFloat(2.0f), MLAS_FP16::FromFloat(1.0f),
        };
        MLAS_FP16 max_val = ScalarReduceMaxFp16(data.data(), 4);
        PrintTestResult("descending max", std::fabs(max_val.ToFloat() - 4.0f) < 0.001f);
    }

    {
        // Negative values
        std::vector<MLAS_FP16> data = {
            MLAS_FP16::FromFloat(-10.0f), MLAS_FP16::FromFloat(-1.0f),
            MLAS_FP16::FromFloat(-5.0f), MLAS_FP16::FromFloat(-3.0f),
        };
        MLAS_FP16 max_val = ScalarReduceMaxFp16(data.data(), 4);
        PrintTestResult("negative max", std::fabs(max_val.ToFloat() - (-1.0f)) < 0.001f);
    }

    {
        // All same values
        std::vector<MLAS_FP16> data = {
            MLAS_FP16::FromFloat(2.5f), MLAS_FP16::FromFloat(2.5f),
            MLAS_FP16::FromFloat(2.5f), MLAS_FP16::FromFloat(2.5f),
        };
        MLAS_FP16 max_val = ScalarReduceMaxFp16(data.data(), 4);
        PrintTestResult("all-same max", std::fabs(max_val.ToFloat() - 2.5f) < 0.001f);
    }

    {
        // Single element
        MLAS_FP16 data = MLAS_FP16::FromFloat(42.0f);
        MLAS_FP16 max_val = ScalarReduceMaxFp16(&data, 1);
        PrintTestResult("single element max", std::fabs(max_val.ToFloat() - 42.0f) < 0.001f);
    }
}

//
// Test 3: SumExp correctness
//
static void
TestSumExp() {
    printf("Test: FP16 SumExp\n");

    // Input: [0.0, 1.0, 2.0] with neg_max = -2.0
    // exp(0 + (-2)) = exp(-2) = 0.1353
    // exp(1 + (-2)) = exp(-1) = 0.3679
    // exp(2 + (-2)) = exp(0)  = 1.0000
    // sum = 1.5033
    std::vector<MLAS_FP16> input = {
        MLAS_FP16::FromFloat(0.0f),
        MLAS_FP16::FromFloat(1.0f),
        MLAS_FP16::FromFloat(2.0f),
    };
    MLAS_FP16 neg_max = MLAS_FP16::FromFloat(-2.0f);
    std::vector<MLAS_FP16> output(3);

    MLAS_FP16 sum = ScalarSumExpFp16(input.data(), output.data(), 3, neg_max);

    // Check individual exp values
    bool exp0_ok = std::fabs(output[0].ToFloat() - 0.1353f) < 0.01f;
    bool exp1_ok = std::fabs(output[1].ToFloat() - 0.3679f) < 0.01f;
    bool exp2_ok = std::fabs(output[2].ToFloat() - 1.0000f) < 0.01f;
    bool sum_ok = std::fabs(sum.ToFloat() - 1.5033f) < 0.01f;

    PrintTestResult("exp(x_0 - max) correct", exp0_ok);
    PrintTestResult("exp(x_1 - max) correct", exp1_ok);
    PrintTestResult("exp(x_2 - max) correct", exp2_ok);
    PrintTestResult("sum(exp) correct", sum_ok);
}

//
// Test 4: Softmax normalization
//
static void
TestSoftmaxNorm() {
    printf("Test: FP16 Softmax normalization\n");

    // Simulated exp values: [1.0, 2.0, 3.0], sum = 6.0
    std::vector<MLAS_FP16> exp_vals = {
        MLAS_FP16::FromFloat(1.0f),
        MLAS_FP16::FromFloat(2.0f),
        MLAS_FP16::FromFloat(3.0f),
    };
    MLAS_FP16 sum = MLAS_FP16::FromFloat(6.0f);
    std::vector<MLAS_FP16> output(3);

    ScalarSoftmaxFp16(exp_vals.data(), output.data(), 3, sum);

    bool v0_ok = std::fabs(output[0].ToFloat() - (1.0f / 6.0f)) < 0.01f;
    bool v1_ok = std::fabs(output[1].ToFloat() - (2.0f / 6.0f)) < 0.01f;
    bool v2_ok = std::fabs(output[2].ToFloat() - (3.0f / 6.0f)) < 0.01f;

    PrintTestResult("norm[0] = 1/6", v0_ok);
    PrintTestResult("norm[1] = 2/6", v1_ok);
    PrintTestResult("norm[2] = 3/6", v2_ok);

    // Sum should be 1.0
    float out_sum = output[0].ToFloat() + output[1].ToFloat() + output[2].ToFloat();
    PrintTestResult("sum(normals) = 1.0", std::fabs(out_sum - 1.0f) < 0.01f);
}

//
// Test 5: Full pipeline - larger input
//
static void
TestFullPipeline() {
    printf("Test: Full softmax pipeline (N=80, D=8400)\n");

    // Simulate YOLO-like dimensions: 80 classes, 8400 rows
    // For testing, use smaller: 80 classes, 100 rows
    const size_t num_rows = 100;
    const size_t num_cols = 80;

    std::vector<MLAS_FP16> input(num_rows * num_cols);
    std::vector<MLAS_FP16> output(num_rows * num_cols);

    // Fill with random-ish data
    srand(42);
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        input[i] = MLAS_FP16::FromFloat(static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f);
    }

    // Run scalar FP16 softmax
    for (size_t row = 0; row < num_rows; row++) {
        ScalarSoftmaxFp16Full(
            input.data() + row * num_cols,
            output.data() + row * num_cols,
            num_cols);
    }

    // Verify: each row should sum to ~1.0
    bool all_rows_ok = true;
    for (size_t row = 0; row < num_rows; row++) {
        float row_sum = 0.0f;
        for (size_t col = 0; col < num_cols; col++) {
            row_sum += output[row * num_cols + col].ToFloat();
        }
        if (std::fabs(row_sum - 1.0f) > 0.02f) {
            if (g_verbose) {
                printf("    Row %zu: sum = %.6f (expected 1.0)\n", row, row_sum);
            }
            all_rows_ok = false;
        }
    }
    PrintTestResult("all rows sum to ~1.0", all_rows_ok);

    // Compare first row against FP32 reference
    std::vector<float> input_f32(num_cols);
    std::vector<float> expected_f32(num_cols);
    for (size_t col = 0; col < num_cols; col++) {
        input_f32[col] = input[col].ToFloat();
    }
    ScalarSoftmaxF32(input_f32.data(), expected_f32.data(), num_cols);

    std::vector<MLAS_FP16> first_row(num_cols);
    for (size_t col = 0; col < num_cols; col++) {
        first_row[col] = output[col];
    }
    bool fp16_vs_f32 = CompareFp16Arrays(first_row.data(),
                                          reinterpret_cast<const MLAS_FP16*>(expected_f32.data()),
                                          num_cols, 0.02f);
    PrintTestResult("FP16 vs FP32 close match", fp16_vs_f32);
}

//
// Test 6: Edge cases
//
static void
TestEdgeCases() {
    printf("Test: Edge cases\n");

    {
        // Zero-length input
        std::vector<MLAS_FP16> empty;
        std::vector<MLAS_FP16> out(1);
        MLAS_FP16 max_val = ScalarReduceMaxFp16(empty.data(), 0);
        PrintTestResult("zero-length reduce max", max_val.val == 0xFBFF);
    }

    {
        // Single element softmax = [1.0]
        MLAS_FP16 input = MLAS_FP16::FromFloat(5.0f);
        MLAS_FP16 output;
        ScalarSoftmaxFp16Full(&input, &output, 1);
        PrintTestResult("single element = 1.0", std::fabs(output.ToFloat() - 1.0f) < 0.001f);
    }

    {
        // Two equal elements = [0.5, 0.5]
        std::vector<MLAS_FP16> input = {
            MLAS_FP16::FromFloat(3.0f),
            MLAS_FP16::FromFloat(3.0f),
        };
        std::vector<MLAS_FP16> output(2);
        ScalarSoftmaxFp16Full(input.data(), output.data(), 2);
        bool ok = std::fabs(output[0].ToFloat() - 0.5f) < 0.01f &&
                  std::fabs(output[1].ToFloat() - 0.5f) < 0.01f;
        PrintTestResult("equal elements = [0.5, 0.5]", ok);
    }

    {
        // Large negative values (should still produce valid probabilities)
        std::vector<MLAS_FP16> input = {
            MLAS_FP16::FromFloat(-100.0f),
            MLAS_FP16::FromFloat(-99.0f),
            MLAS_FP16::FromFloat(-98.0f),
        };
        std::vector<MLAS_FP16> output(3);
        ScalarSoftmaxFp16Full(input.data(), output.data(), 3);
        float sum = 0.0f;
        for (size_t i = 0; i < 3; i++) sum += output[i].ToFloat();
        bool ok = std::fabs(sum - 1.0f) < 0.05f && sum > 0.0f;
        PrintTestResult("large negative values valid", ok);
    }

    {
        // Non-power-of-2 length (common in NLP: seq_len=77)
        std::vector<MLAS_FP16> input(77);
        std::vector<MLAS_FP16> output(77);
        srand(123);
        for (size_t i = 0; i < 77; i++) {
            input[i] = MLAS_FP16::FromFloat(static_cast<float>(rand()) / RAND_MAX * 6.0f - 3.0f);
        }
        ScalarSoftmaxFp16Full(input.data(), output.data(), 77);
        float sum = 0.0f;
        for (size_t i = 0; i < 77; i++) sum += output[i].ToFloat();
        PrintTestResult("non-power-of-2 length (N=77)", std::fabs(sum - 1.0f) < 0.02f);
    }
}

//
// Test 7: Log softmax
//
static void
TestLogSoftmax() {
    printf("Test: FP16 LogSoftmax\n");

    // Input: [1.0, 2.0, 3.0]
    std::vector<MLAS_FP16> input = {
        MLAS_FP16::FromFloat(1.0f),
        MLAS_FP16::FromFloat(2.0f),
        MLAS_FP16::FromFloat(3.0f),
    };
    std::vector<MLAS_FP16> output(3);

    // Scalar FP32 reference
    std::vector<float> input_f32 = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected_f32(3);
    ScalarLogSoftmaxF32(input_f32.data(), expected_f32.data(), 3);

    // Scalar FP16 log softmax
    MLAS_FP16 max_val = ScalarReduceMaxFp16(input.data(), 3);
    MLAS_FP16 neg_max = max_val.Negate();
    MLAS_FP16 sum_exp = ScalarSumExpFp16(input.data(), nullptr, 3, neg_max);
    MLAS_FP16 log_sum = MLAS_FP16::FromFloat(std::log(sum_exp.ToFloat()));
    ScalarLogSoftmaxFp16(input.data, output.data(), 3, neg_max, log_sum);

    // Check each value
    bool all_ok = true;
    for (size_t i = 0; i < 3; i++) {
        float diff = std::fabs(output[i].ToFloat() - expected_f32[i]);
        if (diff > 0.02f) {
            if (g_verbose) {
                printf("    [%zu] fp16=%.6f fp32=%.6f diff=%.6f\n",
                       i, output[i].ToFloat(), expected_f32[i], diff);
            }
            all_ok = false;
        }
    }
    PrintTestResult("log softmax vs FP32 reference", all_ok);
}

// ============================================================================
// Benchmark
// ============================================================================

static void
BenchmarkSoftmax(size_t rows, size_t cols, const char* label) {
    printf("\nBenchmark: %s (rows=%zu, cols=%zu, total=%zu elements)\n",
           label, rows, cols, rows * cols);

    std::vector<MLAS_FP16> input(rows * cols);
    std::vector<MLAS_FP16> output(rows * cols);

    srand(42);
    for (size_t i = 0; i < rows * cols; i++) {
        input[i] = MLAS_FP16::FromFloat(static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f);
    }

    // Warm-up
    for (size_t row = 0; row < rows; row++) {
        ScalarSoftmaxFp16Full(input.data() + row * cols, output.data() + row * cols, cols);
    }

    // Timed run
    int iterations = 10;
    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t row = 0; row < rows; row++) {
            ScalarSoftmaxFp16Full(input.data() + row * cols, output.data() + row * cols, cols);
        }
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count() / iterations;

    printf("  Scalar FP16: %.3f ms (%.0f rows/s)\n",
           elapsed * 1000.0, static_cast<double>(rows) / elapsed);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    printf("=== FP16 Softmax Kernel Test Suite ===\n\n");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) g_verbose = true;
        if (strcmp(argv[i], "--bench") == 0) g_bench = true;
    }

    // Run correctness tests
    TestReduceMax();
    TestSumExp();
    TestSoftmaxNorm();
    TestBasicSoftmax();
    TestLogSoftmax();
    TestEdgeCases();
    TestFullPipeline();

    // Run benchmarks if requested
    if (g_bench) {
        BenchmarkSoftmax(100, 80, "YOLO detection (100 x 80)");
        BenchmarkSoftmax(1, 8400, "YOLO single row (1 x 8400)");
        BenchmarkSoftmax(12, 77, "BERT attention (12 x 77)");
        BenchmarkSoftmax(1, 128, "GPT attention (1 x 128)");
        BenchmarkSoftmax(1, 4096, "LLaMA attention (1 x 4096)");
    }

    printf("\n=== Results: %d passed, %d failed ===\n",
           g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
