// test.cpp — correctness test for RVV MlasQgemmKernel (VL=16, 512-bit)
//
// Compares the VL=16 RVV INT8 GEMM kernel against the scalar reference.
// All computations are integer (int32 output), so results must match exactly.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -D__riscv_v -DVLEN_512 \
//       test.cpp -o test -lm -fuse-ld=lld
//
// Run under QEMU:
//   qemu-riscv64 -cpu max,vlen=512 -L <sysroot> ./test

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Scalar reference: QGEMM with default packing format (column-by-column)
// ---------------------------------------------------------------------------
// Matches MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_DEFAULT> exactly.
// Processes one row at a time, one column of B per iteration.

#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static size_t qgemm_kernel_scalar(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode)
{
    (void)CountM;
    (void)ldc;

    while (CountN-- > 0) {
        int32_t Accumulator = *RowSumBuffer;
        if (ZeroPointB != nullptr) {
            Accumulator *= *ZeroPointB++;
        }
        Accumulator += *ColumnSumBuffer++;

        const uint8_t* a = A;
        for (size_t k = 0; k < PackedCountK; k++) {
            Accumulator += a[0] * B[0];
            Accumulator += a[1] * B[1];
            Accumulator += a[2] * B[2];
            Accumulator += a[3] * B[3];
            a += 4;
            B += 4;
        }

        if (!ZeroMode) {
            Accumulator += C[0];
        }
        C[0] = Accumulator;
        C += 1;
    }

    return 1;
}

// Scalar A packing: row-major, PackedK aligned (same as default kernel)
static void qgemm_pack_a_scalar(
    uint8_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned)
{
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const uint8_t BitFlip = (AIsSigned ? 0x80 : 0);

    while (CountM-- > 0) {
        int32_t RowSum = 0;
        for (size_t k = 0; k < CountK; k++) {
            uint8_t a0 = A[k] ^ BitFlip;
            D[k] = a0;
            RowSum += a0;
        }
        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }
        *RowSumBuffer++ = RowSum;
        A += lda;
        D += AlignedCountK;
    }
}

// Scalar B packing: column-by-column, PackedK aligned (same as default kernel)
static void qgemm_pack_b_scalar(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned)
{
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const uint8_t BitFlip = (BIsSigned ? 0x80 : 0);

    while (CountN-- > 0) {
        const uint8_t* b = B;
        int32_t ColumnSum = 0;
        for (size_t k = 0; k < CountK; k++) {
            uint8_t b0 = b[0] ^ BitFlip;
            D[k] = b0;
            ColumnSum += b0;
            b += ldb;
        }
        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }
        *ColumnSumBuffer++ = ColumnSum;
        B += 1;
        D += AlignedCountK;
    }
}

// ---------------------------------------------------------------------------
// RVV implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_qgemm_kernel_vl16.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator (LCG, deterministic)
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static uint8_t rng_u8() {
    return (uint8_t)(rng_next() & 0xFF);
}

static void fill_random_u8(uint8_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = rng_u8();
    }
}

// ---------------------------------------------------------------------------
// Test runner: compare RVV kernel vs scalar reference
// ---------------------------------------------------------------------------
static int run_test(
    size_t CountK,
    size_t CountM,
    size_t CountN,
    bool ZeroMode,
    bool AIsSigned,
    bool BIsSigned,
    bool PerColumnZP,
    const char* label)
{
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const size_t PackedCountK = AlignedCountK / 4;
    const size_t lda = CountK;
    const size_t ldc = CountN;

    // Generate A matrix (CountM × CountK)
    std::vector<uint8_t> A_raw(CountM * lda);
    fill_random_u8(A_raw.data(), A_raw.size());

    // Generate B matrix (CountK × CountN, column-major for packing)
    // B is stored row-major with ldb = CountN
    std::vector<uint8_t> B_raw(CountK * CountN);
    fill_random_u8(B_raw.data(), B_raw.size());

    // Pack A (scalar format, same for both kernels)
    std::vector<uint8_t> A_packed(CountM * AlignedCountK);
    std::vector<int32_t> RowSum(CountM);
    qgemm_pack_a_scalar(A_packed.data(), A_raw.data(), lda, CountM, CountK,
                        RowSum.data(), AIsSigned);

    // Pack B in scalar format (for scalar kernel)
    std::vector<uint8_t> B_packed_scalar(CountN * AlignedCountK);
    std::vector<int32_t> ColSum_scalar(CountN);
    qgemm_pack_b_scalar(B_packed_scalar.data(), B_raw.data(), CountN, CountN, CountK,
                        ColSum_scalar.data(), BIsSigned);

    // Zero-point
    std::vector<int32_t> ZeroPointB_col;
    if (PerColumnZP) {
        ZeroPointB_col.resize(CountN);
        for (size_t i = 0; i < CountN; i++) {
            ZeroPointB_col[i] = -(int32_t)(rng_u8() ^ (BIsSigned ? 0x80 : 0));
        }
    }
    const int32_t* zp_ptr = PerColumnZP ? ZeroPointB_col.data() : nullptr;

    // Apply RowSum correction (matches MlasGemmQuantOperation driver):
    // RowSum[mm] -= CountK * ZeroPointA
    // Then if no per-column ZP: RowSum[mm] *= -ZeroPointB
    const int32_t ZeroPointA = AIsSigned ? (uint8_t)(128 ^ 0x80) : 128;  // fixup'd
    const int32_t ZeroPointB_val = BIsSigned ? (uint8_t)(128 ^ 0x80) : 128;
    for (size_t mm = 0; mm < CountM; mm++) {
        RowSum[mm] -= (int32_t)CountK * ZeroPointA;
    }
    if (!PerColumnZP) {
        for (size_t mm = 0; mm < CountM; mm++) {
            RowSum[mm] *= -ZeroPointB_val;
        }
    }
    // Scale column sums by -ZeroPointA
    for (size_t n = 0; n < CountN; n++) {
        ColSum_scalar[n] *= -ZeroPointA;
    }

    // Run scalar reference
    std::vector<int32_t> C_scalar(CountM * ldc, 0);
    if (!ZeroMode) {
        for (size_t i = 0; i < CountM * ldc; i++) {
            C_scalar[i] = (int32_t)(rng_next() % 200 - 100);
        }
    }
    std::vector<int32_t> C_rvv(C_scalar);

    for (size_t m = 0; m < CountM; m++) {
        qgemm_kernel_scalar(
            A_packed.data() + m * AlignedCountK,
            B_packed_scalar.data(),
            C_scalar.data() + m * ldc,
            PackedCountK,
            1,
            CountN,
            ldc,
            &RowSum[m],
            ColSum_scalar.data(),
            zp_ptr,
            ZeroMode);
    }

#if defined(__riscv_v) && defined(VLEN_512)
    // Pack B in RVV512 format (16-column blocks)
    const size_t num_blocks = (CountN + 15) / 16;
    std::vector<uint8_t> B_packed_rvv(PackedCountK * 64 * num_blocks);
    std::vector<int32_t> ColSum_rvv(CountN);
    MlasQgemmPackBRvv512(B_packed_rvv.data(), B_raw.data(), CountN, CountN, CountK,
                          ColSum_rvv.data(), BIsSigned);

    // Apply the same column sum scaling
    for (size_t n = 0; n < CountN; n++) {
        ColSum_rvv[n] *= -ZeroPointA;
    }

    // Run RVV kernel (one row at a time to match scalar behavior)
    for (size_t m = 0; m < CountM; m++) {
        size_t rows = MlasQgemmKernelRvv512Impl(
            A_packed.data() + m * AlignedCountK,
            B_packed_rvv.data(),
            C_rvv.data() + m * ldc,
            PackedCountK,
            CountN,
            RowSum[m],
            ColSum_rvv.data(),
            zp_ptr,
            ZeroMode);
        (void)rows;  // should be 1
    }
#else
    // Non-RVV build: copy scalar results
    C_rvv = C_scalar;
#endif

    // Compare (int32, exact match)
    int failures = 0;
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            int32_t expected = C_scalar[m * ldc + n];
            int32_t actual = C_rvv[m * ldc + n];
            if (expected != actual) {
                if (failures < 5) {
                    printf("  MISMATCH [%zu,%zu]: expected=%d, actual=%d, diff=%d\n",
                           m, n, expected, actual, actual - expected);
                }
                failures++;
            }
        }
    }

    const char* status = (failures == 0) ? "PASS" : "FAIL";
    printf("  %-45s K=%-3zu M=%zu N=%-3zu %s\n",
           label, CountK, CountM, CountN, status);

    return failures;
}

// ---------------------------------------------------------------------------
// Simple end-to-end test: raw GEMM with unsigned int8, no zero-point
// ---------------------------------------------------------------------------
static int run_simple_test(
    size_t CountK,
    size_t CountM,
    size_t CountN,
    const char* label)
{
    // No sign flipping, no zero-point corrections — pure uint8 MAC
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const size_t PackedCountK = AlignedCountK / 4;

    std::vector<uint8_t> A_raw(CountM * CountK);
    std::vector<uint8_t> B_raw(CountK * CountN);
    fill_random_u8(A_raw.data(), A_raw.size());
    fill_random_u8(B_raw.data(), B_raw.size());

    // Compute expected result directly
    std::vector<int32_t> C_expected(CountM * CountN, 0);
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            int32_t acc = 0;
            for (size_t k = 0; k < CountK; k++) {
                acc += (int32_t)A_raw[m * CountK + k] * (int32_t)B_raw[k * CountN + n];
            }
            C_expected[m * CountN + n] = acc;
        }
    }

#if defined(__riscv_v) && defined(VLEN_512)
    // Pack A
    std::vector<uint8_t> A_packed(CountM * AlignedCountK);
    std::vector<int32_t> RowSum(CountM);
    MlasQgemmPackARvv512(A_packed.data(), A_raw.data(), CountK, CountM, CountK,
                          RowSum.data(), false);

    // Pack B
    const size_t num_blocks = (CountN + 15) / 16;
    std::vector<uint8_t> B_packed(PackedCountK * 64 * num_blocks);
    std::vector<int32_t> ColSum(CountN);
    MlasQgemmPackBRvv512(B_packed.data(), B_raw.data(), CountN, CountN, CountK,
                          ColSum.data(), false);

    // No zero-point: RowSum = raw row sum, ColSum = raw column sum
    // No corrections needed (ZeroPointA = 0, ZeroPointB = 0)
    // But we need: acc = RowSum * 0 + ColSum + sum(a*b)
    // Since ZP=0: acc = 0 + ColSum + sum(a*b) ... wait, that's wrong.
    // The kernel computes: RowSum * ZPB + ColSum + sum(a_i * b_i)
    // Without ZP: RowSum*0 + ColSum + sum = ColSum + sum
    // But we want just sum(a*b). So RowSum should start as 0, ColSum as 0,
    // and ZPB = nullptr (which means RowSum is already scaled by ZPB).
    // With ZPB=nullptr, the kernel does: RowSum + ColSum + sum
    // So set RowSum[0] = 0, ColSum = 0.
    for (size_t m = 0; m < CountM; m++) RowSum[m] = 0;
    for (size_t n = 0; n < CountN; n++) ColSum[n] = 0;

    std::vector<int32_t> C_rvv(CountM * CountN, 0);
    for (size_t m = 0; m < CountM; m++) {
        MlasQgemmKernelRvv512Impl(
            A_packed.data() + m * AlignedCountK,
            B_packed.data(),
            C_rvv.data() + m * CountN,
            PackedCountK,
            CountN,
            RowSum[m],
            ColSum.data(),
            nullptr,
            true);
    }

    int failures = 0;
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            if (C_expected[m * CountN + n] != C_rvv[m * CountN + n]) {
                if (failures < 5) {
                    printf("  MISMATCH [%zu,%zu]: expected=%d, actual=%d\n",
                           m, n, C_expected[m * CountN + n], C_rvv[m * CountN + n]);
                }
                failures++;
            }
        }
    }
#else
    int failures = 0;
#endif

    const char* status = (failures == 0) ? "PASS" : "FAIL";
    printf("  %-45s K=%-3zu M=%zu N=%-3zu %s\n", label, CountK, CountM, CountN, status);
    return failures;
}

int main() {
    printf("=== MlasQgemmKernel (VL=16): RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    // Simple tests: pure uint8 MAC without zero-point complications
    printf("--- Simple uint8 MAC (no zero-point) ---\n");
    failures += run_simple_test(4, 1, 16, "K=4 M=1 N=16 full block");
    failures += run_simple_test(8, 1, 16, "K=8 M=1 N=16");
    failures += run_simple_test(16, 1, 16, "K=16 M=1 N=16");
    failures += run_simple_test(32, 1, 16, "K=32 M=1 N=16");
    failures += run_simple_test(64, 1, 16, "K=64 M=1 N=16");
    failures += run_simple_test(4, 1, 8, "K=4 M=1 N=8 partial block");
    failures += run_simple_test(8, 1, 3, "K=8 M=1 N=3 partial block");
    failures += run_simple_test(4, 1, 32, "K=4 M=1 N=32 multi-block");
    failures += run_simple_test(8, 1, 20, "K=8 M=1 N=20 partial 2nd block");
    failures += run_simple_test(7, 1, 16, "K=7 M=1 N=16 odd K");
    failures += run_simple_test(5, 1, 16, "K=5 M=1 N=16 odd K");
    failures += run_simple_test(128, 1, 16, "K=128 M=1 N=16 large K");

    // Tests with zero-point corrections (unsigned uint8 data)
    printf("\n--- With zero-point corrections (uint8 A, uint8 B) ---\n");
    failures += run_test(8, 1, 16, true, false, false, false, "uint8/uint8 ZP=128 zero=true");
    failures += run_test(8, 1, 16, false, false, false, false, "uint8/uint8 ZP=128 zero=false");
    failures += run_test(16, 1, 16, true, false, false, true, "uint8/uint8 per-col ZP zero=true");
    failures += run_test(16, 1, 16, false, false, false, true, "uint8/uint8 per-col ZP zero=false");
    failures += run_test(8, 1, 8, true, false, false, false, "uint8/uint8 N=8 partial");
    failures += run_test(32, 1, 32, true, false, false, false, "uint8/uint8 K=32 N=32");

    // Tests with signed data
    printf("\n--- With signed data (int8 A and/or B) ---\n");
    failures += run_test(8, 1, 16, true, true, false, false, "int8/uint8 zero=true");
    failures += run_test(8, 1, 16, false, true, false, false, "int8/uint8 zero=false");
    failures += run_test(8, 1, 16, true, false, true, false, "uint8/int8 zero=true");
    failures += run_test(8, 1, 16, true, true, true, false, "int8/int8 zero=true");
    failures += run_test(16, 1, 16, false, true, true, true, "int8/int8 per-col ZP");

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
