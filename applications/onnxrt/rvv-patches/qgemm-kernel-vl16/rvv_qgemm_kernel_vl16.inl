// rvv_qgemm_kernel_vl16.inl — RVV implementation of MlasQemmKernel (VL=16, 512-bit)
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/riscv64/QgemmKernelRvv512.cpp  (production build)
//   - test.cpp                                                 (correctness test)
//
// This kernel targets 512-bit vector registers (VLEN=512, SEW=32, LMUL=1, VL=16).
// It processes 16 output columns per vector iteration using unsigned widening
// multiply-accumulate (uint8 × uint8 → uint16 → uint32).
//
// Signed/unsigned handling (same as MLAS_GEMM_QUANT_KERNEL_DEFAULT):
//   All data is stored as unsigned uint8. Signed data is XOR'd with 0x80 during
//   packing. Zero-points are fixup'd to match. The RowSum/ColumnSum/ZeroPoint
//   corrections compensate for the XOR offsets mathematically.
//
// Two's complement arithmetic: the kernel uses uint32 throughout. Signed
// corrections (RowSum, ColumnSum, ZeroPointB) are represented as uint32 values
// whose two's complement bit patterns give the correct int32 result. This is
// safe because the final output range fits within int32 for all practical
// QGEMM dimensions.
//
// B packing format (16-column blocks):
//   For each group of 16 columns (last block may be partial):
//     For each PackedK group (4 K elements):
//       k0: [col0 col1 ... col15]  (16 bytes contiguous)
//       k1: [col0 col1 ... col15]
//       k2: [col0 col1 ... col15]
//       k3: [col0 col1 ... col15]
//     = 64 bytes per PackedK group per 16-column block
//   ColumnSumBuffer is indexed by column (same as default kernel).
//
// Prerequisites:
//   - __riscv_v must be defined (V extension)
//   - VLEN_512 must be defined
//   - <riscv_vector.h> must be included before this file

#include <cassert>
#include <cstring>

#if defined(__riscv_v) && defined(VLEN_512)

#include <riscv_vector.h>

// =============================================================================
// Standalone RVV512 INT8 GEMM kernel
// =============================================================================
// Processes 1 row × CountN columns. CountN is processed in 16-column blocks.
// Returns 1 (number of rows processed).

inline size_t
MlasQgemmKernelRvv512Impl(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountN,
    int32_t RowSum,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    constexpr size_t kVlen = 16;
    size_t remaining_n = CountN;
    const uint8_t* b_block = B;

    // Convert RowSum to uint32 for two's complement arithmetic
    const uint32_t row_sum_u = static_cast<uint32_t>(RowSum);

    do {
        const size_t n_this = (remaining_n >= kVlen) ? kVlen : remaining_n;
        const size_t vl = __riscv_vsetvl_e32m1(n_this);
        const size_t vl8 = __riscv_vsetvl_e8mf4(n_this);

        // Initialize accumulator: RowSum * ZeroPointB + ColumnSum (all as uint32)
        vuint32m1_t vacc;

        if (ZeroPointB != nullptr) {
            // Load ZeroPointB as uint32 (two's complement of negated zero-point)
            vuint32m1_t v_zpb;
            // ZeroPointB contains int32 values; load as uint32 for 2's complement
            memcpy(&v_zpb, ZeroPointB, n_this * sizeof(uint32_t));
            ZeroPointB += n_this;
            vacc = __riscv_vmul_vx_u32m1(v_zpb, row_sum_u, vl);
            vuint32m1_t v_colsum;
            memcpy(&v_colsum, ColumnSumBuffer, n_this * sizeof(uint32_t));
            vacc = __riscv_vadd_vv_u32m1(vacc, v_colsum, vl);
        } else {
            vuint32m1_t v_colsum;
            memcpy(&v_colsum, ColumnSumBuffer, n_this * sizeof(uint32_t));
            vacc = __riscv_vadd_vx_u32m1(v_colsum, row_sum_u, vl);
        }
        ColumnSumBuffer += n_this;

        // K-loop: PackedCountK groups, each with 4 K elements
        const uint8_t* a = A;
        const uint8_t* b = b_block;

        for (size_t pk = 0; pk < PackedCountK; pk++) {

            const uint8_t a0 = a[0];
            const uint8_t a1 = a[1];
            const uint8_t a2 = a[2];
            const uint8_t a3 = a[3];

            // K element 0: a0 * B[0..15]
            vuint8mf4_t vb0 = __riscv_vle8_v_u8mf4(b, vl8);
            vuint16mf2_t vp0 = __riscv_vwmulu_vx_u16mf2(vb0, a0, vl8);
            vacc = __riscv_vwaddu_wv_u32m1(vacc, vp0, vl);

            // K element 1: a1 * B[16..31]
            vuint8mf4_t vb1 = __riscv_vle8_v_u8mf4(b + 16, vl8);
            vuint16mf2_t vp1 = __riscv_vwmulu_vx_u16mf2(vb1, a1, vl8);
            vacc = __riscv_vwaddu_wv_u32m1(vacc, vp1, vl);

            // K element 2: a2 * B[32..47]
            vuint8mf4_t vb2 = __riscv_vle8_v_u8mf4(b + 32, vl8);
            vuint16mf2_t vp2 = __riscv_vwmulu_vx_u16mf2(vb2, a2, vl8);
            vacc = __riscv_vwaddu_wv_u32m1(vacc, vp2, vl);

            // K element 3: a3 * B[48..63]
            vuint8mf4_t vb3 = __riscv_vle8_v_u8mf4(b + 48, vl8);
            vuint16mf2_t vp3 = __riscv_vwmulu_vx_u16mf2(vb3, a3, vl8);
            vacc = __riscv_vwaddu_wv_u32m1(vacc, vp3, vl);

            a += 4;
            b += 64;  // 4 K elements × 16 columns = 64 bytes
        }

        // Store results: uint32 → int32 (same bits, two's complement)
        if (n_this == kVlen) {
            if (!ZeroMode) {
                vuint32m1_t v_c = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(C), vl);
                vacc = __riscv_vadd_vv_u32m1(vacc, v_c, vl);
            }
            __riscv_vse32_v_u32m1(reinterpret_cast<uint32_t*>(C), vacc, vl);
            C += kVlen;
        } else {
            uint32_t tmp[16];
            __riscv_vse32_v_u32m1(tmp, vacc, vl);
            for (size_t i = 0; i < n_this; i++) {
                uint32_t val = tmp[i];
                if (!ZeroMode) {
                    val += static_cast<uint32_t>(C[i]);
                }
                C[i] = static_cast<int32_t>(val);
            }
            C += n_this;
        }

        remaining_n -= n_this;
        b_block += PackedCountK * 64;  // Next 16-column block

    } while (remaining_n > 0);

    return 1;
}

// =============================================================================
// Standalone B packing function (16-column block format)
// =============================================================================

inline void
MlasQgemmPackBRvv512(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const size_t PackedCountK = AlignedCountK / 4;

    for (size_t n = 0; n < CountN; n++) {
        ColumnSumBuffer[n] = 0;
    }

    size_t col_start = 0;
    while (col_start < CountN) {
        const size_t n_this = (CountN - col_start >= 16) ? 16 : (CountN - col_start);

        for (size_t pk = 0; pk < PackedCountK; pk++) {
            for (size_t kk = 0; kk < 4; kk++) {
                const size_t k = pk * 4 + kk;
                for (size_t n = 0; n < 16; n++) {
                    const size_t col_idx = col_start + n;
                    if (k < CountK && n < n_this) {
                        const uint8_t val = B[col_idx + k * ldb] ^ BitFlipValue;
                        D[n] = val;
                        ColumnSumBuffer[col_idx] += val;
                    } else {
                        D[n] = 0;
                    }
                }
                D += 16;
            }
        }

        col_start += n_this;
    }
}

// =============================================================================
// Standalone A packing function (row-major, PackedK aligned)
// =============================================================================

inline void
MlasQgemmPackARvv512(
    uint8_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    const size_t AlignedCountK = (CountK + 3) & ~3u;
    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);

    while (CountM-- > 0) {
        int32_t RowSum = 0;
        for (size_t k = 0; k < CountK; k++) {
            const uint8_t a0 = A[k] ^ BitFlipValue;
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

#endif // __riscv_v && VLEN_512
