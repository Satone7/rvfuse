/*++

Copyright (c) RVFuse Project. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmKernelRvv512.cpp

Abstract:

    This module implements the RVV (RISC-V Vector Extension) kernel for the
    quantized integer matrix/matrix multiply operation (QGEMM) targeting
    512-bit vector registers (VLEN=512, SEW=32, LMUL=1, VL=16).

    The implementation is in rvv_qgemm_kernel_vl16.inl (single source of truth),
    shared with the standalone correctness test.

--*/

#include "mlasi.h"
#include "qgemm.h"

#if defined(__riscv_v) && defined(VLEN_512)

#include "rvv_qgemm_kernel_vl16.inl"

struct MLAS_GEMM_U8X8_KERNEL_RVV512
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{1, 128, 128};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{0, 0, 0};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_RVV512::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_RVV512::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_RVV512::PackedStrides;

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_U8X8_KERNEL_RVV512>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_RVV512>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_RVV512::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_RVV512>(
    MLAS_GEMM_U8X8_KERNEL_RVV512::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MlasQgemmPackARvv512(D, A, lda, CountM, CountK, RowSumBuffer, AIsSigned);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_RVV512>(
    MLAS_GEMM_U8X8_KERNEL_RVV512::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MlasQgemmPackBRvv512(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_RVV512>(
    const MLAS_GEMM_U8X8_KERNEL_RVV512::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_RVV512::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    return MlasQgemmKernelRvv512Impl(
        A, B, C, PackedCountK, CountN,
        RowSumBuffer[0], ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchRvv512 = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_RVV512>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_RVV512::PackedK,
    0,
    1
};

#endif // __riscv_v && VLEN_512
