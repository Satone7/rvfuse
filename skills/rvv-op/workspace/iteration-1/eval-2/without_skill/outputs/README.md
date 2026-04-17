# RVV FP16 Softmax Patch for ONNX Runtime MLAS

## Overview

This patch adds RISC-V Vector Extension (RVV) support for FP16 softmax kernels
in ONNX Runtime's MLAS (Multi-platform Linear Algebra Subprograms) library.

**Target**: VLEN=512-bit vector registers with Zvfh (half-precision floating-point
vector extension).

## Files

| File | Description |
|------|-------------|
| `softmax_fp16_rvv512.inl` | Core RVV FP16 softmax implementation (vectorized kernels) |
| `test_softmax_fp16.cpp` | Test harness with scalar reference implementations |
| `patch.diff` | MLAS integration patch for ONNX Runtime |
| `README.md` | This documentation |

## Vector Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| VLEN | 512 bits | Vector register width |
| SEW | 16 bits | FP16 element size |
| LMUL | 1 | Single vector register group |
| VL | 32 elements | Elements per vector register (512/16) |
| Extension | Zvfh | Half-precision FP vector extension |

## Implemented Kernels

The patch implements 5 kernels matching the `MLAS_SOFTMAX_DISPATCH` interface:

### 1. `ReduceMax_Kernel_Fp16_Rvv512`

Finds the maximum value in an FP16 array.

- **Input**: FP16 array, length N
- **Output**: Maximum value as FP16
- **Strategy**: 4x unrolled vector comparison, horizontal reduction via slide-down + vfmax
- **Loop granularity**: 128 elements per iteration (4 x VL=32)

### 2. `SumExp_Kernel_Fp16_Rvv512`

Computes `exp(x - max)` for each element and returns the sum.

- **Input**: FP16 array, negative maximum value
- **Output**: Sum of exponentials (FP16), optionally stores exp values
- **Strategy**: 4x unrolled vectorized exp with polynomial approximation
- **Exp approximation**: Range reduction to log2 domain + Horner polynomial evaluation

### 3. `Softmax_Kernel_Fp16_Rvv512`

Final normalization: `output[i] = input[i] / sum`.

- **Input**: FP16 array of exp values, accumulated sum
- **Output**: Normalized probabilities (FP16)
- **Strategy**: 4x unrolled vector multiply (each element / sum)

### 4. `Exp_Kernel_Fp16_Rvv512`

Computes `exp(x)` for each element.

- **Input**: FP16 array, valid range [-17.3, 11.1]
- **Output**: FP16 array of exp values
- **Strategy**: Full range exp with overflow handling

### 5. `LogSoftmax_Kernel_Fp16_Rvv512`

Computes log softmax: `output[i] = input[i] + neg_max - log_sum`.

- **Input**: Original input, negative maximum, log of exp sum
- **Output**: Log softmax values (FP16)
- **Strategy**: 4x unrolled vector add + subtract

## Algorithm: FP16 Exp Approximation

The exp function uses a polynomial approximation matching the approach used by
the ARM NEON FP16 kernel in MLAS:

```
1. Clamp input to [LowerRange, UpperRange]
2. Range reduction: biased = x * (1/ln2) + rounding_bias
3. Integer part: m = biased - rounding_bias
4. Residual: r = x + m * ln2_high + m * ln2_mid + m * ln2_low
5. Exponent reconstruction: normal = int16(biased) << 10 + max_exponent_bias
6. Polynomial: p = poly_0 + r*(poly_1 + r*(poly_2 + r*(poly_3 + r*(poly_4 + r*poly_56))))
7. Result: p * reinterpret_as_fp16(normal)
```

FP16 bit-pattern constants match the MLAS `ExpConstantsFp16` values from the
ARM NEON implementation (`softmax_kernel_neon_fp16.cpp`).

## Softmax Pipeline (3-step)

The MLAS softmax pipeline processes each row through three stages:

```
Step 1: max = ReduceMax(input, N)
Step 2: sum = SumExp(input, temp, N, -max)
Step 3: Softmax(temp, output, N, sum)
```

For log softmax, step 2 uses `temp=nullptr` and step 3 uses `LogSoftmax` instead.

## Performance Estimates

Compared to scalar (non-vectorized) FP16 softmax:

| Kernel | Scalar | RVV VLEN=512 | Speedup |
|--------|--------|--------------|---------|
| ReduceMax | N comparisons | N/32 vector ops | ~32x |
| SumExp | N scalar exp | N/32 vectorized exp | ~32x |
| Softmax | N divisions | N/32 vector mul | ~32x |

For YOLO inference with 8400 rows x 80 classes:
- **Scalar**: ~672,000 individual FP16 operations per softmax
- **RVV-512**: ~21,000 vector operations per softmax (4x unroll = ~5,250 iterations)

**Note**: Actual speedup depends on memory bandwidth, pipeline latency, and
the exp polynomial approximation cost (which is non-trivial).

## Cross-Platform Reference

This implementation is modeled after the ARM NEON FP16 softmax kernel:
- `onnxruntime/core/mlas/lib/softmax_kernel_neon_fp16.cpp`
- `onnxruntime/core/mlas/lib/softmax.h` (dispatch interface)
- `onnxruntime/core/mlas/lib/fp16_common.h` (NEON FP16 helpers)

Key differences:
- ARM NEON processes 8 (float16x8) or 4 (float16x4) elements per instruction
- RVV VLEN=512 processes 32 elements per instruction (4x NEON width)
- ARM NEON uses `vpaddq_f16` for horizontal reduction; RVV uses `vslidedown` + `vfadd`

## Not Yet Implemented

The following kernels from `MLAS_SOFTMAX_DISPATCH` are left as `nullptr`:

- `Tanh_Fp16` - Hyperbolic tangent (used by GQA softcap)
- `Softcap_Fp16` - Soft capping function

These can be added following the same pattern using the Tanh polynomial
approximation from `softmax_kernel_neon_fp16.cpp`.

## Requirements

- RISC-V 64-bit target (`rv64gcv_zvfh`)
- GCC or Clang with RVV intrinsics support (`<riscv_vector.h>`)
- Zvfh extension (half-precision floating-point vector instructions)
- VLEN=512 (compile-time configuration via `VLEN_512` define)

## Usage

Apply to ONNX Runtime source tree before building:

```bash
cd applications/yolo/ort/vendor/onnxruntime
patch -p1 < ../../../rvv-patches/patch.diff
```

Copy the `.inl` file alongside the new kernel source:

```bash
cp ../../../rvv-patches/softmax_fp16_rvv512.inl \
   onnxruntime/core/mlas/lib/riscv64/softmax_fp16_rvv512.inl
```

Build with RISC-V cross-toolchain:

```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../riscv64-linux-toolchain.cmake \
  -Donnxruntime_MLAS_RVV_VLEN_512=ON \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv_zvfh"
```

## Testing

Build and run the test harness:

```bash
# Cross-compile for RISC-V
riscv64-unknown-linux-gnu-g++ -std=c++17 -O2 \
  -march=rv64gcv_zvfh -DVLEN_512 \
  test_softmax_fp16.cpp -o test_softmax_fp16

# Run under QEMU
qemu-riscv64 -L <sysroot> ./test_softmax_fp16 --verbose --bench
```

For scalar reference testing on x86:

```bash
g++ -std=c++17 -O2 test_softmax_fp16.cpp -o test_softmax_fp16_scalar
./test_softmax_fp16_scalar --verbose
```
