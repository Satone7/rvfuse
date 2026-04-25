# qgemm-kernel-vl16

RVV implementation of `MlasGemmQuantKernel` — quantized INT8 matrix/matrix multiply (uint8 x uint8 -> int32) targeting 512-bit vector registers.

## Status

✅ Tests passing (24/24 under QEMU VLEN=512), BBV profiling complete

**Note**: Vanilla ORT RISC-V INT8 path crashes (segfault at ConvInteger warm-up). This kernel is both a performance optimization and a functional necessity for INT8 inference on RISC-V.

## Files

| File | Purpose |
|------|---------|
| `rvv_qgemm_kernel_vl16.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
size_t MlasQgemmKernelRvv512Impl(
    const uint8_t* A,           // Packed A matrix (row-major, PackedK aligned)
    const uint8_t* B,           // Packed B matrix (16-column blocks)
    int32_t* C,                 // Output matrix
    size_t PackedCountK,        // Number of PackedK groups
    size_t CountN,              // Number of output columns
    int32_t RowSum,             // Row sum correction for current A row
    const int32_t* ColumnSumBuffer,  // Column sum corrections
    const int32_t* ZeroPointB,       // Per-column zero-point (or nullptr)
    bool ZeroMode               // If true, overwrite C; if false, accumulate
);
```

## Algorithm

1. Initialize 16-wide int32 accumulator from RowSum/ColumnSum/ZeroPointB corrections
2. K-loop (PackedCountK iterations, 4 K elements each, unrolled):
   - Load 4 A scalars (uint8)
   - For each K element: vle8.v (16 B cols) + vwmulu.vx (u8*u8->u16) + vwaddu.wv (u32+=u16)
3. Store 16 int32 results to C (handle partial tail blocks)

## B Matrix Packing (16-column blocks)

Unlike the default kernel which packs B column-by-column, this kernel packs B in 16-column blocks for vector-friendly memory access:

```
For each 16-column block:
  For each PackedK group (4 K elements):
    k0: [col0..col15]  (16 bytes contiguous)
    k1: [col0..col15]
    k2: [col0..col15]
    k3: [col0..col15]
  = 64 bytes per PackedK group per block
```

This enables a single `vle8.v` instruction per K element per 16 columns, compared to scalar's sequential byte access.

## VLEN Requirement

- **VLEN >= 512**: Uses RVV intrinsics with VL=16 for 16-column processing
- **VLEN < 512**: Falls back to scalar (not implemented in this .inl)

## Performance Estimates

### Current implementation (per K element, 16 columns)

| Instruction | Latency | Count/K group |
|-------------|---------|---------------|
| vle8.v      | ~3 cyc  | 4 (one per K element) |
| vwmulu.vx   | ~4 cyc  | 4 |
| vwaddu.wv   | ~4 cyc  | 4 |
| **Total**   | **~44 cyc** | **per 4K x 16N** |
| **Per K element** | **~11 cyc** | |

### With vsegdot.vv extension (hypothetical)

| Instruction | Latency | Count/4K group |
|-------------|---------|----------------|
| vle32.v     | ~3 cyc  | 1 (load 4 A elements packed) |
| vle8.v      | ~3 cyc  | 1 (load 64 B elements) |
| vsegdot.vv  | ~7 cyc  | 1 (4×i8 dot→i32, 16 cols) |
| **Total**   | **~13 cyc** | **per 4K x 16N** |
| **Per K element** | **~3.25 cyc** | |

Speedup: 11 / 3.25 ≈ 3.4× for the K-loop

## Build & Test

```bash
# Build standalone test (rv64gcv, VLEN=512)
clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=output/cross-ort/sysroot \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v -DVLEN_512 \
    -I applications/onnxrt/rvv-patches/qgemm-kernel-vl16 \
    applications/onnxrt/rvv-patches/qgemm-kernel-vl16/test.cpp \
    -o output/qgemm_test -lm -fuse-ld=lld

# Run under QEMU
third_party/qemu/build/qemu-riscv64 \
    -cpu max,vlen=512 \
    -L output/cross-ort/sysroot \
    output/qgemm_test
```
