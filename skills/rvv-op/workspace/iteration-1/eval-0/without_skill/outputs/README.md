# gemv-q5_0-8x8-q8_0

RVV implementation of `ggml_gemv_q5_0_q8_0` - Q5_0 weights x Q8_0 activations GEMV (matrix-vector multiplication) with 8x8 interleaved tile blocking.

## Status

**Implementation Complete** - Ready for testing on RISC-V hardware with VLEN >= 256.

## Motivation

Based on the Qwen2.5-0.5B Q4_K_M profiling analysis (`perf_q4_hotspot_analysis.md`), `ggml_vec_dot_q5_0_q8_0_generic` accounts for **55.45%** of total inference time on the Banana Pi K1 (rv64imafdcv). This is the single largest hotspot in the entire inference pipeline.

While llama.cpp already has an RVV implementation of `ggml_vec_dot_q5_0_q8_0` (single vector dot product), the GEMV variant (`ggml_gemv_q5_0_q8_0`) does not exist in the codebase. The GEMV approach provides better performance through:

1. **Interleaved data layout**: 8 columns are processed together, improving cache locality
2. **Reduced function call overhead**: One GEMV call replaces multiple vec_dot calls
3. **Vector register reuse**: Q8_0 activations are loaded once and reused across 8 columns

## Files

| File | Purpose |
|------|---------|
| `rvv_gemv_q5_0_q8_0.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |
| `README.md` | This file |

## Function Signature

```cpp
void ggml_gemv_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
    int nr, int nc);
```

**Parameters:**
- `n`: Total number of elements per column (must be multiple of QK8_0=32)
- `s`: Output vector of `nc` float results
- `bs`: Stride between output elements (unused, `nc` columns are contiguous)
- `vx`: Interleaved Q5_0 weight blocks (`block_q5_0x8[]`, nc/8 groups of n/QK blocks)
- `vy`: Q8_0 activation blocks (`block_q8_0[]`, n/QK blocks)
- `nr`: Number of rows (unused, always 1 for GEMV)
- `nc`: Total number of columns (must be multiple of 8)

## VLEN Requirement

- **VLEN >= 256**: Uses optimized RVV intrinsics (m1 for 32-element ops, m2 for 128-element ops)
- **VLEN < 256**: Falls back to scalar generic implementation
- **Target**: 512-bit VLEN (vlenb = 64 bytes) for optimal performance

## Data Structures

### block_q5_0x8 (new, 176 bytes)

```cpp
struct block_q5_0x8 {
    ggml_half d[8];       // 8 half-precision scale factors (16 bytes)
    uint8_t qh[8][4];     // 8 x 4-byte qh bitmasks (32 bytes)
    uint8_t qs[128];      // interleaved nibble-packed quants (128 bytes)
};
```

This is a custom structure (not derived from the `template<K,N>` in repack.h) because Q5_0 requires a qh bitmask for 5-bit encoding, which doesn't fit the template pattern.

### block_q8_0 (existing, 34 bytes)

```cpp
typedef struct {
    ggml_half d;       // delta (scale factor)
    int8_t  qs[32];    // quantized values
} block_q8_0;
```

### Q5_0 Quantization Format

Each element is stored as 5-bit signed integer (range -16 to +15):
- **qs[]**: Lower 4 bits of each element, packed as nibbles (2 per byte)
- **qh[]**: 5th bit as a bitmask (1 bit per element, 32 bits per block)

Dequantization: `value = ((nibble & 0x0F) | (qh_bit << 4)) - 16`

### Interleaved qs Layout

The 128-byte qs array stores 8 columns of 32 elements each (4-bit packed = 16 bytes per column):

```
Offset: k * 8 * 8 + j * 8 + i
  k: chunk index (0..3, each chunk has 8 bytes = 16 nibbles)
  j: column index (0..7)
  i: byte index within chunk (0..7)

Each byte contains two nibbles:
  Low nibble (bits 0-3): element at position k*8+i (relative to column j)
  High nibble (bits 4-7): element at position k*8+i+16 (relative to column j)
```

## Algorithm

Based on ARM NEON `ggml_vec_dot_q5_0_q8_0` and RISC-V RVV `ggml_gemv_q4_0_8x8_q8_0`:

### Per-block, per-chunk processing (2 chunks of 8 elements each):

1. **Load activations**: 8 int8 values from Q8_0 block (low half and high half)
2. **Load weights**: 32 bytes of interleaved nibbles for 4 columns
3. **Extract nibbles**: `nib_lo = qs & 0x0F`, `nib_hi = qs >> 4`
4. **Apply qh bitmask**:
   - Load 4 bytes of qh per column as b8 mask
   - Invert mask: `mask = ~qh` (need subtract where bit = 0)
   - Masked subtract: `v = vsub(mask, nib, nib, 0x10)` where mask=1
5. **Replicate activations**: Duplicate 8 a_lo values across 4 columns (32 elements)
6. **Widening multiply**: `i8 * i8 -> i16` for all 32 pairs
7. **Add high/low products**: `mul_total = mul_lo + mul_hi`
8. **Horizontal reduction**: Extract 4 groups of 8 i16, reduce each to i32
9. **Scale and accumulate**: `sumf[j] += sumi[j] * d_x[j] * d_y[l]`

### Two column-pair groups:
- Group 0: columns 0,1,2,3
- Group 1: columns 4,5,6,7

## RVV Intrinsics Used

| Operation | Intrinsic | LMUL | Purpose |
|-----------|-----------|------|---------|
| Load uint8 | `vle8_v_u8m1` | m1 | Load 32 bytes of interleaved nibbles |
| AND mask | `vand_vx_u8m1` | m1 | Extract low nibbles (mask 0x0F) |
| Shift right | `vsrl_vx_u8m1` | m1 | Extract high nibbles (shift by 4) |
| Load mask | `vlm_v_b8` | - | Load qh bitmask as b8 mask |
| Mask NOT | `vmnand_mm_b8` | - | Invert mask for sign extension |
| Masked sub | `vsub_vx_i8m1_mu` | m1 | Subtract 0x10 for sign extension |
| Load int8 | `vle8_v_i8m1` | m1 | Load Q8_0 activations |
| Slide-up | `vslideup_vx_i8m1` | m1-m2 | Replicate activations across columns |
| Widen mul | `vwmul_vv_i16m2` | m2 | i8*i8 -> i16 widening multiply |
| Add | `vadd_vv_i16m2` | m2 | Add low and high nibble products |
| Extract | `vget_v_i16m2_i16m1` | m1 | Extract per-column groups |
| Widen redsum | `vwredsum_vs_i16m1_i32m1` | m1 | Reduce i16 -> i32 |

## Build & Test

```bash
# Cross-compile with RVV extensions (VLEN=512)
riscv64-linux-gnu-g++ -std=c++17 -O2 \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -DGGML_USE_RISCV_V -D__riscv_v_fixed_vlen=512 \
    test.cpp -o test_gemv_q5_0 -lm

# Run under QEMU
qemu-riscv64 -L /path/to/sysroot ./test_gemv_q5_0

# Scalar build (no RVV, for reference comparison)
g++ -std=c++17 -O2 test.cpp -o test_scalar -lm
./test_scalar
```

### Integration with llama.cpp

```bash
# Copy .inl to vendor source directory
cp rvv_gemv_q5_0_q8_0.inl \
    vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/

# Apply patch
cd vendor/llama.cpp
git apply ../../rvv-patches/gemv-q5_0-8x8-q8_0/patch.diff

# Rebuild
cd ../..
./build.sh --skip-source --skip-sysroot
```

## Cross-Platform Analysis

| Platform | File | Approach | Key Technique |
|----------|------|----------|---------------|
| **RISC-V RVV** (this) | `arch/riscv/rvv_vec_dot_q5_0_q8_0.inl` | Dual-path (m1/m2) | vlm b8 mask + masked subtract for qh |
| **ARM NEON** | `arch/arm/quants.c:840` | 2-block unrolled | Lookup table for qh, vdotq_s32 |
| **x86 AVX2** | `arch/x86/quants.c:846` | AVX2 intrinsics | bytes_from_bits_32 for qh, mul_sum_i8_pairs_float |
| **Generic C** | `quants.c:308` | Scalar loop | Bit extraction + sign extension per element |

### Key Differences in qh Handling

| Platform | qh Extraction Method |
|----------|---------------------|
| **RISC-V RVV** (vec_dot) | `vlm_v_b8` + `vmnand_mm` + masked `vsub 0x10` |
| **ARM NEON** | Lookup table `table_b2b_1` for `(!b) << 4`, then `vsubq` |
| **x86 AVX2** | `bytes_from_bits_32` (pshufb-based), then `_mm256_or_si256` |
| **Generic** | `(qh >> j) & 1` per element |

## Relationship to Existing Functions

```
ggml_vec_dot_q5_0_q8_0          (already vectorized in llama.cpp)
  |-- Single vector dot product
  |-- Called per-column, per-block in mul_mat
  |-- Signature: (n, s, bs, vx, bx, vy, by, nrc)

ggml_gemv_q5_0_q8_0            (THIS IMPLEMENTATION - new)
  |-- Matrix-vector multiply, 8 columns interleaved
  |-- Processes entire matrix row at once
  |-- Signature: (n, s, bs, vx, vy, nr, nc)
  |-- Requires block_q5_0x8 interleaved format

ggml_gemv_q4_0_8x8_q8_0        (existing RVV in llama.cpp)
  |-- Same GEMV pattern, Q4_0 weights instead of Q5_0
  |-- Reference for interleaved layout and algorithm structure
```

## Performance Expectations

Based on the profiling data:
- **Current hotspot**: `ggml_vec_dot_q5_0_q8_0_generic` = 55.45% of total execution
- **Expected speedup**: 4-8x over scalar (RVV vectorization + interleaved cache benefits)
- **Projected total impact**: Reduce inference time by ~40-50%

## References

- ARM NEON implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (line 840)
- x86 AVX2 implementation: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (line 846)
- RISC-V RVV vec_dot: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/rvv_vec_dot_q5_0_q8_0.inl`
- RISC-V RVV gemv_q4_0: `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp` (line 114)
- Generic scalar: `vendor/llama.cpp/ggml/src/ggml-cpu/quants.c` (line 308)
- Hotspot analysis: `applications/llama.cpp/temp/perf_q4_hotspot_analysis.md`
