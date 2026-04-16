# quantize-q8_0-4x4

RVV implementation of `ggml_quantize_mat_q8_0_4x4` - Quantize FP32 activations to Q8_0 format with 4x4 interleaving.

## Files

| File | Purpose |
|------|---------|
| `rvv_quantize_q8_0_4x4.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Status

✅ **Tests passing** — RVV implementation and scalar reference both produce correct interleaved output matching `vsseg4e32` segment store layout.

## Function Signature

```cpp
void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
```

## Algorithm

1. Load 32 FP32 values from each of 4 rows (128 total)
2. Compute max absolute value for each row (for scale)
3. Scale to int8 range [-127, 127] with rounding
4. Interleave 4 rows using RVV segment store (vsseg4e32)

## Interleaving Format

The `block_q8_0x4` structure (from llama.cpp `repack.h`):
```cpp
template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};
using block_q8_0x4 = block<8, 4>;  // N=4 rows, K=8 (QK8_0)
```

RVV `vsseg4e32` interleaving layout (stride=4):
- Each int32 unit contains 4 consecutive int8 values from the same row
- Memory: row0[0:4], row1[0:4], row2[0:4], row3[0:4], row0[4:8], row1[4:8], ...
- Index formula: `pos = 4*r + (j/4)*16 + (j%4)` for row r, element j

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-quantize-mat-q8-0-4x4-2026-04-15.md`

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test
./build.sh --test
```