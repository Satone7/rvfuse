# vec_dot-q6_K_q8_K

RVV VL512 implementation of `ggml_vec_dot_q6_K_q8_K` - dot product between Q6_K (6-bit quantized) and Q8_K (8-bit quantized) vectors.

## Files

| File | Purpose |
|------|---------|
| `rvv_vec_dot_q6_K_q8_K.inl` | RVV VL512 implementation (single source of truth) |
| `patch.diff` | Patch to integrate into llama.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);
```

## Data Structures

**block_q6_K** (256 elements, 210 bytes):
- `ql[128]`: Lower 4 bits (packed, 2 values per byte)
- `qh[64]`: Upper 2 bits (packed, 4 values per byte)
- `scales[16]`: Per-16-element-group int8 scales
- `d`: Super-block FP16 scale

**block_q8_K** (256 elements, 292 bytes):
- `d`: Block FP32 scale
- `qs[256]`: 8-bit signed quantized values
- `bsums[16]`: Block sums (not used in vec_dot)

## Algorithm

For each super-block (256 elements):
1. Compute `d = x[i].d * y[i].d` (super-block scale)
2. For each sub-block (128 elements, 2 per super-block):
   a. Load qh (32 bytes) and ql (64 bytes)
   b. Unpack 6-bit values:
      - a[0..31] = (ql[0..31] & 0xF) | ((qh >> 0) & 3) << 4) - 32
      - a[32..63] = (ql[32..63] & 0xF) | ((qh >> 2) & 3) << 4) - 32
      - a[64..95] = (ql[0..31] >> 4) | ((qh >> 4) & 3) << 4) - 32
      - a[96..127] = (ql[32..63] >> 4) | ((qh >> 6) & 3) << 4) - 32
   c. Load q8 (128 int8 values)
   d. Multiply: `a[k] * q8[k]` → int16 (widening)
   e. Apply 8 per-16-element scales: `scale[g] * sum_16_elements`
   f. Chain-reduce to scalar
3. Accumulate: `sumf += d * sum_t`

## VL512 Optimization

Compared to VL256, the VL512 implementation:
- Loads 128 q8 values in one `vle8_v_i8m2` instruction (vs 4 loads in VL256)
- Uses `vget_v_i8m2_i8m1` to extract 32-element groups
- Same unpacking algorithm (32-byte loads for qh/ql)
- Same scale application (vwmul_vx_i32m2 with vl=16)

Register usage per sub-block:
- 2 m1 registers for qh/ql loads
- 1 m2 register for q8 load
- 4 m2 registers for int16 products
- 8 m2 registers for int32 scaled sums
- 1 m1 register for reduction accumulator

## Gap Analysis

See: `docs/report/llama.cpp/rvv-gap-analysis-vec_dot-q6_K_q8_K-*.md`

## Build & Test

```bash
# Build llama.cpp with this patch
./build.sh --force --test

# Run standalone test (under QEMU)
./build.sh --test
```

## References

- ARM NEON: Uses bias correction with bsums (different approach)
- x86 AVX2: Uses similar unpacking with VPAND/VPSRL
- Scalar generic: Direct unpacking with per-element scale application