# l2-normalize-f32

RVV implementation of L2 Normalize for float32 — RISC-V Vector Extension.

## Status

✅ Implementation complete — partially vectorized with scalar sqrtf fallback.

## Algorithm

L2 Normalize: `L2Normalize(x) = x / ||x||_2` where `||x||_2 = sqrt(sum(x_i^2))`

Applied per-vector along the channel dimension.

| Step | RVV Instruction | Purpose |
|------|-----------------|---------|
| 1 | `vsetvl_e32m1(N)` | Set vector length (handle tails) |
| 2 | `vle32_v_f32m1` | Load input vector |
| 3 | `vfmul_vv_f32m1` | Square each element (x_i^2) |
| 4 | `vfredusum_vs_f32m1` | Sum-of-squares reduction |
| 5 | **scalar `sqrtf()`** | Compute L2 norm from sum (no vector sqrt in base V) |
| 6 | **scalar division** | Compute 1/norm (no vector reciprocal in base V) |
| 7 | `vfmul_vf_f32m1` | Scale each element by 1/norm |

## Files

| File | Purpose |
|------|---------|
| `rvv_l2_normalize_f32.inl` | RVV L2 Normalize implementation (single source of truth) |

## Function Signatures

```cpp
void MlasL2NormalizeF32_rvv(float* buffer, size_t N);
```

## Key Shapes (SuperPoint)

| Parameter | Value | RVV Implication |
|-----------|-------|-----------------|
| Descriptor dim | 256 | 256 = 16×16 — perfectly aligned with VL=16! |
| Spatial positions | 60×80 = 4800 | Per-position normalize in ORT ReduceL2 |
| Total sqrt+div ops | 256 × 4800 = 1,228,800 | In ORT's ReduceAggregatorL2 path |

## Known Limitations

- **Scalar `sqrtf()` fallback**: RISC-V V extension has no vector square root instruction. The norm computation uses scalar `sqrtf()`. A proposed `vfsqrt.v` or approximate `vfrsqrt7.v` extension would eliminate this bottleneck.
- **Scalar reciprocal**: No vector reciprocal instruction (`vfrec7.v`) in base V extension. The 1/norm computation uses scalar division.
- **Dual L2 Normalize paths in SuperPoint**:
  1. **ORT path** (primary, 0.29% of perf time): The ONNX model contains `ReduceL2(axes=[1])` → `Unsqueeze(axes=[1])` → `Div`, processing the full (1,256,60,80) descriptor tensor.
  2. **C++ runner path** (negligible, 0.00% of perf time): After bilinear descriptor sampling, each 256-dim descriptor is re-normalized. Only ~412 vectors vs. 1.228M in ORT's path.

## SuperPoint Relevance

L2 Normalize (ORT ReduceL2) accounts for 0.29% of total execution time on hardware (Spacemit X60 perf data).
The absolute impact is small, but the scalar sqrt bottleneck affects any workload using ORT's ReduceL2 operator.
The proposed `vfsqrt.v` / `vfrsqrt7.v` extensions would accelerate this path directly.

## VLEN Independence

This implementation is VLEN-agnostic — it uses `vsetvl` for dynamic vector length,
so it works with VLEN=128, 256, 512, or 1024 without modification.
