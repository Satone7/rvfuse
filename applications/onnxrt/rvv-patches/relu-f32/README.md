# relu-f32

RVV implementation of ReLU activation for float32 — RISC-V Vector Extension.

## Status

✅ Implementation complete — element-wise max(0, x) with setvl tail handling.

## Algorithm

ReLU(x) = max(0, x) vectorized using `vfmax.vf` with immediate 0.0f.

| Step | RVV Instruction | Purpose |
|------|-----------------|---------|
| 1 | `vsetvl_e32m1(N)` | Set vector length (handle tails) |
| 2 | `vle32_v_f32m1` | Load input vector |
| 3 | `vfmax_vf_f32m1(v, 0.0f)` | Element-wise max with zero |
| 4 | `vse32_v_f32m1` | Store result |

## Files

| File | Purpose |
|------|---------|
| `rvv_relu_f32.inl` | RVV ReLU implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signatures

```cpp
void MlasReLuF32_rvv(const float* input, float* output, size_t N);
void MlasReLuF32Inplace_rvv(float* buffer, size_t N);
```

## SuperPoint Relevance

ReLU appears ~5% of compute in SuperPoint (after each Conv2d layer in the VGG encoder).
While not the dominant operator, it benefits from RVV due to its simplicity and memory-bound nature.

## VLEN Independence

This implementation is VLEN-agnostic — it uses `vsetvl` for dynamic vector length,
so it works with VLEN=128, 256, 512, or 1024 without modification.
