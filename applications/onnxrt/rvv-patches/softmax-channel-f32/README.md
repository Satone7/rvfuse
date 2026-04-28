# softmax-channel-f32

RVV implementation of channel-wise Softmax for float32 — RISC-V Vector Extension.

## Status

✅ Implementation complete — partially vectorized with scalar expf fallback.

## Algorithm

Channel-wise Softmax: `Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))`

Applied along the channel dimension for each spatial position independently.

| Step | RVV Instruction | Purpose |
|------|-----------------|---------|
| 1 | `vsetvl_e32m1(C)` | Set vector length for channel dim (handle tails) |
| 2 | `vle32_v_f32m1` | Load channel vector |
| 3 | `vfredmax_vs_f32m1` | Find max value across channels (numerical stability) |
| 4 | `vfsub_vf_f32m1` | Subtract max from each element |
| 5 | **scalar `expf()`** | Compute exp per element (no vector exp in base V) |
| 6 | `vfredusum_vs_f32m1` | Sum of exp values across channels |
| 7 | `vfmul_vf_f32m1` | Divide each element by sum |

## Files

| File | Purpose |
|------|---------|
| `rvv_softmax_channel_f32.inl` | RVV channel-wise Softmax implementation (single source of truth) |

## Function Signatures

```cpp
void MlasSoftmaxChannelF32_rvv(
    const float* input, float* output,
    int batch, int channels, int Hc, int Wc);
```

## Key Shapes (SuperPoint)

| Parameter | Value | RVV Implication |
|-----------|-------|-----------------|
| Channels | 65 | 65 = 4×16 + 1 — tail issue at VL=16 |
| Spatial positions | 60×80 = 4800 | Per-position softmax calls |
| Total expf calls | 65 × 4800 = 312,000 | Scalar expf is the bottleneck |

## Known Limitations

- **Scalar `expf()` fallback**: RISC-V V extension has no vector exponential instruction. The exp step uses scalar `expf()` per element. This is the primary performance gap. A proposed `vfexp.v` extension would eliminate this bottleneck.
- **Channel tail handling**: With 65 channels at VL=16, the last iteration processes only 1 element (65 % 16 = 1), causing partial vector utilization.
- **Not integrated into ORT**: This patch targets the C++ runner post-processing path, not ONNX Runtime's MLAS library. SuperPoint's model does not include a Softmax ONNX operator — the softmax is performed entirely in post-processing.

## SuperPoint Relevance

Softmax accounts for ~0.01% of total execution time on hardware (Spacemit X60 perf data).
The absolute performance impact is small because softmax executes outside the dominant SGEMM path,
but the scalar expf bottleneck would become significant if softmax were used more heavily.

## VLEN Independence

This implementation is VLEN-agnostic — it uses `vsetvl` for dynamic vector length,
so it works with VLEN=128, 256, 512, or 1024 without modification.
