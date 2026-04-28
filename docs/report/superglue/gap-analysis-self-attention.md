# Self-Attention RVV Gap Analysis — SuperGlue

**Operator**: Multi-head Self-Attention (Scaled Dot-Product)
**Role**: 5 self-attention layers in SuperGlue GNN (Batch 2 baseline — first self-attention analysis)
**Shapes**: QKV $(N, 256) \times (256, 256)$, QK^T $(4, N, 64) \times (4, 64, N)$, Attn×V $(4, N, N) \times (4, N, 64)$
**Date**: 2026-04-28

## 1. Operator Profile

| Component | Shape | % of Attention | % of Total |
|-----------|-------|----------------|------------|
| QKV Projections | $(N, 256) \times (256, 256)$ ×3 | 40% | ~15% |
| QK^T MatMul | $(4, N, 64) \times (4, 64, N)$ | 25% | ~10% |
| Softmax | $(4, N, N)$ per head | 10% | ~4% |
| Attn×V MatMul | $(4, N, N) \times (4, N, 64)$ | 20% | ~8% |
| Output Projection | $(N, 256) \times (256, 256)$ | 5% | ~2% |

## 2. RVV Vectorization

### QK^T Attention Score Computation

$$\text{Scores} = Q K^T / \sqrt{d_k}$$

Shape: $(B, H, N, d_k) \times (B, H, d_k, N) \rightarrow (B, H, N, N)$ where $B=1, H=4, N=1024, d_k=64$

**RVV strategy**: Process output matrix in 16×16 blocks:
- FMA accumulation over $d_k=64$ dimension
- $d_k = 64 \bmod 16 = 0$ → perfect alignment
- $N = 1024 \bmod 16 = 0$ → perfect alignment

### Key RVV Instructions

| Operation | Instruction | Notes |
|-----------|------------|-------|
| Blocked QK^T | `vfmacc_vf_f32m1` | Broadcast Q, multiply-add with K row |
| Scale | `vfmul_vf_f32m1` | Divide by $\sqrt{d_k}$ |
| Tile loops | `vsetvl_e32m1` | 16×16 output tiles |

## 3. Cross-Platform Comparison

| Platform | Attention QK^T Throughput | Notes |
|----------|--------------------------|-------|
| RVV512 | 512-bit, 16 f32/reg | VL-aligned, optimal for N=1024 |
| AVX-512 | 512-bit, 16 f32/reg | BF16 option for 2× throughput |
| AVX2 | 256-bit, 8 f32/reg | Half throughput of RVV512 |
| SVE (256-bit) | 256-bit, 8 f32/reg | Half throughput of RVV512 |
| NEON | 128-bit, 4 f32/reg | Quarter throughput |

**Key gap for Batch 2**: RVV lacks BF16/FP16 hardware support in the base V extension (Zvfbfmin is proposed). AVX-512 can use BF16 for 2× attention throughput. This gap is shared across all attention operators (self, cross, ViT).

## 4. Batch 2 Baseline Role

This self-attention analysis serves as the **Batch 2 baseline** for ViT-Base/16 and ViT-Base/32 apps. Key findings they should reference:

1. **Shape alignment**: N=1024 is VL-aligned (16×), but ViT's N=197 and N=50 are NOT aligned → tail handling overhead documented here
2. **$d_k=64$ alignment**: 64 = 4×16 → perfectly aligned for all Batch 2 apps
3. **RVV attention pattern**: The block-tiled QK^T approach with `vfmacc.vf` generalizes to all sequence lengths

## 5. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| QKV SGEMM VL=16 | 3.5× | 15% | 0.53 |
| QK^T BatchMatMul RVV | 2.5× | 10% | 0.25 |
| Attn×V BatchMatMul RVV | 2.5× | 8% | 0.20 |
| Scale + Softmax RVV | 2.0× | 4% | 0.08 |
| **Total** | | **37%** | **1.06×** |
