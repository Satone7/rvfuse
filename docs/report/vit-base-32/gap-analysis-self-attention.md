# Self-Attention RVV Gap Analysis — ViT-Base/32

**Operator**: Multi-head Self-Attention (Scaled Dot-Product)
**Application**: Vision Transformer Base/32 — 12 encoder blocks
**Shapes**: QKV (145,768)×(768,2304), QK^T (12,145,64)×(12,64,145), Attn×V (12,145,145)×(12,145,64)
**Date**: 2026-04-28
**Perf Data**: SGEMM dominates at ~87% (QKV+Attn+MLP all via MlasSgemmKernelRvv512)

## 1. Operator Profile

| Component | Shape | % of Attention | % of Total (perf) |
|-----------|-------|----------------|-------------------|
| QKV Projections | (145,768)×(768,2304) ×12 | 50% | ~35% |
| QK^T MatMul | (12,145,64)×(12,64,145) ×12 | 18% | ~15% |
| Softmax | (12,145,145) ×12 | 4% | ~1.1% |
| Attn×V MatMul | (12,145,145)×(12,145,64) ×12 | 18% | ~15% |
| Output Projection | (145,768)×(768,768) ×12 | 10% | ~10% |

**Total attention**: ~60% of total inference time (lower than ViT-Base/16's 66.3% due to smaller N).

### ViT Attention Shape Comparison

| Parameter | ViT-Base/16 | ViT-Base/32 | SuperGlue |
|-----------|-------------|-------------|-----------|
| Input size | 224×224 | 384×384 | Varies |
| Patch size | 16×16 | 32×32 | N/A |
| Sequence length (N) | 197 | **145** | 1024 |
| N mod 16 | **5** | **1** | 0 |
| Hidden dim (d) | 768 | 768 | 256 |
| Head dim (d_k) | 64 | 64 | 64 |
| Num heads (H) | 12 | 12 | 4 |
| d mod 16 | 0 | 0 | 0 |
| d_k mod 16 | 0 | 0 | 0 |

**Key differences from ViT-Base/16**:
1. **N=145 has 1-element tail** (vs N=197's 5-element tail) → better VL=16 alignment
2. **Fewer patches** (144 vs 196) → 26% fewer attention FLOPs
3. **Same hidden dim** (768) → MLP shapes identical
4. **Same number of heads** (12) → same batch processing

## 2. RVV Vectorization

### QKV Projections: SGEMM VL=16 (Already Optimized)

All QKV projections use `MlasSgemmKernelRvv512Impl` with VL=16:
- K=768: 768 mod 16 = 0 → perfect alignment ✓
- N=2304: 2304 mod 16 = 0 → perfect alignment ✓

### QK^T MatMul: Shape Analysis

$(B, H, N, d_k) \times (B, H, d_k, N)$ where B=1, H=12, N=145, d_k=64

- **K dimension (d_k=64)**: 64 mod 16 = 0 → inner loop perfectly aligned ✓
- **N dimension (145)**: 145 mod 16 = **1** → output rows/cols require tail ✗
- **Impact**: 9 full tile rows + 1-element tail per attention head

### Attn×V MatMul: Shape Analysis

$(B, H, N, N) \times (B, H, N, d_k)$ where N=145, d_k=64

- **K dimension (N=145)**: 145 mod 16 = **1** → inner K-loop has tail ✗
- **N dimension (d_k=64)**: 64 mod 16 = 0 → output aligned ✓
- **Impact**: 9 full K iterations + 1-element tail, per output element

### Tail Analysis: ViT-Base/32 vs ViT-Base/16

| Metric | ViT-Base/16 (N=197) | ViT-Base/32 (N=145) |
|--------|---------------------|---------------------|
| QK^T tail columns | 5 | **1** |
| Attn×V tail K elements | 5 | **1** |
| Full VL=16 iterations (QK^T) | 12 | 9 |
| Tail fraction (QK^T) | 5/197 = 2.5% | 1/145 = **0.7%** |
| Softmax tail | 5 elements | **1 element** |
| Total tail overhead | ~2.5% of attention | **~0.7% of attention** |

**Conclusion**: ViT-Base/32's 1-element tail is nearly negligible — the attention operators are almost perfectly vectorized at VL=16.

## 3. Cross-Platform Comparison

| Platform | Attention QK^T Throughput | Tail (N=145) | Notes |
|----------|--------------------------|---------------|-------|
| RVV512 | 512-bit, 16 f32/reg | `vsetvl` (1 tail) | VL-aligned for d_k=64 |
| AVX-512 | 512-bit, 16 f32/reg | k-mask (1 tail) | BF16 option for 2× throughput |
| AVX2 | 256-bit, 8 f32/reg | Scalar (1 tail) | Half throughput |
| SVE (256-bit) | 256-bit, 8 f32/reg | Predicate (1 tail) | Half throughput |
| NEON | 128-bit, 4 f32/reg | Scalar (1 tail) | Quarter throughput |

### Key Gap: BF16/FP16 for Attention

AVX-512 can use BF16 (`vdpbf16ps`) for 2× attention throughput. RVV base V lacks BF16 support:
- **Proposed**: Zvfbfmin extension (already in RVV spec as proposal)
- **Benefit**: 2× throughput for QK^T and Attn×V MatMul at slightly reduced precision
- **ViT-specific**: ViT attention is less sensitive to precision than SuperGlue matching → BF16 is highly applicable

### ViT-Base/32 vs ViT-Base/16: Tail Comparison

For N=145 vs N=197, the tail impact differs:

| Platform | N=197 Tail Mechanism | N=145 Tail Mechanism | Improvement |
|----------|---------------------|---------------------|-------------|
| RVV512 | vsetvl(VL=5) | vsetvl(VL=1) | 80% fewer tail elements |
| AVX-512 | k-mask(5) | k-mask(1) | Minimal (both fast) |
| ARM SVE | predicate(5) | predicate(1) | Minimal (both fast) |
| AVX2 | Scalar(5) | Scalar(1) | 80% fewer scalar ops |
| NEON | Scalar(5) | Scalar(1) | 80% fewer scalar ops |

**Fixed-width ISAs** (AVX2, NEON) benefit more from the smaller tail because their scalar fallback is more expensive per element.

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| VL=16 SGEMM for attention (existing) | 12× vs scalar | 60% | 7.20 |
| N=145 tail optimization | 1.001× | 30% | 0.0003 |
| BF16 for QK^T + Attn×V | 2.0× | 30% | 0.30 |
| Vector exp for softmax | 1.5× | 1.1% | 0.006 |
| **Total proposed** | | | **0.31×** |

**Priority**: BF16 (P0) >> vector exp (P3) > tail optimization (negligible for N=145)

## 5. Comparison with ViT-Base/16 Self-Attention

| Finding | ViT-Base/16 | ViT-Base/32 | Consistency |
|---------|-------------|-------------|-------------|
| Attention % of total | 66.3% | ~60% | Lower due to fewer patches |
| Tail (N mod 16) | 5 | 1 | Consistent pattern |
| BF16 benefit | 0.30× | 0.30× | Same (same d_k, H) |
| No vector exp | Same gap | Same gap | Identical |
| Tail overhead | ~2.5% | ~0.7% | 3.6× less for /32 |

**Shape sensitivity**: The primary difference between ViT-Base/32 and /16 is the tail handling efficiency. The 1-element tail at N=145 makes VL=16 vectorization nearly perfect for attention operators, whereas the 5-element tail at N=197 introduces modest overhead.

---

*Cross-reference: ViT-Base/16 Self-Attention gap analysis at docs/report/vit-base-16/gap-analysis-self-attention.md*
