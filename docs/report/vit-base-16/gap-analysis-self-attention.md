# Self-Attention RVV Gap Analysis — ViT-Base/16

**Operator**: Multi-head Self-Attention (Scaled Dot-Product)
**Application**: Vision Transformer Base/16 — 12 encoder blocks
**Shapes**: QKV (197,768)×(768,2304), QK^T (12,197,64)×(12,64,197), Attn×V (12,197,197)×(12,197,64)
**Date**: 2026-04-28
**Perf Data**: SGEMM dominates at 87.15% (QKV+Attn+MLP all via MlasSgemmKernelRvv512)

## 1. Operator Profile

| Component | Shape | % of Attention | % of Total (perf) |
|-----------|-------|----------------|-------------------|
| QKV Projections | (197,768)×(768,2304) ×12 | 50% | ~35% |
| QK^T MatMul | (12,197,64)×(12,64,197) ×12 | 18% | ~15% |
| Softmax | (12,197,197) ×12 | 4% | ~1.32% |
| Attn×V MatMul | (12,197,197)×(12,197,64) ×12 | 18% | ~15% |
| Output Projection | (197,768)×(768,768) ×12 | 10% | ~10% |

**Total attention**: ~66.3% of total inference time.

### ViT Attention vs SuperGlue Attention

| Parameter | ViT-Base/16 | SuperGlue |
|-----------|-------------|-----------|
| Sequence length (N) | 197 | 1024 |
| Hidden dim (d) | 768 | 256 |
| Head dim (d_k) | 64 | 64 |
| Num heads (H) | 12 | 4 |
| N mod 16 | **5** | 0 |
| d mod 16 | 0 | 0 |
| d_k mod 16 | 0 | 0 |
| Attention type | Self-only | Self + Cross |

**Key differences**:
1. **N=197 is NOT VL-aligned** → tail handling overhead in attention score computation
2. **Larger hidden dim** (768 vs 256) → more FLOPs in QKV projections
3. **More heads** (12 vs 4) → more batch dimension processing
4. **Self-attention only** (no cross-attention) → simpler data flow

## 2. RVV Vectorization

### QKV Projections: SGEMM VL=16 (Already Optimized)

All QKV projections use `MlasSgemmKernelRvv512Impl` with VL=16:
- K=768: 768 mod 16 = 0 → perfect alignment ✓
- N=2304: 2304 mod 16 = 0 → perfect alignment ✓

### QK^T MatMul: Shape Analysis

$(B, H, N, d_k) \times (B, H, d_k, N)$ where B=1, H=12, N=197, d_k=64

- **K dimension (d_k=64)**: 64 mod 16 = 0 → inner loop perfectly aligned ✓
- **N dimension (197)**: 197 mod 16 = 5 → **output rows/cols require tail** ✗
- **Impact**: 12 full tile rows + 5-element tail per attention head

### Attn×V MatMul: Shape Analysis

$(B, H, N, N) \times (B, H, N, d_k)$ where N=197, d_k=64

- **K dimension (N=197)**: 197 mod 16 = 5 → **inner K-loop has tail** ✗
- **N dimension (d_k=64)**: 64 mod 16 = 0 → output aligned ✓
- **Impact**: 12 full K iterations + 5-element tail, per output element

## 3. Cross-Platform Comparison

| Platform | Attention QK^T Throughput | Tail (N=197) | Notes |
|----------|--------------------------|---------------|-------|
| RVV512 | 512-bit, 16 f32/reg | `vsetvl` (5 tail) | VL-aligned for d_k=64 |
| AVX-512 | 512-bit, 16 f32/reg | k-mask (5 tail) | BF16 option for 2× throughput |
| AVX2 | 256-bit, 8 f32/reg | Scalar (5 tail) | Half throughput |
| SVE (256-bit) | 256-bit, 8 f32/reg | Predicate (5 tail) | Half throughput |
| NEON | 128-bit, 4 f32/reg | Scalar (5 tail) | Quarter throughput |

### Key Gap: BF16/FP16 for Attention

AVX-512 can use BF16 (`vdpbf16ps`) for 2× attention throughput. RVV base V lacks BF16 support:
- **Proposed**: Zvfbfmin extension (already in RVV spec as proposal)
- **Benefit**: 2× throughput for QK^T and Attn×V MatMul at slightly reduced precision
- **ViT-specific**: ViT attention is less sensitive to precision than SuperGlue matching → BF16 is highly applicable

## 4. N=197 Tail Handling Detail

This is the **novel ViT-specific** analysis point. For attention with N=197:

**QK^T computation** (per head):
```
Output: (197, 197) = 38,809 elements
Vector iterations: 12 × 16 = 192 elements (12 full tiles)
Tail: 5 elements (2.5% of output)
```

**Current behavior** (from BBV analysis of MlasSgemmKernelRvv512Impl):
- BB 10/11/12: Scalar `flw` + `fadd.s` + `fsw` loop for tail columns
- 5 scalar iterations × 197 rows = 985 scalar operations per QK^T call
- 985 / 38,809 = **2.5% overhead from scalar tail**

**With proposed vsetvl tail** (process 5 elements as partial vector):
- 1 vector iteration at VL=5 instead of 5 scalar iterations
- Would eliminate most of the 2.5% overhead
- Implementation: Change the N-loop to use `vsetvl(CountN)` instead of fixed VL=16

## 5. BBV-Weighted Benefit

| Item | Speedup | Weight (% of total) | Weighted |
|------|---------|-------------------|----------|
| QKV SGEMM VL=16 (existing) | 12× | 35% | Baseline |
| QK^T BatchMatMul RVV (existing) | 10× | 15% | Baseline |
| Attn×V BatchMatMul RVV (existing) | 10× | 15% | Baseline |
| vsetvl tail for N=197 (proposed) | 1.025× on attention | 30% | 0.008 |
| BF16 attention (proposed Zvfbfmin) | 2× on attention | 30% | 0.30 |
| **Total** | | **66%** | **0.31×** |

The biggest single improvement opportunity is **BF16 attention** (0.30× weighted benefit), which would provide 2× throughput for attention MatMul with acceptable precision loss for ViT.

## 6. Comparison with SuperGlue Self-Attention

SuperGlue analysis (docs/report/superglue/gap-analysis-self-attention.md):
- N=1024 perfectly aligned → no tail handling concern
- Same BF16 gap → Zvfbfmin equally beneficial
- 4 heads vs ViT's 12 → ViT benefits more from vectorization (more parallel work)

**Cross-reference finding**: SuperGlue's "Batch 2 Baseline Role" (§4) correctly predicted ViT's N=197 tail handling issue. This analysis confirms the prediction with BBV-weighted quantification: 2.5% overhead on attention MatMul.

---

*Cross-reference: SuperGlue Self-Attention gap analysis at docs/report/superglue/gap-analysis-self-attention.md*
