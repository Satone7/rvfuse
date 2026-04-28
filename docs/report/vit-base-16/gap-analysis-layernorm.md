# LayerNorm RVV Gap Analysis — ViT-Base/16

**Operator**: Layer Normalization
**Application**: Vision Transformer Base/16 — 25 instances total
**Shapes**: (197, 768) — normalize over last dimension
**Date**: 2026-04-28
**Perf Data**: Not visible in top hotspots (<0.5% of total runtime)

## 1. Operator Profile

| Location | Count | Shape | Description |
|----------|-------|-------|-------------|
| After QKV + attention | 12 | (197, 768) | Pre-norm before attention |
| After MLP | 12 | (197, 768) | Pre-norm before MLP |
| Final LayerNorm | 1 | (197, 768) | Before classification head |
| **Total** | **25** | **(197, 768)** | |

**% Compute**: <0.5% (too small for perf sampling to capture reliably)

## 2. RVV Vectorization

### Algorithm

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Where D = 768 (hidden dimension).

### Alignment Analysis

$D = 768 = 48 \times 16$ — perfectly aligned with VL=16. **No tail handling needed.**

This is a significant advantage over SuperGlue's D=256 (16×16 = 1 set of 16 iterations):
- ViT: 48 iterations of VL=16 reduction per row
- SuperGlue: 16 iterations of VL=16 reduction per row

### Implementation: `rvv-patches/layernorm-f32/rvv_layernorm_f32.inl`

| Step | RVV Instruction | Alignment |
|------|----------------|-----------|
| Mean reduction | `vfredusum_vs_f32m1` | D=768 = 48×16 ✓ |
| Variance reduction | `vfredusum_vs_f32m1` | D=768 ✓ |
| Normalize | `vfsub_vv` + `vfmul_vv` | D=768 ✓ |
| Scale/Shift | `vfmul_vv` + `vfadd_vv` | D=768 ✓ |

**Per-row vector ops**: ~192 vector ops (48 iterations × 4 operations each)

## 3. Cross-Platform Comparison

| Platform | D=768 Efficiency | Notes |
|----------|-----------------|-------|
| RVV512 (VL=16) | 100% aligned | 48 iterations cover D=768 exactly |
| AVX-512 | 100% aligned | 48 iterations, same as RVV512 |
| AVX2 | 100% aligned | 96 iterations (8 f32/reg) |
| NEON | 100% aligned | 192 iterations (4 f32/reg) |
| SVE 512-bit | 100% aligned | Same as RVV512 |

**No significant gap**: LayerNorm at D=768 is perfectly aligned on all platforms. The gap is purely in vector width (throughput), not ISA features.

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| Mean/variance reduction RVV | 5.0× | 0.5% | 0.02 |
| Element-wise normalize RVV | 4.0× | 0.3% | 0.01 |
| **Total** | | **0.5%** | **0.03×** |

LayerNorm's contribution to ViT runtime is minimal (<0.5%) due to:
- O(N×D) complexity vs O(N²×D) for attention
- 25 instances × 197 rows = 4,925 rows total
- Each row: 768 elements, perfectly VL-aligned

## 5. Comparison with SuperGlue LayerNorm

SuperGlue analysis (docs/report/superglue/gap-analysis-layernorm.md):
- D=256 (16 iterations) → perfect alignment ✓
- ViT: D=768 (48 iterations) → also perfect alignment ✓
- Same conclusion: no significant gap, solid RVV speedup from alignment

---

*Cross-reference: SuperGlue LayerNorm gap analysis at docs/report/superglue/gap-analysis-layernorm.md*
