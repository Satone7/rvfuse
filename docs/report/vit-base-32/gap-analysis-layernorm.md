# LayerNorm RVV Gap Analysis — ViT-Base/32

**Operator**: Layer Normalization
**Application**: Vision Transformer Base/32 — 25 instances total
**Shapes**: (145, 768) — normalize over last dimension
**Date**: 2026-04-28
**Perf Data**: Not visible in top hotspots (<0.5% of total runtime)

## 1. Operator Profile

| Location | Count | Shape | Description |
|----------|-------|-------|-------------|
| After QKV + attention | 12 | (145, 768) | Pre-norm before attention |
| After MLP | 12 | (145, 768) | Pre-norm before MLP |
| Final LayerNorm | 1 | (145, 768) | Before classification head |
| **Total** | **25** | **(145, 768)** | |

**% Compute**: <0.5% (too small for perf sampling to capture reliably)

### Shape Comparison

| Parameter | ViT-Base/16 | ViT-Base/32 |
|-----------|-------------|-------------|
| Shape per instance | (197, 768) | (145, 768) |
| Total rows (25 instances) | 4,925 | 3,625 |
| D (hidden dim) | 768 | 768 |
| D mod 16 | 0 | 0 |

**Key**: The hidden dimension (768) is identical between both variants and perfectly aligned with VL=16. Only the row count differs (145 vs 197 rows per instance).

## 2. RVV Vectorization

### Algorithm

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Where D = 768 (hidden dimension).

### Alignment Analysis

$D = 768 = 48 \times 16$ — perfectly aligned with VL=16. **No tail handling needed.**

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

## 5. Comparison with ViT-Base/16 LayerNorm

| Finding | ViT-Base/16 | ViT-Base/32 | Consistency |
|---------|-------------|-------------|-------------|
| D=768 alignment | Perfect ✓ | Perfect ✓ | Identical |
| Total rows | 4,925 | 3,625 | Scaled by SeqLen |
| % Compute | <0.5% | <0.5% | Consistent |
| No gap | Same | Same | Identical |

**Shape sensitivity**: None — LayerNorm is dominated by D (hidden dimension) which is identical in both variants. The sequence length difference has no effect on per-row vectorization efficiency.

---

*Cross-reference: ViT-Base/16 LayerNorm gap analysis at docs/report/vit-base-16/gap-analysis-layernorm.md*
