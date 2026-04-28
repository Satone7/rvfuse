# GELU (Erf) RVV Gap Analysis — ViT-Base/32

**Operator**: Standard GELU activation = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))
**Application**: Vision Transformer Base/32 — 12 encoder blocks, each with MLP GELU
**Shapes**: (145, 3072) per instance, 12 instances
**Date**: 2026-04-28
**BBV Data**: QEMU-BBV profiling at VLEN=512 (output/bbv-vit32/)
**Perf Data**: `MlasErfKernel` = 3.36% of total runtime

## 1. Operator Profile

| Location | Count | Shape | Elements |
|----------|-------|-------|----------|
| MLP GELU (each encoder) | 12 | (145, 3072) | 445,440 × 12 |
| **Total** | **12** | | **5,345,280** |

**Note**: ViT uses **standard GELU** (via `erf`), NOT QuickGELU. The `MlasErfKernel` function computes `erf(x)` as the core of GELU.

**Perf confirmation**: `MlasErfKernel` accounts for 3.36% of total inference time (slightly lower than ViT-Base/16's 3.78% due to smaller sequence length).

### Shape Comparison

| Parameter | ViT-Base/16 | ViT-Base/32 |
|-----------|-------------|-------------|
| GELU shape | (197, 3072) | (145, 3072) |
| Elements per instance | 604,224 | 445,440 |
| Total elements (12 blocks) | 7,250,688 | 5,345,280 |
| % Compute | 3.78% | 3.36% |
| D (MLP dim) mod 16 | 0 | 0 |

**Key**: The MLP dimension (3072) is identical between both variants and perfectly aligned with VL=16. The difference is only in the sequence dimension (145 vs 197).

## 2. RVV Vectorization

### Current Implementation: `MlasErfKernel`

Same implementation as ViT-Base/16:
- Uses `vsetivli zero,4,e32,m1,ta,ma` — processes 4 float32 per iteration (VLEN=128 mode within 512-bit)
- Polynomial approximation of erf using Horner's method
- RVV instructions: `vmv.v.i`, `vmv.v.x` for coefficient setup; scalar `flw`/`fsw` in inner loops
- **Not fully vectorized**: The polynomial evaluation uses a mix of vector setup + scalar computation

### Key Gap: No Vector Exponential

RVV base V extension lacks a vector exponential instruction (`vexp` or approximate equivalent). This forces GELU to use a polynomial approximation of `erf`, which requires:
1. Large coefficient table (93 setup instructions for VL=4)
2. Multiple polynomial evaluation stages
3. Scalar fallback for edge cases

**Proposed**: `vfexp_approx` — approximate vector exponential similar to AVX-512's `vexp2ps`.

## 3. Cross-Platform Comparison

| Platform | GELU Implementation | Key Advantage |
|----------|-------------------|---------------|
| RVV512 (current) | `MlasErfKernel` — polynomial erf | VL=4 mode, 93-coeff setup |
| AVX-512 | `vexp2ps` approx + polynomial | Vector exp available, larger VL |
| AVX2 | Polynomial erf (similar to RVV) | No vector exp, 8-wide |
| NEON | Polynomial erf | 4-wide, same limitation |
| SVE | FEXPA + polynomial | Approximate vector exp |

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| VL=16 erf kernel (increase VL from 4→16) | 3.0× | 3.36% | 0.068 |
| Vector exp (proposed) | 1.5× | 3.36% | 0.017 |
| Coefficient table optimization | 1.2× | 3.36% | 0.007 |
| **Total** | | **3.36%** | **0.092×** |

## 5. Comparison with ViT-Base/16 GELU

| Finding | ViT-Base/16 | ViT-Base/32 | Consistency |
|---------|-------------|-------------|-------------|
| % Compute | 3.78% | 3.36% | Consistent (scaled by SeqLen) |
| VL=4 mode | Same | Same | Identical implementation |
| No vector exp | Same gap | Same gap | Identical |
| VL=16 erf kernel benefit | 0.08× | 0.068× | Scaled by weight |

**Shape sensitivity**: GELU efficiency is dominated by the MLP dimension (3072), which is identical between /16 and /32. The sequence length difference (145 vs 197) only affects the total FLOP count, not the per-element vectorization efficiency.

---

*Cross-reference: ViT-Base/16 GELU gap analysis at docs/report/vit-base-16/gap-analysis-gelu-erf.md*
