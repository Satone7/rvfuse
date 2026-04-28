# GELU (Erf) RVV Gap Analysis — ViT-Base/16

**Operator**: Standard GELU activation = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))
**Application**: Vision Transformer Base/16 — 12 encoder blocks, each with MLP GELU
**Shapes**: (197, 3072) per instance, 12 instances
**Date**: 2026-04-28
**BBV Data**: QEMU-BBV profiling at VLEN=512 (output/bbv_rvv512/vit-base-16/erf-gelu/)
**Perf Data**: `MlasErfKernel` = 3.78% of total runtime

## 1. Operator Profile

| Location | Count | Shape | Elements |
|----------|-------|-------|----------|
| MLP GELU (each encoder) | 12 | (197, 3072) | 604,224 × 12 |
| **Total** | **12** | | **7,250,688** |

**Note**: ViT uses **standard GELU** (via `erf`), NOT QuickGELU. The `MlasErfKernel` function computes `erf(x)` as the core of GELU. This is different from YOLO's `QuickGeluAlphaScale` which uses `sigmoid(alpha * x)`.

**Perf confirmation**: `MlasErfKernel` accounts for 3.78% of total inference time.

## 2. RVV Vectorization

### Current Implementation: `MlasErfKernel`

The BBV disassembly (28 BBs) reveals:
- Uses `vsetivli zero,4,e32,m1,ta,ma` — processes 4 float32 per iteration (VLEN=128 mode within 512-bit)
- Polynomial approximation of erf using Horner's method
- RVV instructions: `vmv.v.i`, `vmv.v.x` for coefficient setup; scalar `flw`/`fsw` in inner loops
- **Not fully vectorized**: The polynomial evaluation uses a mix of vector setup + scalar computation

### BBV Hotspot Analysis

From `erf-gelu.disas` (28 BBs):

| BB | Instructions | Role | Key Ops |
|----|-------------|------|---------|
| BB 0 | 22 | Prologue + setup | Stack frame, initial checks |
| BB 2 | 93 | Coefficient loading (huge!) | 93 `vmv.v.x` instructions for polynomial coeff |
| BB 3 | 3 | Branch | Loop condition |
| BB 8 | 11 | Core computation | Polynomial eval |
| BB 11 | 8 | Store result | Output writeback |

**Key observation**: BB 2 has 93 instructions — all `vmv.v.x` for loading polynomial coefficients into vector registers. This is a significant setup cost that could be reduced with a vector broadcast-from-memory instruction or a constant pool approach.

## 3. Cross-Platform Comparison

| Platform | GELU Implementation | Key Advantage |
|----------|-------------------|---------------|
| RVV512 (current) | `MlasErfKernel` — polynomial erf | VL=4 mode, 93-coeff setup |
| AVX-512 | `vexp2ps` approx + polynomial | Vector exp available, larger VL |
| AVX2 | Polynomial erf (similar to RVV) | No vector exp, 8-wide |
| NEON | Polynomial erf | 4-wide, same limitation |
| SVE | FEXPA + polynomial | Approximate vector exp |

### Key Gap: No Vector Exponential

RVV base V extension lacks a vector exponential instruction (`vexp` or approximate equivalent). This forces GELU to use a polynomial approximation of `erf`, which requires:
1. Large coefficient table (93 setup instructions for VL=4)
2. Multiple polynomial evaluation stages
3. Scalar fallback for edge cases

**Proposed**: `vfexp_approx` — approximate vector exponential similar to AVX-512's `vexp2ps`. This would enable:
- Direct `GELU(x) = x * sigmoid(1.702 * x)` via vector exp instead of erf polynomial
- Or simplified `erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715*x³))` using vector operations

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| VL=16 erf kernel (increase VL from 4→16) | 3.0× | 3.78% | 0.08 |
| Vector exp (proposed) | 1.5× | 3.78% | 0.02 |
| Coefficient table optimization | 1.2× | 3.78% | 0.01 |
| **Total** | | **3.78%** | **0.11×** |

GELU's contribution to overall runtime is modest (~3.78%). The main improvement opportunity is increasing the vector processing width from VL=4 to VL=16 in the erf kernel.

## 5. Comparison with QuickGELU Analysis

The existing `rvv-patches/quick-gelu/` analysis (YOLO) covers `QuickGeluAlphaScale` which is:
- `QuickGELU(x) = x * sigmoid(1.702 * x)` — 3 RVV instructions per VL
- Simpler than standard GELU (no erf needed)

**ViT uses standard GELU**, which requires `erf()` — a fundamentally more complex operation. The QuickGELU patch is NOT directly applicable to ViT. A dedicated `MlasErfKernel` optimization is needed.

**Cross-reference**: YOLO QuickGELU gap analysis at docs/report/onnxrt/rvv-gap-analysis-quick-gelu-2026-04-26.md

---

*Review Log*:
| Round | Issues | Fixed | Remaining |
|-------|--------|-------|-----------|
| R1 | 0 | 0 | 0 |
