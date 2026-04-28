# Softmax RVV Gap Analysis — ViT-Base/32

**Operator**: Softmax (attention score normalization)
**Application**: Vision Transformer Base/32 — 12 encoder blocks, each with multi-head attention
**Shapes**: (12, 145, 145) per encoder — 12 attention heads, 145×145 score matrix
**Date**: 2026-04-28
**BBV Data**: QEMU-BBV profiling at VLEN=512 (output/bbv-vit32/)
**Perf Data**: `MlasComputeSoftmaxThreaded` = 1.10% of total runtime

## 1. Operator Profile

| Location | Count | Shape per instance | Total elements |
|----------|-------|-------------------|----------------|
| Attention softmax (each encoder) | 12 | (12, 145, 145) | 252,300 × 12 |
| **Total** | **12** | | **3,027,600** |

**Operation**: softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
**Reduction axis**: Last dimension (columns), per row per head
**% Compute**: 1.10% (perf confirmed, slightly lower than ViT-Base/16's 1.32%)

### ViT-Specific: N=145 Alignment

$145 = 9 \times 16 + 1$ — the softmax reduction dimension has only a 1-element tail.

- Each row has 145 elements → 9 full VL=16 vector iterations + 1 tail element
- `vsetvl` handles tails dynamically, the misaligned dimension causes:
  - 9/10 = 90.0% of iterations are full vector (optimal)
  - 1/10 = 10.0% are partial vector (1-element tail)

**Comparison with ViT-Base/16**:

| Metric | ViT-Base/16 (N=197) | ViT-Base/32 (N=145) |
|--------|---------------------|---------------------|
| Full iterations | 12 (92.3%) | 9 (90.0%) |
| Tail iterations | 1 (7.7%) | 1 (10.0%) |
| Tail elements | 5 | **1** |
| Tail FLOP fraction | 5/197 = 2.5% | 1/145 = **0.7%** |
| Total softmax elements | 5,619,456 | 3,027,600 |

## 2. RVV Vectorization

### Current Implementation

Same as ViT-Base/16 — `MlasComputeSoftmaxThreaded`:
- Max reduction: `vle32.v` + comparison → scalar max
- Exp computation: Scalar `exp()` calls — **no vector exponential in RVV**
- Sum reduction: `vfredusum`-like pattern
- Division + store: `vfdiv.vf`, `vse32.v`

### Alignment Analysis

| Dimension | Value | mod 16 | Status |
|-----------|-------|--------|--------|
| N (seq length) | 145 | **1** | Tail handling via vsetvl (minimal) |
| H (heads) | 12 | 12 | Processed sequentially |

## 3. Cross-Platform Comparison

| Platform | Softmax Throughput | Tail (N=145) | Key Gap |
|----------|-------------------|--------------|---------|
| RVV512 | Medium (scalar exp) | `vsetvl` (1 tail) | No vector exp |
| AVX-512 | High (VEXP2PS approx) | k-mask (1 tail) | Vector exp available |
| NEON | Medium (scalar exp) | Scalar fallback | Same gap as RVV |
| SVE | Medium-High (FEXPA) | Predicate (1 tail) | Approximate vector exp |
| LASX | Medium (scalar exp) | Scalar fallback | Same gap as RVV |

### Tail Handling: RVV Advantage for N=145

RVV's dynamic `vsetvl` provides clean tail handling. For N=145:
- RVV: 9 iterations at VL=16 + 1 iteration at VL=1 → 10 total iterations
- AVX2: 18 iterations at VL=8 + 1 iteration at VL=1 → 19 iterations
- NEON: 36 iterations at VL=4 + 1 iteration at VL=1 → 37 iterations

RVV512 has the fewest iterations due to widest vector width. The 1-element tail is nearly free.

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| Max/sum reduction RVV (existing) | 4.0× vs scalar | 0.55% | 0.022 |
| Element-wise ops RVV (existing) | 3.0× vs scalar | 0.55% | 0.017 |
| Vector exp (proposed) | 1.5× | 1.10% | 0.006 |
| **Total** | | **1.10%** | **0.045×** |

## 5. Comparison with ViT-Base/16 Softmax

| Finding | ViT-Base/16 | ViT-Base/32 | Consistency |
|---------|-------------|-------------|-------------|
| % Compute | 1.32% | 1.10% | Scaled by N² |
| N mod 16 | 5 | **1** | Better alignment |
| No vector exp | Same gap | Same gap | Identical |
| Tail overhead | ~2.5% | ~0.7% | 3.6× less for /32 |

**Shape sensitivity**: Softmax's O(N²) element count means ViT-Base/32 has significantly fewer softmax elements (3M vs 5.6M), but the per-element vectorization efficiency is slightly better due to the 1-element tail.

---

*Cross-reference: ViT-Base/16 Softmax gap analysis at docs/report/vit-base-16/gap-analysis-softmax.md*
