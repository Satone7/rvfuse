# Softmax RVV Gap Analysis — ViT-Base/16

**Operator**: Softmax (attention score normalization)
**Application**: Vision Transformer Base/16 — 12 encoder blocks, each with multi-head attention
**Shapes**: (12, 197, 197) per encoder — 12 attention heads, 197×197 score matrix
**Date**: 2026-04-28
**BBV Data**: QEMU-BBV profiling at VLEN=512 (output/bbv_rvv512/vit-base-16/softmax/)
**Perf Data**: `MlasComputeSoftmaxThreaded` = 1.32% of total runtime

## 1. Operator Profile

| Location | Count | Shape per instance | Total elements |
|----------|-------|-------------------|----------------|
| Attention softmax (each encoder) | 12 | (12, 197, 197) | 468,288 × 12 |
| **Total** | **12** | | **5,619,456** |

**Operation**: softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
**Reduction axis**: Last dimension (columns), per row per head
**% Compute**: 1.32% (perf confirmed)

### ViT-Specific: N=197 Alignment

$197 = 12 \times 16 + 5$ — the softmax reduction dimension is NOT aligned to VL=16.

- Each row has 197 elements → 12 full VL=16 vector iterations + 5 tail elements
- `vsetvl` handles tails dynamically, but the misaligned dimension causes:
  - 12/13 = 92.3% of iterations are full vector (optimal)
  - 1/13 = 7.7% are partial vector (5-element tail, lower throughput)

## 2. RVV Vectorization

### BBV Hotspot Analysis (108 BBs)

The softmax disassembly reveals a complex implementation with 108 basic blocks:

| BB | Instructions | Role | Key RVV Ops |
|----|-------------|------|-------------|
| BB 0 | 34 | Prologue + setup | Register save, dimension checks |
| BB 6 | 44 | Max reduction + subtract | `vle32.v`, `vfsub.vv`, reduction |
| BB 9 | 14 | Exp computation | Scalar exp calls |
| BB 10 | 9 | Sum reduction | `vfredusum`-like pattern |
| BB 12 | 8 | Division + store | `vfdiv.vf`, `vse32.v` |

**Key instructions observed**:
- `vsetvli zero,zero,e32,m1,ta,ma` — dynamic VL (correct for N=197 tail)
- `vle32.v` / `vse32.v` — vector load/store
- `vfmul.vv` / `vfdiv.vf` — element-wise operations
- Scalar `exp()` calls — **no vector exponential in RVV**

### Alignment Analysis

| Dimension | Value | mod 16 | Status |
|-----------|-------|--------|--------|
| N (seq length) | 197 | **5** | Tail handling via vsetvl |
| H (heads) | 12 | 12 | Processed sequentially |

## 3. Cross-Platform Comparison

| Platform | Softmax Throughput | Tail Handling | Key Gap |
|----------|-------------------|---------------|---------|
| RVV512 | Medium (scalar exp) | `vsetvl` (dynamic) | No vector exp |
| AVX-512 | High (VEXP2PS approx) | k-mask | Vector exp available |
| NEON | Medium (scalar exp) | Scalar fallback | Same gap as RVV |
| SVE | Medium-High (FEXPA) | Predicate mask | Approximate vector exp |
| LASX | Medium (scalar exp) | Scalar fallback | Same gap as RVV |

### Key Gap: No Vector Exponential

Same fundamental gap as GELU — RVV lacks vector `exp()`. All platforms except AVX-512 rely on scalar `expf()`. The benefit of a vector exp is limited here because softmax is a small fraction of total compute.

### Tail Handling: RVV Advantage

RVV's dynamic `vsetvl` provides a cleaner tail-handling mechanism than fixed-width ISAs. For N=197:
- RVV: 12 iterations at VL=16 + 1 iteration at VL=5 → 13 total iterations
- AVX2: 24 iterations at VL=8 + 1 iteration at VL=5 (masked) → 25 iterations
- NEON: 49 iterations at VL=4 + 1 iteration at VL=1 (masked) → 50 iterations

RVV512 has the fewest iterations due to widest vector width.

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| Max/sum reduction RVV (existing) | 4.0× vs scalar | 0.66% | 0.020 |
| Element-wise ops RVV (existing) | 3.0× vs scalar | 0.66% | 0.015 |
| Vector exp (proposed) | 1.5× | 1.32% | 0.005 |
| **Total** | | **1.32%** | **0.04×** |

Softmax contributes minimally to overall runtime (~1.32%). The existing RVV reduction instructions (`vfredmax`, `vfredusum`) already provide solid speedup. New instructions would have marginal overall benefit.

## 5. Comparison with SuperGlue Softmax

SuperGlue analysis (docs/report/superglue/gap-analysis-softmax.md) found:
- N=1024, perfectly aligned → no tail overhead
- Same vector exp gap → identical proposed extension

**ViT difference**: N=197 introduces tail handling overhead (~7.7% of softmax iterations), but the overall impact is negligible due to softmax's small compute share.

---

*Cross-reference: SuperGlue Softmax gap analysis at docs/report/superglue/gap-analysis-softmax.md*
