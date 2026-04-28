# Softmax RVV Gap Analysis — SuperGlue

**Operator**: Softmax (attention weights normalization)
**Role**: Normalize attention scores in 18 attention blocks (self + cross, 9 layers × 2 images)
**Shapes**: $(4, N, N)$ per attention head, $N \in [1, 1024]$
**Date**: 2026-04-28

## 1. Operator Profile

- **18 instances**: 9 layers × 2 images (A and B each have attention softmax)
- **Shape per instance**: $(H, N, N) = (4, N, N)$ where N ≤ 1024
- **Operation**: $\text{softmax}(x_i) = \exp(x_i - \max(x)) / \sum_j \exp(x_j - \max(x))$
- **Reduction axis**: Last dimension (columns), per row
- **% Compute**: ~3% (estimated)

## 2. RVV Vectorization

### Strategy

Row-wise softmax with vectorized max/sum reductions and element-wise operations:

| Step | RVV Instruction | Notes |
|------|----------------|-------|
| Max reduction | `vfredmax_vs_f32m1` | Find maximum per row |
| Subtract + exp | `vfsub_vv` + `expf` (scalar) | No vector exp in base V |
| Sum reduction | `vfredusum_vs_f32m1` | Compute partition function |
| Divide | `vfdiv_vf_f32m1` | Normalize by sum |

### Patch: `rvv-patches/softmax-channel-f32/`

The channel-wise softmax pattern adapts to row-wise softmax by changing the reduction axis. Core operations (max reduction, subtract, exp, sum reduction, divide) are identical.

### Alignment

- $N=1024 \bmod 16 = 0$ → perfect VL alignment
- $N$ varies per image → dynamic `vsetvl` handles tails

## 3. Cross-Platform Comparison

| Platform | Softmax Throughput | Key Gap |
|----------|-------------------|---------|
| RVV512 | Medium (scalar expf) | No vector exp instruction |
| AVX-512 | High (VEXP2PS approximate) | Vector exp available |
| NEON | Medium (scalar expf) | Same gap as RVV |
| SVE | Medium-High (FEXPA) | Approximate vector exp |

**Key gap**: RVV base V extension lacks a vector exponential instruction. All platforms except AVX-512 rely on scalar `expf()`. The proposed Zvfbfmin extension would address this.

## 4. BBV-Weighted Benefit

| Item | Speedup | Weight | Weighted |
|------|---------|--------|----------|
| Max/sum reduction RVV | 4.0× | 1.5% | 0.06 |
| Element-wise div RVV | 3.0× | 1.5% | 0.05 |
| Vector exp (proposed) | 1.5× | 3.0% | 0.05 |
| **Total** | | **3%** | **0.15×** |

Softmax contributes modestly to overall runtime (~3%) but is critical for numerical accuracy. RVV reduction instructions (`vfredmax`, `vfredusum`) provide solid speedup.
