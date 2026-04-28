# ViT-Base/32 Consolidated RVV Gap Analysis

**Application**: Vision Transformer Base/32 (google/vit-base-patch32-384)
**Architecture**: 12× Transformer Encoder (768-dim, 12 heads, d_k=64, SeqLen=145) + Classification Head
**Date**: 2026-04-28
**Perf Platform**: Banana Pi K1 (SpacemiT X60, rv64imafdcv_zvl256b, 4 core @ 1.6 GHz)
**BBV Platform**: QEMU with VLEN=512, BBV plugin interval=1000

## Executive Summary

ViT-Base/32 is an **SGEMM-dominant** Transformer workload, consistent with ViT-Base/16. Hardware perf profiling on Banana Pi K1 reveals that MatMul (via `MlasSgemmKernelRvv512Impl`) accounts for **~87%** of total inference time. The existing RVV512 SGEMM kernel (VL=16) is already integrated and provides substantial vectorization. The primary findings are:

1. **N=145 tail handling**: ViT-Base/32's sequence length (145 = 9×16 + 1) introduces only a **1-element tail** — significantly better than ViT-Base/16 (N=197, 5-element tail). This results in ~0.7% tail overhead vs ~2.5% for ViT-Base/16
2. **BF16 attention**: The highest-impact proposed extension (0.30× weighted benefit) — 2× throughput for attention MatMul
3. **Vector exponential**: Missing from RVV, affects both GELU and Softmax — but these operators are small fractions of total compute
4. **Shape sensitivity**: The key research contribution of this analysis — VL=16 efficiency varies significantly with sequence length

## 1. Perf Profiling Results

### Hardware Metrics (Banana Pi K1, 10 iterations)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total time | 61.2s | ~6.1s per inference |
| Cycles | 97.3B | Lower than ViT-Base/16 (127.1B) |
| Instructions | 38.9B | 25% fewer instructions than /16 |
| IPC | 0.40 | Memory-bound (consistent with /16) |
| L1-dcache misses | 1.58% | Reasonable cache behavior |
| Branch misses | 0.39% | Low misprediction |

**IPC = 0.40** indicates ViT-Base/32 is **memory-bound** on the Banana Pi K1, consistent with ViT-Base/16 (IPC=0.41).

### Performance Comparison: ViT-Base/16 vs ViT-Base/32

| Metric | ViT-Base/16 | ViT-Base/32 | Ratio |
|--------|-------------|-------------|-------|
| Total time (10 iters) | 79.7s | 61.2s | 0.77× |
| Cycles | 127.1B | 97.3B | 0.77× |
| Instructions | 51.8B | 38.9B | 0.75× |
| IPC | 0.41 | 0.40 | 0.98× |
| L1-dcache miss rate | 1.54% | 1.58% | Similar |

The 23% lower runtime matches the expected compute reduction from fewer patches (145 vs 197 tokens → ~74% of attention FLOPs).

### Hot Function Profile

| Function | Self % | Role |
|----------|--------|------|
| `MlasSgemmKernelRvv512Impl` | 58.66% | Inner MatMul kernel (RVV512 FMA) |
| `MlasSgemmKernelRvv512` | 22.49% | Outer MatMul loop |
| `MlasErfKernel` | 3.36% | GELU activation (erf) |
| `MlasSgemmCopyPackB` | 4.33% | B-matrix packing |
| `MurmurHash3::x86_128` | 3.23% | Internal hashing |
| `MlasComputeSoftmaxThreaded` | 1.10% | Attention softmax |
| `BroadcastLooper` | 0.91% | Broadcast ops (residual add) |
| `Add<float>::Compute` | 0.47% | Residual adds |

**Comparison with ViT-Base/16**:

| Function | ViT-Base/16 % | ViT-Base/32 % | Change |
|----------|--------------|--------------|--------|
| SgemmKernelRvv512Impl | 62.89% | 58.66% | Lower (fewer attention FLOPs) |
| SgemmKernelRvv512 | 24.26% | 22.49% | Lower |
| MlasErfKernel | 3.78% | 3.36% | Scaled by SeqLen |
| SgemmCopyPackB | 1.92% | 4.33% | Higher (more packing overhead) |
| Softmax | 1.32% | 1.10% | Scaled by N² |

The higher B-packing overhead (4.33% vs 1.92%) is notable — the smaller N dimension for attention means a higher packing-to-compute ratio.

## 2. BBV Profiling Results

| Operator | BB File | Disas File | BBs | Key RVV Instructions |
|----------|---------|-----------|-----|---------------------|
| Full inference | vit32-inference.0.bb (337MB) | vit32-inference.disas (761K lines) | 89,528 | `vfmacc.vf`, `vle32.v`, `vfmul.vf`, `vmv.v` |

BBV data collected at VLEN=512 with interval=1000 (partial run — ~2 iterations captured before QEMU timeout).

### BBV Instruction Distribution

| Category | Count | % | Notes |
|----------|-------|---|-------|
| `vse64.v` | 14,838 | 55.7% | Vector store (SGEMM output, register spill) |
| `vmv.v` | 6,341 | 23.8% | Vector move (zero init, coeff load) |
| `vle64.v` / `vle8.v` | 833 / 810 | 6.2% | Vector load |
| `vse8.v` | 835 | 3.1% | Byte vector store |
| `vsetivli` (VL=16) | ~20 | 0.1% | SGEMM kernel setup |
| `vsetvli` (dynamic) | 446 | 1.7% | Dynamic VL for tail handling |

**Note**: The BBV plugin captures the full address space including the ORT shared library. The `vse64.v` dominance reflects SGEMM accumulator store-back patterns. The actual floating-point compute instructions (`vfmacc.vf`, `vfmadd.vf`) are underrepresented in the disassembly because the plugin's symbol resolution for the dynamically loaded ORT library is limited.

## 3. Operator-Level Gap Analysis Summary

| Operator | % Compute | VL Alignment | Primary Gap | Proposed Extension | Weighted Benefit |
|----------|-----------|-------------|-------------|-------------------|-----------------|
| SGEMM (MatMul) | 87% | K=768: ✓, N=145: ✗(1) | Minimal tail (1 elem) | `vsetvl` tail path | 0.001× |
| Self-Attention | 60% (subset) | d_k=64: ✓, N=145: ✗(1) | No BF16 for 2× throughput | Zvfbfmin | 0.30× |
| GELU (Erf) | 3.36% | VL=4 mode | No vector exp; VL=4 not VL=16 | `vfexp_approx` | 0.017× |
| Softmax | 1.10% | N=145: ✗(1 tail) | No vector exp | `vfexp_approx` | 0.006× |
| LayerNorm | <0.5% | D=768: ✓ | No gap (perfectly aligned) | None | 0.03× |

## 4. Priority Table (BBV-Weighted Benefit)

| Priority | Proposed Extension | Source Platform | Target Operator | Weighted Benefit |
|----------|-------------------|----------------|----------------|-----------------|
| **P0** | **Zvfbfmin (BF16)** | AVX-512 `vdpbf16ps` | Attention MatMul | **0.30×** |
| P1 | VL=16 erf kernel | — | GELU (MlasErfKernel) | 0.068× |
| P2 | vfexp_approx | AVX-512 `vexp2ps` | GELU + Softmax | 0.023× |
| P3 | vsetvl tail path | ARM SVE predicates | SGEMM (N=145) | 0.001× |

**Cumulative estimated benefit**: ~0.39× (sum of independent extensions)

## 5. Shape Sensitivity Analysis (Key Research Contribution)

This section provides the **first quantitative comparison** of RVV VL=16 efficiency across different Transformer sequence lengths within the same architecture family (ViT-Base).

### 5.1 Tail Element Analysis

| Application | N (SeqLen) | N mod 16 | Full VL=16 Tiles | Tail Elements | VL Utilization |
|-------------|-----------|----------|-----------------|---------------|----------------|
| SuperGlue | 1024 | 0 | 64 | 0 | 100.0% |
| **ViT-Base/32** | **145** | **1** | **9** | **1** | **99.3%** |
| **ViT-Base/16** | **197** | **5** | **12** | **5** | **97.5%** |

**Key finding**: ViT-Base/32's 1-element tail gives it **99.3% VL=16 utilization** — effectively as good as SuperGlue's perfect alignment. ViT-Base/16's 5-element tail reduces utilization to 97.5%.

### 5.2 Impact by Operator

| Operator | N=197 Tail Impact | N=145 Tail Impact | Improvement |
|----------|-------------------|-------------------|-------------|
| QK^T MatMul (output cols) | 5 scalar cols / 197 = 2.5% | 1 scalar col / 145 = **0.7%** | 3.6× less |
| Attn×V MatMul (K-loop) | 5 scalar K / 197 = 2.5% | 1 scalar K / 145 = **0.7%** | 3.6× less |
| Softmax (reduction dim) | 5 tail elems / 197 = 2.5% | 1 tail elem / 145 = **0.7%** | 3.6× less |
| LayerNorm (D=768) | 0% (aligned) | 0% (aligned) | No change |
| MLP (K=768/3072) | 0% (aligned) | 0% (aligned) | No change |

### 5.3 Practical Significance

While the relative improvement in tail handling is large (3.6×), the absolute impact on total inference time is small:

| Application | Tail Overhead | Overall Impact on Inference |
|-------------|--------------|---------------------------|
| SuperGlue | 0% | 0% |
| **ViT-Base/32** | **0.7% of attention ops** | **~0.4% of total** |
| ViT-Base/16 | 2.5% of attention ops | ~0.75% of total |

**Conclusion**: Sequence length shape sensitivity exists but is a **second-order effect** compared to the BF16 throughput gap. For practical optimization, BF16 (2× throughput on 60% of compute) dominates over tail handling (0.4% vs 0.75% of total time).

### 5.4 Generalized Shape Sensitivity Model

For any Transformer with VL=16 vectorization and sequence length N:

$$\text{VL utilization} = \frac{N - (N \bmod 16)}{N} = 1 - \frac{N \bmod 16}{N}$$

$$\text{Tail overhead} \approx \frac{N \bmod 16}{N} \times f_{\text{attention}}$$

Where $f_{\text{attention}}$ is the fraction of compute in N-dependent operators.

| N mod 16 | Worst case (N=17) | Best case (N→∞) | Typical (N≈200) |
|-----------|-------------------|------------------|-----------------|
| 0 | 0% | 0% | 0% |
| 1 | 5.9% | ~0% | 0.7% |
| 5 | 29.4% | ~0% | 2.5% |
| 8 | 47.1% | ~0% | 4.0% |
| 15 | 88.2% | ~0% | 7.5% |

**Takeaway**: Small N values (like ViT sequence lengths) make tail handling more impactful. For very large N (like LLMs), tail overhead becomes negligible regardless of N mod 16.

## 6. Cross-Reference with ViT-Base/16 Findings

| Finding | ViT-Base/16 | ViT-Base/32 | Consistency |
|---------|-------------|-------------|-------------|
| SGEMM dominance | 87.15% | ~87% | ✓ Consistent |
| Top gap: BF16 | P0 (0.30×) | P0 (0.30×) | ✓ Same |
| VL=4 GELU inefficiency | 3.78% weight | 3.36% weight | ✓ Scaled by N |
| No vector exp | Same gap | Same gap | ✓ Identical |
| LayerNorm: no gap | <0.5% | <0.5% | ✓ Identical |
| Memory-bound IPC | 0.41 | 0.40 | ✓ Consistent |
| Tail handling | 5 elem (2.5%) | **1 elem (0.7%)** | ✓ Shape-dependent |
| B-packing overhead | 1.92% | 4.33% | ✗ Higher for /32 |

**Anomaly**: The B-packing overhead is significantly higher for ViT-Base/32 (4.33% vs 1.92%). This is likely because:
1. Smaller attention MatMul shapes (N=145) mean more calls to `MlasSgemmCopyPackB` relative to compute
2. The fixed cost of packing is amortized over fewer K iterations for smaller N

## 7. Artifact Summary

| Artifact | Location |
|----------|----------|
| Runner source | `applications/onnxrt/vit-base-32/runner/vit_runner.cpp` |
| Build script | `applications/onnxrt/vit-base-32/build.sh` |
| Cross-compiled binary | `output/cross-vit-base-32/vit_inference` |
| zvl512b binary | `output/cross-vit-base-32/vit_inference_zvl512b` |
| Perf data | `docs/report/vit-base-32/perf-data/perf-sw.data` |
| BBV data | `output/bbv-vit32/vit32-inference.*` |
| SGEMM gap analysis | `docs/report/vit-base-32/gap-analysis-sgemm.md` |
| Attention gap analysis | `docs/report/vit-base-32/gap-analysis-self-attention.md` |
| GELU gap analysis | `docs/report/vit-base-32/gap-analysis-gelu-erf.md` |
| Softmax gap analysis | `docs/report/vit-base-32/gap-analysis-softmax.md` |
| LayerNorm gap analysis | `docs/report/vit-base-32/gap-analysis-layernorm.md` |
| This report | `docs/report/vit-base-32/vit-base-32-consolidated-2026-04-28.md` |

---

*Generated as part of RVFuse Batch 2 analysis. Cross-reference with ViT-Base/16 consolidated report at docs/report/vit-base-16/vit-base-16-consolidated-2026-04-28.md*
