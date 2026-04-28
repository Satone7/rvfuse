# ViT-Base/16 Consolidated RVV Gap Analysis

**Application**: Vision Transformer Base/16 (google/vit-base-patch16-224)
**Architecture**: 12× Transformer Encoder (768-dim, 12 heads, d_k=64) + Classification Head
**Date**: 2026-04-28
**Perf Platform**: Banana Pi K1 (SpacemiT X60, rv64imafdcv_zvl256b, 4 core @ 1.6 GHz)
**BBV Platform**: QEMU with VLEN=512, BBV plugin interval=1000

## Executive Summary

ViT-Base/16 is an **SGEMM-dominant** Transformer workload. Hardware perf profiling on Banana Pi K1 reveals that MatMul (via `MlasSgemmKernelRvv512Impl`) accounts for **87.15%** of total inference time. The existing RVV512 SGEMM kernel (VL=16) is already integrated and provides substantial vectorization. The primary gap analysis finding is:

1. **N=197 tail handling**: ViT's sequence length (197 = 12×16 + 5) introduces 2.5% overhead on attention MatMul via scalar fallback — a novel finding not present in SuperGlue (N=1024, perfectly aligned)
2. **BF16 attention**: The highest-impact proposed extension (0.30× weighted benefit) — 2× throughput for attention MatMul
3. **Vector exponential**: Missing from RVV, affects both GELU and Softmax — but these operators are small fractions of total compute

## 1. Perf Profiling Results

### Hardware Metrics (Banana Pi K1, 10 iterations)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total time | 79.7s | ~8s per inference |
| Cycles | 127.1B | High compute intensity |
| Instructions | 51.8B | Memory-bound (low IPC) |
| IPC | 0.41 | Severe memory bottleneck |
| L1-dcache misses | 1.54% | Reasonable cache behavior |
| Branch misses | 0.35% | Low misprediction |

**IPC = 0.41** indicates ViT-Base/16 is **memory-bound** on the Banana Pi K1, not compute-bound. This means vector width improvements alone may not translate to proportional speedups without corresponding memory system improvements.

### Hot Function Profile

| Function | Self % | Role |
|----------|--------|------|
| `MlasSgemmKernelRvv512Impl` | 62.89% | Inner MatMul kernel (RVV512 FMA) |
| `MlasSgemmKernelRvv512` | 24.26% | Outer MatMul loop |
| `MlasErfKernel` | 3.78% | GELU activation (erf) |
| `MlasSgemmCopyPackB` | 1.92% | B-matrix packing |
| `MlasComputeSoftmaxThreaded` | 1.32% | Attention softmax |
| `MurmurHash3::x86_128` | 1.23% | Internal hashing |
| `BroadcastLooper` | 0.88% | Broadcast ops (residual add) |
| `Add<float>::Compute` | 0.56% | Residual adds |

## 2. BBV Profiling Results

| Operator | BB File | Disas File | BBs | Intervals | Key RVV Instructions |
|----------|---------|-----------|-----|-----------|---------------------|
| SGEMM Kernel | sgemm-kernel.0.bb (29MB) | sgemm-kernel.disas | 47 | 1,276,973 | `vfmacc.vf`, `vle32.v`, `vfmul.vf` |
| SGEMM Impl | sgemm-impl.0.bb (65MB) | sgemm-impl.disas | 13 | 3,182,850 | `vfmacc.vf`, `vfmadd.vf`, `vle32.v` |
| GELU/Erf | erf-gelu.0.bb (34MB) | erf-gelu.disas | 28 | 225,365 | `vmv.v.x`, `vsetivli`, scalar exp |
| Softmax | softmax.0.bb (27MB) | softmax.disas | 108 | 110,694 | `vfmul.vv`, `vsetvli`, scalar exp |

## 3. Operator-Level Gap Analysis Summary

| Operator | % Compute | VL Alignment | Primary Gap | Proposed Extension | Weighted Benefit |
|----------|-----------|-------------|-------------|-------------------|-----------------|
| SGEMM (MatMul) | 87.15% | K=768: ✓, N=197: ✗ | Tail handling for N%16≠0 | `vsetvl` tail path | 0.008× |
| Self-Attention | 66.3% (subset) | d_k=64: ✓, N=197: ✗ | No BF16 for 2× throughput | Zvfbfmin | 0.30× |
| GELU (Erf) | 3.78% | VL=4 mode | No vector exp; VL=4 not VL=16 | `vfexp_approx` | 0.02× |
| Softmax | 1.32% | N=197: ✗ (5 tail) | No vector exp | `vfexp_approx` | 0.005× |
| LayerNorm | <0.5% | D=768: ✓ | No gap (perfectly aligned) | None | 0.03× |

## 4. Priority Table (BBV-Weighted Benefit)

| Priority | Proposed Extension | Source Platform | Target Operator | Weighted Benefit |
|----------|-------------------|----------------|----------------|-----------------|
| **P0** | **Zvfbfmin (BF16)** | AVX-512 `vdpbf16ps` | Attention MatMul | **0.30×** |
| P1 | vsetvl tail path | ARM SVE predicates | SGEMM (N=197) | 0.008× |
| P2 | vfexp_approx | AVX-512 `vexp2ps` | GELU + Softmax | 0.025× |
| P3 | VL=16 erf kernel | — | GELU (MlasErfKernel) | 0.08× |

**Cumulative estimated benefit**: ~0.41× (sum of independent extensions)

## 5. N=197 Tail Analysis (ViT-Specific)

This is the key **novel finding** from ViT-Base/16 analysis, not present in SuperGlue (N=1024 aligned):

| Aspect | Analysis |
|--------|----------|
| Sequence length | 197 = 196 + 1 (CLS token) = 14×14 patches + 1 |
| 197 mod 16 | **5** → 12 full vector tiles + 5-element tail |
| Affected operators | QK^T, Attn×V, Softmax (all with N=197 dimension) |
| Affected fraction of compute | ~30% (attention MatMul + softmax) |
| Tail overhead | ~2.5% of attention MatMul (scalar fallback) |
| Overall impact | ~0.75% of total inference time |

**Comparison across Batch 2 apps**:

| Application | N | N mod 16 | Tail Elements | Tail Overhead |
|-------------|---|----------|---------------|---------------|
| SuperGlue | 1024 | 0 | None | 0% |
| **ViT-Base/16** | **197** | **5** | **5** | **~2.5%** |
| ViT-Base/32 (predicted) | 50 | 2 | 2 | ~4% |

## 6. Cross-Reference with SuperGlue Findings

The SuperGlue gap analysis (docs/report/superglue/) established the Batch 2 baseline for Transformer attention. Key cross-references:

| Finding | SuperGlue | ViT-Base/16 | Consistency |
|---------|-----------|-------------|-------------|
| SGEMM dominance | ~61% | ~87% | ✓ ViT even more SGEMM-heavy |
| N alignment | 1024 = 64×16 ✓ | 197 = 12×16+5 ✗ | Novel ViT finding |
| d_k=64 alignment | 64 = 4×16 ✓ | 64 = 4×16 ✓ | Consistent |
| BF16 gap | Present (0.30×) | Present (0.30×) | Identical gap magnitude |
| Vector exp gap | Present | Present | Identical |
| LayerNorm alignment | D=256 ✓ | D=768 ✓ | Both aligned |
| Self vs Cross attention | Self + Cross | Self-only | ViT simpler |

## 7. Architecture Profile

```
ViT-Base/16 Architecture (rv64gcv_zvl512b):
├── Patch Embedding: Conv2d(3, 768, k=16, s=16) → flatten
├── CLS Token + Position Embedding: (1, 197, 768)
├── 12× Transformer Encoder:
│   ├── Pre-norm LayerNorm(768)         — D=768 aligned ✓
│   ├── QKV Projection: SGEMM (197,768)×(768,2304) — K=768 aligned ✓
│   ├── QK^T MatMul: (12,197,64)×(12,64,197) — N=197 tail ✗
│   ├── Softmax: (12,197,197)            — N=197 tail ✗
│   ├── Attn×V MatMul: (12,197,197)×(12,197,64) — K=197 tail ✗
│   ├── Output Projection: SGEMM (197,768)×(768,768) — aligned ✓
│   ├── Residual Add: broadcast          — trivial
│   ├── Pre-norm LayerNorm(768)         — D=768 aligned ✓
│   ├── MLP Up: SGEMM (197,768)×(768,3072) — aligned ✓
│   ├── GELU (Erf): MlasErfKernel       — VL=4 mode
│   ├── MLP Down: SGEMM (197,3072)×(3072,768) — aligned ✓
│   └── Residual Add: broadcast          — trivial
├── Final LayerNorm(768)                — D=768 aligned ✓
└── Classification Head: Linear(768, 1000)
```

## 8. Artifacts

| Artifact | Path |
|----------|------|
| Runner source | applications/onnxrt/vit-base-16/runner/vit_runner.cpp |
| Build script | applications/onnxrt/vit-base-16/build.sh |
| Cross-compiled binary | output/cross-vit-base-16/vit_inference |
| Toolchain (zvl256b) | applications/onnxrt/vit-base-16/riscv64-linux-toolchain.cmake |
| Toolchain (zvl512b) | applications/onnxrt/vit-base-16/riscv64-linux-zvl512b-toolchain.cmake |
| Perf data | output/perf/vit-base-16/ |
| BBV data | output/bbv_rvv512/vit-base-16/ |
| SGEMM gap analysis | docs/report/vit-base-16/gap-analysis-sgemm.md |
| GELU gap analysis | docs/report/vit-base-16/gap-analysis-gelu-erf.md |
| Softmax gap analysis | docs/report/vit-base-16/gap-analysis-softmax.md |
| Self-Attention gap analysis | docs/report/vit-base-16/gap-analysis-self-attention.md |
| LayerNorm gap analysis | docs/report/vit-base-16/gap-analysis-layernorm.md |

---

*Generated by RVFuse ViT-Base/16 pipeline. Cross-references SuperGlue analysis (docs/report/superglue/) and YOLO analysis (docs/report/onnxrt/).*
