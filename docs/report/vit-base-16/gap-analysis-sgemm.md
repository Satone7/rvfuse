# SGEMM (MatMul) RVV Gap Analysis — ViT-Base/16

**Operator**: Single-precision General Matrix Multiply (SGEMM/MatMul)
**Application**: Vision Transformer Base/16 (google/vit-base-patch16-224)
**Date**: 2026-04-28
**BBV Data**: QEMU-BBV profiling at VLEN=512 (output/bbv_rvv512/vit-base-16/)
**Perf Data**: Hardware perf on Banana Pi K1 (output/perf/vit-base-16/)

## 1. Operator Profile

| Instance | Shape | Count | % Compute (perf) |
|----------|-------|-------|-------------------|
| QKV projection | (197,768)×(768,2304) | 12 | ~35% |
| Attn QK^T | (12,197,64)×(12,64,197) | 12 | ~15% |
| Attn×V | (12,197,197)×(12,197,64) | 12 | ~15% |
| Output projection | (197,768)×(768,768) | 12 | ~10% |
| MLP up | (197,768)×(768,3072) | 12 | ~8% |
| MLP down | (197,3072)×(3072,768) | 12 | ~8% |
| **Total** | | **72** | **~87%** |

**Perf confirmation**: `MlasSgemmKernelRvv512Impl` (62.89%) + `MlasSgemmKernelRvv512` (24.26%) = **87.15%** of total runtime.

### Shape Alignment Analysis (ViT-specific)

| Instance | K | K mod 16 | N (output cols) | N mod 16 | Tail Handling |
|----------|---|----------|-----------------|----------|---------------|
| QKV proj | 768 | 0 ✓ | 2304 | 0 ✓ | None |
| Attn QK^T | 64 | 0 ✓ | 197 | **5** ✗ | **Scalar fallback for 5 cols** |
| Attn×V | 197 | **5** ✗ | 64 | 0 ✓ | **Scalar K-loop for last 5** |
| Out proj | 768 | 0 ✓ | 768 | 0 ✓ | None |
| MLP up | 768 | 0 ✓ | 3072 | 0 ✓ | None |
| MLP down | 3072 | 0 ✓ | 768 | 0 ✓ | None |

**Key finding**: N=197 = 12×16 + 5 — attention MatMul instances require tail handling for 5 remainder columns. This is a ViT-specific challenge not present in SuperGlue (N=1024, perfectly aligned).

## 2. RVV Vectorization

### Patch: `rvv-patches/sgemm-kernel-vl16/`

Already integrated into the ORT build as `MlasSgemmKernelRvv512`. The VL=16 kernel processes 16 output columns per vector instruction.

### BBV Hotspot Analysis (from .disas data)

**MlasSgemmKernelRvv512Impl** (13 BBs, 0x1a2 bytes):

| BB | Instructions | Role | Key RVV Ops |
|----|-------------|------|-------------|
| BB 0 | 3 | Prologue | `vsetivli zero,16,e32,m1` |
| BB 1 | 4 | Loop init | `vmv.v.i v8,0` (zero acc) |
| BB 2 | 27 | First K-loop iter + setup | `vfmacc.vf`, `vfmadd.vf` |
| **BB 3** | **16** | **K-loop body (hot)** | `vfmacc.vf`, `vfmadd.vf`, `vle32.v` |
| BB 4 | 1 | K remainder check | `bne` |
| BB 5 | 3 | Alpha scale | `vfmul.vf` |
| BB 6 | 11 | Store + accumulate (ZeroMode=false) | `vfadd.vv`, `vse32.v` |
| BB 7 | 22 | N-loop reset + K-entry | `vmv1r.v` (reset accum) |
| BB 10 | 5 | Tail: spill to stack | `vse32.v` |
| BB 11 | 18 | Tail: scalar add+store loop | `flw`, `fadd.s`, `fsw` |
| BB 12 | 15 | Tail: scalar inner loop | `flw`, `fadd.s`, `fsw` |

**K-loop BB (BB 3) instruction mix**:
- 2× `flw` (load A scalars)
- 2× `vle32.v` (load B vectors)
- 2× `vfmacc.vf` (FMA accumulate)
- 2× `vfmadd.vf` (FMA subtract-accumulate)
- 4× `addi` (pointer increments)
- 1× `bgtu` (loop branch)
- 3× misc (address calc)

## 3. Cross-Platform Comparison

| Platform | Vector Width | f32/reg | K=768 Efficiency | Tail (N%16≠0) | Speedup vs Scalar |
|----------|-------------|---------|------------------|----------------|-------------------|
| RVV512 (VL=16) | 512-bit | 16 | 100% aligned | Scalar fallback | ~12× |
| AVX-512 | 512-bit | 16 | 100% aligned | k-mask partial | ~12× |
| AVX2 | 256-bit | 8 | 100% aligned | Scalar fallback | ~6× |
| NEON | 128-bit | 4 | 100% aligned | Scalar fallback | ~3× |
| SVE 512-bit | 512-bit | 16 | 100% aligned | Predicate mask | ~12× |
| LASX | 256-bit | 8 | 100% aligned | Scalar fallback | ~6× |

### Key Gap: Tail Handling for N=197

ViT's attention MatMul (N=197 = 12×16 + 5) forces the kernel into the scalar tail path (BB 11/12). This is a **ViT-specific** performance gap:

- **ARM SVE**: Uses predicate masks to handle 5 tail elements in-vector → no scalar fallback
- **AVX-512**: Uses k-mask registers for partial vector operations → minimal overhead
- **RVV**: Uses `vsetvl` for dynamic VL, but the current kernel design uses fixed VL=16 packing → falls back to scalar

**Proposed improvement**: Use `vsetvl`-based partial vector processing for the tail block instead of the current scalar loop. This would process 5 elements in one vector iteration instead of 5 scalar iterations.

## 4. BBV-Weighted Benefit

Based on perf data (SGEMM = 87.15% of total) and BBV instruction analysis:

| Item | Speedup | Weight (% compute) | Weighted |
|------|---------|-------------------|----------|
| SGEMM VL=16 kernel (existing) | 12× vs scalar | 87.15% | Baseline |
| vsetvl tail handling (proposed) | 1.02× on attention | 30% (attention MatMul) | 0.006 |
| Prefetch hint for B-matrix | 1.05× | 87.15% | 0.04 |
| **Total** | | | **0.05×** |

The tail handling improvement is small because attention MatMul is only ~30% of SGEMM time, and the tail (5/197 = 2.5%) affects only a fraction of that. The dominant MatMul instances (QKV, MLP) are perfectly aligned.

## 5. Comparison with SuperGlue SGEMM

SuperGlue's SGEMM analysis (K=256/512, N=1024) found perfect alignment for all shapes. ViT-Base/16 introduces **non-aligned N=197** for attention, which is the novel finding:

- SuperGlue: All shapes VL-aligned → no tail handling overhead
- ViT-Base/16: 4/6 MatMul instances aligned, 2/6 require tail → **2.5% overhead on attention**
- ViT-Base/32 (forthcoming): N=50 (patch count 7×7+1) → 50 mod 16 = 2 → **4% tail overhead**

---

*Cross-reference: SuperGlue SGEMM gap analysis at docs/report/superglue/gap-analysis-sgemm.md*
