# g2o Eigen 6x6 — Cross-Platform RVV Gap Analysis

**Date**: 2026-04-29 | **Operator**: g2o Bundle Adjustment (Eigen 6x6 dense matrix)
**RVV Coverage**: 0 RVV instructions (NOT auto-vectorized) | **Hotspot**: ~16%

## Current Status

Eigen 3.4.0 does NOT auto-vectorize for RISC-V. Both libg2o.so and libORB_SLAM3.so were rebuilt with
`-march=rv64gcv_zvl512b` but contain zero RVV instructions. Eigen's expression templates lack RVV
backend support — unlike NEON (`Eigen/src/Core/arch/NEON/`) and SSE/AVX (`Eigen/src/Core/arch/SSE/`),
there is no `Eigen/src/Core/arch/RVV/` directory.

## Scalar Instruction Sequence (from perf annotate)

### Eigen::internal::dense_assignment_loop (6x6 Matrix Multiply)

The 6×6 matrix multiply C = A × B uses 6^3 = 216 scalar FMA operations:

```
// Per output element C[i][j] = Σ(k=0..5) A[i][k] * B[k][j]
loop over i (0..5):         // 6 iterations
  loop over j (0..5):       // 6 iterations
    loop over k (0..5):     // 6 iterations
      fmadd.d  fa0, fa1, fa2, fa0  // C[i][j] += A[i][k] * B[k][j]
                                  // 216 fma.d instructions total
```

## Proposed RVV512 Implementation

### Design: Column-Major 6x6 Matrix Multiply with vfloat64m1

VLEN=512, SEW=64 (double precision) → VLMAX=8 doubles per vector register.
6 doubles per column fit in a single vfloat64m1_t with vl=6.

```
void eigen_6x6_mul_rvv(double* C, const double* A, const double* B) {
    // Load B columns into vector registers (column-major: columns are contiguous)
    vfloat64m1_t b0 = __riscv_vle64_v_f64m1(&B[0], 6);   // B column 0
    vfloat64m1_t b1 = __riscv_vle64_v_f64m1(&B[6], 6);   // B column 1
    vfloat64m1_t b2 = __riscv_vle64_v_f64m1(&B[12], 6);  // B column 2
    vfloat64m1_t b3 = __riscv_vle64_v_f64m1(&B[18], 6);  // B column 3
    vfloat64m1_t b4 = __riscv_vle64_v_f64m1(&B[24], 6);  // B column 4
    vfloat64m1_t b5 = __riscv_vle64_v_f64m1(&B[30], 6);  // B column 5

    // Compute C column 0 = A[0][0]*b0 + A[1][0]*b1 + ... + A[5][0]*b5
    vfloat64m1_t c0 = __riscv_vfmv_vf_f64m1(0.0, 6);
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[0], b0, 6);   // c0 += A[0][0] * b0
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[6], b1, 6);   // c0 += A[1][0] * b1
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[12], b2, 6);  // c0 += A[2][0] * b2
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[18], b3, 6);  // c0 += A[3][0] * b3
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[24], b4, 6);  // c0 += A[4][0] * b4
    c0 = __riscv_vfmacc_vf_f64m1(c0, A[30], b5, 6);  // c0 += A[5][0] * b5

    // ... repeat for c1..c5 (each column)

    // Store results
    __riscv_vse64_v_f64m1(&C[0], c0, 6);
    // ...
}
```

**Instruction count**: 6 vle64 + 36 vfmacc + 6 vse64 = 48 RVV instructions for 6×6 multiply
**Scalar**: 216 fma.d + 36 load + 36 store = 288 instructions
**RVV speedup**: 6× (288→48 instructions, plus vector pipeline throughput)

### 6x6 Matrix Add (C = A + B)

```
vfloat64m1_t a0 = __riscv_vle64_v_f64m1(&A[0], 6);   // 6 columns
// ... (6 loads)
vfloat64m1_t c0 = __riscv_vfadd_vv_f64m1(a0, b0, 6);  // 6 vfadd.vv
// ... (6 stores)
```

12 instructions vs 36 scalar → 3× speedup.

### 6x6 Triangular Solve (Cholesky Back-substitution)

For factor L (lower triangular) and vector b, solve Lx = b:
```
// Forward substitution: x[i] = (b[i] - Σ(k<i) L[i][k]*x[k]) / L[i][i]
// With RVV: use vfmacc for the Σ term, vfdiv for the division
for (i = 0; i < 6; i++) {
    vfloat64m1_t partial = __riscv_vfmv_vf_f64m1(0.0, 6);
    // Accumulate L[i][0..i-1] * x[0..i-1]
    partial = __riscv_vfmacc_vv_f64m1(partial, L_cols[i], x_vec, i);
    x[i] = (b[i] - __riscv_vfmv_fs_f64m1(partial)) / L[i][i];
}
```

~30 RVV instructions vs ~90 scalar → 3× speedup.

## Cross-Platform Comparison (6x6 Matrix Multiply, f64)

| Platform | Key Instruction(s) | Vector Width | Ops/Iteration | Total Vector Instructions |
|----------|-------------------|-------------|---------------|--------------------------|
| **RVV512** | `vle64.v` + `vfmacc.vf` | 8 doubles | 6×6 in 36 fmas | 48 |
| **x86 AVX2** | `_mm256_load_pd` + `_mm256_fmadd_pd` | 4 doubles | 6×6 in 54 fmas | 60 |
| **x86 AVX-512** | `_mm512_load_pd` + `_mm512_fmadd_pd` | 8 doubles | 6×6 in 36 fmas | 48 |
| **ARM NEON** | `vld1q_f64` + `vfmaq_f64` | 2 doubles | 6×6 in 108 fmas | 114 |
| **ARM SVE** | `ld1d` + `fmla` (VL=512) | 8 doubles | 6×6 in 36 fmas | 48 |
| **LoongArch LSX** | `vld` + `vfmadd.d` | 2 doubles | 6×6 in 108 fmas | 114 |

**RVV advantage**: Matches AVX-512/SVE at 48 instructions. 2.4× fewer than NEON/LSX (114). But the real win for 6×6 is the fixed VL=6 fitting exactly in LMUL=1 — no masking, no tail handling, no loop overhead.

## Key Gap: Eigen Lacks RVV Backend

Eigen has architecture-specific backends for:
- `Eigen/src/Core/arch/SSE/` — x86 SSE/AVX/AVX-512 via `Packet4d`, `Packet8d`
- `Eigen/src/Core/arch/NEON/` — ARM NEON via `Packet2d`
- `Eigen/src/Core/arch/AltiVec/` — Power VSX
- `Eigen/src/Core/arch/ZVector/` — S390X

**No `Eigen/src/Core/arch/RVV/` exists.** This is the root cause of the 0 RVV instruction count.

### What an RVV Backend Needs

```
// Proposed: Eigen/src/Core/arch/RVV/PacketMath.h
template<> struct packet_traits<double> {
    typedef vfloat64m1_t type;        // Packet = 1 vector register
    enum {
        Vectorizable = 1,
        size = 6,                      // 6 doubles per packet (6x6 matrix column)
        HasAdd = 1, HasSub = 1,
        HasMul = 1, HasDiv = 1,
        HasFma = 1,                    // vfmacc.vf
        HasBlend = 1,                  // via mask
        AlignedOnScalar = 1
    };
};

// Packet ops map to RVV intrinsics
template<> EIGEN_STRONG_INLINE
Packet2d padd<Packet2d>(const Packet2d& a, const Packet2d& b) {
    return __riscv_vfadd_vv_f64m1(a, b, 6);
}
```

## Estimated Benefit

| Operation | Scalar (cycles) | RVV512 (cycles) | Speedup |
|-----------|----------------|-----------------|---------|
| 6×6 multiply (C=A*B) | ~300 | ~50 | **6×** |
| 6×6 add (C=A+B) | ~40 | ~15 | **2.7×** |
| 6×6 triangular solve | ~120 | ~40 | **3×** |
| Schur complement update | ~500 | ~100 | **5×** |
| **Overall g2o BA (16% hotspot)** | — | — | **~4× estimated** |

With 4× speedup on 16% of runtime, the overall ORB-SLAM3 speedup from g2o Eigen RVV is ~3-4%.

## Implementation Priority

**HIGH.** This is the largest remaining scalar hotspot. The implementation is well-defined:
1. 48 RVV instructions for matrix multiply
2. Template specialization matches Eigen's existing architecture pattern
3. Test vector: compare against Eigen scalar for 6x6 correctness

See companion implementation: `applications/orb-slam3/rvv-patches/eigen-6x6/eigen_rvv.inl`
