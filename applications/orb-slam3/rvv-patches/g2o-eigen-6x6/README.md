# g2o Eigen 6x6 Matrix Operations — Auto-vectorization Status

**Status**: NOT AUTO-VECTORIZED (RVV implementation needed)
**Date**: 2026-04-29
**Verification**: ORB-SLAM3 rebuilt with `-march=rv64gcv_zvl512b` — 0 RVV instructions in libg2o.so and libORB_SLAM3.so

## Verification Methodology

1. Fixed toolchain file: `-march=rv64gc` → `-march=rv64gcv_zvl512b` (same fix as OpenCV T1)
2. Rebuilt ORB-SLAM3 with ninja (117/117 targets)
3. Disassembled both libg2o.so and libORB_SLAM3.so
4. Counted RVV instructions: **0** in both

## Why No Auto-vectorization?

Eigen 3.4.0 uses C++ expression templates. Auto-vectorization depends on:
1. **Compiler recognizing the loop pattern**: Eigen's template-expanded loops may be too complex for LLVM's auto-vectorizer to recognize as vectorizable
2. **No RVV-specific Eigen backend**: Eigen has NEON/SSE/AVX backends with `#ifdef __ARM_NEON` / `#ifdef __SSE__`, but NO `#ifdef __riscv_vector` backend
3. **6x6 fixed-size matrices**: Small fixed-size operations may not trigger vectorization (compiler often vectorizes when trip count > vector width)

## Key Hot Functions

From Phase 1 perf annotate (16% hotspot):
- `Eigen::internal::dense_assignment_loop` — 6x6 matrix assignment
- `g2o::BlockSolver` — template-heavy sparse solver using Eigen

## RVV Implementation Path (Emergent Task)

A greenfield RVV implementation would target:
1. **Eigen 6x6 matrix multiply**: `vle64.v` for column loads, `vfmacc.vv` for FMA
2. **Eigen 6x6 matrix add**: `vle64.v` + `vfadd.vv` + `vse64.v`
3. **Eigen 6x6 triangular solve**: `vle64.v` + `vfdiv.vv` + `vfmacc.vv` for back-substitution

The implementation would be an `rvv-eigen-inline` header that specializes Eigen's `dense_assignment_loop` for 6x6 matrices on RISC-V.

## Recommendation

Create an emergent task for Eigen 6x6 RVV implementation. Priority: HIGH (16% hotspot). This is the largest remaining scalar hotspot now that GaussianBlur (25%) and FAST (4.4%) are covered by the OpenCV rebuild.

## References

- Eigen source: `output/eigen3/eigen-3.4.0/`
- g2o source: `applications/orb-slam3/vendor/ORB_SLAM3/Thirdparty/g2o/`
- Phase 1 perf: `output/perf/orb-slam3/perf_annotate.txt`
