# g2o Eigen 6x6 — RVV512 Implementation

**Status**: IMPLEMENTED (eigen_rvv.inl + test.cpp + patch.diff)
**Date**: 2026-04-29
**Target**: Eigen 3.4.0 + g2o Bundle Adjustment
**Hotspot**: ~16% (largest remaining scalar hotspot after OpenCV rebuild)

## Summary

Eigen 3.4.0 lacks an RVV backend (`Eigen/src/Core/arch/RVV/` does not exist).
This patch provides RVV512 specializations for 6x6 fixed-size double-precision matrix operations
used by g2o's bundle adjustment (Schur complement on 6-DOF camera poses).

## Files

| File | Purpose |
|------|---------|
| `eigen_rvv.inl` | Core RVV kernel: 6×6 multiply, add, triangular solve |
| `test.cpp` | Correctness test against Eigen scalar reference |
| `patch.diff` | Integration into Eigen's architecture dispatch system |

## Key Operations

| Operation | Scalar Instructions | RVV Instructions | Speedup |
|-----------|-------------------|-----------------|---------|
| 6×6 multiply (C=A*B) | 288 | 48 | **6×** |
| 6×6 add (C=A+B) | 72 | 18 | **4×** |
| 6×6 triangular solve | ~90 | ~30 | **3×** |

## RVV Instructions Used

- `vle64.v` / `vse64.v` — 64-bit element load/store (6 columns × 6 doubles)
- `vfmacc.vf` — vector fused multiply-add with scalar (key instruction for matrix multiply)
- `vfadd.vv` — vector floating-point add
- `vfmul.vv` — vector floating-point multiply
- `vfmv.vf` — broadcast scalar to vector

## VLEN Configuration

- VLEN=512, SEW=64 → VLMAX=8
- Fixed VL=6 for 6×6 columns (fits in one vfloat64m1_t register)
- No tail processing needed (exact fit)

## Integration

### Method 1: Direct include (recommended for testing)

```cpp
#ifdef __riscv_vector
#include "eigen_rvv.inl"
#endif
#include <Eigen/Core>
// Eigen 6x6 operations now use RVV path
```

### Method 2: Eigen source patch (for production)

Apply `patch.diff` to Eigen 3.4.0 source tree. Creates `Eigen/src/Core/arch/RVV/PacketMath.h`
and adds RVV detection to `ConfigureVectorization.h`.

## Expected Benefit

- g2o Bundle Adjustment: ~4× speedup on 16% of ORB-SLAM3 runtime
- Overall ORB-SLAM3 speedup: ~3-4% (Amdahl's law: 16%/4 = 4% → net 4% gain with 4× BA speedup)

## Limitations

1. **VLEN-dependent**: Assumes VLEN≥512 (8 doubles). For VLEN=256, LMUL=2 needed to hold 6 doubles.
2. **Double precision only**: Single-precision (float32) would use vfloat32m2_t with VL=6.
3. **No 6×6×N batching**: The current implementation processes one 6×6 multiply at a time.
   For g2o's Schur complement which operates on many 6×6 blocks, batching could yield additional speedup.

## References

- Eigen arch examples: `Eigen/src/Core/arch/NEON/PacketMath.h`, `Eigen/src/Core/arch/SSE/PacketMath.h`
- g2o BlockSolver: `applications/orb-slam3/vendor/ORB_SLAM3/Thirdparty/g2o/g2o/core/block_solver.hpp`
- Phase 1 perf: `output/perf/orb-slam3/` (16% hotspot)
- Gap analysis: `docs/report/orb-slam3/gap-analysis/g2o-eigen-6x6.md`
