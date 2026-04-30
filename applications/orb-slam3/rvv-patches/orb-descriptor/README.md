# ORB Descriptor (BRIEF) — RVV Analysis

**Status**: ANALYZED (deferred — <1% hotspot, low ROI)
**Date**: 2026-04-29
**Algorithm**: 256-bit BRIEF descriptor via rotated pixel-pair intensity comparisons

## Scalar Algorithm (ORBextractor.cc:107-146)

32 iterations × 16 pixel pairs each:
1. GET_VALUE macro: loads pixel at rotated position via `center[offset]`
2. 16 comparisons: `t0 < t1` packed into 1 byte via bit-shift OR
3. 32 bytes = 256-bit descriptor

## RVV512 Approach (Documented, Not Implemented)

```
for each 4 keypoints (VLEN=512, 8-bit pixels):
  1. vluxei8.v x 32: gather 32 pixel pairs (64 pixels) for each kpt
  2. vmslt.vv: compare pairs → 16-bit mask per kpt
  3. vcompress/pack: convert 16 mask bits → 1 byte
  4. vse8.v: store 32 descriptor bytes
```

Key instructions: `vluxei8.v` (indexed gather), `vmslt.vv` (comparison), `vmandnot.mm` (bit packing)

## Limitation

The pixel access pattern is rotation-dependent (sin/cos-based coordinate transform per keypoint). Unlike FAST (fixed Bresenham circle), each keypoint reads different pixel positions. This makes the RVV implementation gather-heavy and memory-bound. The <1% hotspot confirms the algorithm is already memory-efficient.

## Decision

**DEFERRED.** The <1% runtime contribution does not justify the implementation effort. The 16% g2o Eigen hotspot and 25% GaussianBlur are higher priorities (both now vectorized via OpenCV rebuild).

## References

- Source: `applications/orb-slam3/vendor/ORB_SLAM3/src/ORBextractor.cc` lines 107-146
- Phase 1 perf data: ORB descriptor <1% (g2o Eigen dominates at 16%)
