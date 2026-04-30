# FAST Corner Detection — RVV Vectorization Status

**Status**: ENABLED via OpenCV rebuild (T2 greenfield SUPERSEDED)
**Date**: 2026-04-29
**Method**: OpenCV rebuild with `-DCPU_BASELINE=RVV -DRISCV_RVV_SCALABLE=ON`

## Summary

FAST corner detection (`cv::FAST_t<16>`, `cv::FAST_t<8>`) in `libopencv_features2d.so` is **fully auto-vectorized** when OpenCV is rebuilt with `CPU_BASELINE=RVV`. The rebuilt binary contains **1,525 RVV instructions** (up from 0 in the scalar build).

FAST_t<8> specifically contains 25 RVV instructions in its inner loop including `vmslt.vx`, `vmsgt.vx`, `vse8.v`, `vmv.v.i` — processing 64 pixels per iteration at VLEN=512.

## Build Result

| Metric | Before (Scalar) | After (RVV) |
|--------|-----------------|-------------|
| libopencv_features2d.so RVV instructions | 0 | **1,525** |
| FAST_t<8> inner loop RVV | 0 | 25 (vmslt, vmsgt, vse8, vmv) |

## T2 Decision

**ABANDONED.** The rebuild provides full RVV vectorization for FAST corner detection. No greenfield RVV implementation needed.

## Rebuild Instructions

Same as GaussianBlur — see `rvv-patches/gaussian-blur/README.md`. Include `features2d` in BUILD_LIST.

## References

- OpenCV source: `applications/opencv/vendor/opencv/modules/features2d/src/fast.cpp`
- Phase 1 perf data: `output/perf/orb-slam3/` (FAST at ~4.4% hotspot)
