# GaussianBlur — RVV Vectorization via OpenCV Rebuild

**Status**: ENABLED (no greenfield code needed)
**Date**: 2026-04-29
**Method**: CMake rebuild with `-DCPU_BASELINE=RVV -DRISCV_RVV_SCALABLE=ON`

## Summary

OpenCV 4.10.0 has a complete RVV implementation via its universal intrinsics layer (`CV_SIMD_SCALABLE` + `intrin_rvv_scalable.hpp`). The original Banana Pi build had `CPU_BASELINE=""` and `CPU_DISPATCH=""`, which disabled ALL SIMD paths, resulting in purely scalar code (verified by ET-1).

Rebuilding OpenCV with `-DCPU_BASELINE=RVV` enables the existing RVV SIMD paths, generating correct RVV instructions for the GaussianBlur hot functions without any source code changes.

## Rebuild Instructions

Three changes needed:

### 1. Toolchain file (`applications/opencv/riscv64-linux-toolchain.cmake`)

Line 21: Change `-march=rv64gc` to `-march=rv64gcv_zvl512b`
Line 39: Change `set(CPU_BASELINE ""` to `set(CPU_BASELINE "RVV"`

### 2. OpenCV internal (`vendor/opencv/cmake/OpenCVCompilerOptimizations.cmake`)

Line 397: Change `-march=rv64gcv` to `-march=rv64gcv_zvl512b` (add VLEN-specific extension)

### 3. CMake configure

```bash
cmake -S vendor/opencv -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=riscv64-linux-toolchain.cmake \
  -DBUILD_LIST="core,imgproc,features2d,imgcodecs" \
  -DCPU_BASELINE=RVV \
  -DRISCV_RVV_SCALABLE=ON \
  -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON
ninja -j$(nproc) install
```

## Verification

### Build Result

| Metric | Before (Scalar) | After (RVV) |
|--------|-----------------|-------------|
| libopencv_imgproc.so size | 16.6 MB | 18.1 MB |
| RVV instructions | 0 | **8,137** |
| Key RVV ops present | — | vle8, vle16, vse8, vse16, vwmulu, vwmul, vadd.vv, vnclipu, vnsrl, vwcvtu |

### Hot Functions (RVV Confirmed)

- `hlineSmoothONa_yzy_a<uint8_t,ufixedpoint16>`: Vectorized via universal intrinsics (vwmulu + vadd + vnclipu)
- `vlineSmoothONa_yzy_a<uint8_t,ufixedpoint16>`: Vectorized via universal intrinsics (vwmul + vadd + vnclipu + vnsrl)

### Expected Speedup

- **Inner loop**: 8-16x (32 elements/iteration at VLEN=512 vs 1 element scalar)
- **Overall GaussianBlur**: ~4-8x (Amdahl's law with dispatch/border overhead)

## Challenges

1. **LLVM 22 scalable-to-fixed-width error**: Some OpenCV modules fail with `-march=rv64gcv_zvl512b` due to LLVM backend issues mixing scalable and fixed-width vector types. Workaround: build only needed modules.

2. **CMake flag misnomer**: `RISCV_RVV_SCALABLE=ON` alone does not enable RVV — it only selects the scalable API variant. `CPU_BASELINE=RVV` is the actual enabler.

3. **FORCE flags in toolchain**: `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` use `FORCE` which blocks CLI overrides. Must edit file directly.

## References

- ET-1 verification: `applications/orb-slam3/rvv-patches/gaussian-blur/verification.md`
- ET-1 test compile: `applications/orb-slam3/rvv-patches/gaussian-blur/test_compile.cpp`
- Phase 1 perf data: `output/perf/orb-slam3/`
