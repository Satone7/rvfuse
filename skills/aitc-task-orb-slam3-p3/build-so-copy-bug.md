# ORB-SLAM3 RVV Build Operational Notes

## Build Script .so Copy Bug

**Problem**: `build.sh` `cross_compile_orbslam()` copies `libORB_SLAM3.so` from `${BUILD_DIR}/lib/` which doesn't exist.

**Root Cause**: ORB-SLAM3's CMakeLists.txt sets `CMAKE_LIBRARY_OUTPUT_DIRECTORY` to `${PROJECT_SOURCE_DIR}/lib`, overriding the default build-tree output. The actual .so files end up in:
- `vendor/ORB_SLAM3/lib/libORB_SLAM3.so` (main lib)
- `vendor/ORB_SLAM3/Thirdparty/g2o/lib/libg2o.so` (g2o)
- `vendor/ORB_SLAM3/lib/libDBoW2.so` (DBoW2, bundled into same dir)

**Fix**: Either:
1. Override `CMAKE_LIBRARY_OUTPUT_DIRECTORY` in cmake configure: `-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${BUILD_DIR}/lib`
2. Update `build.sh` to copy from `${ORBSLAM_SOURCE}/lib/` instead of `${BUILD_DIR}/lib/`
3. Use the `find` fallback (already works for g2o/DBoW2 but NOT libORB_SLAM3.so due to the initial `cp` that silently fails)

## LLVM 22 Auto-vectorization Gaps for ORB-SLAM3

### vcpop.m Not Generated
LLVM 22 does NOT generate `vcpop.m` for popcount loops. The auto-vectorizer produces a software popcount sequence (shift-mask-add-multiply-reduce) even when `-march=rv64gcv` is specified. This is confirmed in `DescriptorDistance` function.

### Eigen Fixed-Size Matrix Not Vectorized
Eigen `Matrix<double,6,6>` operations in `Optimizer::Marginalize` are NOT auto-vectorized. Only memory copy (vle64/vse64) instructions appear. The auto-vectorizer cannot handle Eigen's complex template expression trees for small fixed-size matrices. Contrast with `G2oTypes::linearizeOplus` which DOES vectorize because it unrolls to explicit scalar float ops that the SLP vectorizer can combine.

## Correct Artifacts Location After Rebuild

When verifying RVV in rebuilt ORB-SLAM3, check the **vendor source tree**, not `output/orb-slam3/lib/`:
```bash
# Correct location (has RVV):
llvm-objdump -d applications/orb-slam3/vendor/ORB_SLAM3/lib/libORB_SLAM3.so | grep -c vsetvli

# Wrong location (stale old build, no RVV):
llvm-objdump -d output/orb-slam3/lib/libORB_SLAM3.so | grep -c vsetvli
```

## Rebuild Command

```bash
LLVM_INSTALL=/home/pren/wsp/cx/rvfuse/third_party/llvm-install \
SYSROOT=/home/pren/wsp/cx/rvfuse/output/orb-slam3/sysroot \
bash applications/orb-slam3/build.sh --skip-sysroot --skip-opencv --skip-source
```

The `--skip-source` flag is safe — patches are only applied if their `.applied` marker file is missing.
