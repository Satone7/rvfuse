# get-feature-maps

RVV implementation of HOG gradient magnitude computation in `getFeatureMaps()` — FHOG feature extraction for KCF tracker.

## Status

⚠️ In development — Test pending QEMU verification

## Files

| File | Purpose |
|------|---------|
| `rvv_get_feature_maps.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into KCFcpp fhog.cpp |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signatures

```cpp
// Simple magnitude: sqrt(dx^2 + dy^2)
void computeMagnitudeSimple_rvv(
    const float* dx_data,
    const float* dy_data,
    float* magnitude,
    size_t count);

// Partial norm: sum of squares for normalization
void computePartialNorm_rvv(
    const float* map_data,
    float* partOfNorm,
    int numFeatures,
    int p,
    size_t count);

// Truncation: clamp values above threshold
void truncateFeatureMap_rvv(
    float* data,
    size_t count,
    float threshold);
```

## Algorithm

### 1. Gradient Magnitude (computeMagnitudeSimple)
- Input: dx, dy gradient arrays
- Output: magnitude = sqrt(dx^2 + dy^2)
- RVV strategy:
  - Load dx, dy pairs into vector registers
  - Square each component (vfmul)
  - Add squares (vfadd)
  - Compute sqrt (vfsqrt)
  - Store result

### 2. Partial Norm (computePartialNorm)
- Input: feature map data
- Output: sum of squares for first p elements per position
- RVV strategy:
  - Vectorized inner loop for sum of squares
  - Reduction using vfredsum for accumulation

### 3. Truncation (truncateFeatureMap)
- Input: feature map, threshold value
- Output: clamped values (min(v, threshold))
- RVV strategy:
  - Single-pass vfmin for clamping

## VLEN Requirement

- VLEN >= 128: Basic RVV support required
- VLEN >= 512: Better throughput for large arrays

## Hotspot Analysis Context

From `output/perf/opencv-kcf/analysis_report.md`:

| Function | Self % | Category |
|----------|--------|----------|
| `getFeatureMaps` | 18.23% | HOG extraction (P0) |
| `normalizeAndTruncate` | 6.12% | Feature normalization (P1) |

The RVV implementation targets:
- Magnitude computation in `getFeatureMaps` inner loops
- Norm computation in `normalizeAndTruncate`
- Truncation in `normalizeAndTruncate`

Total potential improvement: ~24% of execution time.

## Build & Test

### Standalone Test

```bash
# Build test binary
cd applications/opencv/kcf/rvv-patches/get-feature-maps

clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu \
    --sysroot=$SYSROOT \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v_intrinsic -D__riscv_v_fixed_vlen=512 \
    -I. test.cpp -o test -lm

# Run under QEMU
qemu-riscv64 -L $SYSROOT -cpu rv64,v=true,vlen=512 ./test
```

### Integration Test

```bash
# Apply patch and rebuild KCF tracker
cd applications/opencv/kcf
./build.sh --rvv  # (requires build.sh modification to enable RVV)
```

## Known Limitations

1. **Channel selection logic**: The max-magnitude channel selection in `getFeatureMaps` has conditional update semantics that don't map cleanly to RVV. Current implementation keeps this scalar.

2. **Orientation binning**: The dot-product based orientation binning loop is kept scalar due to the `-dotProd > max` conditional branch.

3. **OpenCV RVV compatibility**: LLVM 22 has RVV intrinsics compatibility issues with OpenCV's existing RVV code. This implementation uses standard intrinsics that should work.