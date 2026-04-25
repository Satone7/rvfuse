# KCF Tracker Demo

Demonstration application for OpenCV's KCF (Kernelized Correlation Filters) tracker.

## Build

Requires OpenCV built for RISC-V (run `../build.sh` first).

```bash
# Compile demo with OpenCV
./build.sh
```

## Usage

```bash
# Run on test video
./output/kcf_demo test_video.mp4

# Profile with BBV
qemu-riscv64 -L output/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/kcf.bbv \
  ./output/kcf_demo test_video.mp4
```

## RVV Optimization Targets

Key hotspots identified for vectorization:

| Function | Operation | RVV Potential |
|----------|-----------|---------------|
| `getSubWindow` | Patch extraction | Memory copy |
| `extractFeatures` | HOG computation | Vector arithmetic |
| `gaussianCorrelation` | Kernel evaluation | FFT, vector math |
| `detectPeak` | Correlation peak | Vector search |

See `rvv-patches/` for optimization implementations.