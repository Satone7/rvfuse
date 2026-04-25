# OpenCV Cross-Compilation for RISC-V

OpenCV 4.10.0 with RISC-V RVV vector extension support.

## Application Profile

| Field | Value |
|-------|-------|
| Name | opencv |
| Source | https://github.com/opencv/opencv.git |
| Version | **4.10.0** (commit `71d3237a093b60a27601c20e9ee6c3e52154e8b1`) |
| Extra source | opencv_contrib 4.10.0 (tracking module) |
| Build system | CMake |
| Target arch | rv64gcv_zvl256b |
| ABI | lp64d |
| Extra sysroot packages | libgomp-dev (OpenMP) |
| Smoke test binary | opencv_version |
| LLVM | 22.1.3 |

> **Note**: OpenCV has native RVV support via `CPU_BASELINE=RVV` option. The `zvl256b` extension specifies VLEN=256-bit vector registers.

## Build Status

| Component | Status |
|-----------|--------|
| libopencv_core.so | ✓ RISC-V ELF (rv64gcv) |
| libopencv_imgproc.so | ✓ RISC-V ELF (rv64gcv) |
| libopencv_tracking.so | ✓ RISC-V ELF (KCF tracker) |
| opencv_version | ✓ RISC-V ELF (smoke test passed) |

## Directory Structure

```
opencv/
├── build.sh                  # Cross-compilation script
├── riscv64-linux-toolchain.cmake  # CMake toolchain
├── README.md                 # This file
├── .gitignore                # Build artifact exclusions
├── vendor/                   # Source code (populated by build.sh)
│   ├── opencv/               # OpenCV 4.10.0
│   └── opencv_contrib/       # opencv_contrib 4.10.0
├── rvv-patches/              # RVV optimization patches
├── kcf/                      # KCF tracker demo
└── benchmark/                # Performance benchmark scripts
```

## Build

```bash
# Full build (first time)
./build.sh

# Force rebuild
./build.sh --force

# Skip sysroot extraction (use existing)
./build.sh --skip-sysroot

# Parallel build with 8 jobs
./build.sh -j 8
```

### Build Options

The build script configures OpenCV with minimal modules for KCF tracking:

| Option | Value | Purpose |
|--------|-------|---------|
| BUILD_LIST | core,imgproc,imgcodecs,video,videoio,highgui,tracking | Minimal build |
| CPU_BASELINE | RVV | Enable RVV optimizations |
| RISCV_RVV_SCALABLE | ON | Use scalable RVV API |
| BUILD_ZLIB/PNG/JPEG | ON | Build 3rdparty libs (no sysroot deps) |
| WITH_FFMPEG/GSTREAMER/V4L | OFF | Disable video I/O (simplify deps) |

## Output Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Core library | `output/opencv/lib/libopencv_core.so` | Core OpenCV functions |
| Image processing | `output/opencv/lib/libopencv_imgproc.so` | Image operations |
| Tracking module | `output/opencv/lib/libopencv_tracking.so` | KCF and other trackers |
| Headers | `output/opencv/include/opencv2/` | Include files |
| Version binary | `output/opencv/bin/opencv_version` | Smoke test |
| Sysroot | `output/opencv/sysroot/` | RISC-V runtime libs |

## Run with QEMU

```bash
# Smoke test (opencv_version)
third_party/qemu/build/qemu-riscv64 \
    -L output/opencv/sysroot \
    -E LD_LIBRARY_PATH=output/opencv/lib \
    -cpu rv64,v=true,vlen=256 \
    output/opencv/bin/opencv_version

# Expected output: 4.10.0
```

**VLEN Warning**: The binary is compiled with `zvl256b`, so QEMU must use `-cpu rv64,v=true,vlen=256`. Default QEMU (VLEN=128) would cause `Illegal instruction` errors.

## KCF Tracker

The KCF (Kernelized Correlation Filters) tracker is in the `tracking` module from opencv_contrib.

```cpp
#include <opencv2/tracking.hpp>

cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
tracker->init(frame, roi);
tracker->update(frame, roi);
```

See `kcf/` directory for demo application.

## RVV Optimization

OpenCV has native RVV support in core modules. Key RVV-optimized functions:

| Function | Module | RVV Status |
|----------|--------|------------|
| Arithmetic operations | core | ✓ RVV intrinsics |
| Image filtering | imgproc | ✓ RVV paths |
| Color conversion | imgproc | ✓ RVV paths |
| Geometric transforms | imgproc | ✓ RVV paths |
| KCF correlation | tracking | Scalar (needs optimization) |

## Version

| Component | Version | Commit |
|-----------|---------|--------|
| OpenCV | **4.10.0** | `71d3237a093b60a27601c20e9ee6c3e52154e8b1` |
| opencv_contrib | **4.10.0** | `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` |
| LLVM | 22.1.3 | - |
| Target | rv64gcv_zvl256b | - |

> **Do not upgrade** OpenCV version without verifying RVV intrinsics compatibility in `modules/core/include/opencv2/core/hal/intrin_rvv*.hpp`.

## References

- [OpenCV](https://github.com/opencv/opencv) - Main repository
- [opencv_contrib](https://github.com/opencv/opencv_contrib) - Extra modules (tracking)
- [KCF Paper](https://arxiv.org/abs/1404.7584) - High-Speed Tracking with Kernelized Correlation Filters