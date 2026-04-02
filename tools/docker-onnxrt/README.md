# ONNX Runtime RISC-V Cross-Compilation Environment

This directory contains the Docker-based build environment for cross-compiling
ONNX Runtime and YOLO inference runner for RISC-V 64-bit Linux.

## Prerequisites

- Docker (19.03+)
- ~20GB disk space (build artifacts and intermediate images)
- Internet connection for downloading ONNX Runtime source and dependencies

## Quick Start

```bash
./build.sh
```

This builds the complete environment and extracts artifacts to `output/`:
- `yolo_inference`: RISC-V ELF with DWARF debug info
- `yolo11n.onnx`: YOLO11n model file
- `test_image.jpg`: Test image (COCO bus.jpg sample)

## Build Architecture

Multi-stage Docker build:

| Stage | Purpose |
|-------|---------|
| base | Ubuntu 22.04 + RISC-V cross-compiler |
| protobuf-builder | Cross-compile protobuf for RISC-V |
| onnxrt-builder | ONNX Runtime minimal static build |
| runner-builder | YOLO inference runner compilation |
| artifacts | Download model and test image |
| export | Final artifact extraction |

## Build Configuration

ONNX Runtime is built with:
- Minimal build (no training, no MLAS)
- CPU Execution Provider only
- Static linking (`libonnxruntime.a`)
- DWARF debug info (`-O2 -g`)
- Target: `riscv64-linux-gnu`

## Troubleshooting

### Build takes too long
ONNX Runtime build can take 30-60 minutes. Use `--parallel $(nproc)` for faster builds.

### Protobuf build fails
Ensure sysroot has correct RISC-V libraries. Check `/opt/riscv-sysroot/lib/` exists.

### Missing debug info
Verify with: `riscv64-linux-gnu-readelf --debug-dump=info output/yolo_inference`

## File Reference

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build definition |
| `build.sh` | Build entrypoint script |
| `toolchain-riscv64.cmake` | CMake toolchain for RISC-V |
| `README.md` | This documentation |
