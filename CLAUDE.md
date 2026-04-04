# RVFuse Development Guidelines

Auto-generated from all feature designs. Last updated: 2026-03-31

## Active Technologies

- C++17 (yolo_runner.cpp), Docker (RISC-V native build), Python 3 (analyze_bbv.py, prepare_model.sh), Git submodules (QEMU, ONNX Runtime source)

## Project Structure

```text
[ACTUAL STRUCTURE FROM DESIGNS]
```

## Commands

```bash
# Export YOLO11n ONNX model and download test image
./prepare_model.sh

# Build QEMU with BBV plugin support (first time only)
./verify_bbv.sh

# Docker build ONNX Runtime + YOLO runner for RISC-V
./tools/docker-onnxrt/build.sh

# Run BBV profiling on the YOLO binary
qemu-riscv64 -plugin third_party/qemu/build/contrib/plugins/libbbv.so,interval=10000,outfile=output/yolo.bbv \
  ./output/yolo_inference ./output/yolo11n.onnx ./output/test.jpg

# Generate hotspot report from BBV data
python3 tools/analyze_bbv.py --bbv output/yolo.bbv --elf output/yolo_inference

# Run analyze_bbv.py tests
cd tools && python3 -m pytest test_analyze_bbv.py -v
```

## Code Style

- C++: camelCase functions/variables, PascalCase for ONNX Runtime API types, single-responsibility functions under 50 lines
- Python: snake_case, type hints on public functions, stdlib-only (no external deps for analyze_bbv.py)
- Shell: `set -euo pipefail`, `SCRIPT_DIR` pattern for paths, no unquoted variables

## Recent Changes

- 001-riscv-fusion-setup: Added N/A (documentation-only phase) + Git (for submodule integration), Markdown rendering
- ONNX Runtime + YOLO native build: Added Docker pipeline, YOLO runner (C++), BBV analysis tool (Python), model preparation and QEMU verification scripts

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
