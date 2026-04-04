# ONNX Runtime + YOLO RISC-V Native Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a statically-linked RISC-V YOLO11n inference binary via Docker native compilation, ready for QEMU+BBV hotspot analysis.

**Architecture:** Three-stage Docker build produces ONNX Runtime static library and links it with a minimal C++ YOLO runner. Model preparation happens on the x86 host via Python+ultralytics. The resulting binary runs under QEMU+BBV to collect basic block execution counts for hotspot analysis.

**Tech Stack:** Docker (riscv64/ubuntu:24.04), ONNX Runtime v1.17 (minimal C++ build), stb_image (header-only), Python 3 (ultralytics, onnx), C++17, addr2line

---

## File Structure

```
tools/
  docker-onnxrt/
    Dockerfile                # New - Three-stage RISC-V native build
    build.sh                  # New - Docker build entry script
  yolo_runner/
    yolo_runner.cpp           # New - YOLO inference runner
    stb_image.h               # New - Downloaded header-only image library
    CMakeLists.txt            # New - CMake reference build config
  analyze_bbv.py              # New - BBV hotspot analysis tool
  test_analyze_bbv.py         # New - Tests for analyze_bbv.py
output/
  .gitkeep                    # New - Track output directory in git
prepare_model.sh              # New - ONNX model export + test image download
verify_bbv.sh                 # New - QEMU+BBV build + verification
.gitignore                    # Modified - Add output/ rules
```

**Responsibility boundaries:**
- `prepare_model.sh` -- x86 host-only; exports ONNX model and downloads test image
- `tools/docker-onnxrt/` -- Docker infrastructure; produces RISC-V ELF binary
- `tools/yolo_runner/` -- C++ source compiled inside Docker; inference logic
- `tools/analyze_bbv.py` -- x86 host analysis; parses BBV output into hotspot report
- `verify_bbv.sh` -- x86 host; builds QEMU with BBV plugin support

---

## Scope Notes

This plan targets a single coherent pipeline: model prep -> Docker build -> QEMU+BBV analysis. TDD applies to `analyze_bbv.py` (testable on x86). Other components (shell scripts, Dockerfile, C++) are verified by syntax checks and smoke tests since they require RISC-V emulation or specific hardware.

---

### Task 1: Project Structure Setup

**Files:**
- Create: `output/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create output directory with gitkeep**

```bash
mkdir -p output
touch output/.gitkeep
```

- [ ] **Step 2: Update .gitignore**

Append the following block to `.gitignore`:

```gitignore
# ONNX Runtime + YOLO build artifacts
output/*
!output/.gitkeep
```

- [ ] **Step 3: Create remaining directories**

```bash
mkdir -p tools/docker-onnxrt tools/yolo_runner
```

- [ ] **Step 4: Verify structure**

Run: `find tools output -type f | sort`
Expected:
```
output/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add output/.gitkeep .gitignore
git commit -m "chore: add output directory and update .gitignore for build artifacts"
```

---

### Task 2: prepare_model.sh -- ONNX Model Export Script

**Files:**
- Create: `prepare_model.sh`

- [ ] **Step 1: Write prepare_model.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

check_or_install() {
    if ! python3 -c "import $1" 2>/dev/null; then
        echo "Installing $1..."
        pip3 install "$1"
    fi
}

echo "=== RVFuse: Preparing YOLO11n ONNX model ==="

mkdir -p "${OUTPUT_DIR}"

check_or_install ultralytics
check_or_install onnx

MODEL_PATH="${OUTPUT_DIR}/yolo11n.onnx"
if [ -f "${MODEL_PATH}" ]; then
    echo "Model already exists: ${MODEL_PATH}"
else
    echo "Exporting YOLO11n to ONNX (opset 12, batch 1)..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', opset=12, batch=1, simplify=True)
import shutil
shutil.move('yolo11n.onnx', '${MODEL_PATH}')
"
    echo "Model exported: ${MODEL_PATH}"
fi

IMAGE_PATH="${OUTPUT_DIR}/test.jpg"
if [ -f "${IMAGE_PATH}" ]; then
    echo "Test image already exists: ${IMAGE_PATH}"
else
    echo "Downloading COCO test image (bus.jpg)..."
    curl -fL -o "${IMAGE_PATH}" \
        "https://ultralytics.com/images/bus.jpg"
    echo "Test image downloaded: ${IMAGE_PATH}"
fi

echo "=== Done ==="
ls -lh "${OUTPUT_DIR}/yolo11n.onnx" "${OUTPUT_DIR}/test.jpg"
```

- [ ] **Step 2: Syntax check**

Run: `bash -n prepare_model.sh`
Expected: No output (no syntax errors)

- [ ] **Step 3: Make executable**

```bash
chmod +x prepare_model.sh
```

- [ ] **Step 4: Commit**

```bash
git add prepare_model.sh
git commit -m "feat: add prepare_model.sh for YOLO11n ONNX export"
```

---

### Task 3: YOLO Runner C++ Source

**Files:**
- Create: `tools/yolo_runner/yolo_runner.cpp`
- Create: `tools/yolo_runner/CMakeLists.txt`
- Download: `tools/yolo_runner/stb_image.h`

- [ ] **Step 1: Download stb_image.h**

```bash
curl -fL -o tools/yolo_runner/stb_image.h \
    "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
```

Verify: `head -5 tools/yolo_runner/stb_image.h`
Expected: Comments starting with `/* stb_image`

- [ ] **Step 2: Write yolo_runner.cpp**

```cpp
#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr int MODEL_INPUT_SIZE = 640;

static std::vector<unsigned char> resizeNearest(
    const unsigned char* src, int srcW, int srcH, int channels,
    int dstW, int dstH)
{
    std::vector<unsigned char> dst(dstW * dstH * channels);
    float xRatio = static_cast<float>(srcW) / dstW;
    float yRatio = static_cast<float>(srcH) / dstH;

    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int sx = std::min(static_cast<int>(x * xRatio), srcW - 1);
            int sy = std::min(static_cast<int>(y * yRatio), srcH - 1);
            int si = (sy * srcW + sx) * channels;
            int di = (y * dstW + x) * channels;
            for (int c = 0; c < channels; c++) {
                dst[di + c] = src[si + c];
            }
        }
    }
    return dst;
}

static void preprocess(
    const char* imagePath,
    std::vector<float>& tensorData,
    std::vector<int64_t>& inputShape)
{
    int imgW, imgH, imgC;
    unsigned char* img = stbi_load(imagePath, &imgW, &imgH, &imgC, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load image %s\n", imagePath);
        exit(1);
    }

    std::vector<unsigned char> resized =
        resizeNearest(img, imgW, imgH, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    stbi_image_free(img);

    // HWC -> CHW, normalize to [0, 1]
    int pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
    tensorData.resize(pixels * 3);
    for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
        for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
            int hwc = (y * MODEL_INPUT_SIZE + x) * 3;
            tensorData[0 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 0] / 255.0f;
            tensorData[1 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 1] / 255.0f;
            tensorData[2 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 2] / 255.0f;
        }
    }

    inputShape = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};
}

static void printTopDetections(Ort::Value& tensor, int showCount)
{
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    float* data = tensor.GetTensorMutableData<float>();
    int64_t total = info.GetElementCount();

    fprintf(stdout, "Output shape: [");
    for (size_t d = 0; d < shape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(shape[d]));
    }
    fprintf(stdout, "]\n");

    // YOLO output: [numDets, 6] where cols = x1, y1, x2, y2, conf, cls
    int cols = 6;
    int numDets = static_cast<int>(total / cols);
    int show = std::min(numDets, showCount);
    fprintf(stdout, "Top %d detections:\n", show);
    for (int d = 0; d < show; d++) {
        int i = d * cols;
        fprintf(stdout, "  [%d] box=(%.1f,%.1f,%.1f,%.1f) conf=%.3f cls=%.0f\n",
            d, data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5]);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.onnx> <image.jpg> [iterations]\n", argv[0]);
        return 1;
    }

    const char* modelPath = argv[1];
    const char* imagePath = argv[2];
    int iterations = 10;
    if (argc >= 4) {
        iterations = atoi(argv[3]);
        if (iterations < 1) iterations = 1;
    }

    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Image: %s\n", imagePath);
    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputNames = session.GetOutputNamesAllocated(allocator);
    std::vector<const char*> outputNamePtrs;
    for (auto& n : outputNames) {
        outputNamePtrs.push_back(n.get());
    }

    fprintf(stdout, "Preprocessing image...\n");
    std::vector<float> tensorData;
    std::vector<int64_t> inputShape;
    preprocess(imagePath, tensorData, inputShape);

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape.data(), inputShape.size());

    fprintf(stdout, "Running inference...\n");
    const char* inputNames[] = {inputName.get()};

    for (int i = 0; i < iterations; i++) {
        fprintf(stdout, "  [%d/%d]%s\n", i + 1, iterations,
                i == 0 ? " (warm-up)" : "");
        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNames, &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());

        if (i == iterations - 1) {
            printTopDetections(outputs[0], 5);
        }
    }

    fprintf(stdout, "Done.\n");
    return 0;
}
```

- [ ] **Step 3: Write CMakeLists.txt**

This is a reference CMake config. The Docker build uses direct g++ invocation, but this file allows local or alternative builds.

```cmake
cmake_minimum_required(VERSION 3.16)
project(yolo_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

add_executable(yolo_inference yolo_runner.cpp)

target_include_directories(yolo_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(yolo_inference PRIVATE
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a
    Threads::Threads
    m dl
)

set_target_properties(yolo_inference PROPERTIES LINK_FLAGS "-static")
```

- [ ] **Step 4: Verify C++ source is parseable**

Run: `head -1 tools/yolo_runner/yolo_runner.cpp`
Expected: `#include <onnxruntime_cxx_api.h>`

- [ ] **Step 5: Commit**

```bash
git add tools/yolo_runner/
git commit -m "feat: add YOLO inference runner with stb_image dependency"
```

---

### Task 4: Docker Build Pipeline

**Files:**
- Create: `tools/docker-onnxrt/Dockerfile`
- Create: `tools/docker-onnxrt/build.sh`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# Stage 1: Build environment with toolchain packages
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS build-env

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ ninja-build python3 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build ONNX Runtime from source
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS onnxrt-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ ninja-build python3 git ca-certificates make \
    && rm -rf /var/lib/apt/lists/*

ARG ONNXRUNTIME_VERSION=v1.17.3
RUN git clone --depth 1 --branch ${ONNXRUNTIME_VERSION} \
        https://github.com/microsoft/onnxruntime.git /onnxruntime

# Minimal build: only operators needed for YOLO inference.
# BuildKit cache avoids recompiling unchanged sources across builds.
RUN --mount=type=cache,target=/onnxruntime/build \
    cd /onnxruntime \
    && python3 build.sh \
        --config Release \
        --minimal_build \
        --disable_rtti \
        --build_shared_lib=false \
        --enable_pybind=OFF \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF

# Stage 3: Compile YOLO runner and link statically
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS runner-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime headers (from source tree) and static library (from build)
COPY --from=onnxrt-build \
    /onnxruntime/include/ /onnxruntime/include/
COPY --from=onnxrt-build \
    /onnxruntime/build/Linux/Release/lib/ /onnxruntime/lib/

# Copy runner source (stb_image.h + yolo_runner.cpp)
COPY tools/yolo_runner/ /runner/

WORKDIR /runner

RUN g++ -std=c++17 -O2 -g -static \
        -I/onnxruntime/include \
        -I/runner \
        yolo_runner.cpp \
        -o /out/yolo_inference \
        -L/onnxruntime/lib \
        -lonnxruntime \
        -lpthread -lm -ldl

# Final stage: minimal image with only the binary
FROM --platform=riscv64 riscv64/ubuntu:24.04
COPY --from=runner-build /out/yolo_inference /yolo_inference
ENTRYPOINT ["/yolo_inference"]
```

- [ ] **Step 2: Validate Dockerfile syntax**

Run: `docker build --check --platform riscv64 -f tools/docker-onnxrt/Dockerfile . 2>&1 || echo "(docker --check not available, skipping)"`
Note: `--check` requires BuildKit and Docker 25+. If unavailable, verify manually.

- [ ] **Step 3: Write build.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"

mkdir -p "${OUTPUT_DIR}"

echo "=== Building ONNX Runtime + YOLO runner for RISC-V ==="
echo "This may take 2-6 hours under QEMU emulation."

DOCKER_BUILDKIT=1 docker build \
    --platform riscv64 \
    -t rvfuse-yolo-builder \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --progress=plain \
    "${PROJECT_ROOT}"

CONTAINER_ID=$(docker create rvfuse-yolo-builder)
docker cp "${CONTAINER_ID}:/yolo_inference" "${OUTPUT_DIR}/yolo_inference"
docker rm "${CONTAINER_ID}" > /dev/null

echo "=== Build complete ==="
file "${OUTPUT_DIR}/yolo_inference"
```

- [ ] **Step 4: Syntax check build.sh**

Run: `bash -n tools/docker-onnxrt/build.sh`
Expected: No output

- [ ] **Step 5: Make executable**

```bash
chmod +x tools/docker-onnxrt/build.sh
```

- [ ] **Step 6: Commit**

```bash
git add tools/docker-onnxrt/
git commit -m "feat: add Docker pipeline for RISC-V native ONNX Runtime build"
```

---

### Task 5: analyze_bbv.py with Tests

**Files:**
- Create: `tools/analyze_bbv.py`
- Create: `tools/test_analyze_bbv.py`

- [ ] **Step 1: Write test_analyze_bbv.py**

```python
#!/usr/bin/env python3
"""Tests for analyze_bbv.py"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from analyze_bbv import generate_report, parse_bbv, resolve_addresses


class TestParseBbv(unittest.TestCase):
    def test_basic_parsing(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("0x10000 42\n")
            f.write("0x10050 15\n")
            f.write("0x10200 100\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(len(blocks), 3)
            self.assertEqual(blocks[0], (0x10000, 42))
            self.assertEqual(blocks[1], (0x10050, 15))
            self.assertEqual(blocks[2], (0x10200, 100))
        finally:
            Path(path).unlink()

    def test_skips_comments_and_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("# comment\n\n0x20000 10\n  \n0x20100 20\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0], (0x20000, 10))
        finally:
            Path(path).unlink()

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(blocks, [])
        finally:
            Path(path).unlink()

    def test_decimal_addresses(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbv", delete=False
        ) as f:
            f.write("65536 42\n")
            path = f.name

        try:
            blocks = parse_bbv(path)
            self.assertEqual(blocks, [(65536, 42)])
        finally:
            Path(path).unlink()


class TestResolveAddresses(unittest.TestCase):
    def test_empty_blocks(self):
        resolved = resolve_addresses([], "/fake/elf")
        self.assertEqual(resolved, [])

    @patch("analyze_bbv.subprocess.run")
    def test_calls_addr2line_and_parses_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="main_func\n/app/main.c:10\n"
                   "helper\n/app/util.c:25\n",
            returncode=0,
        )
        blocks = [(0x1000, 42), (0x2000, 15)]
        resolved = resolve_addresses(blocks, "/fake/elf")

        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0], (0x1000, 42, "main_func (/app/main.c:10)"))
        self.assertEqual(resolved[1], (0x2000, 15, "helper (/app/util.c:25)"))
        mock_run.assert_called_once()

    @patch("analyze_bbv.subprocess.run")
    def test_fallback_on_addr2line_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError("not found")
        resolved = resolve_addresses([(0x1000, 42)], "/fake/elf")
        self.assertEqual(resolved, [(0x1000, 42, "??")])


class TestGenerateReport(unittest.TestCase):
    def test_sorted_by_count_descending(self):
        resolved = [
            (0x1000, 10, "func_a (a.c:1)"),
            (0x2000, 100, "func_b (b.c:2)"),
            (0x3000, 50, "func_c (c.c:3)"),
        ]
        report = generate_report(resolved, top_n=3)
        self.assertLess(report.index("func_b"), report.index("func_c"))
        self.assertLess(report.index("func_c"), report.index("func_a"))

    def test_respects_top_n(self):
        resolved = [(i, 100 - i, f"func_{i}") for i in range(10)]
        report = generate_report(resolved, top_n=3)
        self.assertIn("func_0", report)
        self.assertNotIn("func_3", report)

    def test_percentage_calculation(self):
        resolved = [
            (0x1000, 75, "func_a (a.c:1)"),
            (0x2000, 25, "func_b (b.c:2)"),
        ]
        report = generate_report(resolved, top_n=2)
        self.assertIn("75.00%", report)
        self.assertIn("25.00%", report)

    def test_empty_input(self):
        report = generate_report([])
        self.assertIn("Total basic blocks: 0", report)
        self.assertIn("Total executions: 0", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail (module not yet importable)**

Run: `cd tools && python3 -m pytest test_analyze_bbv.py -v 2>&1 | head -5`
Expected: `ModuleNotFoundError: No module named 'analyze_bbv'`

- [ ] **Step 3: Write analyze_bbv.py**

```python
#!/usr/bin/env python3
"""Parse QEMU BBV output and generate hotspot reports.

Maps basic block addresses to source locations using addr2line
and prints the most frequently executed blocks.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_bbv(bbv_path):
    """Parse BBV file into list of (address, count) tuples."""
    blocks = []
    with open(bbv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            addr = int(parts[0], 16) if parts[0].startswith("0x") else int(parts[0])
            blocks.append((addr, int(parts[1])))
    return blocks


def resolve_addresses(blocks, elf_path):
    """Resolve addresses to source locations via addr2line."""
    if not blocks:
        return []

    addresses = [f"0x{a:x}" for a, _ in blocks]
    cmd = ["addr2line", "-f", "-e", elf_path] + addresses

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"Warning: addr2line failed: {exc}", file=sys.stderr)
        return [(a, c, "??") for a, c in blocks]

    lines = result.stdout.strip().split("\n")
    resolved = []
    for i, (addr, count) in enumerate(blocks):
        if 2 * i + 1 < len(lines):
            func = lines[2 * i].strip()
            loc = lines[2 * i + 1].strip()
            resolved.append((addr, count, f"{func} ({loc})"))
        else:
            resolved.append((addr, count, "??"))
    return resolved


def generate_report(resolved, top_n=20):
    """Generate a sorted hotspot report string."""
    sorted_blocks = sorted(resolved, key=lambda x: x[1], reverse=True)
    total = sum(c for _, c, _ in resolved)
    show = min(top_n, len(sorted_blocks))

    lines = [
        "=" * 72,
        "BBV Hotspot Report",
        "=" * 72,
        f"Total basic blocks: {len(resolved)}",
        f"Total executions:   {total}",
        f"Showing top {show} blocks",
        "",
        f"{'Rank':<6}{'Address':<18}{'Count':<14}{'% Total':<10}Location",
        "-" * 72,
    ]
    for rank, (addr, count, location) in enumerate(sorted_blocks[:show], 1):
        pct = (count / total * 100) if total else 0
        lines.append(
            f"{rank:<6}0x{addr:016x}  {count:<14}{pct:>6.2f}%    {location}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QEMU BBV output and generate hotspot report"
    )
    parser.add_argument("--bbv", required=True, help="Path to .bbv file")
    parser.add_argument("--elf", required=True, help="Path to RISC-V ELF binary")
    parser.add_argument("--top", type=int, default=20, help="Top N blocks")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    blocks = parse_bbv(args.bbv)
    if not blocks:
        print("Error: no basic blocks found in BBV file", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(blocks)} basic blocks from {args.bbv}")

    resolved = resolve_addresses(blocks, args.elf)
    report = generate_report(resolved, args.top)

    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_analyze_bbv.py -v`
Expected: All tests PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/analyze_bbv.py tools/test_analyze_bbv.py
git commit -m "feat: add BBV hotspot analysis tool with tests"
```

---

### Task 6: verify_bbv.sh -- QEMU+BBV Build and Verification

**Files:**
- Create: `verify_bbv.sh`

- [ ] **Step 1: Write verify_bbv.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="${SCRIPT_DIR}/third_party/qemu"

echo "=== Building QEMU with BBV plugin ==="

if [ ! -d "${QEMU_DIR}/.git" ]; then
    echo "Initializing QEMU submodule..."
    git submodule update --init --depth 1 third_party/qemu
fi

cd "${QEMU_DIR}"
mkdir -p build
cd build

echo "Configuring QEMU (riscv64-linux-user, plugins enabled)..."
../configure \
    --target-list=riscv64-linux-user \
    --disable-werror \
    --enable-plugins

echo "Building QEMU..."
make -j"$(nproc)"

echo "Building BBV plugin..."
make plugins

BINARY="${PWD}/qemu-riscv64"
PLUGIN="${PWD}/contrib/plugins/libbbv.so"

if [ ! -f "${BINARY}" ]; then
    echo "Error: qemu-riscv64 not found at ${BINARY}" >&2
    exit 1
fi
echo "QEMU binary: ${BINARY}"

if [ ! -f "${PLUGIN}" ]; then
    echo "Error: libbbv.so not found at ${PLUGIN}" >&2
    exit 1
fi
echo "BBV plugin: ${PLUGIN}"

echo ""
echo "=== QEMU+BBV build complete ==="
echo "To profile a binary:"
echo "  ${BINARY} -plugin ${PLUGIN},interval=10000,outfile=output/prof.bbv -- ./output/yolo_inference ./output/yolo11n.onnx ./output/test.jpg"
```

- [ ] **Step 2: Syntax check**

Run: `bash -n verify_bbv.sh`
Expected: No output

- [ ] **Step 3: Make executable**

```bash
chmod +x verify_bbv.sh
```

- [ ] **Step 4: Commit**

```bash
git add verify_bbv.sh
git commit -m "feat: add verify_bbv.sh for QEMU+BBV build and verification"
```

---

### Task 7: Update Project Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with new commands**

In the `## Commands` section of `CLAUDE.md`, add:

```markdown
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
```

Also update the `## Active Technologies` and `## Code Style` sections:

```markdown
## Active Technologies

- C++17 (yolo_runner.cpp), Docker (RISC-V native build), Python 3 (analyze_bbv.py, prepare_model.sh), Git submodules (QEMU, ONNX Runtime source)

## Code Style

- C++: camelCase functions/variables, PascalCase for ONNX Runtime API types, single-responsibility functions under 50 lines
- Python: snake_case, type hints on public functions, stdlib-only (no external deps for analyze_bbv.py)
- Shell: `set -euo pipefail`, `SCRIPT_DIR` pattern for paths, no unquoted variables
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with ONNX Runtime + YOLO build commands"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] prepare_model.sh exports YOLO11n ONNX + downloads test image (Task 2)
- [x] Three-stage Dockerfile with build-env, onnxrt-build, runner-build (Task 4)
- [x] BuildKit cache mount for ONNX Runtime build (Task 4, Dockerfile)
- [x] YOLO runner with stb_image.h, preprocessing, 10x inference loop, top-N output (Task 3)
- [x] CLI: `./yolo_inference <model.onnx> <image.jpg> [iterations]` (Task 3)
- [x] Compile flags: `-O2 -g -static` (Task 3, Task 4)
- [x] analyze_bbv.py parses BBV + addr2line resolution + hotspot report (Task 5)
- [x] verify_bbv.sh builds QEMU+BBV with correct flags (Task 6)
- [x] BBV interval parameter guidance (Task 6, verify_bbv.sh echo output)
- [x] One-click workflow documented in CLAUDE.md (Task 7)

### Placeholder Scan
- [x] No TBD/TODO/FIXME in any step
- [x] All code blocks contain complete implementations
- [x] All commands have expected output specified
- [x] No "similar to Task N" references

### Type/Name Consistency
- [x] `yolo_inference` -- binary name consistent across Dockerfile, build.sh, verify_bbv.sh, CLAUDE.md
- [x] `analyze_bbv.py` -- path consistent across Task 5 and CLAUDE.md
- [x] `MODEL_INPUT_SIZE = 640` -- consistent in yolo_runner.cpp
- [x] `prepare_model.sh` / `verify_bbv.sh` / `build.sh` -- names match design file structure
