# =============================================================================
# Multi-stage Dockerfile for ONNX Runtime RISC-V cross-compilation
# Build context: project root (set via build.sh)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base - Ubuntu 22.04 with RISC-V cross-compiler and sysroot
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-pip \
        wget \
        riscv64-linux-gnu-gcc \
        riscv64-linux-gnu-g++ \
        riscv64-linux-gnu-binutils \
    && rm -rf /var/lib/apt/lists/*

# Create RISC-V sysroot and populate with system libraries and headers
RUN mkdir -p /opt/riscv-sysroot/lib /opt/riscv-sysroot/include \
    && cp -a /usr/riscv64-linux-gnu/lib/* /opt/riscv-sysroot/lib/ \
    && cp -a /usr/riscv64-linux-gnu/include/* /opt/riscv-sysroot/include/

# ---------------------------------------------------------------------------
# Stage 2: Protobuf Builder - cross-compile protobuf for RISC-V
# ---------------------------------------------------------------------------
FROM base AS protobuf-builder

WORKDIR /build

# Copy toolchain file for CMake cross-compilation
COPY tools/docker-onnxrt/toolchain-riscv64.cmake /build/

# Clone protobuf v3.21.12
RUN git clone --depth 1 --branch v3.21.12 https://github.com/protocolbuffers/protobuf.git

WORKDIR /build/protobuf

# CMake configure: cross-compile with toolchain, Release, static, no tests/examples/zlib
RUN cmake -B build \
        -DCMAKE_TOOLCHAIN_FILE=/build/toolchain-riscv64.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/riscv-sysroot \
        -DBUILD_SHARED_LIBS=OFF \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_EXAMPLES=OFF \
        -Dprotobuf_WITH_ZLIB=OFF \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build

# ---------------------------------------------------------------------------
# Stage 3: ONNX Runtime Builder - minimal static build for RISC-V
# ---------------------------------------------------------------------------
FROM protobuf-builder AS onnxrt-builder

WORKDIR /build

# Clone onnxruntime v1.17.0
RUN git clone --depth 1 --branch v1.17.0 https://github.com/microsoft/onnxruntime.git

WORKDIR /build/onnxruntime

# Build ONNX Runtime: Release, static, minimal, no RTTI, RISC-V target
RUN python3 build.py \
        --config Release \
        --build_shared_lib=OFF \
        --minimal_build=ON \
        --disable_rtti \
        --target_triple riscv64-linux-gnu \
        --cmake_extra_defines \
            CMAKE_TOOLCHAIN_FILE=/build/toolchain-riscv64.cmake \
            CMAKE_SYSROOT=/opt/riscv-sysroot \
    && ls -lh build/Linux/Release/libonnxruntime.a

# ---------------------------------------------------------------------------
# Stage 4: Runner Builder - compile YOLO inference runner
# ---------------------------------------------------------------------------
FROM onnxrt-builder AS runner-builder

WORKDIR /build/yolo_runner

# Copy runner source files from project (build context = project root)
COPY tools/yolo_runner/yolo_runner.cpp /build/yolo_runner/
COPY tools/yolo_runner/stb_image.h /build/yolo_runner/
COPY tools/yolo_runner/CMakeLists.txt /build/yolo_runner/

# Set ONNX Runtime library and include paths
ENV ONNXRUNTIME_LIB=/build/onnxruntime/build/Linux/Release/libonnxruntime.a
ENV ONNXRUNTIME_INCLUDE=/build/onnxruntime/include/onnxruntime/core/session

# CMake configure with toolchain and sysroot
RUN cmake -B build \
        -DCMAKE_TOOLCHAIN_FILE=/build/toolchain-riscv64.cmake \
        -DCMAKE_SYSROOT=/opt/riscv-sysroot \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_LIB=${ONNXRUNTIME_LIB} \
        -DONNXRUNTIME_INCLUDE=${ONNXRUNTIME_INCLUDE} \
    && cmake --build build --parallel $(nproc)

# Verify: debug info present and file is RISC-V ELF
RUN riscv64-linux-gnu-readelf -S build/yolo_inference | grep -q .debug_info \
    && file build/yolo_inference | grep -q "ELF 64-bit LSB executable, UCB RISC-V"

# ---------------------------------------------------------------------------
# Stage 5: Artifacts - collect executable, model, and test image
# ---------------------------------------------------------------------------
FROM runner-builder AS artifacts

WORKDIR /artifacts

# Copy the built executable
COPY --from=runner-builder /build/yolo_runner/build/yolo_inference /artifacts/

# Download YOLO11n ONNX model from ultralytics assets (v8.3.0)
RUN wget -q -O /artifacts/yolo11n.onnx \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx

# Download test image (COCO bus.jpg sample from ultralytics)
RUN wget -q -O /artifacts/test_image.jpg \
        https://ultralytics.com/images/bus.jpg

# Verify all artifacts are present and non-empty
RUN ls -lh /artifacts/yolo_inference \
    && ls -lh /artifacts/yolo11n.onnx \
    && ls -lh /artifacts/test_image.jpg \
    && test -s /artifacts/yolo_inference \
    && test -s /artifacts/yolo11n.onnx \
    && test -s /artifacts/test_image.jpg

# ---------------------------------------------------------------------------
# Stage 6: Export - minimal scratch image with artifacts only
# ---------------------------------------------------------------------------
FROM scratch AS export

COPY --from=artifacts /artifacts/yolo_inference /yolo_inference
COPY --from=artifacts /artifacts/yolo11n.onnx /yolo11n.onnx
COPY --from=artifacts /artifacts/test_image.jpg /test_image.jpg
