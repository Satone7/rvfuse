# ONNX Runtime v1.24.4 RISC-V Native Build
# Uses LLVM 18 (Ubuntu 24.04 default) targeting riscv64gcv
#
# Build output stays in this image (tag: rvfuse-onnxrt-1.24.4)
# No extraction to host - use docker cp or docker run to access artifacts

FROM --platform=riscv64 riscv64/ubuntu:24.04 AS onnxrt-build

# Install LLVM 18 + build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang lld llvm cmake ninja-build python3 git ca-certificates make \
    && rm -rf /var/lib/apt/lists/

# Configure compiler environment
ENV CC=clang
ENV CXX=clang++
ENV LDFLAGS="-fuse-ld=lld"

# Copy pre-cloned sources (managed by build.sh on host)
COPY tools/docker-onnxrt-1.24.4/vendor/onnxruntime/ /onnxruntime/
COPY tools/docker-onnxrt-1.24.4/vendor/eigen/ /eigen/

# Build ONNX Runtime v1.24.4 (FULL build, not minimal)
# Target: riscv64gcv (G: IMAFD base, C: Compressed, V: Vector)
RUN --mount=type=cache,target=/onnxruntime/build \
    cd /onnxruntime \
    && bash build.sh \
        --config Release \
        --build_shared_lib \
        --parallel 8 \
        --allow_running_as_root \
        --skip_submodule_sync \
        --skip_tests \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
            CMAKE_C_FLAGS="-march=riscv64gcv -mtune=riscv64" \
            CMAKE_CXX_FLAGS="-march=riscv64gcv -mtune=riscv64"

# Organize build output for easy access
RUN mkdir -p /onnxruntime/build-output \
    && cp /onnxruntime/build/Linux/Release/libonnxruntime.so* \
       /onnxruntime/build-output/ \
    && cp /onnxruntime/build/Linux/Release/onnxruntime_config.h \
       /onnxruntime/build-output/ \
    && cp -r /onnxruntime/include /onnxruntime/build-output/ 2>/dev/null || true \
    && echo "Build output organized in /onnxruntime/build-output/"