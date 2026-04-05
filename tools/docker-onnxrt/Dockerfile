# Stage 1: Build ONNX Runtime shared library
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS onnxrt-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ ninja-build python3 git ca-certificates make \
    && rm -rf /var/lib/apt/lists/

# Source trees are pre-cloned on the host by build.sh
COPY tools/docker-onnxrt/vendor/onnxruntime/ /onnxruntime/
COPY tools/docker-onnxrt/vendor/eigen/ /eigen/

# Build ONNX Runtime as a shared library so CMake resolves all transitive deps.
RUN --mount=type=cache,target=/onnxruntime/build \
    cd /onnxruntime \
    && bash build.sh \
        --config Release \
        --minimal_build \
        --build_shared_lib \
        --disable_rtti \
        --allow_running_as_root \
        --parallel 8 \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen \
            CMAKE_CXX_FLAGS="-Wno-stringop-overflow" \
    && mkdir -p /onnxruntime/build-output \
    && cp /onnxruntime/build/Linux/Release/libonnxruntime.so* /onnxruntime/build-output/ \
    && cp /onnxruntime/build/Linux/Release/onnxruntime_config.h /onnxruntime/build-output/ \
    && cp -r /onnxruntime/build/Linux/Release/onnx /onnxruntime/build-output/

# Stage 2: Compile YOLO runner
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS runner-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/

# Copy ONNX Runtime headers and shared library
COPY --from=onnxrt-build \
    /onnxruntime/include/ /onnxruntime/include/
COPY --from=onnxrt-build \
    /onnxruntime/build-output/ /onnxruntime/build-output/

# Copy runner source (stb_image.h + yolo_runner.cpp)
COPY tools/yolo_runner/ /runner/

WORKDIR /runner

RUN mkdir -p /out \
    && g++ -std=c++17 -O2 -g \
        -I/onnxruntime/include/onnxruntime/core/session \
        -I/onnxruntime/include \
        -I/onnxruntime/build-output \
        -I/runner \
        yolo_runner.cpp \
        -o /out/yolo_inference \
        -L/onnxruntime/build-output \
        -lonnxruntime \
        -Wl,-rpath,'$ORIGIN'

# Final stage: minimal image with binary and shared library
FROM --platform=riscv64 riscv64/ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 libc6 \
    && rm -rf /var/lib/apt/lists/

COPY --from=runner-build /out/yolo_inference /yolo_inference
COPY --from=onnxrt-build /onnxruntime/build-output/libonnxruntime.so* /usr/lib/riscv64-linux-gnu/

ENTRYPOINT ["/yolo_inference"]
