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
        --skip_submodule_sync `# submodules not needed for minimal build` \
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

# Stage 2: Compile YOLO runner + test programs
FROM --platform=riscv64 riscv64/ubuntu:24.04 AS runner-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make ca-certificates \
        libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/

# Copy ONNX Runtime headers and shared library
COPY --from=onnxrt-build \
    /onnxruntime/include/ /onnxruntime/include/
COPY --from=onnxrt-build \
    /onnxruntime/build-output/ /onnxruntime/build-output/

# Copy runner source (stb_image.h + yolo_runner.cpp + test_*.cpp)
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

# Test: preprocess (FFmpeg only, no ORT)
RUN g++ -std=c++17 -O2 -g \
        yolo_preprocess.cpp \
        -o /out/yolo_preprocess \
        -lavcodec -lavformat -lswscale -lavutil

# Test: postprocess (stb_image only, no ORT/FFmpeg)
RUN g++ -std=c++17 -O2 -g \
        -I/runner \
        yolo_postprocess.cpp \
        -o /out/yolo_postprocess

# Final stage: minimal image with binaries and shared libraries
FROM --platform=riscv64 riscv64/ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 libc6 \
        libavcodec60 libavformat60 libswscale7 libavutil58 \
    && rm -rf /var/lib/apt/lists/

COPY --from=runner-build /out/yolo_inference /yolo_inference
COPY --from=runner-build /out/yolo_preprocess /yolo_preprocess
COPY --from=runner-build /out/yolo_postprocess /yolo_postprocess
COPY --from=onnxrt-build /onnxruntime/build-output/libonnxruntime.so* /usr/lib/riscv64-linux-gnu/
