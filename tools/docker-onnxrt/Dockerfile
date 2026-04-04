# Stage 1: Build ONNX Runtime from source
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

# Stage 2: Compile YOLO runner and link statically
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

RUN mkdir -p /out && g++ -std=c++17 -O2 -g -static \
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
